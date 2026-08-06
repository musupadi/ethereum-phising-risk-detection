"""
ml/fetch_etherscan_edges.py
---------------------------
Fetch transaction edges between hybrid-dataset addresses using Etherscan txlist.

For each address in hybrid_dataset.csv, this script queries Etherscan for normal
transactions, internal transactions, and ERC-20 token transfers, then records
every counterparty that is ALSO in the hybrid dataset (12,815 nodes at 1:10). The resulting
edge list supplements the sparse MulDiGraph edges used by GNN-GraphSAGE V2.

Output:
  data/raw/etherscan_edges.csv       — final edge list (src, dst)
  data/raw/etherscan_edges.partial.csv — checkpoint (auto-deleted when done)

Usage:
  python ml/fetch_etherscan_edges.py

Environment variables (.env):
  ETHERSCAN_API_KEY         — required
  ETHERSCAN_REQUEST_DELAY   — seconds between requests (default 0.4)
  ETHERSCAN_PER_RUN_LIMIT   — max addresses per run, 0 = all (default 0)
  ETHERSCAN_MAX_PAGES       — max pages per source per address (default 3, 1 page = 10k recs)
  ETHERSCAN_INCLUDE_INTERNAL — include internal transactions (default 1)
  ETHERSCAN_INCLUDE_TOKENS   — include ERC-20 token transfers (default 1)

Runtime estimate (free API key, 0.4s delay):
  Each address queries up to 3 sources (normal + internal + token), so expect
  ~3× the time of a single-source run. Disable internal/token via env to speed up.
  Script is fully resumable — rerun safely after interruption.

How to use the output in GNN training:
  The train_dataset_scenarios.py script auto-detects etherscan_edges.csv
  and merges it with MulDiGraph edges when building the GNN graph.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from data_fetchers import EtherscanFetcher

ROOT         = Path(__file__).resolve().parent.parent
DATA_RAW     = ROOT / "data" / "raw"
DATA_PROC    = ROOT / "data" / "processed"
HYBRID_CSV   = DATA_PROC / "hybrid_dataset.csv"
OUTPUT_CSV   = DATA_RAW  / "etherscan_edges.csv"
CHECKPOINT   = DATA_RAW  / "etherscan_edges.partial.csv"


def main() -> None:
    load_dotenv()
    api_key      = os.getenv("ETHERSCAN_API_KEY", "")
    delay        = float(os.getenv("ETHERSCAN_REQUEST_DELAY", "0.4"))
    per_run_raw  = os.getenv("ETHERSCAN_PER_RUN_LIMIT", "0")
    per_run      = int(per_run_raw) if per_run_raw.isdigit() else 0
    max_pages    = int(os.getenv("ETHERSCAN_MAX_PAGES", "3"))
    inc_internal = os.getenv("ETHERSCAN_INCLUDE_INTERNAL", "1") != "0"
    inc_tokens   = os.getenv("ETHERSCAN_INCLUDE_TOKENS", "1") != "0"

    if not api_key:
        print("ERROR: ETHERSCAN_API_KEY is missing. Set it in .env or as env variable.")
        return

    if not HYBRID_CSV.exists():
        print(f"ERROR: hybrid_dataset.csv not found at {HYBRID_CSV}")
        print("Run ml/build_hybrid_dataset.py first.")
        return

    # Load every node address in the hybrid dataset
    df = pd.read_csv(HYBRID_CSV)
    all_addresses = set(df["node_id"].astype(str).str.lower())
    address_list  = sorted(all_addresses)  # deterministic order
    print(f"Total hybrid-dataset addresses : {len(all_addresses)}")

    # Load already-processed checkpoint
    done_src: set[str] = set()
    existing_edges: list[dict] = []
    if CHECKPOINT.exists():
        ckpt = pd.read_csv(CHECKPOINT)
        done_src = set(ckpt["src"].astype(str).str.lower())
        existing_edges = ckpt.to_dict("records")
        print(f"Resuming from checkpoint       : {len(done_src)} addresses already done")
    elif OUTPUT_CSV.exists():
        prev = pd.read_csv(OUTPUT_CSV)
        done_src = set(prev["src"].astype(str).str.lower())
        existing_edges = prev.to_dict("records")
        print(f"Resuming from output file      : {len(done_src)} addresses already done")

    pending = [a for a in address_list if a not in done_src]
    if per_run > 0:
        pending = pending[:per_run]

    sources = 1 + int(inc_internal) + int(inc_tokens)
    print(f"Pending this run               : {len(pending)}")
    print(f"Request delay                  : {delay}s  |  Max pages/address: {max_pages}")
    print(f"Sources per address            : {sources} (txlist"
          f"{'+internal' if inc_internal else ''}{'+token' if inc_tokens else ''})")
    print(f"Estimated time                 : ~{len(pending) * delay * max_pages * sources / 60:.0f} min")
    print("-" * 60)

    if not pending:
        print("Nothing to fetch. Edge list is complete.")
        if CHECKPOINT.exists():
            CHECKPOINT.unlink()
        return

    fetcher = EtherscanFetcher(api_key=api_key, chain_id=1, request_delay=delay)
    new_edges: list[dict] = []
    new_done: set[str] = set()
    batch_cross = 0          # edges found in the current 10-address batch
    addrs_with_edges = 0     # how many addresses contributed >=1 cross edge

    for i, src in enumerate(pending, start=1):
        neighbors = fetcher.get_transaction_neighbors(
            src, max_pages=max_pages,
            include_internal=inc_internal, include_tokens=inc_tokens,
        )
        # Keep only edges where the counterparty is also in our dataset
        cross_edges = [n for n in neighbors if n in all_addresses]
        for dst in cross_edges:
            new_edges.append({"src": src, "dst": dst})
        new_done.add(src)
        batch_cross += len(cross_edges)
        if cross_edges:
            addrs_with_edges += 1

        if i % 10 == 0 or i == len(pending):
            # Save checkpoint every 10 addresses
            all_edges = existing_edges + new_edges
            pd.DataFrame(all_edges).to_csv(CHECKPOINT, index=False)
            cross_count = len(all_edges)
            print(f"  [{i:>4}/{len(pending)}] last_batch: +{batch_cross} edges  "
                  f"| addrs_with_edges={addrs_with_edges}  total_edges={cross_count}")
            batch_cross = 0

    # Merge all and deduplicate
    all_edges_df = pd.DataFrame(existing_edges + new_edges)
    if not all_edges_df.empty:
        all_edges_df = all_edges_df.drop_duplicates(subset=["src", "dst"])
        # Add reverse edges (undirected)
        reverse = all_edges_df.rename(columns={"src": "dst", "dst": "src"})[["src", "dst"]]
        all_edges_df = pd.concat([all_edges_df, reverse]).drop_duplicates(subset=["src", "dst"])

    all_edges_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved {len(all_edges_df)} unique directed edges -> {OUTPUT_CSV}")
    print(f"Unique address pairs           : {len(all_edges_df) // 2}")

    if CHECKPOINT.exists():
        CHECKPOINT.unlink()
    print("Done.")


if __name__ == "__main__":
    main()
