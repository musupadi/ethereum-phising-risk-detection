from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv

from data_fetchers import EtherscanFetcher

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
INPUT_CSV = DATA_RAW / "addresses.csv"
OUTPUT_CSV = DATA_RAW / "addresses_etherscan_enriched.csv"
CHECKPOINT_CSV = DATA_RAW / "addresses_etherscan_enriched.partial.csv"


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    request_delay = float(os.getenv("ETHERSCAN_REQUEST_DELAY", "0.4"))
    per_run_limit_raw = os.getenv("ETHERSCAN_PER_RUN_LIMIT", "0")
    per_run_limit = int(per_run_limit_raw) if per_run_limit_raw.isdigit() else 0
    if not api_key:
        print("ETHERSCAN_API_KEY is missing")
        return

    if not INPUT_CSV.exists():
        print(f"Input not found: {INPUT_CSV}")
        print("Create data/raw/addresses.csv with column: address")
        return

    source_df = pd.read_csv(INPUT_CSV)
    if "address" not in source_df.columns:
        print("Input CSV must contain 'address' column")
        return

    source_df = source_df.dropna(subset=["address"]).copy()
    source_df["address"] = source_df["address"].astype(str)
    source_df = source_df[source_df["address"].str.match(r"^0x[a-fA-F0-9]{40}$", na=False)].drop_duplicates("address")

    done_df = pd.DataFrame()
    if CHECKPOINT_CSV.exists():
        done_df = pd.read_csv(CHECKPOINT_CSV)
    elif OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV)

    if not done_df.empty and "address" in done_df.columns:
        done_df["address"] = done_df["address"].astype(str)
        done_df = done_df[done_df["address"].isin(set(source_df["address"].astype(str)))].copy()

    done_addresses = set(done_df["address"].astype(str)) if not done_df.empty and "address" in done_df.columns else set()
    pending = source_df[~source_df["address"].isin(done_addresses)].copy()

    if per_run_limit > 0:
        pending = pending.head(per_run_limit).copy()

    print(f"Total valid addresses : {len(source_df)}")
    print(f"Already enriched      : {len(done_addresses)}")
    print(f"Pending               : {len(pending)}")
    print(f"Request delay         : {request_delay}")
    print(f"Per run limit         : {per_run_limit if per_run_limit > 0 else 'ALL'}")

    if pending.empty:
        print("No pending addresses. Nothing to fetch.")
        return

    fetcher = EtherscanFetcher(api_key=api_key, chain_id=1, request_delay=request_delay)
    rows = []

    for idx, address in enumerate(pending["address"], start=1):
        print(f"[{idx}/{len(pending)}] {address}")
        data = fetcher.fetch_all(address)
        source_row = pending[pending["address"] == address].iloc[0].to_dict()
        source_row.update(data)
        source_row["address"] = address
        rows.append(source_row)

        if idx % 25 == 0:
            chunk_df = pd.DataFrame(rows)
            merged = pd.concat([done_df, chunk_df], ignore_index=True)
            merged.to_csv(CHECKPOINT_CSV, index=False)
            print(f"Checkpoint saved: {CHECKPOINT_CSV}")

    new_df = pd.DataFrame(rows)
    final_df = pd.concat([done_df, new_df], ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["address"], keep="last")
    final_df.to_csv(OUTPUT_CSV, index=False)

    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()

    print(f"Saved final enriched data: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
