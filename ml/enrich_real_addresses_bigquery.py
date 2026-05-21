from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd
from dotenv import load_dotenv
from google.cloud import bigquery

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
INPUT_CSV = DATA_RAW / "addresses.csv"
OUTPUT_CSV = DATA_RAW / "addresses_bigquery_enriched.csv"
CHECKPOINT_CSV = DATA_RAW / "addresses_bigquery_enriched.partial.csv"


def chunked(items: list[str], chunk_size: int) -> Iterable[list[str]]:
    for i in range(0, len(items), chunk_size):
        yield items[i:i + chunk_size]


def run_bigquery_batch(
    client: bigquery.Client,
    addresses: list[str],
    days_window: int,
    table_fqn: str,
) -> pd.DataFrame:
    query = f"""
    WITH target AS (
      SELECT address
      FROM UNNEST(@addresses) AS address
    ),
    tx AS (
      SELECT
        LOWER(from_address) AS from_address,
        LOWER(to_address) AS to_address,
        CAST(value AS FLOAT64) / 1e18 AS value_eth,
        DATE(block_timestamp) AS tx_day,
        block_timestamp
      FROM `{table_fqn}`
      WHERE block_timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL @days_window DAY)
        AND (
          LOWER(from_address) IN (SELECT address FROM target)
          OR LOWER(to_address) IN (SELECT address FROM target)
        )
    ),
    out_agg AS (
      SELECT
        from_address AS address,
        COUNT(1) AS bq_tx_count_out,
        SUM(value_eth) AS bq_total_out_eth,
        COUNT(DISTINCT to_address) AS bq_unique_to,
        COUNT(DISTINCT tx_day) AS bq_active_days_out,
        MIN(block_timestamp) AS bq_first_seen_out,
        MAX(block_timestamp) AS bq_last_seen_out
      FROM tx
      GROUP BY from_address
    ),
    in_agg AS (
      SELECT
        to_address AS address,
        COUNT(1) AS bq_tx_count_in,
        SUM(value_eth) AS bq_total_in_eth,
        COUNT(DISTINCT from_address) AS bq_unique_from,
        COUNT(DISTINCT tx_day) AS bq_active_days_in,
        MIN(block_timestamp) AS bq_first_seen_in,
        MAX(block_timestamp) AS bq_last_seen_in
      FROM tx
      GROUP BY to_address
    )
    SELECT
      t.address,
      IFNULL(o.bq_tx_count_out, 0) AS bq_tx_count_out,
      IFNULL(i.bq_tx_count_in, 0) AS bq_tx_count_in,
      IFNULL(o.bq_total_out_eth, 0.0) AS bq_total_out_eth,
      IFNULL(i.bq_total_in_eth, 0.0) AS bq_total_in_eth,
      IFNULL(o.bq_unique_to, 0) AS bq_unique_to,
      IFNULL(i.bq_unique_from, 0) AS bq_unique_from,
      IFNULL(o.bq_active_days_out, 0) AS bq_active_days_out,
      IFNULL(i.bq_active_days_in, 0) AS bq_active_days_in,
      UNIX_SECONDS(
        COALESCE(
          LEAST(o.bq_first_seen_out, i.bq_first_seen_in),
          o.bq_first_seen_out,
          i.bq_first_seen_in
        )
      ) AS bq_first_seen_ts,
      UNIX_SECONDS(
        COALESCE(
          GREATEST(o.bq_last_seen_out, i.bq_last_seen_in),
          o.bq_last_seen_out,
          i.bq_last_seen_in
        )
      ) AS bq_last_seen_ts
    FROM target t
    LEFT JOIN out_agg o USING (address)
    LEFT JOIN in_agg i USING (address)
    """

    config = bigquery.QueryJobConfig(
        query_parameters=[
        bigquery.ArrayQueryParameter("addresses", "STRING", addresses),
        bigquery.ScalarQueryParameter("days_window", "INT64", days_window),
        ]
    )

    rows = [dict(row.items()) for row in client.query(query, job_config=config).result()]
    df = pd.DataFrame(rows)
    if df.empty:
      df = pd.DataFrame(columns=[
        "address",
        "bq_tx_count_out",
        "bq_tx_count_in",
        "bq_total_out_eth",
        "bq_total_in_eth",
        "bq_unique_to",
        "bq_unique_from",
        "bq_active_days_out",
        "bq_active_days_in",
        "bq_first_seen_ts",
        "bq_last_seen_ts",
      ])
    df["address"] = df["address"].astype(str).str.lower()
    return df


def main() -> None:
    load_dotenv()

    project_id = os.getenv("BIGQUERY_PROJECT_ID", "").strip()
    table_fqn = os.getenv("BIGQUERY_TABLE", "bigquery-public-data.crypto_ethereum.transactions").strip()
    days_window = int(os.getenv("BIGQUERY_DAYS_WINDOW", "90"))
    per_run_limit_raw = os.getenv("BIGQUERY_PER_RUN_LIMIT", "0")
    per_run_limit = int(per_run_limit_raw) if per_run_limit_raw.isdigit() else 0
    chunk_size = int(os.getenv("BIGQUERY_CHUNK_SIZE", "300"))

    if not project_id:
        print("BIGQUERY_PROJECT_ID is missing")
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
    source_df["address"] = source_df["address"].astype(str).str.lower().str.strip()
    source_df = source_df[source_df["address"].str.match(r"^0x[a-f0-9]{40}$", na=False)].drop_duplicates("address")

    done_df = pd.DataFrame()
    if CHECKPOINT_CSV.exists():
        done_df = pd.read_csv(CHECKPOINT_CSV)
    elif OUTPUT_CSV.exists():
        done_df = pd.read_csv(OUTPUT_CSV)

    if not done_df.empty and "address" in done_df.columns:
        done_df["address"] = done_df["address"].astype(str).str.lower().str.strip()
        done_df = done_df[done_df["address"].isin(set(source_df["address"]))].copy()

    done_addresses = set(done_df["address"].astype(str)) if not done_df.empty and "address" in done_df.columns else set()
    pending_df = source_df[~source_df["address"].isin(done_addresses)].copy()

    if per_run_limit > 0:
        pending_df = pending_df.head(per_run_limit).copy()

    pending_addresses = pending_df["address"].tolist()

    print(f"Total valid addresses : {len(source_df)}")
    print(f"Already enriched      : {len(done_addresses)}")
    print(f"Pending               : {len(pending_addresses)}")
    print(f"BigQuery project      : {project_id}")
    print(f"BigQuery table        : {table_fqn}")
    print(f"Days window           : {days_window}")
    print(f"Chunk size            : {chunk_size}")
    print(f"Per run limit         : {per_run_limit if per_run_limit > 0 else 'ALL'}")

    if not pending_addresses:
        print("No pending addresses. Nothing to fetch.")
        return

    client = bigquery.Client(project=project_id)
    rows = []

    batches = list(chunked(pending_addresses, chunk_size))
    for batch_idx, batch in enumerate(batches, start=1):
        print(f"[Batch {batch_idx}/{len(batches)}] querying {len(batch)} addresses")
        bq_df = run_bigquery_batch(
            client=client,
            addresses=batch,
            days_window=days_window,
            table_fqn=table_fqn,
        )

        merged_chunk = pending_df[pending_df["address"].isin(batch)].merge(bq_df, on="address", how="left")

        for col in [
            "bq_tx_count_out",
            "bq_tx_count_in",
            "bq_total_out_eth",
            "bq_total_in_eth",
            "bq_unique_to",
            "bq_unique_from",
            "bq_active_days_out",
            "bq_active_days_in",
            "bq_first_seen_ts",
            "bq_last_seen_ts",
        ]:
            if col not in merged_chunk.columns:
                merged_chunk[col] = 0

        rows.append(merged_chunk)

        checkpoint_df = pd.concat([done_df] + rows, ignore_index=True)
        checkpoint_df = checkpoint_df.drop_duplicates(subset=["address"], keep="last")
        checkpoint_df.to_csv(CHECKPOINT_CSV, index=False)
        print(f"Checkpoint saved: {CHECKPOINT_CSV}")

    final_df = pd.concat([done_df] + rows, ignore_index=True)
    final_df = final_df.drop_duplicates(subset=["address"], keep="last")
    final_df.to_csv(OUTPUT_CSV, index=False)

    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()

    print(f"Saved final BigQuery enriched data: {OUTPUT_CSV}")
    print("Done.")


if __name__ == "__main__":
    main()
