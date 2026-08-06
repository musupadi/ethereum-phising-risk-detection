"""
Build scenario-ready phishing datasets by combining:
1) Graph-derived behavioral features from MulDiGraph.pkl
2) Real wallet attributes from addresses_etherscan_enriched.csv
3) BigQuery transaction aggregates from addresses_bigquery_enriched.csv

Join key: Ethereum address

Outputs:
- data/processed/hybrid_dataset_full.csv   (all matched rows)
- data/processed/hybrid_dataset.csv        (optionally rebalanced)
- data/processed/hybrid_train.csv
- data/processed/hybrid_test.csv
- ml/artifacts/hybrid_feature_schema.json

Environment variables:
- HYBRID_NORMAL_RATIO=0  -> keep all available normal rows (default)
- HYBRID_NORMAL_RATIO=3  -> keep at most 3 normal rows per phishing row
"""

from __future__ import annotations

import json
import logging
import os
import pickle
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

try:
    from preprocess import extract_node_features
except ImportError:
    from ml.preprocess import extract_node_features

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"

GRAPH_CANDIDATES = [
    DATA_RAW / "MulDiGraph.pkl",
    ROOT / "Ethereum Phishing Transaction Network" / "MulDiGraph.pkl",
]

ENRICHED_CSV = DATA_RAW / "addresses_etherscan_enriched.csv"
LABEL_SOURCE_CSV = DATA_RAW / "addresses.csv"
BIGQUERY_ENRICHED_CSV = DATA_RAW / "addresses_bigquery_enriched.csv"
HYBRID_FULL_CSV = DATA_PROCESSED / "hybrid_dataset_full.csv"
HYBRID_FINAL_CSV = DATA_PROCESSED / "hybrid_dataset.csv"
HYBRID_TRAIN_CSV = DATA_PROCESSED / "hybrid_train.csv"
HYBRID_TEST_CSV = DATA_PROCESSED / "hybrid_test.csv"
HYBRID_SCHEMA_JSON = ARTIFACTS / "hybrid_feature_schema.json"

RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


NUMERIC_ONCHAIN_COLS = [
    "balance",
    "transaction_count",
    "activity_score",
    "avg_tx_value",
    "tx_frequency",
    "unique_counterparties",
]

NUMERIC_BIGQUERY_COLS = [
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
]

META_COLS = [
    "address",
    "label",
    "class_name",
    "source_dataset",
]

# graph_present is a join-quality audit flag (1 = address found in MulDiGraph),
# not a wallet-behavior signal. It must never enter the ML feature list.
AUDIT_COLS = ["graph_present"]


def find_graph_path() -> Path | None:
    for path in GRAPH_CANDIDATES:
        if path.exists():
            return path
    return None


def rebalance_by_ratio(df: pd.DataFrame, normal_ratio: int) -> pd.DataFrame:
    """Optionally downsample normal class to desired ratio relative to phishing."""
    if normal_ratio <= 0:
        return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    phishing_df = df[df["label"] == 1].copy()
    normal_df = df[df["label"] == 0].copy()

    if phishing_df.empty or normal_df.empty:
        return df.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)

    keep_normals = min(len(normal_df), len(phishing_df) * normal_ratio)
    normal_df = normal_df.sample(n=keep_normals, random_state=RANDOM_SEED)

    out = pd.concat([phishing_df, normal_df], ignore_index=True)
    out = out.sample(frac=1.0, random_state=RANDOM_SEED).reset_index(drop=True)
    return out


def prepare_enriched_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["address"] = out["address"].astype(str).str.lower().str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)

    for col in NUMERIC_ONCHAIN_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
        else:
            out[col] = 0.0

    if "created_at" in out.columns:
        created = pd.to_datetime(out["created_at"], errors="coerce", utc=True)
        out["created_at_ts"] = (created.astype("int64") // 10**9).where(~created.isna(), 0)
    else:
        out["created_at_ts"] = 0

    keep_cols = META_COLS + ["created_at_ts"] + NUMERIC_ONCHAIN_COLS
    keep_cols = [c for c in keep_cols if c in out.columns]

    out = out[keep_cols].drop_duplicates(subset=["address"]).reset_index(drop=True)
    return out


def prepare_label_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["address"] = out["address"].astype(str).str.lower().str.strip()
    out["label"] = pd.to_numeric(out["label"], errors="coerce").fillna(0).astype(int)

    if "class_name" not in out.columns:
        out["class_name"] = out["label"].map({1: "phishing", 0: "normal"}).fillna("unknown")
    if "source_dataset" not in out.columns:
        out["source_dataset"] = "addresses"

    keep_cols = [c for c in META_COLS if c in out.columns]
    out = out[keep_cols].drop_duplicates(subset=["address"]).reset_index(drop=True)
    return out


def prepare_bigquery_df(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "address" not in out.columns:
        return pd.DataFrame(columns=["address"] + NUMERIC_BIGQUERY_COLS)

    out["address"] = out["address"].astype(str).str.lower().str.strip()

    for col in NUMERIC_BIGQUERY_COLS:
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
        else:
            out[col] = 0

    keep_cols = ["address"] + NUMERIC_BIGQUERY_COLS
    out = out[keep_cols].drop_duplicates(subset=["address"]).reset_index(drop=True)
    return out


def extract_graph_features_for_addresses(graph, addresses: list[str]) -> pd.DataFrame:
    records = []
    total = len(addresses)

    for i, address in enumerate(addresses, start=1):
        if i == 1 or i % 250 == 0 or i == total:
            log.info("Graph feature extraction: %d / %d", i, total)

        if address not in graph:
            continue

        feats = extract_node_features(graph, address)
        feats["address"] = address
        feats["graph_label"] = int(graph.nodes[address].get("isp", 0) or 0)
        records.append(feats)

    return pd.DataFrame(records)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    if not LABEL_SOURCE_CSV.exists() and not ENRICHED_CSV.exists():
        log.error("Both label source and enriched CSV are missing.")
        log.error("Expected one of: %s OR %s", LABEL_SOURCE_CSV, ENRICHED_CSV)
        sys.exit(1)

    graph_path = find_graph_path()
    if graph_path is None:
        log.error("MulDiGraph.pkl not found.")
        sys.exit(1)

    normal_ratio = int(os.getenv("HYBRID_NORMAL_RATIO", "0"))

    if LABEL_SOURCE_CSV.exists():
        log.info("Loading base labeled addresses from %s", LABEL_SOURCE_CSV)
        base_df = prepare_label_df(pd.read_csv(LABEL_SOURCE_CSV))
    else:
        log.info("Label source not found, falling back to enriched labels from %s", ENRICHED_CSV)
        base_df = prepare_enriched_df(pd.read_csv(ENRICHED_CSV))[META_COLS]

    log.info("Base labeled rows available: %d", len(base_df))
    log.info("Class distribution (base): %s", base_df["label"].value_counts().to_dict())

    etherscan_df = pd.DataFrame(columns=["address", "created_at_ts"] + NUMERIC_ONCHAIN_COLS)
    if ENRICHED_CSV.exists():
        log.info("Loading Etherscan enrichment from %s", ENRICHED_CSV)
        etherscan_df = prepare_enriched_df(pd.read_csv(ENRICHED_CSV))
        log.info("Etherscan rows available: %d", len(etherscan_df))
    else:
        log.info("Etherscan enrichment not found, proceeding without it")

    if not etherscan_df.empty:
        etherscan_feature_cols = [c for c in ["address", "created_at_ts"] + NUMERIC_ONCHAIN_COLS if c in etherscan_df.columns]
        merged_base = base_df.merge(etherscan_df[etherscan_feature_cols], on="address", how="left")
    else:
        merged_base = base_df.copy()

    for col in ["created_at_ts"] + NUMERIC_ONCHAIN_COLS:
        if col not in merged_base.columns:
            merged_base[col] = 0.0
        merged_base[col] = pd.to_numeric(merged_base[col], errors="coerce").fillna(0)

    bq_df = pd.DataFrame(columns=["address"] + NUMERIC_BIGQUERY_COLS)
    if BIGQUERY_ENRICHED_CSV.exists():
        log.info("Loading BigQuery enrichment from %s", BIGQUERY_ENRICHED_CSV)
        bq_df = prepare_bigquery_df(pd.read_csv(BIGQUERY_ENRICHED_CSV))
        log.info("BigQuery rows available: %d", len(bq_df))
    else:
        log.info("BigQuery enrichment not found, proceeding with Etherscan-only enrichment")

    log.info("Loading graph from %s", graph_path)
    with open(graph_path, "rb") as f:
        graph = pickle.load(f)

    addresses = merged_base["address"].dropna().astype(str).tolist()
    graph_df = extract_graph_features_for_addresses(graph, addresses)

    if graph_df.empty:
        log.error("No graph features could be extracted for the enriched addresses.")
        sys.exit(1)

    graph_feature_cols = [c for c in graph_df.columns if c not in {"address", "graph_label"}]

    # Keep all enriched rows and backfill missing graph features with zeros.
    # This allows larger training sets when some addresses are absent in MulDiGraph.
    merged_df = merged_base.merge(graph_df, on="address", how="left")
    for col in graph_feature_cols:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0)

    # Track graph coverage after widening the join.
    merged_df["graph_present"] = merged_df[graph_feature_cols].sum(axis=1).gt(0).astype(int)
    graph_coverage = float(merged_df["graph_present"].mean() * 100.0)
    log.info("Graph coverage in merged dataset: %.2f%%", graph_coverage)

    if not bq_df.empty:
        merged_df = merged_df.merge(bq_df, on="address", how="left")
    merged_df["node_id"] = merged_df["address"]

    mismatch_mask = merged_df["graph_label"].notna()
    mismatch_count = int((merged_df.loc[mismatch_mask, "graph_label"] != merged_df.loc[mismatch_mask, "label"]).sum())
    if mismatch_count > 0:
        log.warning("Found %d graph/enriched label mismatches", mismatch_count)

    merged_df = merged_df.drop(columns=["graph_label"], errors="ignore")

    merged_df.to_csv(HYBRID_FULL_CSV, index=False)
    log.info("Saved full hybrid dataset → %s", HYBRID_FULL_CSV)

    final_df = rebalance_by_ratio(merged_df, normal_ratio=normal_ratio)
    final_df.to_csv(HYBRID_FINAL_CSV, index=False)
    log.info("Saved final hybrid dataset → %s", HYBRID_FINAL_CSV)
    log.info("Class distribution (final): %s", final_df["label"].value_counts().to_dict())

    feature_cols = [
        c for c in final_df.columns
        if c not in {"address", "node_id", "label", "class_name", "source_dataset", "error", "graph_label", *AUDIT_COLS}
    ]
    etherscan_features = [c for c in ["created_at_ts"] + NUMERIC_ONCHAIN_COLS if c in final_df.columns]
    bigquery_features = [c for c in NUMERIC_BIGQUERY_COLS if c in final_df.columns]
    graph_features = [c for c in feature_cols if c not in set(etherscan_features) | set(bigquery_features)]

    train_df, test_df = train_test_split(
        final_df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=final_df["label"],
    )
    train_df.to_csv(HYBRID_TRAIN_CSV, index=False)
    test_df.to_csv(HYBRID_TEST_CSV, index=False)

    schema = {
        "features": feature_cols,
        "audit_columns": AUDIT_COLS,
        "feature_groups": {
            "graph": graph_features,
            "etherscan": etherscan_features,
            "bigquery": bigquery_features,
        },
        "scenarios": {
            "muldigraph_only": graph_features,
            "muldigraph_plus_etherscan": graph_features + etherscan_features,
            "muldigraph_plus_bigquery": graph_features + bigquery_features,
            "muldigraph_plus_etherscan_plus_bigquery": graph_features + etherscan_features + bigquery_features,
        },
        "label": "label",
        "node_id": "node_id",
        "dataset": "hybrid_graph_plus_etherscan_with_optional_bigquery",
        "normal_ratio": normal_ratio,
        "rows_full": int(len(merged_df)),
        "rows_final": int(len(final_df)),
        "bigquery_enrichment_used": bool(BIGQUERY_ENRICHED_CSV.exists() and not bq_df.empty),
    }
    with open(HYBRID_SCHEMA_JSON, "w", encoding="utf-8") as f:
        json.dump(schema, f, indent=2)

    log.info("Saved hybrid train split → %s", HYBRID_TRAIN_CSV)
    log.info("Saved hybrid test split  → %s", HYBRID_TEST_CSV)
    log.info("Saved hybrid schema      → %s", HYBRID_SCHEMA_JSON)
    log.info("Hybrid dataset build complete.")


if __name__ == "__main__":
    main()
