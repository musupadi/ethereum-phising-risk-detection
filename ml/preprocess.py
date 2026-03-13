"""
ml/preprocess.py
----------------
Extracts node-level graph features from MulDiGraph.pkl,
builds a labelled dataset, splits 80/20, and saves to data/processed/.

Features per node (all computed from the transaction graph):
  in_degree, out_degree,
  total_received, total_sent,
  unique_senders, unique_receivers,
  avg_in_amount, avg_out_amount,
  max_in_amount, max_out_amount,
  min_in_amount, min_out_amount,
  std_in_amount, std_out_amount,
  in_out_ratio, sent_received_ratio,
  avg_in_timestamp, avg_out_timestamp,
  timestamp_spread_in, timestamp_spread_out

Label: isp (1 = phishing, 0 = normal)
"""

import os
import sys
import json
import pickle
import random
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"

# Try both locations for the pickle
GRAPH_CANDIDATES = [
    DATA_RAW / "MulDiGraph.pkl",
    ROOT / "Ethereum Phishing Transaction Network" / "MulDiGraph.pkl",
]

TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
SCHEMA_JSON = ARTIFACTS / "feature_schema.json"

# ── Settings ──────────────────────────────────────────────────────────────────
NEGATIVE_SAMPLE_RATIO = 10   # non-phishing : phishing sampling ratio
RANDOM_SEED = 42

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

# ── Feature extraction helpers ────────────────────────────────────────────────

def _safe_stats(values: list) -> dict:
    """Return descriptive stats for a list of numeric values."""
    if not values:
        return dict(total=0.0, mean=0.0, mx=0.0, mn=0.0, std=0.0)
    arr = np.asarray(values, dtype=float)
    return dict(
        total=float(arr.sum()),
        mean=float(arr.mean()),
        mx=float(arr.max()),
        mn=float(arr.min()),
        std=float(arr.std()) if len(arr) > 1 else 0.0,
    )


def extract_node_features(G, node) -> dict:
    """
    Extract a flat feature dict for *node* from MultiDiGraph G.
    Handles parallel edges (MultiDiGraph allows multiple edges between
    the same pair of nodes).
    """
    # ── Incoming ──────────────────────────────────────────────────────────────
    in_amounts, in_timestamps = [], []
    unique_senders = set()

    for pred in G.predecessors(node):
        unique_senders.add(pred)
        for edge_data in G[pred][node].values():
            in_amounts.append(edge_data.get("amount", 0) or 0)
            ts = edge_data.get("timestamp")
            if ts is not None:
                in_timestamps.append(float(ts))

    # ── Outgoing ──────────────────────────────────────────────────────────────
    out_amounts, out_timestamps = [], []
    unique_receivers = set()

    for succ in G.successors(node):
        unique_receivers.add(succ)
        for edge_data in G[node][succ].values():
            out_amounts.append(edge_data.get("amount", 0) or 0)
            ts = edge_data.get("timestamp")
            if ts is not None:
                out_timestamps.append(float(ts))

    in_s = _safe_stats(in_amounts)
    out_s = _safe_stats(out_amounts)

    n_in = len(in_amounts)
    n_out = len(out_amounts)

    # Timestamp spread (max - min); 0 if no transactions
    ts_spread_in = (max(in_timestamps) - min(in_timestamps)) if len(in_timestamps) > 1 else 0.0
    ts_spread_out = (max(out_timestamps) - min(out_timestamps)) if len(out_timestamps) > 1 else 0.0

    return {
        "in_degree": n_in,
        "out_degree": n_out,
        "unique_senders": len(unique_senders),
        "unique_receivers": len(unique_receivers),
        "total_received": in_s["total"],
        "total_sent": out_s["total"],
        "avg_in_amount": in_s["mean"],
        "avg_out_amount": out_s["mean"],
        "max_in_amount": in_s["mx"],
        "max_out_amount": out_s["mx"],
        "min_in_amount": in_s["mn"],
        "min_out_amount": out_s["mn"],
        "std_in_amount": in_s["std"],
        "std_out_amount": out_s["std"],
        "in_out_ratio": n_in / (n_out + 1),
        "sent_received_ratio": out_s["total"] / (in_s["total"] + 1e-9),
        "timestamp_spread_in": ts_spread_in,
        "timestamp_spread_out": ts_spread_out,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ── Load graph ────────────────────────────────────────────────────────────
    graph_path = None
    for candidate in GRAPH_CANDIDATES:
        if candidate.exists():
            graph_path = candidate
            break

    if graph_path is None:
        log.error(
            "MulDiGraph.pkl not found. Searched:\n  %s",
            "\n  ".join(str(p) for p in GRAPH_CANDIDATES),
        )
        sys.exit(1)

    log.info("Loading graph from %s  (this may take 1–3 minutes)…", graph_path)
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    log.info("Graph loaded: %d nodes, %d edges", G.number_of_nodes(), G.number_of_edges())

    # ── Split nodes by label ──────────────────────────────────────────────────
    log.info("Partitioning nodes by label…")
    phishing_nodes = []
    normal_nodes = []

    for node, data in G.nodes(data=True):
        label = data.get("isp", 0)
        if label == 1:
            phishing_nodes.append(node)
        else:
            normal_nodes.append(node)

    n_phishing = len(phishing_nodes)
    n_sample = min(n_phishing * NEGATIVE_SAMPLE_RATIO, len(normal_nodes))
    log.info("Phishing nodes: %d  |  Normal nodes: %d (sampling %d)", n_phishing, len(normal_nodes), n_sample)

    random.seed(RANDOM_SEED)
    sampled_normal = random.sample(normal_nodes, n_sample)

    # ── Extract features ──────────────────────────────────────────────────────
    all_nodes = [(n, 1) for n in phishing_nodes] + [(n, 0) for n in sampled_normal]

    log.info("Extracting features for %d nodes…", len(all_nodes))
    records = []
    for i, (node, label) in enumerate(all_nodes):
        if i % 500 == 0:
            log.info("  %d / %d", i, len(all_nodes))
        feats = extract_node_features(G, node)
        feats["label"] = label
        feats["node_id"] = str(node)
        records.append(feats)

    df = pd.DataFrame(records)
    log.info("Dataset shape: %s", df.shape)

    # ── Feature schema ────────────────────────────────────────────────────────
    feature_cols = [c for c in df.columns if c not in ("label", "node_id")]
    schema = {"features": feature_cols, "label": "label", "node_id": "node_id"}

    with open(SCHEMA_JSON, "w") as f:
        json.dump(schema, f, indent=2)
    log.info("Feature schema saved → %s", SCHEMA_JSON)

    # ── Train / test split ────────────────────────────────────────────────────
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=RANDOM_SEED,
        stratify=df["label"],
    )
    log.info("Train: %d  |  Test: %d", len(train_df), len(test_df))

    train_df.to_csv(TRAIN_CSV, index=False)
    test_df.to_csv(TEST_CSV, index=False)
    log.info("Saved → %s", TRAIN_CSV)
    log.info("Saved → %s", TEST_CSV)

    # ── Class balance check ───────────────────────────────────────────────────
    pos = df["label"].sum()
    log.info("Class balance — phishing: %d (%.1f%%)  normal: %d (%.1f%%)",
             pos, 100 * pos / len(df), len(df) - pos, 100 * (1 - pos / len(df)))


if __name__ == "__main__":
    main()
