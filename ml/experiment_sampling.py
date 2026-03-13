"""
ml/experiment_sampling.py
--------------------------
Compares sampling ratios 1:5 vs 1:10 (phishing : normal).
Prints a comparison table with accuracy, precision, recall, F1.
Used to justify the 1:10 choice in thesis.
"""

import json
import logging
import random
import sys
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"
GRAPH_CANDIDATES = [
    ROOT / "data" / "raw" / "MulDiGraph.pkl",
    ROOT / "Ethereum Phishing Transaction Network" / "MulDiGraph.pkl",
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Import feature extractor from preprocess ──────────────────────────────────
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from ml.preprocess import extract_node_features

RANDOM_SEED = 42

def run_experiment(ratio: int, phishing_nodes: list, normal_nodes: list, G) -> dict:
    """Run a full train/test cycle for a given sampling ratio."""
    log.info("=== Ratio 1:%d ===", ratio)
    n_sample = min(len(phishing_nodes) * ratio, len(normal_nodes))
    random.seed(RANDOM_SEED)
    sampled_normal = random.sample(normal_nodes, n_sample)
    all_nodes = [(n, 1) for n in phishing_nodes] + [(n, 0) for n in sampled_normal]

    log.info("  Extracting features for %d nodes…", len(all_nodes))
    records = []
    for node, label in all_nodes:
        feats = extract_node_features(G, node)
        feats["label"] = label
        records.append(feats)

    df = pd.DataFrame(records)
    feature_cols = [c for c in df.columns if c != "label"]
    X = df[feature_cols].fillna(0).values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    # ── LR ────────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X_tr_s = scaler.fit_transform(X_train)
    X_te_s = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=RANDOM_SEED, class_weight="balanced")
    lr.fit(X_tr_s, y_train)
    lr_pred = lr.predict(X_te_s)

    # ── RF ────────────────────────────────────────────────────────────────────
    rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    def metrics(y_true, y_pred, name):
        return {
            "ratio": f"1:{ratio}",
            "model": name,
            "n_total": len(df),
            "n_phishing": int(y.sum()),
            "n_normal": int((y == 0).sum()),
            "accuracy":  round(accuracy_score(y_true, y_pred), 4),
            "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
            "recall":    round(recall_score(y_true, y_pred, zero_division=0), 4),
            "f1":        round(f1_score(y_true, y_pred, zero_division=0), 4),
        }

    return [
        metrics(y_test, lr_pred, "LogisticRegression"),
        metrics(y_test, rf_pred, "RandomForest"),
    ]


def main():
    graph_path = next((p for p in GRAPH_CANDIDATES if p.exists()), None)
    if not graph_path:
        log.error("MulDiGraph.pkl not found"); sys.exit(1)

    log.info("Loading graph (may take ~30s if cached)…")
    with open(graph_path, "rb") as f:
        G = pickle.load(f)
    log.info("Graph loaded: %d nodes", G.number_of_nodes())

    phishing_nodes = [n for n, d in G.nodes(data=True) if d.get("isp", 0) == 1]
    normal_nodes   = [n for n, d in G.nodes(data=True) if d.get("isp", 0) == 0]

    all_rows = []
    for ratio in [5, 10]:
        rows = run_experiment(ratio, phishing_nodes, normal_nodes, G)
        all_rows.extend(rows)

    df_result = pd.DataFrame(all_rows)
    print("\n" + "=" * 80)
    print("Sampling Ratio Comparison: 1:5 vs 1:10")
    print("=" * 80)
    print(df_result.to_string(index=False))
    print("=" * 80)

    # Save results
    out_path = ARTIFACTS / "sampling_experiment.json"
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(all_rows, f, indent=2)
    log.info("Results saved → %s", out_path)


if __name__ == "__main__":
    main()
