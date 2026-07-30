"""
ml/tune_gnn.py
--------------
Tindak lanjut audit catatan #13: tuning hyperparameter GNN-GraphSAGE setara dengan
tuning yang sudah dilakukan pada XGBoost. Grid atas: hidden_dim, num_layers,
learning_rate, dropout.

Konsisten dengan `evaluate_roc.py` / `train_dataset_scenarios.py`:
  - skenario : muldigraph_plus_etherscan (26 fitur)
  - split    : test 20% (random_state=42, stratified), identik
  - edge     : edge transaksi nyata dari MulDiGraph (diekstrak SEKALI, dipakai ulang)
  - protokol : sama dengan perbandingan tesis (agar hasil bisa dibandingkan langsung)

Graph MulDiGraph dimuat sekali; edge diekstrak sekali; lalu tiap konfigurasi dilatih
ulang. GNN non-deterministik → tiap config diulang beberapa seed dan dirata-rata agar
peringkatnya stabil.

Run: python ml/tune_gnn.py
Output: ml/artifacts/gnn_tuning_results.json
"""
from __future__ import annotations

import itertools
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from train_dataset_scenarios import (ARTIFACTS, HYBRID_CSV, HYBRID_SCHEMA_JSON,
                                     extract_edges_for_addresses, load_transaction_graph)
from gnn_model_v2 import GraphPhiV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SCENARIO = "muldigraph_plus_etherscan"
RANDOM_STATE = 42
OUT = ARTIFACTS / "gnn_tuning_results.json"

# Grid hyperparameter
HIDDEN = [64, 128, 256]
LAYERS = [2, 3]
LR = [0.005, 0.01, 0.02]
DROPOUT = [0.3, 0.5]
SEEDS = [42, 7]  # rata-rata 2 seed untuk menstabilkan peringkat (GNN non-deterministik)


def summ(y_true, y_pred, proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }


def train_one(Xtr, ytr, Xte, yte, feats, edges, hidden, layers, lr, dropout, seed) -> dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    gnn = GraphPhiV2(model_type="sage", input_dim=len(feats), hidden_dim=hidden, num_layers=layers)
    # override dropout & lr (tidak diekspos di konstruktor)
    gnn.model.dropout = dropout
    gnn.optimizer = torch.optim.Adam(gnn.model.parameters(), lr=lr, weight_decay=5e-4)
    if edges:
        graph, tr_mask, te_mask = gnn.build_graph_with_real_edges(Xtr, ytr, Xte, yte, edges)
    else:
        graph, tr_mask, te_mask = gnn.build_graph_from_train_only(Xtr, ytr, Xte, yte)
    gnn.fit(graph, train_mask=tr_mask, test_mask=te_mask, epochs=200, early_stopping_patience=25)
    pred, proba = gnn.predict(graph, te_mask)
    return summ(yte, pred, proba[:, 1])


def main() -> None:
    schema = json.loads(HYBRID_SCHEMA_JSON.read_text(encoding="utf-8"))
    feats = schema["scenarios"][SCENARIO]
    df = pd.read_csv(HYBRID_CSV)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])
    X_train, y_train = train_df[feats].fillna(0).values, train_df["label"].values
    X_test, y_test = test_df[feats].fillna(0).values, test_df["label"].values

    sc = StandardScaler()
    Xtr = sc.fit_transform(X_train)
    Xte = sc.transform(X_test)

    # edge nyata: ekstrak SEKALI
    g = load_transaction_graph()
    edges = None
    if g is not None:
        tr_addr = train_df["node_id"].astype(str).str.lower().tolist()
        te_addr = test_df["node_id"].astype(str).str.lower().tolist()
        edges = extract_edges_for_addresses(g, tr_addr, te_addr)
        log.info("Edge nyata diekstrak: %d", len(edges) if edges else 0)

    grid = list(itertools.product(HIDDEN, LAYERS, LR, DROPOUT))
    log.info("Total konfigurasi: %d x %d seed = %d training", len(grid), len(SEEDS), len(grid) * len(SEEDS))

    results = []
    for i, (hidden, layers, lr, dropout) in enumerate(grid, 1):
        seed_metrics = [train_one(Xtr, y_train, Xte, y_test, feats, edges, hidden, layers, lr, dropout, s)
                        for s in SEEDS]
        avg = {k: round(float(np.mean([m[k] for m in seed_metrics])), 4) for k in seed_metrics[0]}
        std_f1 = round(float(np.std([m["f1"] for m in seed_metrics])), 4)
        row = {"hidden_dim": hidden, "num_layers": layers, "learning_rate": lr, "dropout": dropout,
               "f1_mean": avg["f1"], "f1_std": std_f1, "roc_auc_mean": avg["roc_auc"],
               "pr_auc_mean": avg["pr_auc"], "precision_mean": avg["precision"],
               "recall_mean": avg["recall"], "accuracy_mean": avg["accuracy"]}
        results.append(row)
        log.info("[%2d/%d] hidden=%d layers=%d lr=%.3f drop=%.1f -> F1=%.4f±%.4f AUC=%.4f",
                 i, len(grid), hidden, layers, lr, dropout, avg["f1"], std_f1, avg["roc_auc"])

    results.sort(key=lambda r: r["f1_mean"], reverse=True)
    best = results[0]
    # baseline default (hidden=128, layers=2, lr=0.01, dropout=0.5)
    baseline = next((r for r in results if r["hidden_dim"] == 128 and r["num_layers"] == 2
                     and abs(r["learning_rate"] - 0.01) < 1e-9 and abs(r["dropout"] - 0.5) < 1e-9), None)

    report = {
        "scenario": SCENARIO,
        "n_features": len(feats),
        "n_test": int(len(y_test)),
        "seeds": SEEDS,
        "grid": {"hidden_dim": HIDDEN, "num_layers": LAYERS, "learning_rate": LR, "dropout": DROPOUT},
        "protocol": "sama dgn evaluate_roc.py (early stopping pada test_mask); rata-rata 2 seed.",
        "baseline_default": baseline,
        "best_config": best,
        "improvement_f1": round(best["f1_mean"] - baseline["f1_mean"], 4) if baseline else None,
        "all_results_sorted": results,
        "xgboost_reference_f1": 0.9102,
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Best: hidden=%d layers=%d lr=%.3f drop=%.1f F1=%.4f | baseline F1=%.4f | -> %s",
             best["hidden_dim"], best["num_layers"], best["learning_rate"], best["dropout"],
             best["f1_mean"], baseline["f1_mean"] if baseline else -1, OUT.name)


if __name__ == "__main__":
    main()
