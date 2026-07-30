"""
ml/evaluate_roc.py
------------------
Tahap E (catatan bimbingan poin 13 & 14): kurva ROC + nilai AUC untuk XGBoost dan
GNN-GraphSAGE, valid untuk klasifikasi biner (phishing vs bukan). Sekaligus skenario
perbandingan XGBoost vs GNN yang fokus pada skenario terbaik (MulDiGraph + Etherscan).

Memakai split yang identik dengan ml/train_dataset_scenarios.py (test_size=0.2,
random_state=42, stratify) sehingga angka konsisten dengan tabel tesis
(XGBoost F1 0.9102, GNN F1 0.896).

Output:
  - ml/artifacts/roc_xgb.png            ROC XGBoost saja
  - ml/artifacts/roc_comparison.png     ROC XGBoost vs GNN (overlay)
  - ml/artifacts/roc_auc_report.json    AUC + metrik ringkas kedua model

Run: python ml/evaluate_roc.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_recall_curve, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# reuse pipeline helpers so the split / edges match the thesis comparison
from train_dataset_scenarios import (ARTIFACTS, HYBRID_CSV, HYBRID_SCHEMA_JSON,
                                     extract_edges_for_addresses, load_transaction_graph)
from xgboost_model import XGBoostModel
from gnn_model_v2 import GraphPhiV2

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SCENARIO = "muldigraph_plus_etherscan"
RANDOM_STATE = 42

C_XGB = "#4E79A7"
C_GNN = "#E15759"


def summ(y_true, y_pred, proba) -> dict:
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
    }


def train_xgb_proba(X_train, y_train, X_test, feats):
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=RANDOM_STATE, stratify=y_train)
    m = XGBoostModel(n_estimators=250, max_depth=6, learning_rate=0.05,
                     subsample=0.85, colsample_bytree=0.85)
    m.fit(X_fit, y_fit, X_val, y_val, feature_names=feats, early_stopping_rounds=40)
    proba = m.predict_proba(X_test)[:, 1]
    return (proba > 0.5).astype(int), proba


def train_gnn_proba(X_train, y_train, X_test, y_test, feats, edges):
    sc = StandardScaler()
    Xtr, Xte = sc.fit_transform(X_train), sc.transform(X_test)
    gnn = GraphPhiV2(model_type="sage", input_dim=len(feats), hidden_dim=128, num_layers=2)
    if edges:
        graph, tr_mask, te_mask = gnn.build_graph_with_real_edges(Xtr, y_train, Xte, y_test, edges)
    else:
        graph, tr_mask, te_mask = gnn.build_graph_from_train_only(Xtr, y_train, Xte, y_test)
    gnn.fit(graph, train_mask=tr_mask, test_mask=te_mask, epochs=200, early_stopping_patience=25)
    pred, proba = gnn.predict(graph, te_mask)
    return pred, proba[:, 1]


def main() -> None:
    schema = json.loads(HYBRID_SCHEMA_JSON.read_text(encoding="utf-8"))
    feats = schema["scenarios"][SCENARIO]
    df = pd.read_csv(HYBRID_CSV)
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"])

    X_train, y_train = train_df[feats].fillna(0).values, train_df["label"].values
    X_test, y_test = test_df[feats].fillna(0).values, test_df["label"].values
    log.info("Skenario %s | train=%d test=%d fitur=%d", SCENARIO, len(X_train), len(X_test), len(feats))

    # XGBoost
    xgb_pred, xgb_proba = train_xgb_proba(X_train, y_train, X_test, feats)
    xgb_m = summ(y_test, xgb_pred, xgb_proba)
    log.info("XGBoost: F1=%.4f AUC=%.4f", xgb_m["f1"], xgb_m["roc_auc"])

    # GNN (butuh MulDiGraph untuk edge nyata)
    g = load_transaction_graph()
    edges = None
    if g is not None:
        tr_addr = train_df["node_id"].astype(str).str.lower().tolist()
        te_addr = test_df["node_id"].astype(str).str.lower().tolist()
        edges = extract_edges_for_addresses(g, tr_addr, te_addr)
    gnn_pred, gnn_proba = train_gnn_proba(X_train, y_train, X_test, y_test, feats, edges)
    gnn_m = summ(y_test, gnn_pred, gnn_proba)
    log.info("GNN: F1=%.4f AUC=%.4f", gnn_m["f1"], gnn_m["roc_auc"])

    # ── ROC XGBoost saja ────────────────────────────────────────────────────────
    fx, tx, _ = roc_curve(y_test, xgb_proba)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(fx, tx, color=C_XGB, lw=2, label=f"XGBoost (AUC = {xgb_m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC — XGBoost (MulDiGraph + Etherscan)")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(ARTIFACTS / "roc_xgb.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── ROC overlay XGBoost vs GNN ───────────────────────────────────────────────
    fg, tg, _ = roc_curve(y_test, gnn_proba)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(fx, tx, color=C_XGB, lw=2, label=f"XGBoost (AUC = {xgb_m['roc_auc']:.3f})")
    ax.plot(fg, tg, color=C_GNN, lw=2, label=f"GNN-GraphSAGE (AUC = {gnn_m['roc_auc']:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("Perbandingan ROC — XGBoost vs GNN")
    ax.legend(loc="lower right"); fig.tight_layout()
    fig.savefig(ARTIFACTS / "roc_comparison.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    # ── Kurva Precision-Recall (lebih informatif utk data imbalance) ─────────────
    base_rate = float(y_test.sum()) / len(y_test)  # baseline = proporsi kelas positif
    px, rx, _ = precision_recall_curve(y_test, xgb_proba)
    pg, rg, _ = precision_recall_curve(y_test, gnn_proba)
    fig, ax = plt.subplots(figsize=(6, 5.4))
    ax.plot(rx, px, color=C_XGB, lw=2, label=f"XGBoost (PR-AUC = {xgb_m['pr_auc']:.3f})")
    ax.plot(rg, pg, color=C_GNN, lw=2, label=f"GNN-GraphSAGE (PR-AUC = {gnn_m['pr_auc']:.3f})")
    ax.axhline(base_rate, ls="--", color="#999", lw=1, label=f"baseline ({base_rate:.3f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Kurva Precision-Recall — XGBoost vs GNN")
    ax.legend(loc="lower left"); fig.tight_layout()
    fig.savefig(ARTIFACTS / "pr_curve_comparison.png", dpi=200, bbox_inches="tight"); plt.close(fig)

    report = {
        "scenario": SCENARIO,
        "n_test": int(len(y_test)),
        "n_phishing_test": int(y_test.sum()),
        "task": "binary (phishing vs bukan phishing) — ROC/AUC valid",
        "base_rate_positive": round(float(y_test.sum()) / len(y_test), 4),
        "XGBoost": xgb_m,
        "GNN_GraphSAGE": gnn_m,
        "winner_by_auc": "XGBoost" if xgb_m["roc_auc"] >= gnn_m["roc_auc"] else "GNN_GraphSAGE",
        "winner_by_f1": "XGBoost" if xgb_m["f1"] >= gnn_m["f1"] else "GNN_GraphSAGE",
    }
    (ARTIFACTS / "roc_auc_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Simpan roc_xgb.png, roc_comparison.png, pr_curve_comparison.png, roc_auc_report.json | SELESAI")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
