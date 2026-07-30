"""
ml/feature_selection.py
-----------------------
Seleksi fitur (catatan bimbingan poin 3,5,9,10): dari 25 fitur skenario
MulDiGraph+Etherscan, tentukan subset terbaik memakai importance (XGBoost gain + RF)
dan pemangkasan korelasi tinggi. Bandingkan F1 set penuh vs tereduksi.

Anti-leakage: capping (p1..p99) & scaling di-fit HANYA pada data train, lalu diterapkan
ke test. Imbalance ditangani lewat scale_pos_weight (bukan buang data).

Sumber : data/processed/hybrid_dataset_clean.csv
Output : ml/artifacts/feature_selection_report.json
         ml/artifacts/eda/06_feature_importance.png

Run: python ml/feature_selection.py
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"
OUTDIR = ARTIFACTS / "eda"
OUTDIR.mkdir(parents=True, exist_ok=True)

CLEAN = PROCESSED / "hybrid_dataset_clean.csv"
REPORT = ARTIFACTS / "feature_selection_report.json"

# 25 fitur skenario MulDiGraph+Etherscan (prod 26 - created_at_ts yang sudah dibuang)
SCENARIO_FEATS = [
    "in_degree", "out_degree", "unique_senders", "unique_receivers",
    "total_received", "total_sent", "avg_in_amount", "avg_out_amount",
    "max_in_amount", "max_out_amount", "min_in_amount", "min_out_amount",
    "std_in_amount", "std_out_amount", "in_out_ratio", "sent_received_ratio",
    "timestamp_spread_in", "timestamp_spread_out", "graph_present",
    "balance", "transaction_count", "activity_score", "avg_tx_value",
    "tx_frequency", "unique_counterparties",
]
CORR_THRESHOLD = 0.95
C_BAR = "#4E79A7"


def fit_caps(X: pd.DataFrame):
    return {c: (X[c].quantile(.01), X[c].quantile(.99)) for c in X.columns}


def apply_caps(X: pd.DataFrame, caps):
    X = X.copy()
    for c, (lo, hi) in caps.items():
        if hi > lo:
            X[c] = X[c].clip(lo, hi)
    return X


def evaluate(feats, Xtr, ytr, Xte, yte, spw):
    m = XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, scale_pos_weight=spw,
        eval_metric="logloss", n_jobs=-1, random_state=42,
    )
    m.fit(Xtr[feats], ytr)
    p = m.predict(Xte[feats])
    return {
        "n_features": len(feats),
        "accuracy": round(accuracy_score(yte, p), 4),
        "precision": round(precision_score(yte, p), 4),
        "recall": round(recall_score(yte, p), 4),
        "f1": round(f1_score(yte, p), 4),
    }


def main():
    df = pd.read_csv(CLEAN)
    feats = [f for f in SCENARIO_FEATS if f in df.columns]
    log.info("Loaded %d rows | %d fitur skenario", len(df), len(feats))

    Xtr, Xte, ytr, yte = train_test_split(
        df[feats], df["label"].values, test_size=0.2, random_state=42, stratify=df["label"],
    )
    caps = fit_caps(Xtr)            # di-fit pada TRAIN saja
    Xtr, Xte = apply_caps(Xtr, caps), apply_caps(Xte, caps)
    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    log.info("scale_pos_weight = %.2f", spw)

    # --- Importance: XGBoost (gain) + RandomForest ---
    xgb = XGBClassifier(n_estimators=250, max_depth=6, learning_rate=0.05,
                        subsample=0.85, colsample_bytree=0.85, scale_pos_weight=spw,
                        eval_metric="logloss", n_jobs=-1, random_state=42)
    xgb.fit(Xtr, ytr)
    rf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                n_jobs=-1, random_state=42)
    rf.fit(Xtr, ytr)

    imp = pd.DataFrame({
        "feature": feats,
        "xgb_gain": xgb.feature_importances_,
        "rf": rf.feature_importances_,
    })
    # skor gabungan = rata-rata importance ternormalisasi
    imp["xgb_n"] = imp["xgb_gain"] / imp["xgb_gain"].sum()
    imp["rf_n"] = imp["rf"] / imp["rf"].sum()
    imp["combined"] = (imp["xgb_n"] + imp["rf_n"]) / 2
    imp = imp.sort_values("combined", ascending=False).reset_index(drop=True)
    log.info("Top-5 fitur: %s", list(imp["feature"].head()))

    # --- Pemangkasan korelasi tinggi (buang yg importance-nya lebih rendah) ---
    corr = Xtr[feats].corr().abs()
    rank = {f: i for i, f in enumerate(imp["feature"])}  # 0 = paling penting
    drop_corr = set()
    for i in range(len(feats)):
        for j in range(i + 1, len(feats)):
            a, b = feats[i], feats[j]
            if corr.loc[a, b] >= CORR_THRESHOLD:
                worse = a if rank[a] > rank[b] else b
                drop_corr.add(worse)
    log.info("Dibuang karena korelasi >= %.2f: %s", CORR_THRESHOLD, sorted(drop_corr) or "(tidak ada)")

    # --- Bandingkan beberapa ukuran subset ---
    ranked = list(imp["feature"])
    ranked_pruned = [f for f in ranked if f not in drop_corr]
    experiments = {
        "full_25": ranked,
        "corr_pruned": ranked_pruned,
        "top_20": ranked[:20],
        "top_18": ranked[:18],
        "top_15": ranked[:15],
    }
    results = {name: evaluate(fl, Xtr, ytr, Xte, yte, spw) for name, fl in experiments.items()}
    for name, r in results.items():
        log.info("%-12s -> F1=%.4f  P=%.4f  R=%.4f  (n=%d)", name, r["f1"], r["precision"], r["recall"], r["n_features"])

    best = max(results, key=lambda k: results[k]["f1"])
    log.info("TERBAIK by F1: %s (F1=%.4f)", best, results[best]["f1"])

    # --- Figur importance ---
    top = imp.head(25).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 8))
    ax.barh(top["feature"], top["combined"], color=C_BAR)
    ax.set_xlabel("Importance gabungan (XGBoost gain + RF, ternormalisasi)")
    ax.set_title("Peringkat Kepentingan Fitur")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(OUTDIR / "06_feature_importance.png", bbox_inches="tight", dpi=150)
    plt.close(fig)
    log.info("Figur -> 06_feature_importance.png")

    # --- Simpan report ---
    REPORT.write_text(json.dumps({
        "scenario": "muldigraph_plus_etherscan (clean, created_at_ts dibuang)",
        "n_features_input": len(feats),
        "scale_pos_weight": round(spw, 3),
        "importance_ranking": imp[["feature", "xgb_n", "rf_n", "combined"]].round(5).to_dict("records"),
        "correlation_threshold": CORR_THRESHOLD,
        "dropped_by_correlation": sorted(drop_corr),
        "experiments": {k: {"features": v, **results[k]} for k, v in experiments.items()},
        "best_by_f1": best,
        "recommended_features": experiments[best],
    }, indent=2), encoding="utf-8")
    log.info("Report -> %s | SELESAI", REPORT.name)


if __name__ == "__main__":
    main()
