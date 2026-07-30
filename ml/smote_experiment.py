"""
ml/smote_experiment.py
----------------------
Tindak lanjut audit catatan #8: menguji SMOTE / oversampling secara EMPIRIS (bukan
sekadar alasan konseptual) dan membandingkannya dengan strategi imbalance lain.

Konsisten dengan `tune_xgboost.py`:
  - data  : hybrid_dataset_clean.csv, fitur top_20 hasil seleksi
  - split : test 20% (random_state=42, stratified), identik
  - caps  : p1..p99 di-fit HANYA pada train (anti-leakage)
  - model : XGBoost dgn best_params dari xgb_tuning_results.json

PENTING (anti-leakage): resampling (SMOTE/oversampling) HANYA diterapkan pada data
train; test dibiarkan apa adanya. Model berbasis resampling memakai scale_pos_weight=1
(data sudah diseimbangkan), agar tidak dobel-koreksi.

Run: python ml/smote_experiment.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE, BorderlineSMOTE, RandomOverSampler
from imblearn.combine import SMOTETomek
from sklearn.metrics import (accuracy_score, average_precision_score, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"
CLEAN = PROCESSED / "hybrid_dataset_clean.csv"
FS_REPORT = ARTIFACTS / "feature_selection_report.json"
TUNE_REPORT = ARTIFACTS / "xgb_tuning_results.json"
OUT = ARTIFACTS / "smote_experiment.json"
RANDOM_STATE = 42


def fit_caps(X: pd.DataFrame):
    return {c: (X[c].quantile(.01), X[c].quantile(.99)) for c in X.columns}


def apply_caps(X: pd.DataFrame, caps) -> pd.DataFrame:
    X = X.copy()
    for c, (lo, hi) in caps.items():
        if hi > lo:
            X[c] = X[c].clip(lo, hi)
    return X


def evaluate(model, Xte, yte) -> dict:
    proba = model.predict_proba(Xte)[:, 1]
    pred = (proba >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(yte, pred)), 4),
        "precision": round(float(precision_score(yte, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(yte, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(yte, pred, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(yte, proba)), 4),
        "pr_auc": round(float(average_precision_score(yte, proba)), 4),
    }


def main() -> None:
    df = pd.read_csv(CLEAN)
    feats = json.loads(FS_REPORT.read_text(encoding="utf-8"))["experiments"]["top_20"]["features"]
    best_params = json.loads(TUNE_REPORT.read_text(encoding="utf-8"))["best_params"]
    log.info("Data=%d baris | %d fitur (top_20) | best_params dari tuning", len(df), len(feats))

    Xtr_full, Xte, ytr_full, yte = train_test_split(
        df[feats], df["label"].values, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )
    caps = fit_caps(Xtr_full)
    Xtr_c = apply_caps(Xtr_full, caps)
    Xte_c = apply_caps(Xte, caps)

    spw_full = float((ytr_full == 0).sum() / (ytr_full == 1).sum())
    n_pos, n_neg = int((ytr_full == 1).sum()), int((ytr_full == 0).sum())
    log.info("Train: %d normal / %d phishing (spw=%.3f)", n_neg, n_pos, spw_full)

    def make_model(scale_pos_weight=1.0):
        return XGBClassifier(**best_params, scale_pos_weight=scale_pos_weight,
                             eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)

    results = {}

    # ── Acuan: scale_pos_weight (tanpa resampling) ──────────────────────────────
    m = make_model(scale_pos_weight=spw_full).fit(Xtr_c, ytr_full)
    results["scale_pos_weight"] = evaluate(m, Xte_c, yte)
    log.info("scale_pos_weight       F1=%.4f", results["scale_pos_weight"]["f1"])

    # ── Resampling (hanya di train). Model pakai spw=1 agar tidak dobel koreksi ──
    resamplers = {
        "SMOTE": SMOTE(random_state=RANDOM_STATE),
        "BorderlineSMOTE": BorderlineSMOTE(random_state=RANDOM_STATE),
        "SMOTETomek": SMOTETomek(random_state=RANDOM_STATE),
        "RandomOverSampler": RandomOverSampler(random_state=RANDOM_STATE),
    }
    for name, rs in resamplers.items():
        Xr, yr = rs.fit_resample(Xtr_c, ytr_full)
        m = make_model(scale_pos_weight=1.0).fit(Xr, yr)
        results[name] = evaluate(m, Xte_c, yte)
        results[name]["train_size_after"] = int(len(yr))
        results[name]["balance_after"] = f"{int((yr==0).sum())}:{int((yr==1).sum())}"
        log.info("%-18s F1=%.4f (train %d, %s)", name, results[name]["f1"],
                 len(yr), results[name]["balance_after"])

    best = max(results, key=lambda k: results[k]["f1"])
    report = {
        "dataset": "hybrid_dataset_clean.csv",
        "features": feats,
        "n_features": len(feats),
        "test_size": int(len(yte)),
        "n_phishing_test": int(yte.sum()),
        "train_original_balance": f"{n_neg}:{n_pos}",
        "note": ("Resampling hanya pada train (anti-leakage); test asli. Model resampling "
                 "memakai scale_pos_weight=1 agar tidak dobel-koreksi imbalance."),
        "results": results,
        "best_by_f1": best,
        "conclusion": (f"'{best}' memberikan F1 tertinggi ({results[best]['f1']}). "
                       "Dibandingkan dengan hasil di xgb_tuning_results.json untuk keputusan akhir."),
    }
    OUT.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    log.info("Hasil -> %s | SELESAI", OUT.name)
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
