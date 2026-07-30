"""
ml/tune_xgboost.py
------------------
Tahap D (catatan bimbingan poin 2, 8, 12): hyperparameter tuning XGBoost +
perbandingan strategi penanganan imbalance. Tujuannya membuktikan apakah F1 bisa
naik dari baseline 0.9102 menuju target >= 0.95 (minimal 0.85).

Alur:
  1. Muat data bersih (hybrid_dataset_clean.csv), pakai subset fitur hasil seleksi
     (top_20 dari feature_selection.py) sesuai arahan dosen "coba pakai 20 fitur".
  2. Capping p1..p99 di-fit HANYA pada train (anti-leakage), lalu diterapkan ke test.
  3. RandomizedSearchCV (StratifiedKFold 5-fold, scoring=f1) untuk cari hyperparameter
     terbaik XGBoost -- tanpa dependency tambahan (tidak butuh Optuna).
  4. Threshold tuning: cari ambang proba terbaik untuk F1 pada validation (bukan test).
  5. D2 -- bandingkan 3 strategi imbalance dengan hyperparameter terbaik:
        (a) undersampling 1:5   (b) scale_pos_weight (tanpa buang data)   (c) class-weight sampel
  6. Simpan ml/artifacts/xgb_tuning_results.json + kurva.

Run: python ml/tune_xgboost.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import randint, uniform
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, precision_recall_curve)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, train_test_split
from xgboost import XGBClassifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"
CLEAN = PROCESSED / "hybrid_dataset_clean.csv"
FS_REPORT = ARTIFACTS / "feature_selection_report.json"
OUT = ARTIFACTS / "xgb_tuning_results.json"

RANDOM_STATE = 42


# ── Fitur: pakai top_20 hasil seleksi (arahan dosen ~20 fitur) ──────────────────
def load_features() -> list[str]:
    rep = json.loads(FS_REPORT.read_text(encoding="utf-8"))
    return rep["experiments"]["top_20"]["features"]


def fit_caps(X: pd.DataFrame):
    return {c: (X[c].quantile(.01), X[c].quantile(.99)) for c in X.columns}


def apply_caps(X: pd.DataFrame, caps) -> pd.DataFrame:
    X = X.copy()
    for c, (lo, hi) in caps.items():
        if hi > lo:
            X[c] = X[c].clip(lo, hi)
    return X


def metrics_at(y_true, proba, thr) -> dict:
    p = (proba >= thr).astype(int)
    return {
        "threshold": round(float(thr), 4),
        "accuracy": round(float(accuracy_score(y_true, p)), 4),
        "precision": round(float(precision_score(y_true, p, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, p, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, p, zero_division=0)), 4),
    }


def best_threshold(y_val, proba_val) -> float:
    """Ambang yang memaksimalkan F1 pada set validasi."""
    prec, rec, thr = precision_recall_curve(y_val, proba_val)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros_like(prec), where=(prec + rec) > 0)
    # thr punya panjang len(prec)-1; sejajarkan
    best_i = int(np.nanargmax(f1[:-1])) if len(thr) else 0
    return float(thr[best_i]) if len(thr) else 0.5


def main() -> None:
    df = pd.read_csv(CLEAN)
    feats = load_features()
    log.info("Data bersih: %d baris | %d fitur (top_20 hasil seleksi)", len(df), len(feats))

    # split test 20% (sama seperti pipeline lain: random_state=42, stratify)
    Xtr_full, Xte, ytr_full, yte = train_test_split(
        df[feats], df["label"].values, test_size=0.2, random_state=RANDOM_STATE, stratify=df["label"],
    )
    # sisihkan validasi 20% dari train untuk threshold tuning
    Xtr, Xval, ytr, yval = train_test_split(
        Xtr_full, ytr_full, test_size=0.2, random_state=RANDOM_STATE, stratify=ytr_full,
    )

    caps = fit_caps(Xtr)                       # fit di train saja
    Xtr_c, Xval_c, Xte_c = apply_caps(Xtr, caps), apply_caps(Xval, caps), apply_caps(Xte, caps)
    # untuk refit akhir pada train+val, fit caps pada train_full
    caps_full = fit_caps(Xtr_full)
    Xtrf_c = apply_caps(Xtr_full, caps_full)

    spw = float((ytr == 0).sum() / (ytr == 1).sum())
    log.info("scale_pos_weight (train) = %.3f", spw)

    # ── 1. Baseline (parameter lama train_production_model) untuk pembanding ─────
    base = XGBClassifier(
        n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85,
        colsample_bytree=0.85, scale_pos_weight=spw, eval_metric="logloss",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    base.fit(Xtr_c, ytr)
    base_test = metrics_at(yte, base.predict_proba(Xte_c)[:, 1], 0.5)
    log.info("Baseline (param lama, thr=0.5) F1_test=%.4f", base_test["f1"])

    # ── 2. RandomizedSearchCV ───────────────────────────────────────────────────
    param_dist = {
        "n_estimators": randint(150, 600),
        "max_depth": randint(3, 10),
        "learning_rate": uniform(0.01, 0.19),
        "subsample": uniform(0.6, 0.4),
        "colsample_bytree": uniform(0.6, 0.4),
        "min_child_weight": randint(1, 8),
        "gamma": uniform(0.0, 0.5),
        "reg_lambda": uniform(0.5, 3.0),
    }
    search = RandomizedSearchCV(
        estimator=XGBClassifier(
            scale_pos_weight=spw, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE,
        ),
        param_distributions=param_dist,
        n_iter=60,
        scoring="f1",
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE),
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    log.info("Menjalankan RandomizedSearchCV (60 iterasi x 5-fold) ...")
    search.fit(Xtr_c, ytr)
    best_params = search.best_params_
    log.info("CV F1 terbaik=%.4f | params=%s", search.best_value if hasattr(search, "best_value") else search.best_score_, best_params)

    # ── 3. Threshold tuning pada validasi ───────────────────────────────────────
    tuned = XGBClassifier(
        **best_params, scale_pos_weight=spw, eval_metric="logloss",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    tuned.fit(Xtr_c, ytr)
    proba_val = tuned.predict_proba(Xval_c)[:, 1]
    thr = best_threshold(yval, proba_val)
    log.info("Threshold terbaik (val) = %.4f", thr)

    # ── 4. Refit akhir pada train+val, evaluasi test pada thr default & thr tuned ─
    spw_full = float((ytr_full == 0).sum() / (ytr_full == 1).sum())
    final = XGBClassifier(
        **best_params, scale_pos_weight=spw_full, eval_metric="logloss",
        n_jobs=-1, random_state=RANDOM_STATE,
    )
    final.fit(Xtrf_c, ytr_full)
    proba_te = final.predict_proba(Xte_c)[:, 1]
    tuned_default = metrics_at(yte, proba_te, 0.5)
    tuned_thr = metrics_at(yte, proba_te, thr)
    log.info("Tuned thr=0.5  F1_test=%.4f | tuned thr=%.3f F1_test=%.4f",
             tuned_default["f1"], thr, tuned_thr["f1"])

    # ── 5. D2: perbandingan strategi imbalance (pakai best_params) ──────────────
    imbalance = {}
    # (a) undersampling 1:5 pada train_full
    tr = Xtrf_c.copy()
    tr["label"] = ytr_full
    pos = tr[tr.label == 1]
    neg = tr[tr.label == 0].sample(n=min(len(tr[tr.label == 0]), 5 * len(pos)), random_state=RANDOM_STATE)
    us = pd.concat([pos, neg]).sample(frac=1, random_state=RANDOM_STATE)
    m_us = XGBClassifier(**best_params, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)
    m_us.fit(us[feats], us["label"].values)
    imbalance["undersampling_1to5"] = metrics_at(yte, m_us.predict_proba(Xte_c)[:, 1], 0.5)

    # (b) scale_pos_weight (tanpa buang data) -> sama seperti final@0.5
    imbalance["scale_pos_weight"] = tuned_default

    # (c) class-weight per-sampel (sample_weight) tanpa scale_pos_weight
    sw = np.where(ytr_full == 1, spw_full, 1.0)
    m_cw = XGBClassifier(**best_params, eval_metric="logloss", n_jobs=-1, random_state=RANDOM_STATE)
    m_cw.fit(Xtrf_c, ytr_full, sample_weight=sw)
    imbalance["sample_weight"] = metrics_at(yte, m_cw.predict_proba(Xte_c)[:, 1], 0.5)

    best_strategy = max(imbalance, key=lambda k: imbalance[k]["f1"])
    log.info("Strategi imbalance terbaik: %s (F1=%.4f)", best_strategy, imbalance[best_strategy]["f1"])

    # ── Simpan ──────────────────────────────────────────────────────────────────
    result = {
        "dataset": "hybrid_dataset_clean.csv",
        "n_rows": len(df),
        "features": feats,
        "n_features": len(feats),
        "cv": "StratifiedKFold(5), scoring=f1, RandomizedSearchCV(n_iter=60)",
        "scale_pos_weight_full": round(spw_full, 3),
        "baseline_old_params": base_test,
        "best_params": {k: (int(v) if isinstance(v, (np.integer,)) else float(v) if isinstance(v, (np.floating,)) else v)
                        for k, v in best_params.items()},
        "cv_best_f1": round(float(search.best_score_), 4),
        "tuned_threshold_val": round(thr, 4),
        "tuned_test_thr_default": tuned_default,
        "tuned_test_thr_tuned": tuned_thr,
        "imbalance_comparison": imbalance,
        "best_imbalance_strategy": best_strategy,
        "target": {"goal_f1": 0.95, "min_f1": 0.85,
                   "achieved_f1": max(tuned_default["f1"], tuned_thr["f1"]),
                   "meets_goal": max(tuned_default["f1"], tuned_thr["f1"]) >= 0.95,
                   "meets_min": max(tuned_default["f1"], tuned_thr["f1"]) >= 0.85},
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    log.info("Hasil -> %s | SELESAI", OUT.name)
    print(json.dumps({k: result[k] for k in
                      ["baseline_old_params", "cv_best_f1", "tuned_test_thr_default",
                       "tuned_test_thr_tuned", "best_imbalance_strategy", "target"]}, indent=2))


if __name__ == "__main__":
    main()
