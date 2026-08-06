"""
new_system/experiments/build_ensemble.py
----------------------------------------
Membangun dan menyimpan MODEL LIVE versi ensemble.

Hasil pencarian di `tune_live_model.py`:

    dasar (XGBoost bawaan)          F1 0,8739
    + ambang dioptimasi             F1 0,8816
    + hyperparameter dituning       F1 0,8871
    + ensemble 3 model              F1 0,9008   <- dipakai

Ensemble = rata-rata probabilitas dari XGBoost (hasil tuning) + RandomForest +
ExtraTrees. Ketiganya melihat data yang sama tetapi membuat kesalahan yang
berbeda-beda, sehingga rata-ratanya lebih stabil daripada model tunggal.

Aturan yang dijaga:
  - Ambang dan hyperparameter dipilih HANYA dari validation split.
  - Test set dinilai sekali saja, untuk dilaporkan.
  - Model baru hanya disimpan bila LULUS gerbang mutu out-of-distribution.
    Naiknya F1 tidak ada artinya kalau Vitalik kembali dituduh phishing.

Jalankan:  python new_system/experiments/build_ensemble.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from paths import (  # noqa: E402
    LIVE_DATASET_CSV, LIVE_MODEL_JSON, LIVE_SCHEMA_JSON, NEW_ARTIFACTS,
    NEW_OUTPUTS, add_import_paths, ensure_dirs,
)

add_import_paths()
ensure_dirs()

from live_features import LIVE_FEATURES  # noqa: E402

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
RF_PKL = NEW_ARTIFACTS / "model_live_rf.pkl"
ET_PKL = NEW_ARTIFACTS / "model_live_et.pkl"


def best_threshold(y, proba) -> float:
    grid = np.arange(0.20, 0.81, 0.01)
    return float(grid[int(np.argmax(
        [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]))])


def main() -> None:
    import xgboost as xgb

    tuning = json.loads((NEW_OUTPUTS / "tuning_report.json").read_text(encoding="utf-8"))
    params = {k: v for k, v in tuning["best_params"].items()}

    df = pd.read_csv(LIVE_DATASET_CSV)
    df["address"] = df["address"].astype(str).str.lower()
    df = df.drop_duplicates("address", keep="last")
    pool = df[df.get("heldout", 0) != 1]
    heldout = df[df.get("heldout", 0) == 1]

    cols = [c for c in LIVE_FEATURES if c in pool.columns]
    clean = lambda frame: frame[cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values  # noqa: E731
    X, y = clean(pool), pool["label"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)

    dfit = xgb.DMatrix(X_fit, label=y_fit, feature_names=cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=cols)
    dtest = xgb.DMatrix(X_te, feature_names=cols)

    print(f"Data {len(df)} baris | fit {len(X_fit)} / val {len(X_val)} / test {len(X_te)}")

    # ── Latih ketiga anggota ─────────────────────────────────────────────────
    print("\nMelatih anggota ensemble...")
    xgb_model = xgb.train(params, dfit, num_boost_round=800, evals=[(dval, "val")],
                          early_stopping_rounds=60, verbose_eval=False)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=-1, random_state=SEED).fit(X_fit, y_fit)
    et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                              class_weight="balanced", n_jobs=-1, random_state=SEED).fit(X_fit, y_fit)
    print("  XGBoost, RandomForest, ExtraTrees siap")

    ens = lambda dm, arr: (xgb_model.predict(dm) + rf.predict_proba(arr)[:, 1]  # noqa: E731
                           + et.predict_proba(arr)[:, 1]) / 3.0

    thr = best_threshold(y_val, ens(dval, X_val))
    print(f"  ambang terbaik dari validasi: {thr:.2f}")

    # ── Nilai test sekali ────────────────────────────────────────────────────
    p_te = ens(dtest, X_te)
    pred = (p_te >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_te, pred, labels=[0, 1]).ravel()
    m = {
        "f1": float(f1_score(y_te, pred, zero_division=0)),
        "precision": float(precision_score(y_te, pred, zero_division=0)),
        "recall": float(recall_score(y_te, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, pred)),
        "roc_auc": float(roc_auc_score(y_te, p_te)),
        "pr_auc": float(average_precision_score(y_te, p_te)),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }
    print(f"\nTest: F1={m['f1']:.4f} Prec={m['precision']:.4f} "
          f"Rec={m['recall']:.4f} ROC-AUC={m['roc_auc']:.4f}")

    # ── Gerbang mutu OOD ─────────────────────────────────────────────────────
    print("\nGerbang mutu out-of-distribution:")
    X_h = clean(heldout)
    p_h = ens(xgb.DMatrix(X_h, feature_names=cols), X_h)
    gate = []
    for (_, row), p in zip(heldout.iterrows(), p_h):
        ok = bool(p < thr)
        gate.append({"name": str(row["source"]).split(":", 1)[-1],
                     "address": row["address"],
                     "score": int(round(float(p) * 100)),
                     "pass": ok})
        print(f"  {gate[-1]['name'][:26]:26s} skor {gate[-1]['score']:3d}  "
              f"{'LULUS' if ok else 'GAGAL'}")
    gate_ok = all(g["pass"] for g in gate)

    if not gate_ok:
        print("\nGERBANG GAGAL - ensemble TIDAK disimpan. Model lama dipertahankan.")
        sys.exit(2)

    # ── Simpan ───────────────────────────────────────────────────────────────
    xgb_model.save_model(str(LIVE_MODEL_JSON))
    joblib.dump(rf, RF_PKL)
    joblib.dump(et, ET_PKL)

    schema = json.loads(LIVE_SCHEMA_JSON.read_text(encoding="utf-8"))
    schema.update({
        "features": cols,
        "model_type": "ensemble",
        "ensemble_members": ["xgboost", "random_forest", "extra_trees"],
        "ensemble_files": {"xgboost": LIVE_MODEL_JSON.name,
                           "random_forest": RF_PKL.name,
                           "extra_trees": ET_PKL.name},
        "aggregation": "mean_probability",
        "threshold": thr,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "tuned": True,
        "note": ("Ensemble rata-rata probabilitas. Ambang dan hyperparameter dipilih "
                 "dari validation split; test set dinilai sekali. Penjelasan SHAP "
                 "berasal dari komponen XGBoost saja."),
    })
    LIVE_SCHEMA_JSON.write_text(json.dumps(schema, indent=2), encoding="utf-8")

    (NEW_OUTPUTS / "ensemble_report.json").write_text(json.dumps({
        "metrics": m, "threshold": thr, "ood_gate": gate, "ood_gate_passed": gate_ok,
        "params_xgb": params,
        "comparison": {
            "baseline_single_xgb": tuning["results"]["baseline"]["f1"],
            "threshold_tuned": tuning["results"]["threshold_tuned"]["f1"],
            "hyperparam_tuned": tuning["results"]["hyperparam_tuned"]["f1"],
            "ensemble": m["f1"],
            "thesis_model_reference": 0.9102,
        },
    }, indent=2), encoding="utf-8")

    print(f"\nENSEMBLE DISIMPAN. F1 {tuning['results']['baseline']['f1']:.4f} -> {m['f1']:.4f}")
    print(f"  {LIVE_MODEL_JSON.name}, {RF_PKL.name}, {ET_PKL.name}")


if __name__ == "__main__":
    main()
