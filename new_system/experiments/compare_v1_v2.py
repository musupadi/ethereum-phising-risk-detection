"""
new_system/experiments/compare_v1_v2.py
---------------------------------------
Membandingkan fitur generasi pertama (26) dengan generasi kedua (37) secara adil.

Perbandingan dibuat seketat mungkin:
  - populasi alamat SAMA PERSIS (2.700 wallet yang sama),
  - pembagian train/val/test SAMA PERSIS (seed 42),
  - arsitektur model SAMA (ensemble XGBoost + RandomForest + ExtraTrees),
  - hyperparameter XGBoost SAMA (hasil tuning sebelumnya),
  - ambang dipilih dari validasi, test dinilai sekali.

Dengan begitu satu-satunya yang berbeda adalah HIMPUNAN FITUR-nya. Kalau F1 naik,
kenaikan itu memang berasal dari informasi baru.

Model v2 hanya diadopsi bila:
  1. unggul >= 0,005 dari v1  (agar tidak mengejar derau), DAN
  2. lulus gerbang mutu out-of-distribution.

Keluaran:
  new_system/outputs/v1_vs_v2_report.json
  bila menang: artefak model diperbarui + skema menyebut 37 fitur

Jalankan:  python new_system/experiments/compare_v1_v2.py
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
    LIVE_MODEL_JSON, LIVE_SCHEMA_JSON, NEW_ARTIFACTS, NEW_DATA, NEW_OUTPUTS,
    add_import_paths, ensure_dirs,
)

add_import_paths()
ensure_dirs()

from live_features import LIVE_FEATURES, LIVE_FEATURES_V2, V2_FEATURES  # noqa: E402

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
V2_CSV = NEW_DATA / "live_dataset_v2.csv"
MIN_GAIN = 0.005


def best_threshold(y, proba) -> float:
    grid = np.arange(0.20, 0.81, 0.01)
    return float(grid[int(np.argmax(
        [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]))])


def evaluate(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "accuracy": float(accuracy_score(y, pred)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "pr_auc": float(average_precision_score(y, proba)),
        "threshold": round(float(thr), 3),
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def run_ensemble(df, cols, xgb_params, label: str):
    """Latih ensemble pada himpunan fitur `cols`; kembalikan metrik + model."""
    import xgboost as xgb

    pool = df[df.get("heldout", 0) != 1]
    heldout = df[df.get("heldout", 0) == 1]
    clean = lambda f: f[cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values  # noqa: E731
    X, y = clean(pool), pool["label"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)

    dfit = xgb.DMatrix(X_fit, label=y_fit, feature_names=cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=cols)
    dtest = xgb.DMatrix(X_te, feature_names=cols)

    xgb_model = xgb.train(xgb_params, dfit, num_boost_round=800, evals=[(dval, "v")],
                          early_stopping_rounds=60, verbose_eval=False)
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                                n_jobs=-1, random_state=SEED).fit(X_fit, y_fit)
    et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2, class_weight="balanced",
                              n_jobs=-1, random_state=SEED).fit(X_fit, y_fit)

    ens = lambda dm, arr: (xgb_model.predict(dm) + rf.predict_proba(arr)[:, 1]  # noqa: E731
                           + et.predict_proba(arr)[:, 1]) / 3.0
    thr = best_threshold(y_val, ens(dval, X_val))
    metrics = evaluate(y_te, ens(dtest, X_te), thr)
    print(f"  {label:24s} F1={metrics['f1']:.4f}  P={metrics['precision']:.4f}  "
          f"R={metrics['recall']:.4f}  AUC={metrics['roc_auc']:.4f}")

    gate = []
    if not heldout.empty:
        Xh = clean(heldout)
        ph = ens(xgb.DMatrix(Xh, feature_names=cols), Xh)
        for (_, r), p in zip(heldout.iterrows(), ph):
            gate.append({"name": str(r["source"]).split(":", 1)[-1],
                         "score": int(round(float(p) * 100)), "pass": bool(p < thr)})

    return {"metrics": metrics, "gate": gate, "threshold": thr,
            "models": (xgb_model, rf, et), "cols": cols}


def main() -> None:
    if not V2_CSV.exists():
        print(f"{V2_CSV} belum ada. Jalankan dulu:\n"
              f"  python new_system\\ml\\rebuild_dataset_v2.py")
        sys.exit(1)

    tuning = json.loads((NEW_OUTPUTS / "tuning_report.json").read_text(encoding="utf-8"))
    xgb_params = dict(tuning["best_params"])

    df = pd.read_csv(V2_CSV)
    df["address"] = df["address"].astype(str).str.lower()
    df = df.drop_duplicates("address", keep="last")
    print(f"Dataset v2: {len(df)} baris | label {df['label'].value_counts().to_dict()}\n")

    cols_v1 = [c for c in LIVE_FEATURES if c in df.columns]
    cols_v2 = [c for c in LIVE_FEATURES_V2 if c in df.columns]
    print(f"Fitur v1: {len(cols_v1)} | v2: {len(cols_v2)} (+{len(cols_v2) - len(cols_v1)} baru)\n")

    print("Melatih pada split yang identik:")
    res_v1 = run_ensemble(df, cols_v1, xgb_params, "v1 (26 fitur)")
    res_v2 = run_ensemble(df, cols_v2, xgb_params, "v2 (37 fitur)")

    gain = res_v2["metrics"]["f1"] - res_v1["metrics"]["f1"]
    gate_ok = all(g["pass"] for g in res_v2["gate"]) if res_v2["gate"] else False
    print(f"\nSelisih F1: {gain:+.4f}")
    print(f"Gerbang OOD v2: {sum(g['pass'] for g in res_v2['gate'])}/{len(res_v2['gate'])} lulus")
    for g in res_v2["gate"]:
        print(f"    {g['name'][:26]:26s} skor {g['score']:3d}  {'LULUS' if g['pass'] else 'GAGAL'}")

    # ── Sumbangan tiap fitur baru ────────────────────────────────────────────
    xgb_v2 = res_v2["models"][0]
    gains = xgb_v2.get_score(importance_type="gain")
    new_ranked = sorted(((f, gains.get(f, 0.0)) for f in V2_FEATURES),
                        key=lambda kv: kv[1], reverse=True)
    print("\nSumbangan fitur baru (gain):")
    for f, g in new_ranked:
        print(f"    {f:24s} {g:8.1f}")

    adopt = bool(gain >= MIN_GAIN and gate_ok)
    report = {
        "v1": {"n_features": len(cols_v1), **res_v1["metrics"]},
        "v2": {"n_features": len(cols_v2), **res_v2["metrics"]},
        "gain_f1": round(gain, 4),
        "ood_gate_v2": res_v2["gate"],
        "ood_gate_passed": gate_ok,
        "new_feature_gain": [{"feature": f, "gain": float(g)} for f, g in new_ranked],
        "adopted": adopt,
        "min_gain_required": MIN_GAIN,
        "note": ("Populasi, split, arsitektur, dan hyperparameter identik; hanya "
                 "himpunan fitur yang berbeda."),
    }
    (NEW_OUTPUTS / "v1_vs_v2_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if adopt:
        xgb_model, rf, et = res_v2["models"]
        xgb_model.save_model(str(LIVE_MODEL_JSON))
        joblib.dump(rf, NEW_ARTIFACTS / "model_live_rf.pkl")
        joblib.dump(et, NEW_ARTIFACTS / "model_live_et.pkl")
        schema = json.loads(LIVE_SCHEMA_JSON.read_text(encoding="utf-8"))
        schema.update({
            "features": cols_v2,
            "threshold": res_v2["threshold"],
            "feature_generation": "v2",
            "metrics": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in res_v2["metrics"].items()},
        })
        LIVE_SCHEMA_JSON.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(f"\nMODEL DIPERBARUI ke v2: F1 {res_v1['metrics']['f1']:.4f} -> "
              f"{res_v2['metrics']['f1']:.4f}")
    else:
        why = "kenaikan di bawah ambang" if gain < MIN_GAIN else "gerbang OOD gagal"
        print(f"\nModel TIDAK diganti ({why}). Artefak lama dipertahankan.")

    print(f"Laporan -> {NEW_OUTPUTS / 'v1_vs_v2_report.json'}")


if __name__ == "__main__":
    main()
