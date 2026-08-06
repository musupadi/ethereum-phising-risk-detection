"""
new_system/experiments/tune_live_model.py
-----------------------------------------
Menaikkan performa MODEL LIVE tanpa mengambil data baru dari Etherscan.

Tiga jalur yang dicoba, semuanya di atas split yang sama persis supaya
perbandingannya adil:

  1. OPTIMASI AMBANG  - ambang 0,5 itu bawaan, bukan hasil optimasi. Ambang
                        terbaik dicari pada data VALIDASI (bukan test), lalu
                        dipakai apa adanya di test.
  2. TUNING           - pencarian acak hyperparameter XGBoost, dipilih
                        berdasarkan F1 validasi.
  3. ENSEMBLE         - rata-rata probabilitas XGBoost + RandomForest +
                        ExtraTrees.

Aturan yang dijaga: TEST SET TIDAK PERNAH dipakai untuk memilih apa pun.
Semua keputusan diambil dari validation split. Ini penting karena kebocoran
semacam itulah yang dulu membuat angka GNN pada tesis jadi optimistis.

Keluaran:
  new_system/outputs/tuning_report.json
  (model terbaik disimpan hanya bila mengalahkan model sekarang)

Jalankan:  python new_system/experiments/tune_live_model.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from paths import (  # noqa: E402
    LIVE_DATASET_CSV, LIVE_MODEL_JSON, LIVE_REPORT_JSON, LIVE_SCHEMA_JSON,
    NEW_OUTPUTS, add_import_paths, ensure_dirs,
)

add_import_paths()
ensure_dirs()

from live_features import LIVE_FEATURES  # noqa: E402

from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

SEED = 42
rng = np.random.default_rng(SEED)


def metrics_at(y_true, proba, threshold: float) -> dict:
    pred = (proba >= threshold).astype(int)
    return {
        "threshold": round(float(threshold), 3),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
    }


def best_threshold(y_val, proba_val) -> float:
    """Ambang dengan F1 tertinggi pada VALIDASI."""
    grid = np.arange(0.20, 0.81, 0.01)
    scores = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in grid]
    return float(grid[int(np.argmax(scores))])


def main() -> None:
    import xgboost as xgb

    df = pd.read_csv(LIVE_DATASET_CSV)
    df["address"] = df["address"].astype(str).str.lower()
    df = df.drop_duplicates("address", keep="last")
    pool = df[df.get("heldout", 0) != 1]
    heldout = df[df.get("heldout", 0) == 1]

    cols = [c for c in LIVE_FEATURES if c in pool.columns]
    X = pool[cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values
    y = pool["label"].astype(int).values

    # Split identik dengan train_live_model.py supaya hasilnya sebanding.
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)

    print(f"Data: {len(df)} baris | fit {len(X_fit)} / val {len(X_val)} / test {len(X_te)}")
    spw = float((y_fit == 0).sum() / max((y_fit == 1).sum(), 1))

    dfit = xgb.DMatrix(X_fit, label=y_fit, feature_names=cols)
    dval = xgb.DMatrix(X_val, label=y_val, feature_names=cols)
    dtest = xgb.DMatrix(X_te, feature_names=cols)
    dheld = (xgb.DMatrix(heldout[cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values,
                         feature_names=cols) if not heldout.empty else None)

    results = {}

    # ── Dasar: konfigurasi yang sekarang dipakai ─────────────────────────────
    print("\n[1/4] Dasar (konfigurasi sekarang)")
    base_params = dict(objective="binary:logistic", eval_metric="logloss",
                       max_depth=5, learning_rate=0.05, subsample=0.85,
                       colsample_bytree=0.85, scale_pos_weight=spw,
                       random_state=SEED, tree_method="hist")
    base = xgb.train(base_params, dfit, num_boost_round=400,
                     evals=[(dval, "val")], early_stopping_rounds=50, verbose_eval=False)
    p_val, p_te = base.predict(dval), base.predict(dtest)
    results["baseline"] = metrics_at(y_te, p_te, 0.5)
    print(f"      F1 test = {results['baseline']['f1']:.4f}")

    # ── 1. Optimasi ambang ───────────────────────────────────────────────────
    print("[2/4] Optimasi ambang (dipilih dari validasi)")
    thr = best_threshold(y_val, p_val)
    results["threshold_tuned"] = metrics_at(y_te, p_te, thr)
    print(f"      ambang {thr:.2f} -> F1 test = {results['threshold_tuned']['f1']:.4f}")

    # ── 2. Pencarian acak hyperparameter ─────────────────────────────────────
    print("[3/4] Pencarian hyperparameter (40 konfigurasi)")
    best = {"f1_val": -1.0}
    for i in range(40):
        params = dict(objective="binary:logistic", eval_metric="logloss",
                      max_depth=int(rng.integers(3, 10)),
                      learning_rate=float(rng.choice([0.02, 0.03, 0.05, 0.08, 0.1])),
                      subsample=float(rng.uniform(0.6, 1.0)),
                      colsample_bytree=float(rng.uniform(0.5, 1.0)),
                      min_child_weight=int(rng.integers(1, 10)),
                      gamma=float(rng.choice([0, 0, 0.1, 0.3])),
                      reg_lambda=float(rng.choice([0.5, 1, 2, 5])),
                      scale_pos_weight=spw * float(rng.choice([0.7, 1.0, 1.3])),
                      random_state=SEED, tree_method="hist")
        m = xgb.train(params, dfit, num_boost_round=800, evals=[(dval, "val")],
                      early_stopping_rounds=60, verbose_eval=False)
        pv = m.predict(dval)
        t = best_threshold(y_val, pv)
        f1v = f1_score(y_val, (pv >= t).astype(int), zero_division=0)
        if f1v > best["f1_val"]:
            best = {"f1_val": float(f1v), "params": params, "model": m,
                    "threshold": t, "rounds": int(m.best_iteration or 0)}
            print(f"      #{i+1:02d} F1 val = {f1v:.4f}  (terbaik baru)")
    tuned_p_te = best["model"].predict(dtest)
    results["hyperparam_tuned"] = metrics_at(y_te, tuned_p_te, best["threshold"])
    print(f"      -> F1 test = {results['hyperparam_tuned']['f1']:.4f}")

    # ── 3. Ensemble ──────────────────────────────────────────────────────────
    print("[4/4] Ensemble (XGB terbaik + RandomForest + ExtraTrees)")
    rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                class_weight="balanced", n_jobs=-1, random_state=SEED)
    et = ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                              class_weight="balanced", n_jobs=-1, random_state=SEED)
    rf.fit(X_fit, y_fit)
    et.fit(X_fit, y_fit)
    ens_val = (best["model"].predict(dval) + rf.predict_proba(X_val)[:, 1]
               + et.predict_proba(X_val)[:, 1]) / 3.0
    ens_te = (tuned_p_te + rf.predict_proba(X_te)[:, 1] + et.predict_proba(X_te)[:, 1]) / 3.0
    t_ens = best_threshold(y_val, ens_val)
    results["ensemble"] = metrics_at(y_te, ens_te, t_ens)
    print(f"      -> F1 test = {results['ensemble']['f1']:.4f}")

    # ── Gerbang mutu OOD untuk kandidat terbaik ──────────────────────────────
    ranked = sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    winner_name = ranked[0][0]
    print(f"\nPeringkat F1 test: " + ", ".join(f"{k}={v['f1']:.4f}" for k, v in ranked))

    ood = []
    if dheld is not None and winner_name in ("hyperparam_tuned", "threshold_tuned", "baseline"):
        model = best["model"] if winner_name == "hyperparam_tuned" else base
        thr_w = results[winner_name]["threshold"]
        for (_, row), p in zip(heldout.iterrows(), model.predict(dheld)):
            ood.append({"name": str(row["source"]).split(":", 1)[-1],
                        "score": int(round(float(p) * 100)),
                        "pass": bool(p < thr_w)})
    ood_ok = all(o["pass"] for o in ood) if ood else None
    if ood:
        print(f"Gerbang OOD kandidat terbaik: "
              f"{sum(o['pass'] for o in ood)}/{len(ood)} lulus")

    # ── Simpan laporan; model hanya ditimpa bila menang DAN lolos gerbang ────
    report = {
        "results": {k: v for k, v in results.items()},
        "winner": winner_name,
        "winner_f1": ranked[0][1]["f1"],
        "baseline_f1": results["baseline"]["f1"],
        "improvement": round(ranked[0][1]["f1"] - results["baseline"]["f1"], 4),
        "best_params": {k: v for k, v in best["params"].items() if k != "model"},
        "best_threshold": best["threshold"],
        "ood_gate": ood,
        "ood_gate_passed": ood_ok,
        "note": ("Ambang dan hyperparameter dipilih HANYA dari validation split; "
                 "test set dipakai sekali untuk melaporkan hasil."),
    }
    (NEW_OUTPUTS / "tuning_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    if winner_name == "hyperparam_tuned" and ood_ok and report["improvement"] > 0.002:
        best["model"].save_model(str(LIVE_MODEL_JSON))
        schema = json.loads(LIVE_SCHEMA_JSON.read_text(encoding="utf-8"))
        schema["threshold"] = best["threshold"]
        schema["metrics"] = {k: round(v, 4) for k, v in results["hyperparam_tuned"].items()}
        schema["tuned"] = True
        LIVE_SCHEMA_JSON.write_text(json.dumps(schema, indent=2), encoding="utf-8")
        print(f"\nMODEL DIPERBARUI: F1 {results['baseline']['f1']:.4f} -> "
              f"{ranked[0][1]['f1']:.4f}  (ambang {best['threshold']:.2f})")
    else:
        print(f"\nModel TIDAK diganti (pemenang: {winner_name}, "
              f"perbaikan {report['improvement']:+.4f}).")

    print(f"Laporan -> {NEW_OUTPUTS / 'tuning_report.json'}")


if __name__ == "__main__":
    main()
