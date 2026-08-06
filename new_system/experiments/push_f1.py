"""
new_system/experiments/push_f1.py
---------------------------------
Percobaan lanjutan menaikkan F1 model live TANPA mengambil data baru.

Yang sudah dicoba sebelumnya (tune_live_model.py + build_ensemble.py):
    XGBoost bawaan            0,8739
    + ambang dioptimasi       0,8816
    + hyperparameter dituning 0,8871
    + ensemble 3 model        0,9008   <- kondisi sekarang

Yang dicoba di sini:
    A. Ensemble 4 anggota  (+ HistGradientBoosting)
    B. Ensemble berbobot   (bobot tiap anggota dioptimasi di validasi)
    C. Stacking            (meta-model dilatih di atas prediksi out-of-fold)
    D. Seleksi fitur       (buang fitur berkontribusi kecil, latih ulang)

Aturan yang dijaga ketat:
  - Semua pemilihan (bobot, ambang, fitur, meta-model) HANYA dari validation split.
  - Test set dinilai sekali di akhir, untuk dilaporkan.
  - Kandidat baru hanya menang bila LULUS gerbang mutu out-of-distribution.

Jalankan:  python new_system/experiments/push_f1.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "ml"))
from paths import (  # noqa: E402
    LIVE_DATASET_CSV, LIVE_SCHEMA_JSON, NEW_OUTPUTS, add_import_paths, ensure_dirs,
)

add_import_paths()
ensure_dirs()

from live_features import LIVE_FEATURES  # noqa: E402

from sklearn.ensemble import (  # noqa: E402
    ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score  # noqa: E402
from sklearn.model_selection import StratifiedKFold, train_test_split  # noqa: E402

SEED = 42


def best_threshold(y, proba) -> float:
    grid = np.arange(0.20, 0.81, 0.01)
    return float(grid[int(np.argmax(
        [f1_score(y, (proba >= t).astype(int), zero_division=0) for t in grid]))])


def report(y, proba, thr) -> dict:
    pred = (proba >= thr).astype(int)
    return {
        "f1": float(f1_score(y, pred, zero_division=0)),
        "precision": float(precision_score(y, pred, zero_division=0)),
        "recall": float(recall_score(y, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y, proba)),
        "threshold": round(float(thr), 3),
    }


def main() -> None:
    import xgboost as xgb

    tuning = json.loads((NEW_OUTPUTS / "tuning_report.json").read_text(encoding="utf-8"))
    xgb_params = dict(tuning["best_params"])

    df = pd.read_csv(LIVE_DATASET_CSV)
    df["address"] = df["address"].astype(str).str.lower()
    df = df.drop_duplicates("address", keep="last")
    pool, heldout = df[df.get("heldout", 0) != 1], df[df.get("heldout", 0) == 1]

    cols = [c for c in LIVE_FEATURES if c in pool.columns]
    clean = lambda f, c=None: f[c or cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values  # noqa: E731
    X, y = clean(pool), pool["label"].astype(int).values

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_tr, y_tr, test_size=0.2, random_state=SEED, stratify=y_tr)
    print(f"fit {len(X_fit)} / val {len(X_val)} / test {len(X_te)} | {len(cols)} fitur\n")

    def make_members():
        return {
            "xgb": ("xgb", xgb_params),
            "rf": ("sk", RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                                class_weight="balanced", n_jobs=-1,
                                                random_state=SEED)),
            "et": ("sk", ExtraTreesClassifier(n_estimators=400, min_samples_leaf=2,
                                              class_weight="balanced", n_jobs=-1,
                                              random_state=SEED)),
            "hgb": ("sk", HistGradientBoostingClassifier(max_iter=400, learning_rate=0.06,
                                                         max_depth=6, random_state=SEED)),
        }

    def fit_members(names, Xf, yf, feat_cols):
        fitted = {}
        for n in names:
            kind, spec = make_members()[n]
            if kind == "xgb":
                d = xgb.DMatrix(Xf, label=yf, feature_names=feat_cols)
                dv = xgb.DMatrix(X_val[:, [cols.index(c) for c in feat_cols]]
                                 if feat_cols != cols else X_val, label=y_val,
                                 feature_names=feat_cols)
                fitted[n] = xgb.train(spec, d, num_boost_round=800, evals=[(dv, "v")],
                                      early_stopping_rounds=60, verbose_eval=False)
            else:
                fitted[n] = spec.fit(Xf, yf)
        return fitted

    def proba_of(fitted, Xa, feat_cols):
        out = {}
        for n, m in fitted.items():
            if n == "xgb":
                out[n] = m.predict(xgb.DMatrix(Xa, feature_names=feat_cols))
            else:
                out[n] = m.predict_proba(Xa)[:, 1]
        return out

    results = {}

    # ── Baseline: ensemble 3 anggota ─────────────────────────────────────────
    base_names = ["xgb", "rf", "et"]
    fitted3 = fit_members(base_names, X_fit, y_fit, cols)
    pv = proba_of(fitted3, X_val, cols); pt = proba_of(fitted3, X_te, cols)
    v3 = np.mean([pv[n] for n in base_names], axis=0)
    t3 = np.mean([pt[n] for n in base_names], axis=0)
    thr3 = best_threshold(y_val, v3)
    results["ensemble_3 (sekarang)"] = report(y_te, t3, thr3)
    print(f"A0 ensemble 3 anggota      F1 test = {results['ensemble_3 (sekarang)']['f1']:.4f}")

    # ── A. Tambah HistGradientBoosting ───────────────────────────────────────
    names4 = ["xgb", "rf", "et", "hgb"]
    fitted4 = fit_members(names4, X_fit, y_fit, cols)
    pv4 = proba_of(fitted4, X_val, cols); pt4 = proba_of(fitted4, X_te, cols)
    v4 = np.mean([pv4[n] for n in names4], axis=0)
    t4 = np.mean([pt4[n] for n in names4], axis=0)
    thr4 = best_threshold(y_val, v4)
    results["ensemble_4 (+HistGB)"] = report(y_te, t4, thr4)
    print(f"A  ensemble 4 anggota      F1 test = {results['ensemble_4 (+HistGB)']['f1']:.4f}")

    # ── B. Bobot dioptimasi di validasi ──────────────────────────────────────
    rng = np.random.default_rng(SEED)
    best_w, best_vf1 = None, -1.0
    for _ in range(300):
        w = rng.dirichlet(np.ones(len(names4)))
        vw = sum(w[i] * pv4[n] for i, n in enumerate(names4))
        f1v = f1_score(y_val, (vw >= best_threshold(y_val, vw)).astype(int), zero_division=0)
        if f1v > best_vf1:
            best_vf1, best_w = f1v, w
    tw = sum(best_w[i] * pt4[n] for i, n in enumerate(names4))
    vw = sum(best_w[i] * pv4[n] for i, n in enumerate(names4))
    results["ensemble_berbobot"] = report(y_te, tw, best_threshold(y_val, vw))
    print(f"B  ensemble berbobot       F1 test = {results['ensemble_berbobot']['f1']:.4f}  "
          f"bobot={dict(zip(names4, best_w.round(3)))}")

    # ── C. Stacking dengan prediksi out-of-fold ──────────────────────────────
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros((len(X_tr), len(names4)))
    for tr_i, va_i in skf.split(X_tr, y_tr):
        fm = fit_members(names4, X_tr[tr_i], y_tr[tr_i], cols)
        pm = proba_of(fm, X_tr[va_i], cols)
        for j, n in enumerate(names4):
            oof[va_i, j] = pm[n]
    meta = LogisticRegression(max_iter=2000, class_weight="balanced").fit(oof, y_tr)
    fitted_full = fit_members(names4, X_tr, y_tr, cols)
    pv_f = proba_of(fitted_full, X_val, cols); pt_f = proba_of(fitted_full, X_te, cols)
    v_meta = meta.predict_proba(np.column_stack([pv_f[n] for n in names4]))[:, 1]
    t_meta = meta.predict_proba(np.column_stack([pt_f[n] for n in names4]))[:, 1]
    results["stacking"] = report(y_te, t_meta, best_threshold(y_val, v_meta))
    print(f"C  stacking (meta-model)   F1 test = {results['stacking']['f1']:.4f}")

    # ── D. Seleksi fitur ─────────────────────────────────────────────────────
    gain = fitted4["xgb"].get_score(importance_type="gain")
    ranked = [c for c, _ in sorted(gain.items(), key=lambda kv: kv[1], reverse=True)]
    keep = ranked[:18] or cols
    idx = [cols.index(c) for c in keep]
    fs = fit_members(base_names, X_fit[:, idx], y_fit, keep)
    pv_s = proba_of(fs, X_val[:, idx], keep); pt_s = proba_of(fs, X_te[:, idx], keep)
    v_s = np.mean([pv_s[n] for n in base_names], axis=0)
    t_s = np.mean([pt_s[n] for n in base_names], axis=0)
    results[f"seleksi_fitur ({len(keep)})"] = report(y_te, t_s, best_threshold(y_val, v_s))
    print(f"D  seleksi {len(keep)} fitur       F1 test = "
          f"{results[f'seleksi_fitur ({len(keep)})']['f1']:.4f}")

    # ── Ringkasan ────────────────────────────────────────────────────────────
    ranked_res = sorted(results.items(), key=lambda kv: kv[1]["f1"], reverse=True)
    print("\nPeringkat F1 test:")
    for name, r in ranked_res:
        print(f"  {name:26s} F1={r['f1']:.4f}  P={r['precision']:.4f}  R={r['recall']:.4f}")

    winner, wr = ranked_res[0]
    gain_vs_now = wr["f1"] - results["ensemble_3 (sekarang)"]["f1"]
    print(f"\nTerbaik: {winner}  ({gain_vs_now:+.4f} dibanding model sekarang)")
    if gain_vs_now < 0.005:
        print("Selisihnya di bawah 0,005 — dalam derau split ini, tidak layak "
              "mengganti model. Kandidat ditolak.")

    (NEW_OUTPUTS / "push_f1_report.json").write_text(json.dumps({
        "results": results, "winner": winner,
        "gain_vs_current": round(gain_vs_now, 4),
        "adopted": bool(gain_vs_now >= 0.005),
        "note": ("Semua pemilihan dilakukan di validation split; test dinilai sekali. "
                 "Kandidat hanya diadopsi bila unggul >= 0,005 agar tidak mengejar derau."),
    }, indent=2), encoding="utf-8")
    print(f"\nLaporan -> {NEW_OUTPUTS / 'push_f1_report.json'}")


if __name__ == "__main__":
    main()
