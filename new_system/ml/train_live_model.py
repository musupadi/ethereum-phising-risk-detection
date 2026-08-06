"""
ml/train_live_model.py
----------------------
Melatih MODEL LIVE (model B) — model untuk memindai wallet DI LUAR dataset EPTransNet.

Model ini TERPISAH dari model tesis (`model_prod_xgb.json`, F1 0,9102) dan tidak
menggantikannya. Model tesis tetap melayani wallet yang ada di dalam dataset;
angka-angka pada naskah tesis tidak berubah sama sekali.

Uji kelulusan (gerbang mutu) — dijalankan pada alamat HELD-OUT yang tidak pernah
ikut dilatih, dan yang dulu SALAH dinilai oleh model lama:
    Vitalik Buterin        harus BUKAN phishing   (model lama: 92 HIGH)
    Uniswap V2 Router      harus BUKAN phishing   (model lama: 100 HIGH)
    ETH2 Deposit Contract  harus BUKAN phishing   (model lama: 84 HIGH)

Keluaran:
    ml/artifacts/model_live_xgb.json      booster
    ml/artifacts/live_feature_schema.json skema + metrik
    ml/artifacts/live_model_report.json   laporan evaluasi lengkap
    ml/artifacts/cm_live_model.png        confusion matrix
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    ConfusionMatrixDisplay, accuracy_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import (  # noqa: E402
    LIVE_DATASET_CSV, LIVE_MODEL_JSON, LIVE_REPORT_JSON, LIVE_SCHEMA_JSON,
    NEW_ARTIFACTS, add_import_paths, ensure_dirs,
)

add_import_paths()
ensure_dirs()

from live_features import LIVE_FEATURES  # noqa: E402
from xgboost_model import XGBoostModel  # noqa: E402  (dipakai ulang dari sistem tesis)

ARTIFACTS = NEW_ARTIFACTS
# Bisa diarahkan ke checkpoint parsial lewat env, berguna untuk uji coba lebih awal
# selagi pengambilan data masih berjalan.
LIVE_CSV = Path(os.getenv("LIVE_DATASET_CSV", str(LIVE_DATASET_CSV)))

MODEL_OUT = LIVE_MODEL_JSON
SCHEMA_OUT = LIVE_SCHEMA_JSON
REPORT_OUT = LIVE_REPORT_JSON
CM_OUT = NEW_ARTIFACTS / "cm_live_model.png"

SEED = 42
THRESHOLD = 0.5

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("train_live")


def metrics(y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
        "pr_auc": float(average_precision_score(y_true, y_proba)) if len(set(y_true)) > 1 else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


def main() -> None:
    if not LIVE_CSV.exists():
        log.error("%s belum ada. Jalankan dulu: python new_system/ml/build_live_dataset.py", LIVE_CSV)
        sys.exit(1)

    df = pd.read_csv(LIVE_CSV)
    df["address"] = df["address"].astype(str).str.lower()
    df = df.drop_duplicates("address", keep="last")
    log.info("Dataset live: %d baris | label: %s", len(df), df["label"].value_counts().to_dict())

    # Alamat held-out tidak boleh ikut dilatih maupun ikut test acak:
    # ia dipakai khusus sebagai gerbang mutu out-of-distribution.
    heldout = df[df.get("heldout", 0) == 1].copy()
    pool = df[df.get("heldout", 0) != 1].copy()
    log.info("Held-out OOD: %d alamat | kolam latih/uji: %d", len(heldout), len(pool))

    feature_cols = [c for c in LIVE_FEATURES if c in pool.columns]
    if len(feature_cols) != len(LIVE_FEATURES):
        log.warning("Fitur hilang: %s", set(LIVE_FEATURES) - set(feature_cols))

    X = pool[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values
    y = pool["label"].astype(int).values

    X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
        X, y, np.arange(len(pool)), test_size=0.2, random_state=SEED, stratify=y)
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train)

    log.info("Train %d | Val %d | Test %d", len(X_fit), len(X_val), len(X_test))

    model = XGBoostModel(n_estimators=400, max_depth=5, learning_rate=0.05,
                         subsample=0.85, colsample_bytree=0.85)
    model.fit(X_fit, y_fit, X_val, y_val, feature_names=feature_cols,
              early_stopping_rounds=50)

    import xgboost as xgb
    dtest = xgb.DMatrix(X_test, feature_names=feature_cols)
    proba_test = model.model.predict(dtest)
    pred_test = (proba_test >= THRESHOLD).astype(int)
    m = metrics(y_test, pred_test, proba_test)
    log.info("Test: F1=%.4f Prec=%.4f Rec=%.4f Acc=%.4f",
             m["f1"], m["precision"], m["recall"], m["accuracy"])

    # ── Gerbang mutu: alamat sah terkenal yang held-out ───────────────────────
    ood_rows = []
    if not heldout.empty:
        Xo = heldout[feature_cols].fillna(0.0).replace([np.inf, -np.inf], 0.0).values
        proba_o = model.model.predict(xgb.DMatrix(Xo, feature_names=feature_cols))
        for (_, row), p in zip(heldout.iterrows(), proba_o):
            score = int(round(float(p) * 100))
            ood_rows.append({
                "address": row["address"],
                "name": str(row.get("source", "")).split(":", 1)[-1],
                "score": score,
                "level": "LOW" if score < 40 else ("MEDIUM" if score < 70 else "HIGH"),
                "verdict": "Phishing" if p >= THRESHOLD else "Bukan Phishing",
                "pass": bool(p < THRESHOLD),
            })
        n_pass = sum(r["pass"] for r in ood_rows)
        log.info("Gerbang OOD: %d/%d alamat sah dinilai BUKAN phishing", n_pass, len(ood_rows))
        for r in ood_rows:
            log.info("   %-24s skor %3d  %-6s  %s", r["name"][:24], r["score"],
                     r["level"], "LULUS" if r["pass"] else "GAGAL")

    # ── Kepentingan fitur ─────────────────────────────────────────────────────
    gain = model.model.get_score(importance_type="gain")
    top_gain = sorted(gain.items(), key=lambda kv: kv[1], reverse=True)[:12]
    log.info("Fitur paling berpengaruh: %s", [f"{k}={v:.0f}" for k, v in top_gain])

    # ── Simpan ────────────────────────────────────────────────────────────────
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_OUT)
    SCHEMA_OUT.write_text(json.dumps({
        "features": feature_cols,
        "label": "label",
        "node_id": "address",
        "model_type": "xgboost",
        "scenario": "live_etherscan_hard_negatives",
        "threshold": THRESHOLD,
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in m.items()},
        "note": ("Model untuk wallet DI LUAR dataset EPTransNet. Fitur diambil live dari "
                 "Etherscan. Kelas negatif memuat hard negatives (bursa, router DEX, "
                 "kontrak populer, wallet acak lintas era). TIDAK menggantikan "
                 "model_prod_xgb.json yang dipakai untuk wallet dalam dataset."),
    }, indent=2), encoding="utf-8")

    report = {
        "dataset": {
            "rows": int(len(df)),
            "label_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
            "source_counts": {str(k): int(v) for k, v in
                              df["source"].str.split(":").str[0].value_counts().items()},
            "heldout": int(len(heldout)),
        },
        "split": {"fit": int(len(X_fit)), "val": int(len(X_val)), "test": int(len(X_test))},
        "test_metrics": m,
        "ood_gate": ood_rows,
        "ood_gate_passed": bool(ood_rows and all(r["pass"] for r in ood_rows)),
        "top_features_by_gain": [{"feature": k, "gain": float(v)} for k, v in top_gain],
        "threshold": THRESHOLD,
    }
    REPORT_OUT.write_text(json.dumps(report, indent=2), encoding="utf-8")

    disp = ConfusionMatrixDisplay(confusion_matrix(y_test, pred_test, labels=[0, 1]),
                                  display_labels=["Normal", "Phishing"])
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    disp.plot(ax=ax, colorbar=False, cmap="Blues", values_format="d")
    ax.set_title("Model Live (Etherscan + hard negatives)")
    plt.tight_layout()
    plt.savefig(CM_OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)

    log.info("Model    -> %s", MODEL_OUT)
    log.info("Skema    -> %s", SCHEMA_OUT)
    log.info("Laporan  -> %s", REPORT_OUT)
    print("\n" + json.dumps({"test_metrics": m, "ood_gate_passed": report["ood_gate_passed"]},
                            indent=2))


if __name__ == "__main__":
    main()
