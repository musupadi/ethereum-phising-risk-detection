"""Generate Model-Artifact figures that match the SERVED production model.

The Streamlit "Model Artifacts" panel used to show the Random-Forest baseline
plots, which do not match the XGBoost model that is actually served by the API.
This script recomputes both figures directly from `model_prod_xgb.json` on the
held-out `prod_demo.csv` set, so the UI reflects the real production model.

Outputs (ml/artifacts/):
  - prod_cm_xgb.png                 confusion matrix on held-out set
  - prod_feature_importance_xgb.png feature importance (gain) of served model
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "ml" / "artifacts"
DEMO = ROOT / "data" / "processed" / "prod_demo.csv"

MODEL = ART / "model_prod_xgb.json"
SCHEMA = ART / "prod_feature_schema.json"


def main() -> None:
    schema = json.loads(SCHEMA.read_text(encoding="utf-8"))
    feats = schema["features"]

    df = pd.read_csv(DEMO)
    X = df[feats].fillna(0).values
    y = df["label"].astype(int).values

    booster = xgb.Booster()
    booster.load_model(str(MODEL))
    proba = booster.inplace_predict(X.astype(np.float32))
    y_pred = (proba >= 0.5).astype(int)

    f1 = f1_score(y, y_pred)
    prec = precision_score(y, y_pred)
    rec = recall_score(y, y_pred)
    print(f"Held-out ({len(y)} rows)  F1={f1*100:.2f}  "
          f"Precision={prec*100:.2f}  Recall={rec*100:.2f}")

    # ── Confusion matrix ──────────────────────────────────────────────────────
    cm = confusion_matrix(y, y_pred)
    fig, ax = plt.subplots(figsize=(5.4, 4.8))
    ConfusionMatrixDisplay(cm, display_labels=["Normal", "Phishing"]).plot(
        ax=ax, cmap="Blues", colorbar=False, values_format="d"
    )
    ax.set_title(f"XGBoost (production) - Confusion Matrix\n"
                 f"held-out n={len(y)}  |  F1={f1*100:.1f}%")
    fig.tight_layout()
    out_cm = ART / "prod_cm_xgb.png"
    fig.savefig(out_cm, dpi=130)
    plt.close(fig)
    print("wrote", out_cm)

    # ── Feature importance (gain) ─────────────────────────────────────────────
    score = booster.get_score(importance_type="gain")
    imp = np.array([score.get(f, 0.0) for f in feats])
    order = np.argsort(imp)  # ascending for barh
    fig, ax = plt.subplots(figsize=(6.2, 6.4))
    ax.barh([feats[i] for i in order], imp[order], color="#4C78A8")
    ax.set_xlabel("Importance (gain)")
    ax.set_title(f"Top Feature Importances (XGBoost production, {len(feats)} features)")
    fig.tight_layout()
    out_fi = ART / "prod_feature_importance_xgb.png"
    fig.savefig(out_fi, dpi=130)
    plt.close(fig)
    print("wrote", out_fi)


if __name__ == "__main__":
    main()
