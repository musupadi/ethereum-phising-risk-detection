"""
ml/predict.py
-------------
Inference helper for the *production* model served by the API / Streamlit demo:
XGBoost on the best thesis scenario (MulDiGraph + Etherscan, 26 features, F1 ≈ 91%).

Loads:
  - ml/artifacts/model_prod_xgb.json     (XGBoost booster)
  - ml/artifacts/prod_feature_schema.json (26 ordered feature names)
Falls back to the legacy Random Forest (model.pkl + feature_schema.json) if the
production artifacts are missing, so older setups keep working.

Generate the production artifacts with:  python ml/train_production_model.py
"""

import json
import logging
from pathlib import Path

import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"

PROD_MODEL = ARTIFACTS / "model_prod_xgb.json"
PROD_SCHEMA = ARTIFACTS / "prod_feature_schema.json"
LEGACY_MODEL = ARTIFACTS / "model.pkl"
LEGACY_SCHEMA = ARTIFACTS / "feature_schema.json"

log = logging.getLogger(__name__)

# ── Module-level singletons (lazy-loaded) ──────────────────────────────────────
_model = None
_schema = None
_kind = None  # "xgboost" | "sklearn"


def _load():
    global _model, _schema, _kind
    if _model is not None:
        return

    if PROD_MODEL.exists() and PROD_SCHEMA.exists():
        import xgboost as xgb

        booster = xgb.Booster()
        booster.load_model(str(PROD_MODEL))
        _model = booster
        _schema = json.loads(PROD_SCHEMA.read_text(encoding="utf-8"))
        _kind = "xgboost"
        log.info("Loaded production XGBoost model (%d features) from %s",
                 len(_schema["features"]), PROD_MODEL)
        return

    # ── Fallback: legacy Random Forest ────────────────────────────────────────
    import joblib

    if not LEGACY_MODEL.exists() or not LEGACY_SCHEMA.exists():
        raise FileNotFoundError(
            f"No model found. Run `python ml/train_production_model.py` to create "
            f"{PROD_MODEL.name}, or `python ml/train.py` for the legacy model."
        )
    _model = joblib.load(LEGACY_MODEL)
    _schema = json.loads(LEGACY_SCHEMA.read_text(encoding="utf-8"))
    _kind = "sklearn"
    log.info("Loaded legacy sklearn model from %s", LEGACY_MODEL)


def predict_proba(features: dict) -> float:
    """
    Compute probability that the given wallet is phishing.

    Parameters
    ----------
    features : dict
        Keys should include the feature names listed in the active schema.
        Missing keys default to 0.

    Returns
    -------
    float
        Probability in [0, 1] that the address is phishing.
    """
    _load()
    feature_cols: list = _schema["features"]
    row = np.array([[float(features.get(col, 0.0)) for col in feature_cols]], dtype=float)

    if _kind == "xgboost":
        import xgboost as xgb

        dmatrix = xgb.DMatrix(row, feature_names=feature_cols)
        return float(_model.predict(dmatrix)[0])

    return float(_model.predict_proba(row)[0, 1])


def risk_score(features: dict) -> int:
    """Return integer risk score 0–100."""
    return round(predict_proba(features) * 100)


def risk_level(score: int) -> str:
    """Map numeric score to LOW / MEDIUM / HIGH."""
    if score < 40:
        return "LOW"
    elif score < 70:
        return "MEDIUM"
    else:
        return "HIGH"


def predict_full(features: dict) -> dict:
    """
    Convenience wrapper – returns probability, risk_score and risk_level.

    Returns
    -------
    dict with keys: probability, risk_score, risk_level
    """
    prob = predict_proba(features)
    score = round(prob * 100)
    return {
        "probability": prob,
        "risk_score": score,
        "risk_level": risk_level(score),
    }


# ── Quick smoke-test ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dummy = {
        "in_degree": 5,
        "out_degree": 2,
        "unique_senders": 3,
        "unique_receivers": 2,
        "total_received": 1.5,
        "total_sent": 0.8,
        "avg_in_amount": 0.3,
        "avg_out_amount": 0.4,
        "max_in_amount": 0.9,
        "max_out_amount": 0.5,
        "min_in_amount": 0.1,
        "min_out_amount": 0.3,
        "std_in_amount": 0.25,
        "std_out_amount": 0.1,
        "in_out_ratio": 2.5,
        "sent_received_ratio": 0.53,
        "timestamp_spread_in": 86400,
        "timestamp_spread_out": 3600,
    }
    result = predict_full(dummy)
    print(result)
