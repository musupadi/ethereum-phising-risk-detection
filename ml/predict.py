"""
ml/predict.py
-------------
Inference helper: loads model.pkl + feature_schema.json and
exposes predict_proba(features_dict) -> float (probability of phishing).
"""

import json
import logging
from pathlib import Path
from typing import Union

import joblib
import numpy as np

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"

log = logging.getLogger(__name__)

# ── Module-level singletons (lazy-loaded) ──────────────────────────────────────
_model = None
_schema = None


def _load():
    global _model, _schema
    if _model is None:
        model_path = ARTIFACTS / "model.pkl"
        schema_path = ARTIFACTS / "feature_schema.json"

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model not found at {model_path}. Run ml/train.py first."
            )
        if not schema_path.exists():
            raise FileNotFoundError(
                f"Feature schema not found at {schema_path}. Run ml/train.py first."
            )

        _model = joblib.load(model_path)
        with open(schema_path) as f:
            _schema = json.load(f)
        log.info("Model and schema loaded from %s", ARTIFACTS)


def predict_proba(features: dict) -> float:
    """
    Compute probability that the given wallet is phishing.

    Parameters
    ----------
    features : dict
        Keys must include (at least) the feature names listed in
        ml/artifacts/feature_schema.json.  Missing keys default to 0.

    Returns
    -------
    float
        Probability in [0, 1] that the address is phishing.
    """
    _load()
    feature_cols: list = _schema["features"]
    row = np.array([[features.get(col, 0.0) for col in feature_cols]], dtype=float)
    proba: float = float(_model.predict_proba(row)[0, 1])
    return proba


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
