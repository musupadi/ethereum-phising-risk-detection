"""
api/services/ml_service.py
--------------------------
Thin wrapper around ml/predict.py.
Also supports "demo mode": look up a row from data/processed/test.csv by
integer row_id so the Streamlit UI doesn't need a large feature form.
"""

import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Make root importable regardless of working directory
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.predict import predict_full         # noqa: E402
from api.config import TEST_CSV              # noqa: E402

_test_df: Optional[pd.DataFrame] = None


def _load_test_df() -> pd.DataFrame:
    global _test_df
    if _test_df is None:
        _test_df = pd.read_csv(TEST_CSV)
    return _test_df


def predict_from_features(features: Dict[str, Any]) -> dict:
    """Run ML inference for an explicit feature dict."""
    return predict_full(features)


def predict_from_row(row_id: int) -> dict:
    """
    Run ML inference for a row in test.csv (demo mode).
    Returns prediction dict plus the raw feature values used.
    """
    df = _load_test_df()
    if row_id < 0 or row_id >= len(df):
        raise IndexError(f"row_id {row_id} is out of range (test set has {len(df)} rows)")

    row = df.iloc[row_id]
    drop_cols = {"label", "node_id"}
    features = {k: float(v) for k, v in row.items() if k not in drop_cols}
    result = predict_full(features)
    result["features"] = features
    result["true_label"] = int(row.get("label", -1))
    result["node_id"] = str(row.get("node_id", ""))
    return result


def get_test_row_count() -> int:
    """Total rows available in test.csv for demo mode."""
    return len(_load_test_df())
