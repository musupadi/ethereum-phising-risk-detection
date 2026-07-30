"""
api/services/ml_service.py
--------------------------
Thin wrapper around ml/predict.py.
Also supports "demo mode": look up a row from data/processed/test.csv by
integer row_id so the Streamlit UI doesn't need a large feature form.
"""

import re
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import pandas as pd

# Make root importable regardless of working directory
ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.predict import predict_full         # noqa: E402
from api.config import TEST_CSV, LOOKUP_CSV  # noqa: E402

_test_df: Optional[pd.DataFrame] = None
_lookup_df: Optional[pd.DataFrame] = None


def _load_test_df() -> pd.DataFrame:
    """Held-out demo rows (prod_demo.csv) for 'Demo (row_id)' mode."""
    global _test_df
    if _test_df is None:
        _test_df = pd.read_csv(TEST_CSV)
    return _test_df


def _load_lookup_df() -> pd.DataFrame:
    """Full sampled dataset indexed by lowercase wallet address (node_id).

    Used by 'scan by address' so a user can paste a wallet and get its
    precomputed features (incl. Etherscan) without knowing a row_id. Covers all
    ~8.595 sampled wallets in hybrid_dataset.csv.
    """
    global _lookup_df
    if _lookup_df is None:
        df = pd.read_csv(LOOKUP_CSV)
        df["_addr"] = df["node_id"].astype(str).str.strip().str.lower()
        _lookup_df = df.drop_duplicates("_addr", keep="first").set_index("_addr")
    return _lookup_df


# Non-feature / string metadata columns that must never be fed to the model.
_META_COLS = {"label", "node_id", "address", "class_name", "source_dataset",
              "error", "_split", "_addr"}


def _row_features(row) -> Dict[str, float]:
    """Numeric feature dict from a dataset row (drops meta/string columns)."""
    out = {}
    for k, v in row.items():
        if k in _META_COLS:
            continue
        try:
            out[k] = float(v)
        except (TypeError, ValueError):
            continue  # skip any stray non-numeric column
    return out


# Berapa banyak fitur teratas yang disertakan sebagai penjelasan asal skor (SHAP).
_EXPLAIN_TOP = 5


def predict_from_features(features: Dict[str, Any]) -> dict:
    """Run ML inference for an explicit feature dict."""
    return predict_full(features, explain_top=_EXPLAIN_TOP)


def predict_from_row(row_id: int) -> dict:
    """
    Run ML inference for a row in test.csv (demo mode).
    Returns prediction dict plus the raw feature values used.
    """
    df = _load_test_df()
    if row_id < 0 or row_id >= len(df):
        raise IndexError(f"row_id {row_id} is out of range (test set has {len(df)} rows)")

    row = df.iloc[row_id]
    features = _row_features(row)
    result = predict_full(features, explain_top=_EXPLAIN_TOP)
    result["features"] = features
    result["true_label"] = int(row.get("label", -1))
    result["node_id"] = str(row.get("node_id", ""))
    return result


def predict_from_address(wallet: str) -> dict:
    """
    'Scan by address' mode: look up a wallet in the sampled dataset (train+test)
    by its address, pull its precomputed graph features, and run inference.

    Raises KeyError if the address is not present in the dataset.
    """
    df = _load_lookup_df()
    key = str(wallet).strip().lower()
    # Catch malformed/truncated addresses early with a clearer hint.
    if not re.fullmatch(r"0x[0-9a-f]{40}", key):
        raise KeyError(
            f"Alamat '{wallet}' tidak valid — harus 42 karakter (0x + 40 hex). "
            f"Sepertinya kepotong/typo saat menempel. Periksa kembali."
        )
    if key not in df.index:
        raise KeyError(
            f"Wallet {wallet} tidak ada di dataset MulDiGraph (snapshot 2019). "
            f"Gunakan mode Manual atau pilih alamat contoh dari dataset."
        )
    row = df.loc[key]
    features = _row_features(row)
    result = predict_full(features, explain_top=_EXPLAIN_TOP)
    result["features"] = features
    result["true_label"] = int(row.get("label", -1))
    result["node_id"] = str(row.get("node_id", wallet))
    result["split"] = "dataset"
    return result


def get_test_row_count() -> int:
    """Total rows available in test.csv for demo mode."""
    return len(_load_test_df())
