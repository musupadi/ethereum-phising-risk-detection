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
from api.config import (                     # noqa: E402
    TEST_CSV, LOOKUP_CSV, LIVE_SCAN_ENABLED, LIVE_MAX_PAGES, LIVE_REQUEST_DELAY,
)

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


def _predict_live(wallet: str) -> dict:
    """
    Mode scan live: wallet TIDAK ada di dataset, jadi ke-26 fiturnya dihitung
    langsung dari Etherscan lalu dinilai oleh MODEL PRODUKSI YANG SAMA.

    Tidak ada model kedua dan tidak ada pelatihan ulang di sini — yang berganti
    hanyalah SUMBER fitur (CSV precomputed -> Etherscan API). Lihat
    `ml/live_features.py` untuk pemetaan edge MulDiGraph ke `txlist`.

    Hasilnya ditandai `out_of_distribution=True`: model dilatih pada snapshot
    MulDiGraph 2019, sehingga metrik tesis tidak berlaku untuk wallet ini dan
    skornya tidak boleh dicatat ke blockchain.
    """
    # Impor tertunda supaya `requests`/kunci API hanya dibutuhkan bila mode live
    # benar-benar dipakai. Dengan LIVE_SCAN_ENABLED=false (bawaan), baris di
    # bawah tidak pernah dijalankan.
    from ml.live_features import (  # noqa: E402
        LiveScanUnavailableError, WalletNotFoundError, predict_address,
    )

    try:
        result = predict_address(
            wallet, max_pages=LIVE_MAX_PAGES, request_delay=LIVE_REQUEST_DELAY,
            explain_top=_EXPLAIN_TOP,
        )
    except WalletNotFoundError as exc:
        # KeyError -> HTTP 404 di api/main.py
        raise KeyError(str(exc)) from exc
    except LiveScanUnavailableError as exc:
        raise KeyError(
            f"Wallet {wallet} tidak ada di dataset MulDiGraph (snapshot 2019), "
            f"dan scan live belum bisa dijalankan: {exc}"
        ) from exc

    result["node_id"] = str(wallet).strip().lower()
    result["true_label"] = None          # tidak ada label kebenaran untuk wallet baru
    result["split"] = "live_etherscan"
    result["model_used"] = "production (fitur live Etherscan)"
    warning = (
        "Wallet ini di luar dataset penelitian. Fitur diambil dari Etherscan "
        "terkini lalu dinilai model produksi yang sama — hasilnya ESTIMASI, "
        "tidak dijamin oleh metrik tesis (F1 91% hanya berlaku pada populasi "
        "MulDiGraph 2019), dan tidak dicatat ke blockchain."
    )
    if result.get("insufficient_data"):
        warning += (
            f" Selain itu wallet ini hanya punya {result.get('n_transactions', 0)} "
            f"transaksi — terlalu sedikit untuk dinilai secara bermakna. "
            f"Perlakukan hasilnya sebagai tidak dapat disimpulkan."
        )
    result["warning"] = warning
    return result


def predict_from_address(wallet: str) -> dict:
    """
    'Scan by address' mode: look up a wallet in the sampled dataset (train+test)
    by its address, pull its precomputed graph features, and run inference.

    Jika alamat tidak ada di dataset dan LIVE_SCAN_ENABLED=true, sistem beralih ke
    mode live (fitur diambil dari Etherscan, dinilai model B). Bila mode live mati,
    perilakunya seperti semula: KeyError -> HTTP 404.
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
        if LIVE_SCAN_ENABLED:
            return _predict_live(key)
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
    result["model_used"] = "dataset"
    result["out_of_distribution"] = False
    return result


def get_test_row_count() -> int:
    """Total rows available in test.csv for demo mode."""
    return len(_load_test_df())
