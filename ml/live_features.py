"""
ml/live_features.py
-------------------
Ekstraksi 25 fitur produksi (18 graf + 7 Etherscan) untuk wallet DI LUAR dataset
penelitian, langsung dari Etherscan — tanpa MulDiGraph, tanpa model baru, tanpa
retraining. `graph_present` TIDAK dihitung di sini: kolom itu hanya audit
kualitas join MulDiGraph di dataset training, bukan fitur ML.

Alasan ini mungkin: ke-20 fitur graf di `ml/preprocess.py:extract_node_features`
semuanya bersifat LOKAL 1-hop (siapa mengirim ke node ini, siapa yang dikirimi,
berapa jumlahnya, kapan). MulDiGraph hanya berperan sebagai sumber daftar edge.
Etherscan `txlist` menyediakan tuple yang sama persis — (from, to, value,
timeStamp) — sehingga rumus fitur yang identik bisa diterapkan ulang.

Modul ini SENGAJA mencerminkan `preprocess.py` baris demi baris (lihat
`_safe_stats` dan `_graph_features_from_txs`). Kalau `preprocess.py` berubah,
modul ini harus ikut berubah, atau fitur live jadi tidak sebanding dengan fitur
yang dipakai saat training.

Model yang dipakai tetap `ml/artifacts/model_prod_xgb.json` — TIDAK diubah,
TIDAK dilatih ulang. Metrik tesis (F1 ≈ 91%) tetap berlaku apa adanya untuk
populasi penelitian, dan TIDAK berlaku untuk wallet yang dinilai lewat modul ini.
"""

import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ml.data_fetchers import EtherscanFetcher  # noqa: E402
from ml.predict import predict_full            # noqa: E402

log = logging.getLogger(__name__)

# Di bawah ambang ini, statistik sebaran (std, spread, rasio) tidak bermakna:
# hasilnya tetap dikembalikan tapi ditandai `insufficient_data=True`.
MIN_MEANINGFUL_TX = 5


class WalletNotFoundError(Exception):
    """Alamat valid secara format, tetapi tidak punya riwayat transaksi on-chain."""


class LiveScanUnavailableError(Exception):
    """Prasyarat scan live belum terpenuhi (mis. ETHERSCAN_API_KEY kosong)."""


def is_available() -> bool:
    """True bila scan live bisa dijalankan (butuh ETHERSCAN_API_KEY)."""
    return bool(os.getenv("ETHERSCAN_API_KEY", "").strip())


# ── Pengambilan transaksi ─────────────────────────────────────────────────────

def _fetch_txlist(fetcher: EtherscanFetcher, address: str, max_pages: int) -> List[dict]:
    """
    Ambil transaksi ETH normal (`txlist`) berhalaman, terurut menaik.

    Hanya `txlist` yang dipakai untuk fitur graf — BUKAN `txlistinternal` atau
    `tokentx`. Alasannya menjaga kesebandingan: edge MulDiGraph adalah transfer
    ETH dengan atribut `amount`, sedangkan transfer token tidak membawa nilai ETH
    dan akan mencemari statistik `*_amount`.
    """
    import time

    import requests

    txs: List[dict] = []
    for page in range(1, max_pages + 1):
        time.sleep(fetcher.request_delay)
        params = {
            "chainid": fetcher.chain_id,
            "module": "account",
            "action": "txlist",
            "address": address,
            "startblock": 0,
            "endblock": 99999999,
            "page": page,
            "offset": 10000,
            "sort": "asc",
            "apikey": fetcher.api_key,
        }
        try:
            resp = requests.get(fetcher.base_url, params=params, timeout=20)
            resp.raise_for_status()
            data = resp.json()
        except Exception as e:
            log.error("txlist gagal untuk %s (halaman %d): %s", address, page, e)
            break

        if data.get("status") != "1":
            # "No transactions found" juga masuk sini — ditangani pemanggil
            # lewat daftar kosong.
            break

        records = data.get("result", [])
        if not records:
            break
        txs.extend(records)
        if len(records) < 10000:
            break  # halaman terakhir

    return txs


# ── Fitur graf: cerminan persis dari preprocess.py ────────────────────────────

def _safe_stats(values: list) -> dict:
    """Salinan `ml/preprocess.py:_safe_stats` — jangan diubah sepihak."""
    if not values:
        return dict(total=0.0, mean=0.0, mx=0.0, mn=0.0, std=0.0)
    arr = np.asarray(values, dtype=float)
    return dict(
        total=float(arr.sum()),
        mean=float(arr.mean()),
        mx=float(arr.max()),
        mn=float(arr.min()),
        std=float(arr.std()) if len(arr) > 1 else 0.0,
    )


def _graph_features_from_txs(address: str, txs: List[dict]) -> Dict[str, float]:
    """
    Hitung 20 fitur graf dari daftar transaksi Etherscan.

    Pemetaan ke MulDiGraph:
      predecessor (pengirim)  -> tx dengan `to`   == address, `from` = tetangga
      successor   (penerima)  -> tx dengan `from` == address, `to`   = tetangga
      edge_data["amount"]     -> tx["value"] / 1e18   (Wei -> ETH)
      edge_data["timestamp"]  -> tx["timeStamp"]      (epoch detik)

    Transaksi gagal (`isError == "1"`) dibuang: transfer nilainya tidak pernah
    terjadi, jadi memasukkannya akan menggelembungkan degree dan jumlah.
    """
    addr = address.lower()

    in_amounts: List[float] = []
    in_timestamps: List[float] = []
    unique_senders = set()

    out_amounts: List[float] = []
    out_timestamps: List[float] = []
    unique_receivers = set()

    for tx in txs:
        if str(tx.get("isError", "0")) == "1":
            continue
        frm = (tx.get("from") or "").lower()
        to = (tx.get("to") or "").lower()
        try:
            amount = float(tx.get("value", 0) or 0) / 1e18
        except (TypeError, ValueError):
            amount = 0.0
        try:
            ts = float(tx.get("timeStamp", 0) or 0)
        except (TypeError, ValueError):
            ts = 0.0

        # Self-transaction dihitung dua arah, sama seperti self-loop di MulDiGraph.
        if to == addr and frm:
            unique_senders.add(frm)
            in_amounts.append(amount)
            in_timestamps.append(ts)
        if frm == addr and to:
            unique_receivers.add(to)
            out_amounts.append(amount)
            out_timestamps.append(ts)

    in_s = _safe_stats(in_amounts)
    out_s = _safe_stats(out_amounts)
    n_in = len(in_amounts)
    n_out = len(out_amounts)

    ts_spread_in = (max(in_timestamps) - min(in_timestamps)) if len(in_timestamps) > 1 else 0.0
    ts_spread_out = (max(out_timestamps) - min(out_timestamps)) if len(out_timestamps) > 1 else 0.0

    return {
        "in_degree": float(n_in),
        "out_degree": float(n_out),
        "unique_senders": float(len(unique_senders)),
        "unique_receivers": float(len(unique_receivers)),
        "total_received": in_s["total"],
        "total_sent": out_s["total"],
        "avg_in_amount": in_s["mean"],
        "avg_out_amount": out_s["mean"],
        "max_in_amount": in_s["mx"],
        "max_out_amount": out_s["mx"],
        "min_in_amount": in_s["mn"],
        "min_out_amount": out_s["mn"],
        "std_in_amount": in_s["std"],
        "std_out_amount": out_s["std"],
        "in_out_ratio": n_in / (n_out + 1),
        "sent_received_ratio": out_s["total"] / (in_s["total"] + 1e-9),
        "timestamp_spread_in": ts_spread_in,
        "timestamp_spread_out": ts_spread_out,
    }


# ── Perakitan 25 fitur ────────────────────────────────────────────────────────

def build_live_features(
    address: str,
    api_key: Optional[str] = None,
    max_pages: int = 5,
    request_delay: float = 0.25,
) -> Dict[str, Any]:
    """
    Susun 25 fitur produksi untuk `address` dari data Etherscan terkini.

    Raises
    ------
    LiveScanUnavailableError : ETHERSCAN_API_KEY tidak tersedia.
    WalletNotFoundError      : alamat tidak punya riwayat transaksi ETH normal.

    Returns
    -------
    dict dengan kunci: features (25 fitur), n_transactions, insufficient_data.
    """
    key = api_key or os.getenv("ETHERSCAN_API_KEY", "").strip()
    if not key:
        raise LiveScanUnavailableError(
            "ETHERSCAN_API_KEY tidak tersedia. Isi di .env untuk memakai scan live."
        )

    addr = str(address).strip().lower()
    fetcher = EtherscanFetcher(api_key=key, chain_id=1, request_delay=request_delay)

    # 1) Riwayat transaksi lebih dulu — sekaligus jadi uji "wallet ditemukan?".
    #    Etherscan tidak pernah menjawab "address not found" (setiap string 40-hex
    #    adalah alamat sah); ketiadaan riwayat transaksi adalah definisi
    #    operasional kita untuk "tidak ditemukan".
    txs = _fetch_txlist(fetcher, addr, max_pages=max_pages)
    if not txs:
        raise WalletNotFoundError(
            f"Wallet {address} tidak ditemukan di Ethereum mainnet — alamat ini "
            f"belum pernah melakukan transaksi ETH. Periksa kembali alamatnya."
        )

    features: Dict[str, float] = _graph_features_from_txs(addr, txs)

    # 2) Enam fitur Etherscan — memakai `fetch_all()`, JALUR KODE YANG SAMA PERSIS
    #    dengan `ml/enrich_real_addresses_etherscan.py` saat membangun dataset
    #    training. Ini disengaja: `get_wallet_activity()` mengambil hanya 100
    #    transaksi terakhir, sehingga activity_score/tx_frequency/
    #    unique_counterparties ikut terpotong di ~100 persis seperti saat training.
    #    Menghitungnya "lebih benar" di sini justru akan membuat fitur live keluar
    #    dari sebaran yang pernah dilihat model.
    info = fetcher.fetch_all(addr)
    features["balance"] = float(info.get("balance", 0.0) or 0.0)
    features["transaction_count"] = float(info.get("transaction_count", 0) or 0)
    features["activity_score"] = float(info.get("activity_score", 0) or 0)
    features["avg_tx_value"] = float(info.get("avg_tx_value", 0.0) or 0.0)
    features["tx_frequency"] = float(info.get("tx_frequency", 0.0) or 0.0)
    features["unique_counterparties"] = float(info.get("unique_counterparties", 0) or 0)

    # 3) created_at_ts konstan 0 di hybrid_dataset.csv (Etherscan tidak pernah
    #    mengisi created_at saat enrichment), jadi XGBoost tidak pernah membuat
    #    split atasnya — gain-nya 0.00%. Diisi di sini demi kesebandingan dengan
    #    training, bukan demi akurasi. `graph_present` SENGAJA tidak dihitung:
    #    itu kolom audit join MulDiGraph, bukan salah satu dari 25 fitur model
    #    (lihat prod_feature_schema.json -> audit_columns).
    features["created_at_ts"] = 0.0

    n_tx = len(txs)
    return {
        "features": features,
        "n_transactions": n_tx,
        "insufficient_data": n_tx < MIN_MEANINGFUL_TX,
    }


def predict_address(
    address: str,
    api_key: Optional[str] = None,
    max_pages: int = 5,
    request_delay: float = 0.25,
    explain_top: int = 5,
) -> Dict[str, Any]:
    """
    Ambil fitur live lalu nilai dengan MODEL PRODUKSI yang tidak diubah.

    Hasilnya selalu ditandai `out_of_distribution=True`: model dilatih dan
    dievaluasi pada snapshot MulDiGraph 2019, sedangkan fitur di sini berasal
    dari kondisi rantai terkini. Skornya adalah estimasi, bukan klaim tesis.
    """
    built = build_live_features(
        address, api_key=api_key, max_pages=max_pages, request_delay=request_delay
    )
    features = built["features"]

    result = predict_full(features, explain_top=explain_top)
    result["features"] = features
    result["n_transactions"] = built["n_transactions"]
    result["insufficient_data"] = built["insufficient_data"]
    result["out_of_distribution"] = True
    return result


# ── Uji cepat ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass

    target = sys.argv[1] if len(sys.argv) > 1 else "0xd8da6bf26964af9d7eed9e03e53415d37aa96045"
    try:
        out = predict_address(target)
    except (WalletNotFoundError, LiveScanUnavailableError) as e:
        print(f"[GAGAL] {e}")
        sys.exit(1)

    print(f"\nWallet     : {target}")
    print(f"Transaksi  : {out['n_transactions']}")
    print(f"Skor risiko: {out['risk_score']}  ({out['risk_level']})")
    print(f"Keputusan  : {out['verdict']}")
    print("\nKontributor teratas:")
    for c in out.get("top_contributors", []):
        print(f"  {c['feature']:24s} nilai={c['value']:<14.4f} kontribusi={c['contribution']:+.4f}")
