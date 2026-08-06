"""
ml/live_features.py
-------------------
Ekstraksi fitur langsung dari Etherscan untuk wallet APA PUN (tidak harus ada di
MulDiGraph). Dipakai oleh:
  - ml/build_live_dataset.py   (membangun data latih model live)
  - api/services/ml_service.py (mode scan wallet bebas)

Perbedaan dengan ml/preprocess.py:
  preprocess.py  : sumber = MulDiGraph.pkl (snapshot EPTransNet 2019)
  live_features  : sumber = Etherscan API (riwayat penuh sampai hari ini)

18 fitur perilaku dihitung dengan RUMUS YANG IDENTIK dengan
`preprocess.extract_node_features` sehingga tetap sebanding. Ditambah fitur akun
dan beberapa rasio tak-bergantung-skala yang dirancang untuk mengatasi kegagalan
out-of-distribution model lama (lihat `laporan belajar/13_uji_wallet_luar_dataset.md`).

Catatan penting soal Etherscan v2:
  - Maksimum 1.000 record per halaman, berapa pun nilai `offset` (terverifikasi).
  - `created_at_ts` SENGAJA tidak dipakai sebagai fitur: seluruh wallet phishing
    berasal dari era 2019, sehingga fitur waktu absolut membuat model bisa
    mengambil jalan pintas "wallet tua = phishing".
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any

import numpy as np
import requests

log = logging.getLogger(__name__)


class _RateLimiter:
    """
    Pembatas laju global lintas thread.

    Latensi satu panggilan Etherscan ~1,2 detik, sementara kuotanya 5 panggilan
    per detik. Menunggu berurutan berarti hanya 0,8 panggilan/detik — 6× lebih
    lambat dari yang diizinkan. Dengan beberapa thread ditambah pembatas laju
    bersama ini, kuota terpakai penuh tanpa pernah melampauinya.
    """

    def __init__(self, min_interval: float = 0.22):
        self._min_interval = min_interval
        self._lock = threading.Lock()
        self._next_at = 0.0

    def set_interval(self, min_interval: float) -> None:
        with self._lock:
            self._min_interval = max(0.0, min_interval)

    def relax(self, factor: float = 1.15, ceiling: float = 1.2) -> None:
        """Melambat saat kena rate limit; berlaku untuk semua thread sekaligus."""
        with self._lock:
            self._min_interval = min(ceiling, self._min_interval * factor)

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            wait_for = max(0.0, self._next_at - now)
            self._next_at = max(now, self._next_at) + self._min_interval
        if wait_for > 0:
            time.sleep(wait_for)


LIMITER = _RateLimiter()


# Sesi HTTP per-thread: koneksi TCP dipakai ulang, memangkas biaya handshake.
_thread_local = threading.local()


def _session() -> requests.Session:
    sess = getattr(_thread_local, "session", None)
    if sess is None:
        sess = requests.Session()
        _thread_local.session = sess
    return sess

API_BASE = "https://api.etherscan.io/v2/api"
PAGE_SIZE = 1000          # batas keras Etherscan v2
DEFAULT_MAX_PAGES = 20    # 20.000 transaksi; cukup untuk hampir semua wallet
CHAIN_ID = 1

# ── Skema fitur model live ────────────────────────────────────────────────────
# 18 fitur perilaku (rumus identik preprocess.py)
BEHAVIOUR_FEATURES = [
    "in_degree", "out_degree", "unique_senders", "unique_receivers",
    "total_received", "total_sent", "avg_in_amount", "avg_out_amount",
    "max_in_amount", "max_out_amount", "min_in_amount", "min_out_amount",
    "std_in_amount", "std_out_amount", "in_out_ratio", "sent_received_ratio",
    "timestamp_spread_in", "timestamp_spread_out",
]
# 4 fitur tingkat akun
ACCOUNT_FEATURES = ["balance", "transaction_count", "unique_counterparties", "is_contract"]
# 4 rasio tak-bergantung-skala (dirancang untuk generalisasi ke wallet aktif yang sah)
RATIO_FEATURES = [
    "sender_receiver_ratio",   # phishing: banyak pengirim, sedikit penerima
    "repeat_ratio",            # bursa/router: interaksi berulang dgn pihak sama
    "balance_ratio",           # phishing: saldo dikuras habis
    "activity_span_days",      # phishing: umur pendek
]
# ── Fitur generasi kedua (v2) ────────────────────────────────────────────────
# Ditambahkan setelah pencarian model mentok di F1 0,9008: semua pendekatan
# pemodelan sudah dicoba, sehingga kenaikan berikutnya harus datang dari INFORMASI
# BARU, bukan dari mengutak-atik model.
#
# Yang ditangkap di sini adalah pola yang tidak terlihat oleh 26 fitur lama:
#
#   POLA SAPU     Wallet phishing menerima dana korban lalu memindahkannya keluar
#                 dalam hitungan menit. Wallet normal menahan dananya. Ke-26 fitur
#                 lama sama sekali tidak melihat URUTAN WAKTU antara terima dan
#                 kirim — hanya rentang total. Inilah celah terbesarnya.
#
#   IRAMA WAKTU   Phishing aktif meledak dalam waktu singkat lalu ditinggalkan;
#                 wallet biasa tersebar merata sepanjang waktu.
#
#   TOKEN ERC-20  Penguras dompet sering memindahkan token, bukan hanya ETH.
#                 Butuh satu panggilan API tambahan (`tokentx`).
SWEEP_FEATURES = [
    "sweep_gap_median_s",    # jeda tengah antara dana masuk dan pengiriman berikutnya
    "sweep_gap_min_s",       # penyapuan tercepat yang pernah terjadi
    "sweep_ratio_1h",        # bagian dana masuk yang disapu keluar dalam 1 jam
    "sweep_ratio_24h",       # bagian dana masuk yang disapu keluar dalam 24 jam
]
TEMPORAL_FEATURES = [
    "active_days",           # jumlah hari berbeda dengan aktivitas
    "tx_per_active_day",     # kepadatan aktivitas
    "max_tx_in_day",         # puncak aktivitas satu hari
    "failed_tx_ratio",       # bagian transaksi yang gagal
]
TOKEN_FEATURES = [
    "token_tx_count",        # jumlah transfer ERC-20
    "token_unique_contracts",  # banyaknya token berbeda yang disentuh
    "token_out_ratio",       # kecenderungan mengirim token keluar
]
V2_FEATURES = SWEEP_FEATURES + TEMPORAL_FEATURES + TOKEN_FEATURES

LIVE_FEATURES = BEHAVIOUR_FEATURES + ACCOUNT_FEATURES + RATIO_FEATURES
LIVE_FEATURES_V2 = LIVE_FEATURES + V2_FEATURES

# Penanda asal modul. Sistem tesis punya `ml/live_features.py` dengan nama sama
# tetapi antarmuka berbeda; penanda ini dipakai untuk memastikan modul yang
# ter-import benar-benar milik new_system.
MODULE_VARIANT = "new_system"


class EtherscanError(RuntimeError):
    pass


# ── Lapisan HTTP ──────────────────────────────────────────────────────────────

def _request(params: dict, api_key: str, delay: float, timeout: int = 25,
             retries: int = 7) -> dict:
    """Panggil Etherscan dengan retry untuk rate-limit / gangguan jaringan."""
    params = {**params, "chainid": CHAIN_ID, "apikey": api_key}
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            # `delay` dipertahankan demi kompatibilitas pemanggil lama, tetapi
            # pengaturan laju sesungguhnya dilakukan oleh LIMITER global agar
            # aman saat dijalankan multi-thread.
            LIMITER.wait()
            resp = _session().get(API_BASE, params=params, timeout=timeout)
            resp.raise_for_status()
            data = resp.json()
            # Hanya periksa field bertipe string. Hasil dari modul `proxy` bisa
            # berupa dict blok berisi ratusan transaksi; men-stringify-nya di
            # setiap panggilan sangat memboroskan waktu.
            result = data.get("result")
            msg = str(data.get("message", ""))
            if isinstance(result, str):
                msg += result
            if "rate limit" in msg.lower():
                # Backoff eksponensial + longgarkan laju global sedikit, supaya
                # seluruh thread ikut melambat, bukan hanya thread ini.
                LIMITER.relax()
                time.sleep(min(8.0, 0.75 * (2 ** attempt)))
                last_exc = EtherscanError("rate limit")
                continue
            return data
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            time.sleep(0.6 * (attempt + 1))
    raise EtherscanError(f"Gagal setelah {retries} percobaan: {last_exc}")


def _fetch_tx_pages(address: str, action: str, api_key: str, delay: float,
                    max_pages: int) -> list[dict]:
    """Ambil seluruh transaksi (paging penuh sampai habis atau batas halaman)."""
    out: list[dict] = []
    for page in range(1, max_pages + 1):
        data = _request({
            "module": "account", "action": action, "address": address,
            "startblock": 0, "endblock": 99999999,
            "page": page, "offset": PAGE_SIZE, "sort": "asc",
        }, api_key, delay)
        if data.get("status") != "1":
            break  # "No transactions found" juga masuk sini — wajar
        batch = data.get("result", [])
        if not isinstance(batch, list):
            break
        out.extend(batch)
        if len(batch) < PAGE_SIZE:
            break
    return out


def _fetch_balance_eth(address: str, api_key: str, delay: float) -> float:
    data = _request({"module": "account", "action": "balance",
                     "address": address, "tag": "latest"}, api_key, delay)
    if data.get("status") != "1":
        return 0.0
    try:
        return float(data.get("result", 0)) / 1e18
    except (TypeError, ValueError):
        return 0.0


def _fetch_is_contract(address: str, api_key: str, delay: float) -> int:
    """
    1 = alamat kontrak sungguhan, 0 = EOA (wallet pengguna).

    Hati-hati dengan EIP-7702: sejak Pectra, sebuah EOA boleh menitipkan kode
    delegasi berupa 23 byte berawalan `0xef0100`. Alamat semacam itu TETAP wallet
    pengguna, bukan kontrak. Contoh nyata: alamat Vitalik Buterin mengembalikan
    `0xef0100...` dan sempat salah diklasifikasikan sebagai kontrak.
    """
    data = _request({"module": "proxy", "action": "eth_getCode",
                     "address": address, "tag": "latest"}, api_key, delay)
    code = data.get("result", "0x")
    if not isinstance(code, str) or len(code) <= 2:
        return 0
    if code.lower().startswith("0xef0100"):
        return 0  # EOA dengan delegasi EIP-7702
    return 1


# ── Statistik (disalin dari ml/preprocess.py agar identik) ────────────────────

def _sweep_features(in_events: list[tuple[float, float]],
                    out_events: list[tuple[float, float]]) -> dict[str, float]:
    """
    Ukur pola "terima lalu langsung sapu keluar".

    Untuk setiap dana masuk, dicari pengiriman keluar PERTAMA yang terjadi
    sesudahnya, lalu jaraknya dihitung. Wallet phishing memindahkan dana korban
    dalam hitungan menit; wallet biasa menahan dananya berhari-hari atau tidak
    memindahkannya sama sekali.

    in_events / out_events : daftar (timestamp, jumlah), tidak harus terurut.
    """
    empty = {"sweep_gap_median_s": 0.0, "sweep_gap_min_s": 0.0,
             "sweep_ratio_1h": 0.0, "sweep_ratio_24h": 0.0}
    if not in_events or not out_events:
        return empty

    ins = sorted(in_events)
    outs = sorted(out_events)
    out_ts = [t for t, _ in outs]

    import bisect

    gaps: list[float] = []
    for t_in, _amt in ins:
        j = bisect.bisect_right(out_ts, t_in)
        if j < len(out_ts):
            gaps.append(out_ts[j] - t_in)

    if not gaps:
        return empty

    arr = np.asarray(gaps, dtype=float)
    return {
        "sweep_gap_median_s": float(np.median(arr)),
        "sweep_gap_min_s": float(arr.min()),
        "sweep_ratio_1h": float((arr <= 3600).mean()),
        "sweep_ratio_24h": float((arr <= 86400).mean()),
    }


def _temporal_features(timestamps: list[float], n_failed: int, n_total: int) -> dict[str, float]:
    """Irama aktivitas: seberapa terpusat atau tersebar transaksinya sepanjang waktu."""
    if not timestamps:
        return {"active_days": 0.0, "tx_per_active_day": 0.0,
                "max_tx_in_day": 0.0, "failed_tx_ratio": 0.0}
    days: dict[int, int] = {}
    for ts in timestamps:
        d = int(ts // 86400)
        days[d] = days.get(d, 0) + 1
    n_days = len(days)
    return {
        "active_days": float(n_days),
        "tx_per_active_day": float(len(timestamps) / max(n_days, 1)),
        "max_tx_in_day": float(max(days.values())),
        "failed_tx_ratio": float(n_failed / max(n_total, 1)),
    }


def _safe_stats(values: list) -> dict:
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


# ── API publik ────────────────────────────────────────────────────────────────

def extract_live_features(
    address: str,
    api_key: str | None = None,
    delay: float = 0.25,
    max_pages: int = DEFAULT_MAX_PAGES,
    include_internal: bool = True,
    include_tokens: bool = True,
) -> dict[str, Any]:
    """
    Hitung fitur model live untuk satu alamat Ethereum.

    Mengembalikan dict berisi seluruh `LIVE_FEATURES` plus metadata `_*`
    (jumlah transaksi, apakah terpotong batas halaman, dsb).
    """
    api_key = api_key or os.getenv("ETHERSCAN_API_KEY", "")
    if not api_key:
        raise EtherscanError("ETHERSCAN_API_KEY tidak tersedia.")

    addr = address.strip().lower()

    txs = _fetch_tx_pages(addr, "txlist", api_key, delay, max_pages)
    n_external = len(txs)
    truncated = n_external >= max_pages * PAGE_SIZE
    if include_internal:
        txs = txs + _fetch_tx_pages(addr, "txlistinternal", api_key, delay, max_pages)

    in_amounts, in_ts, senders = [], [], set()
    out_amounts, out_ts, receivers = [], [], set()
    in_events: list[tuple[float, float]] = []   # (waktu, jumlah) untuk pola sapu
    out_events: list[tuple[float, float]] = []
    all_ts: list[float] = []
    n_failed = 0

    for tx in txs:
        if str(tx.get("isError", "0")) == "1":
            n_failed += 1
            continue  # transaksi gagal tidak memindahkan dana
        frm = str(tx.get("from") or "").lower()
        to = str(tx.get("to") or "").lower()
        if not frm or not to:
            continue
        try:
            value = float(tx.get("value", 0)) / 1e18
            ts = float(tx.get("timeStamp", 0))
        except (TypeError, ValueError):
            continue

        all_ts.append(ts)
        if to == addr and frm != addr:
            senders.add(frm)
            in_amounts.append(value)
            in_ts.append(ts)
            in_events.append((ts, value))
        elif frm == addr and to != addr:
            receivers.add(to)
            out_amounts.append(value)
            out_ts.append(ts)
            out_events.append((ts, value))

    i_s, o_s = _safe_stats(in_amounts), _safe_stats(out_amounts)
    n_in, n_out = len(in_amounts), len(out_amounts)
    spread_in = (max(in_ts) - min(in_ts)) if len(in_ts) > 1 else 0.0
    spread_out = (max(out_ts) - min(out_ts)) if len(out_ts) > 1 else 0.0

    balance = _fetch_balance_eth(addr, api_key, delay)
    is_contract = _fetch_is_contract(addr, api_key, delay)

    n_senders, n_receivers = len(senders), len(receivers)
    total_tx = n_in + n_out
    counterparties = len(senders | receivers)

    # ── Fitur v2 ─────────────────────────────────────────────────────────────
    sweep = _sweep_features(in_events, out_events)
    temporal = _temporal_features(all_ts, n_failed, len(txs))

    token_feats = {"token_tx_count": 0.0, "token_unique_contracts": 0.0,
                   "token_out_ratio": 0.0}
    if include_tokens:
        # Satu panggilan tambahan. Penguras dompet sering memindahkan token
        # ERC-20, bukan hanya ETH — aktivitas yang sama sekali tak terlihat
        # pada txlist.
        tok = _fetch_tx_pages(addr, "tokentx", api_key, delay, max_pages=min(max_pages, 3))
        if tok:
            contracts = {str(t.get("contractAddress", "")).lower() for t in tok}
            n_out_tok = sum(1 for t in tok if str(t.get("from", "")).lower() == addr)
            token_feats = {
                "token_tx_count": float(len(tok)),
                "token_unique_contracts": float(len(contracts)),
                "token_out_ratio": float(n_out_tok / len(tok)),
            }

    feats: dict[str, Any] = {
        **sweep, **temporal, **token_feats,
        # 18 fitur perilaku — rumus identik preprocess.extract_node_features
        "in_degree": n_in,
        "out_degree": n_out,
        "unique_senders": n_senders,
        "unique_receivers": n_receivers,
        "total_received": i_s["total"],
        "total_sent": o_s["total"],
        "avg_in_amount": i_s["mean"],
        "avg_out_amount": o_s["mean"],
        "max_in_amount": i_s["mx"],
        "max_out_amount": o_s["mx"],
        "min_in_amount": i_s["mn"],
        "min_out_amount": o_s["mn"],
        "std_in_amount": i_s["std"],
        "std_out_amount": o_s["std"],
        "in_out_ratio": n_in / (n_out + 1),
        "sent_received_ratio": o_s["total"] / (i_s["total"] + 1e-9),
        "timestamp_spread_in": spread_in,
        "timestamp_spread_out": spread_out,
        # fitur tingkat akun
        "balance": balance,
        "transaction_count": float(total_tx),
        "unique_counterparties": float(counterparties),
        "is_contract": float(is_contract),
        # rasio tak-bergantung-skala
        "sender_receiver_ratio": n_senders / (n_receivers + 1),
        "repeat_ratio": total_tx / (counterparties + 1),
        "balance_ratio": balance / (i_s["total"] + 1e-9),
        "activity_span_days": max(spread_in, spread_out) / 86400.0,
        # metadata (bukan fitur)
        "_n_external_tx": n_external,
        "_n_total_tx": len(txs),
        "_truncated": int(truncated),
    }
    return feats


def features_only(feats: dict[str, Any]) -> dict[str, float]:
    """Ambil hanya kolom yang menjadi input model, sesuai urutan skema."""
    return {k: float(feats.get(k, 0.0)) for k in LIVE_FEATURES}


if __name__ == "__main__":
    import json
    import sys

    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)
    target = sys.argv[1] if len(sys.argv) > 1 else "0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045"
    print(json.dumps(extract_live_features(target), indent=2))
