"""
ml/build_live_dataset.py
------------------------
Membangun data latih untuk MODEL LIVE (model B) — model yang dipakai saat pengguna
memindai wallet di luar dataset EPTransNet.

Latar belakang masalah (lihat `laporan belajar/13_uji_wallet_luar_dataset.md`):
model produksi lama memberi skor HIGH kepada Vitalik Buterin (92), Uniswap V2 Router
(100), dan ETH2 Deposit Contract (84). Penyebabnya kelas negatif pada data latih
didominasi wallet beraktivitas rendah, sehingga model mempelajari jalan pintas
"banyak pengirim unik = phishing".

Dua perbaikan yang diterapkan di sini:
  1. HARD NEGATIVES — kelas negatif diisi wallet sah yang benar-benar ramai:
     alamat terkenal (bursa, router DEX, bridge, kontrak staking) + wallet acak
     yang diambil dari blok sungguhan lintas era 2017–sekarang.
  2. FITUR SATU SUMBER — fitur kedua kelas dihitung dari Etherscan dengan rumus
     yang sama, menghilangkan ketidakcocokan sumber MulDiGraph vs Etherscan.

Keluaran:
  data/processed/live_dataset.csv           — dataset final
  data/processed/live_dataset.partial.csv   — checkpoint (dihapus saat selesai)

Skrip ini RESUMABLE: jalankan ulang kapan saja, alamat yang sudah diambil dilewati.

Env (.env):
  ETHERSCAN_API_KEY        wajib
  LIVE_REQUEST_DELAY       jeda antar panggilan, detik (default 0.25)
  LIVE_NEG_TARGET          jumlah hard negative yang dikumpulkan (default 3500)
  LIVE_MAX_PAGES           batas halaman txlist per alamat (default 10 = 10.000 tx)
"""

from __future__ import annotations

import logging
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))

from live_features import (  # noqa: E402
    LIMITER, EtherscanError, LIVE_FEATURES, _request, extract_live_features,
)

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"

PHISHING_CSV = DATA_RAW / "addresses_phishing.csv"
HYBRID_CSV = DATA_PROCESSED / "hybrid_dataset.csv"
OUT_CSV = DATA_PROCESSED / "live_dataset.csv"
CHECKPOINT_CSV = DATA_PROCESSED / "live_dataset.partial.csv"

SEED = 42
EARLIEST_BLOCK = 2_900_000  # ~Januari 2017
PER_BLOCK = 60  # alamat diambil per blok; menekan jumlah panggilan API

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("build_live")

# ── Hard negatives kurasi: alamat sah yang SANGAT ramai ───────────────────────
# Justru alamat semacam inilah yang salah dinilai oleh model lama. Sebagian
# ditahan sebagai data uji (held-out) agar evaluasinya tidak sirkular.
CURATED_LEGIT = {
    # --- dipakai untuk MELATIH ---
    "train": [
        ("0x28c6c06298d514db089934071355e5743bf21d60", "Binance 14"),
        ("0x21a31ee1afc51d94c2efccaa2092ad1028285549", "Binance 15"),
        ("0xdfd5293d8e347dfe59e90efd55b2956a1343963d", "Binance 16"),
        ("0x56eddb7aa87536c09ccc2793473599fd21a8b17f", "Binance 17"),
        ("0x9696f59e4d72e237be84ffd425dcad154bf96976", "Binance 18"),
        ("0x4976a4a02f38326660d17bf34b431dc6e2eb2327", "Binance 19"),
        ("0xd551234ae421e3bcba99a0da6d736074f22192ff", "Binance 2"),
        ("0x564286362092d8e7936f0549571a803b203aaced", "Binance 3"),
        ("0x0681d8db095565fe8a346fa0277bffde9c0edbbf", "Binance 4"),
        ("0xfe9e8709d3215310075d67e3ed32a380ccf451c8", "Binance 5"),
        ("0x2faf487a4414fe77e2327f0bf4ae2a264a776ad2", "FTX/Alameda"),
        ("0x267be1c1d684f78cb4f6a176c4911b741e4ffdc0", "Kraken 4"),
        ("0xa83b11093c858c86321fbc4c20fe82cdbd58e09e", "Kraken 5"),
        ("0xe93381fb4c4f14bda253907b18fad305d799241a", "Huobi 10"),
        ("0xfdb16996831753d5331ff813c29a93c76834a0ad", "Huobi 12"),
        ("0x6748f50f686bfbca6fe8ad62b22228b87f31ff2b", "Huobi 13"),
        ("0x1522900b6dafac587d499a862861c0869be6e428", "Poloniex"),
        ("0xb794f5ea0ba39494ce839613fffba74279579268", "Poloniex old"),
        ("0x742d35cc6634c0532925a3b844bc454e4438f44e", "Bitfinex 2"),
        ("0x876eabf441b2ee5b5b0554fd502a8e0600950cfa", "Bitfinex 3"),
        ("0xe853c56864a2ebe4576a807d26fdc4a0ada51919", "Kraken 3"),
        ("0x3f5ce5fbfe3e9af3971dd833d26ba9b5c936f0be", "Binance 1"),
        ("0xe592427a0aece92de3edee1f18e0157c05861564", "Uniswap V3 Router"),
        ("0x68b3465833fb72a70ecdf485e0e4c7bd8665fc45", "Uniswap Router 2"),
        ("0xd9e1ce17f2641f24ae83637ab66a2cca9c378b9f", "SushiSwap Router"),
        ("0x1111111254eeb25477b68fb85ed929f73a960582", "1inch V5 Router"),
        ("0x881d40237659c251811cec9c364ef91dc08d300c", "Metamask Swap Router"),
        ("0xdef1c0ded9bec7f1a1670819833240f027b25eff", "0x Exchange Proxy"),
        ("0x7d2768de32b0b80b7a3454c06bdac94a69ddc7a9", "Aave LendingPool V2"),
        ("0x5d3a536e4d6dbd6114cc1ead35777bab948e3643", "Compound cDAI"),
        ("0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "USDC token"),
        ("0xdac17f958d2ee523a2206206994597c13d831ec7", "USDT token"),
        ("0x6b175474e89094c44da98b954eedeac495271d0f", "DAI token"),
        ("0xc02aaa39b223fe8d0a0e5c4f27ead9083c756cc2", "WETH"),
        ("0x2260fac5e5542a773aa44fbcfedf7c193bc2c599", "WBTC"),
        ("0x3ee18b2214aff97000d974cf647e7c347e8fa585", "Wormhole Bridge"),
        ("0xa090e606e30bd747d4e6245a1517ebe430f0057e", "Coinbase Miner"),
        ("0x503828976d22510aad0201ac7ec88293211d23da", "Coinbase 2"),
        ("0xddfabcdc4d8ffc6d5beaf154f18b778f892a0740", "Coinbase 3"),
        ("0x71660c4005ba85c37ccec55d0c4493e66fe775d3", "Coinbase 1"),
    ],
    # --- DITAHAN, hanya untuk menguji (tidak pernah dilatih) ---
    "heldout": [
        ("0xd8da6bf26964af9d7eed9e03e53415d37aa96045", "Vitalik Buterin"),
        ("0x7a250d5630b4cf539739df2c5dacb4c659f2488d", "Uniswap V2 Router"),
        ("0x00000000219ab540356cbb839cbe05303d7705fa", "ETH2 Deposit Contract"),
        ("0x220866b1a2219f40e72f5c628b65d54268ca3a9d", "Vitalik 3"),
        ("0xab5801a7d398351b8be11c439e05c5b3259aec9b", "Vitalik 2"),
        ("0x5041ed759dd4afc3a72b8192c143f72f4724081a", "OKX"),
        ("0x0a59649758aa4d66e25f08dd01271e891fe52199", "Maker PSM USDC"),
        ("0x1f9840a85d5af5bf1d1762f925bdaddc4201f984", "UNI token"),
    ],
}


def _load_done() -> pd.DataFrame:
    for path in (CHECKPOINT_CSV, OUT_CSV):
        if path.exists():
            df = pd.read_csv(path)
            if "address" in df.columns:
                df["address"] = df["address"].astype(str).str.lower()
                return df
    return pd.DataFrame()


def _save(rows: list[dict], done: pd.DataFrame, path: Path) -> pd.DataFrame:
    merged = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if rows else done
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["address"], keep="last")
        merged.to_csv(path, index=False)
    return merged


def sample_addresses_from_blocks(
    api_key: str, delay: float, n_target: int, exclude: set[str]
) -> list[str]:
    """Ambil alamat EOA acak dari blok sungguhan, tersebar lintas era 2017-sekarang."""
    latest_hex = _request({"module": "proxy", "action": "eth_blockNumber"},
                          api_key, delay).get("result", "0x0")
    latest = int(latest_hex, 16)
    log.info("Blok terbaru: %d — menyampel blok acak dari %d sampai %d",
             latest, EARLIEST_BLOCK, latest)

    rng = random.Random(SEED)
    found: list[str] = []
    seen: set[str] = set(exclude)
    attempts = 0

    while len(found) < n_target and attempts < n_target * 3:
        attempts += 1
        block_no = rng.randint(EARLIEST_BLOCK, latest)
        try:
            data = _request({"module": "proxy", "action": "eth_getBlockByNumber",
                             "tag": hex(block_no), "boolean": "true"}, api_key, delay)
        except EtherscanError as exc:
            log.warning("Blok %d gagal diambil: %s", block_no, exc)
            continue

        txs = (data.get("result") or {}).get("transactions") or []
        rng.shuffle(txs)
        # Alamat dalam satu blok tidak saling berhubungan, jadi aman mengambil
        # banyak sekaligus. Ini memangkas jumlah panggilan API secara drastis.
        before = len(found)
        for tx in txs[:PER_BLOCK]:
            for key in ("from", "to"):
                a = str(tx.get(key) or "").lower()
                if len(a) == 42 and a.startswith("0x") and a not in seen:
                    seen.add(a)
                    found.append(a)
        if len(found) >= n_target:
            break

        if attempts % 10 == 0:
            log.info("  blok ke-%d (#%d): +%d alamat, total %d/%d",
                     attempts, block_no, len(found) - before, len(found), n_target)

    log.info("Penyampelan blok selesai: %d alamat dari %d blok", len(found), attempts)
    return found[:n_target]


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    if not api_key:
        log.error("ETHERSCAN_API_KEY tidak ada di .env")
        sys.exit(1)

    delay = float(os.getenv("LIVE_REQUEST_DELAY", "0.25"))
    neg_target = int(os.getenv("LIVE_NEG_TARGET", "3500"))
    max_pages = int(os.getenv("LIVE_MAX_PAGES", "10"))
    workers = int(os.getenv("LIVE_WORKERS", "8"))
    LIMITER.set_interval(delay)  # pembatas laju global dipakai semua thread

    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

    # ── Susun daftar kerja ────────────────────────────────────────────────────
    phishing = pd.read_csv(PHISHING_CSV)["address"].astype(str).str.lower().tolist()
    log.info("Wallet phishing (positif): %d", len(phishing))

    tasks: list[tuple[str, int, str, int]] = [  # (address, label, source, heldout)
        (a, 1, "phishing_eptransnet", 0) for a in phishing
    ]
    for a, name in CURATED_LEGIT["train"]:
        tasks.append((a, 0, f"curated:{name}", 0))
    for a, name in CURATED_LEGIT["heldout"]:
        tasks.append((a, 0, f"curated_heldout:{name}", 1))

    exclude = set(phishing) | {a for a, _ in CURATED_LEGIT["train"]} \
        | {a for a, _ in CURATED_LEGIT["heldout"]}
    if HYBRID_CSV.exists():
        exclude |= set(pd.read_csv(HYBRID_CSV)["node_id"].astype(str).str.lower())

    done = _load_done()
    done_addrs = set(done["address"]) if not done.empty else set()

    # Sampel blok hanya diambil bila memang masih kurang (hemat panggilan API).
    have_block_neg = 0
    if not done.empty and "source" in done.columns:
        have_block_neg = int((done["source"] == "block_sample").sum())
    need_block = max(0, neg_target - have_block_neg)
    if need_block > 0:
        log.info("Menyampel %d kandidat alamat dari blok...", need_block)
        sampled = sample_addresses_from_blocks(api_key, delay, need_block, exclude)
        tasks += [(a, 0, "block_sample", 0) for a in sampled]
        log.info("Kandidat dari blok terkumpul: %d", len(sampled))
    else:
        log.info("Hard negative dari blok sudah cukup (%d)", have_block_neg)

    pending = [t for t in tasks if t[0] not in done_addrs]
    log.info("Total tugas: %d | sudah selesai: %d | tersisa: %d",
             len(tasks), len(done_addrs), len(pending))
    if not pending:
        log.info("Tidak ada yang perlu diambil.")
        if CHECKPOINT_CSV.exists():
            _save([], done, OUT_CSV)
            CHECKPOINT_CSV.unlink()
        return

    est_min = len(pending) * 4.3 / max(1.0, 1.0 / delay) / 60
    log.info("Perkiraan waktu: %.0f menit (%d thread, laju %.1f panggilan/detik)",
             est_min, workers, 1.0 / delay)

    # ── Ambil fitur secara paralel ────────────────────────────────────────────
    # Latensi Etherscan ~1,2 dtk/panggilan sementara kuotanya 5/detik, jadi
    # menunggu berurutan hanya memakai ~20% kuota. Beberapa thread + pembatas
    # laju global memakai kuota itu sepenuhnya tanpa melanggarnya.
    def _work(task: tuple[str, int, str, int]) -> dict | None:
        addr, label, source, heldout = task
        feats = extract_live_features(addr, api_key=api_key, delay=delay,
                                      max_pages=max_pages)
        if feats["_n_total_tx"] == 0:
            return None  # alamat tanpa transaksi tidak informatif
        row = {"address": addr, "label": label, "source": source, "heldout": heldout}
        row.update({k: feats[k] for k in LIVE_FEATURES})
        row["n_tx"] = feats["_n_total_tx"]
        row["truncated"] = feats["_truncated"]
        return row

    rows: list[dict] = []
    failed = 0
    completed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_work, t): t for t in pending}
        for future in as_completed(futures):
            addr = futures[future][0]
            completed += 1
            try:
                row = future.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.warning("[%d/%d] %s GAGAL: %s", completed, len(pending), addr, exc)
                row = None
            if row is not None:
                rows.append(row)

            if completed % 100 == 0:
                done = _save(rows, done, CHECKPOINT_CSV)
                rows = []
                elapsed = time.time() - t_start
                rate = completed / max(elapsed, 1e-9)
                eta = (len(pending) - completed) / max(rate, 1e-9) / 60
                log.info("[%d/%d] checkpoint (%d baris, %d gagal) "
                         "%.1f alamat/dtk, sisa ~%.0f menit",
                         completed, len(pending), len(done), failed, rate, eta)

    done = _save(rows, done, OUT_CSV)
    if CHECKPOINT_CSV.exists():
        CHECKPOINT_CSV.unlink()

    log.info("SELESAI. Total baris: %d | gagal: %d", len(done), failed)
    log.info("Distribusi label: %s", done["label"].value_counts().to_dict())
    log.info("Distribusi sumber:\n%s", done["source"].str.split(":").str[0].value_counts())
    log.info("Tersimpan -> %s", OUT_CSV)


if __name__ == "__main__":
    main()
