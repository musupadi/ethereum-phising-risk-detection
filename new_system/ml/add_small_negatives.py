"""
ml/add_small_negatives.py
-------------------------
Menambal celah cakupan pada kelas negatif model live.

Masalah yang ditemukan saat evaluasi model sementara:
    Alamat yang disampel dari blok sungguhan didominasi wallet beraktivitas tinggi
    (bot, relayer, MEV) dengan ribuan transaksi. Sementara itu wallet phishing
    median-nya hanya ~13 transaksi. Akibatnya kelas negatif nyaris tidak memuat
    wallet normal BERUKURAN KECIL, dan model cenderung menyimpulkan
    "wallet kecil = phishing".

    Gejalanya: dari 12 wallet acak segar, 2 yang salah dinilai phishing keduanya
    berukuran kecil (6 dan 12 transaksi).

Skrip ini menyampel alamat tambahan lalu HANYA menyimpan yang jumlah transaksinya
kecil, sehingga sebaran ukuran pada kedua kelas menjadi sebanding.

Ini bukan pemilihan data yang bias terhadap label — kelasnya tetap negatif apa
adanya; yang diseimbangkan hanyalah cakupan ukuran aktivitas (stratifikasi).

Keluaran: menambahkan baris ber-`source="block_sample_small"` ke
data/processed/live_dataset.csv

Env:
  LIVE_SMALL_TARGET     jumlah wallet kecil yang dikumpulkan (default 500)
  LIVE_SMALL_MAX_TX     batas atas "kecil" (default 60 transaksi)
  LIVE_REQUEST_DELAY    jeda antar panggilan (default 0.32)
  LIVE_WORKERS          jumlah thread (default 6)
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
from paths import LIVE_DATASET_CSV, PHISHING_CSV, add_import_paths  # noqa: E402

add_import_paths()

from live_features import LIMITER, LIVE_FEATURES, _request, extract_live_features  # noqa: E402

OUT_CSV = LIVE_DATASET_CSV

EARLIEST_BLOCK = 2_900_000
SEED = 2026  # seed berbeda dari build_live_dataset agar alamatnya tidak berulang

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("small_neg")


def sample_candidates(api_key: str, delay: float, n: int, exclude: set[str]) -> list[str]:
    """Kumpulkan calon alamat; utamakan penerima (`to`) karena lebih sering wallet biasa."""
    latest = int(_request({"module": "proxy", "action": "eth_blockNumber"},
                          api_key, delay).get("result", "0x0"), 16)
    rng = random.Random(SEED)
    found, seen = [], set(exclude)
    attempts = 0
    while len(found) < n and attempts < 400:
        attempts += 1
        block_no = rng.randint(EARLIEST_BLOCK, latest)
        try:
            data = _request({"module": "proxy", "action": "eth_getBlockByNumber",
                             "tag": hex(block_no), "boolean": "true"}, api_key, delay)
        except Exception:  # noqa: BLE001
            continue
        txs = (data.get("result") or {}).get("transactions") or []
        rng.shuffle(txs)
        for tx in txs[:80]:
            # "to" lebih dulu: penerima cenderung wallet pengguna biasa,
            # sedangkan "from" pada blok modern banyak diisi bot/relayer.
            for key in ("to", "from"):
                a = str(tx.get(key) or "").lower()
                if len(a) == 42 and a.startswith("0x") and a not in seen:
                    seen.add(a)
                    found.append(a)
        if len(found) >= n:
            break
    log.info("Kandidat terkumpul: %d dari %d blok", len(found), attempts)
    return found


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    if not api_key:
        log.error("ETHERSCAN_API_KEY tidak ada"); sys.exit(1)

    delay = float(os.getenv("LIVE_REQUEST_DELAY", "0.32"))
    workers = int(os.getenv("LIVE_WORKERS", "6"))
    target = int(os.getenv("LIVE_SMALL_TARGET", "500"))
    max_tx = int(os.getenv("LIVE_SMALL_MAX_TX", "60"))
    LIMITER.set_interval(delay)

    if not OUT_CSV.exists():
        log.error("%s belum ada. Jalankan new_system/ml/build_live_dataset.py dulu.", OUT_CSV)
        sys.exit(1)

    existing = pd.read_csv(OUT_CSV)
    existing["address"] = existing["address"].astype(str).str.lower()
    have_small = int(((existing["label"] == 0) & (existing["n_tx"] <= max_tx)).sum())
    log.info("Dataset saat ini: %d baris | negatif kecil (<=%d tx): %d",
             len(existing), max_tx, have_small)

    exclude = set(existing["address"])
    exclude |= set(pd.read_csv(PHISHING_CSV)["address"].astype(str).str.lower())

    # Butuh banyak kandidat karena sebagian besar akan tersaring keluar (terlalu besar).
    candidates = sample_candidates(api_key, delay, target * 6, exclude)

    kept: list[dict] = []
    checked = 0
    t0 = time.time()

    def screen(addr: str) -> int:
        """
        Saringan murah: satu panggilan dengan offset = max_tx+1.

        Kalau yang kembali sebanyak max_tx+1 record, berarti wallet ini lebih besar
        dari ambang dan langsung dibuang — tanpa perlu 3 panggilan lanjutan untuk
        transaksi internal, saldo, dan bytecode. Ini memangkas biaya penyaringan
        dari 4 panggilan menjadi 1 untuk mayoritas kandidat yang tertolak.
        """
        data = _request({"module": "account", "action": "txlist", "address": addr,
                         "startblock": 0, "endblock": 99999999,
                         "page": 1, "offset": max_tx + 1, "sort": "asc"},
                        api_key, delay)
        if data.get("status") != "1":
            return 0
        result = data.get("result")
        return len(result) if isinstance(result, list) else 0

    def work(addr: str) -> dict | None:
        n_screen = screen(addr)
        if n_screen == 0 or n_screen > max_tx:
            return None  # kosong, atau terlalu besar -> lewati sebelum panggilan mahal

        feats = extract_live_features(addr, api_key=api_key, delay=delay, max_pages=1)
        n_tx = feats["_n_total_tx"]
        if n_tx == 0 or n_tx > max_tx:
            return None  # bukan wallet kecil -> lewati
        row = {"address": addr, "label": 0, "source": "block_sample_small", "heldout": 0}
        row.update({k: feats[k] for k in LIVE_FEATURES})
        row["n_tx"] = n_tx
        row["truncated"] = feats["_truncated"]
        return row

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, a): a for a in candidates}
        for fut in as_completed(futures):
            checked += 1
            try:
                row = fut.result()
            except Exception:  # noqa: BLE001
                row = None
            if row is not None:
                kept.append(row)
            if checked % 100 == 0:
                rate = checked / max(time.time() - t0, 1e-9)
                log.info("diperiksa %d/%d, tersimpan %d (%.1f/dtk)",
                         checked, len(candidates), len(kept), rate)
            if len(kept) >= target:
                for f in futures:
                    f.cancel()
                break

    if not kept:
        log.warning("Tidak ada wallet kecil yang ditemukan.")
        return

    merged = pd.concat([existing, pd.DataFrame(kept)], ignore_index=True)
    merged = merged.drop_duplicates(subset=["address"], keep="first")
    merged.to_csv(OUT_CSV, index=False)

    log.info("Ditambahkan %d wallet normal kecil. Total baris: %d", len(kept), len(merged))
    log.info("Label: %s", merged["label"].value_counts().to_dict())
    log.info("Sebaran n_tx per kelas:\n%s",
             merged.groupby("label")["n_tx"].describe()[["count", "25%", "50%", "75%"]].to_string())


if __name__ == "__main__":
    main()
