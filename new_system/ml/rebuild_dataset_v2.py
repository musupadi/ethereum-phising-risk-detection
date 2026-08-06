"""
new_system/ml/rebuild_dataset_v2.py
-----------------------------------
Membangun ulang dataset dengan FITUR GENERASI KEDUA.

Populasinya PERSIS SAMA dengan `live_dataset.csv` — alamat, label, dan penanda
held-out disalin apa adanya. Tidak ada sampling baru. Yang berubah hanya fiturnya:
26 fitur lama + 11 fitur baru (pola sapu, irama waktu, token ERC-20).

Populasi dijaga tetap sama supaya perbandingannya adil: bila F1 naik, kenaikan itu
benar-benar berasal dari fitur baru, bukan dari data yang kebetulan lebih mudah.

Kenapa harus mengambil ulang dari Etherscan? Karena `live_dataset.csv` hanya
menyimpan fitur akhir, bukan transaksi mentahnya. Fitur baru butuh urutan waktu
tiap transaksi, jadi datanya harus diambil lagi.

Keluaran: new_system/data/live_dataset_v2.csv  (resumable via .partial.csv)

Env:
  LIVE_REQUEST_DELAY   jeda antar panggilan (default 0.32)
  LIVE_WORKERS         jumlah thread (default 6)
  LIVE_MAX_PAGES       batas halaman per sumber (default 5)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent))
from paths import LIVE_DATASET_CSV, NEW_DATA, add_import_paths, ensure_dirs  # noqa: E402

add_import_paths()
ensure_dirs()

from live_features import LIMITER, LIVE_FEATURES_V2, extract_live_features  # noqa: E402

OUT_CSV = NEW_DATA / "live_dataset_v2.csv"
CHECKPOINT = NEW_DATA / "live_dataset_v2.partial.csv"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("rebuild_v2")


def _load_done() -> pd.DataFrame:
    for path in (CHECKPOINT, OUT_CSV):
        if path.exists():
            df = pd.read_csv(path)
            df["address"] = df["address"].astype(str).str.lower()
            return df
    return pd.DataFrame()


def _save(rows: list[dict], done: pd.DataFrame, path: Path) -> pd.DataFrame:
    merged = pd.concat([done, pd.DataFrame(rows)], ignore_index=True) if rows else done
    if not merged.empty:
        merged = merged.drop_duplicates(subset=["address"], keep="last")
        merged.to_csv(path, index=False)
    return merged


def main() -> None:
    load_dotenv()
    api_key = os.getenv("ETHERSCAN_API_KEY", "")
    if not api_key:
        log.error("ETHERSCAN_API_KEY tidak ada di .env")
        sys.exit(1)

    delay = float(os.getenv("LIVE_REQUEST_DELAY", "0.32"))
    workers = int(os.getenv("LIVE_WORKERS", "6"))
    max_pages = int(os.getenv("LIVE_MAX_PAGES", "5"))
    LIMITER.set_interval(delay)

    if not LIVE_DATASET_CSV.exists():
        log.error("%s belum ada.", LIVE_DATASET_CSV)
        sys.exit(1)

    base = pd.read_csv(LIVE_DATASET_CSV)
    base["address"] = base["address"].astype(str).str.lower()
    base = base.drop_duplicates("address", keep="last")
    log.info("Populasi: %d alamat | label: %s", len(base), base["label"].value_counts().to_dict())

    done = _load_done()
    done_addrs = set(done["address"]) if not done.empty else set()
    pending = base[~base["address"].isin(done_addrs)]
    log.info("Sudah selesai: %d | tersisa: %d", len(done_addrs), len(pending))
    if pending.empty:
        _save([], done, OUT_CSV)
        CHECKPOINT.unlink(missing_ok=True)
        log.info("Semua sudah lengkap -> %s", OUT_CSV)
        return

    est = len(pending) * 6.0 * delay / 60
    log.info("Perkiraan waktu: %.0f menit (%d thread)", est, workers)

    def work(row: dict) -> dict | None:
        feats = extract_live_features(row["address"], api_key=api_key, delay=delay,
                                      max_pages=max_pages, include_tokens=True)
        if feats["_n_total_tx"] == 0:
            return None
        out = {"address": row["address"], "label": int(row["label"]),
               "source": row.get("source", ""), "heldout": int(row.get("heldout", 0))}
        out.update({k: feats[k] for k in LIVE_FEATURES_V2})
        out["n_tx"] = feats["_n_total_tx"]
        out["truncated"] = feats["_truncated"]
        return out

    rows: list[dict] = []
    failed = completed = 0
    t0 = time.time()
    tasks = pending.to_dict("records")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(work, t): t["address"] for t in tasks}
        for fut in as_completed(futures):
            completed += 1
            try:
                row = fut.result()
            except Exception as exc:  # noqa: BLE001
                failed += 1
                log.warning("[%d/%d] %s GAGAL: %s", completed, len(tasks),
                            futures[fut][:12], exc)
                row = None
            if row is not None:
                rows.append(row)
            if completed % 100 == 0:
                done = _save(rows, done, CHECKPOINT)
                rows = []
                rate = completed / max(time.time() - t0, 1e-9)
                eta = (len(tasks) - completed) / max(rate, 1e-9) / 60
                log.info("[%d/%d] checkpoint (%d baris, %d gagal) %.2f/dtk, sisa ~%.0f menit",
                         completed, len(tasks), len(done), failed, rate, eta)

    done = _save(rows, done, OUT_CSV)
    CHECKPOINT.unlink(missing_ok=True)
    log.info("SELESAI. %d baris | %d gagal", len(done), failed)
    log.info("Label: %s", done["label"].value_counts().to_dict())
    log.info("Tersimpan -> %s", OUT_CSV)


if __name__ == "__main__":
    main()
