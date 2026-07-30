"""
ml/clean_hybrid_dataset.py
--------------------------
Data-centric cleaning untuk hybrid_dataset (menjawab catatan bimbingan poin 1,4,5,6,7,11).

Aksi cleaning (semua deterministik & terdokumentasi):
  1. Deduplikasi berdasarkan alamat wallet (node_id/address, lowercased).
  2. Buang fitur konstan `created_at_ts` (bernilai 0 di seluruh baris -> nol informasi).
  3. Verifikasi tidak ada NaN & tidak ada nilai negatif (salah input).
  4. Pertahankan baris low-activity yang tetap punya graph_present (bukan noise).
  5. Tangani outlier via winsorize (capping) persentil 1-99 -> versi *_capped.

Output:
  - data/processed/hybrid_dataset_clean.csv    (dedup + drop fitur mati, nilai asli)
  - data/processed/hybrid_dataset_capped.csv   (clean + outlier di-cap p1..p99)
  - ml/artifacts/cleaning_report.json          (ringkasan semua keputusan + threshold)

CATATAN LEAKAGE: file *_capped di sini di-cap memakai persentil seluruh data (deskriptif,
untuk EDA/paper). Saat MODELING, capping sebaiknya di-fit pada data TRAIN saja lalu diterapkan
ke test. Threshold p1/p99 per fitur disimpan di cleaning_report.json agar bisa dipakai ulang.

Run:  python ml/clean_hybrid_dataset.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"

SRC = PROCESSED / "hybrid_dataset.csv"
OUT_CLEAN = PROCESSED / "hybrid_dataset_clean.csv"
OUT_CAPPED = PROCESSED / "hybrid_dataset_capped.csv"
REPORT = ARTIFACTS / "cleaning_report.json"

META_COLS = ["address", "label", "class_name", "source_dataset", "node_id"]
DROP_FEATURES = ["created_at_ts"]  # konstan 0 di semua baris
CAP_LOW, CAP_HIGH = 0.01, 0.99     # winsorize p1..p99


def main() -> None:
    df = pd.read_csv(SRC)
    n0 = len(df)
    report: dict = {"source": str(SRC.name), "rows_input": n0, "cols_input": df.shape[1]}
    log.info("Loaded %d rows x %d cols", n0, df.shape[1])

    # 1) Deduplikasi ---------------------------------------------------------
    key = df["node_id"].astype(str).str.lower()
    n_dup = int(key.duplicated().sum())
    df = df[~key.duplicated(keep="first")].reset_index(drop=True)
    report["duplicates_removed"] = n_dup
    log.info("Duplikat wallet dihapus: %d", n_dup)

    # 2) Buang fitur konstan / mati -----------------------------------------
    dropped = []
    for c in DROP_FEATURES:
        if c in df.columns and df[c].nunique(dropna=False) <= 1:
            df = df.drop(columns=[c])
            dropped.append(c)
    report["constant_features_dropped"] = dropped
    log.info("Fitur mati dibuang: %s", dropped or "(tidak ada)")

    feature_cols = [c for c in df.columns if c not in META_COLS]

    # 3) Verifikasi NaN & nilai negatif -------------------------------------
    nan_total = int(df[feature_cols].isna().sum().sum())
    neg_counts = {c: int((df[c] < 0).sum()) for c in feature_cols
                  if pd.api.types.is_numeric_dtype(df[c]) and (df[c] < 0).any()}
    report["nan_in_features"] = nan_total
    report["negative_value_columns"] = neg_counts
    log.info("NaN di fitur: %d | kolom dgn nilai negatif: %d", nan_total, len(neg_counts))
    if nan_total:  # imputasi mean hanya jika perlu (di data ini seharusnya 0)
        for c in feature_cols:
            if df[c].isna().any():
                df[c] = df[c].fillna(df[c].mean())
        report["nan_imputed_with_mean"] = True

    # 4) Baris low-activity (informasi tetap valid via graph_present) -------
    # Dihitung atas fitur INTI (graph+etherscan), tanpa fitur bq_* yang sparse by design.
    core_cols = [c for c in feature_cols if not c.startswith("bq_")]
    zero_row = (df[core_cols] == 0).sum(axis=1)
    n_lowact = int((zero_row >= len(core_cols) - 3).sum())  # >= (n-3) dari fitur inti bernilai 0
    report["low_activity_rows_kept"] = n_lowact  # DIPERTAHANKAN, bukan noise
    log.info("Baris low-activity dipertahankan (punya graph_present): %d", n_lowact)

    # Simpan versi CLEAN (nilai asli) ---------------------------------------
    df.to_csv(OUT_CLEAN, index=False)
    log.info("Clean tersimpan -> %s (%d rows x %d cols)", OUT_CLEAN.name, len(df), df.shape[1])

    # 5) Outlier: winsorize p1..p99 -> versi CAPPED -------------------------
    capped = df.copy()
    thresholds: dict = {}
    total_capped = 0
    for c in feature_cols:
        if not pd.api.types.is_numeric_dtype(capped[c]):
            continue
        lo, hi = capped[c].quantile(CAP_LOW), capped[c].quantile(CAP_HIGH)
        if hi <= lo:  # fitur biner/konstan (mis. graph_present) -> lewati
            continue
        n_out = int(((capped[c] < lo) | (capped[c] > hi)).sum())
        capped[c] = capped[c].clip(lower=lo, upper=hi)
        thresholds[c] = {"p1": float(lo), "p99": float(hi), "n_clipped": n_out}
        total_capped += n_out
    capped.to_csv(OUT_CAPPED, index=False)
    report["cap_percentiles"] = {"low": CAP_LOW, "high": CAP_HIGH}
    report["cap_thresholds"] = thresholds
    report["total_values_clipped"] = total_capped
    log.info("Capped tersimpan -> %s | total nilai di-clip: %d", OUT_CAPPED.name, total_capped)

    # Ringkasan akhir --------------------------------------------------------
    report["rows_output"] = len(df)
    report["feature_count_after"] = len(feature_cols)
    report["label_distribution"] = {int(k): int(v) for k, v in df["label"].value_counts().items()}
    REPORT.write_text(json.dumps(report, indent=2), encoding="utf-8")
    log.info("Report tersimpan -> %s", REPORT.name)
    log.info("SELESAI. %d -> %d baris, %d fitur.", n0, len(df), len(feature_cols))


if __name__ == "__main__":
    main()
