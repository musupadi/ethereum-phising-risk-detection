"""
ml/eda_visualize.py
-------------------
Figur EDA untuk paper (menjawab catatan bimbingan poin 4,8 + bagian data-centric):
distribusi kelas, persentase outlier per fitur, boxplot sebelum/sesudah capping,
histogram fitur skewed, dan heatmap korelasi.

Sumber : data/processed/hybrid_dataset_clean.csv  (+ *_capped untuk perbandingan)
Output : ml/artifacts/eda/*.png

Run: python ml/eda_visualize.py
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-7s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
PROCESSED = ROOT / "data" / "processed"
OUTDIR = ROOT / "ml" / "artifacts" / "eda"
OUTDIR.mkdir(parents=True, exist_ok=True)

# Palet kategorikal CVD-safe (urutan tetap): normal=biru, phishing=merah-oranye
C_NORMAL, C_PHISH = "#4E79A7", "#E15759"
C_BAR, C_BAR2 = "#4E79A7", "#F28E2B"
META = ["address", "label", "class_name", "source_dataset", "node_id"]

plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": 150, "font.size": 10,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

KEY_FEATS = ["balance", "total_received", "total_sent", "transaction_count",
             "in_degree", "out_degree", "avg_tx_value", "unique_counterparties"]


def _save(fig, name):
    p = OUTDIR / name
    fig.tight_layout()
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    log.info("Tersimpan -> %s", p.name)


def plot_class_distribution(df):
    counts = df["label"].value_counts().sort_index()
    labels = ["Normal (0)", "Phishing (1)"]
    fig, ax = plt.subplots(figsize=(5.2, 4))
    bars = ax.bar(labels, counts.values, color=[C_NORMAL, C_PHISH], width=0.6)
    total = counts.sum()
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x() + b.get_width() / 2, v, f"{v:,}\n({v/total*100:.1f}%)",
                ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("Jumlah wallet")
    ax.set_title("Distribusi Kelas — Imbalance 1 : 6,4")
    ax.set_ylim(0, counts.max() * 1.15)
    _save(fig, "01_class_distribution.png")


def plot_outlier_pct(df, feats):
    rows = []
    for c in feats:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):
            continue
        q1, q3 = s.quantile(.25), s.quantile(.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        pct = 100 * ((s < lo) | (s > hi)).mean()
        rows.append((c, pct))
    rows.sort(key=lambda x: x[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    fig, ax = plt.subplots(figsize=(7, max(4, 0.32 * len(names))))
    ax.barh(names, vals, color=C_BAR)
    for y, v in enumerate(vals):
        ax.text(v + 0.2, y, f"{v:.1f}%", va="center", fontsize=8)
    ax.set_xlabel("Persentase outlier (metode IQR)")
    ax.set_title("Proporsi Outlier per Fitur")
    ax.grid(axis="y", visible=False)
    _save(fig, "02_outlier_percentage.png")


def plot_box_before_after(clean, capped, feats):
    feats = [f for f in feats if f in clean.columns][:8]
    fig, axes = plt.subplots(2, 4, figsize=(14, 7))
    for ax, f in zip(axes.ravel(), feats):
        data = [np.log1p(clean[f].clip(lower=0)), np.log1p(capped[f].clip(lower=0))]
        bp = ax.boxplot(data, labels=["asli", "capped"], patch_artist=True,
                        widths=0.55, showfliers=True, flierprops=dict(marker=".", markersize=3, alpha=0.4))
        for patch, col in zip(bp["boxes"], [C_BAR, C_BAR2]):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)
        for med in bp["medians"]:
            med.set_color("black")
        ax.set_title(f, fontsize=9)
        ax.set_ylabel("log(1+x)")
    fig.suptitle("Sebaran Fitur: Sebelum vs Sesudah Capping (p1–p99)", fontsize=12)
    _save(fig, "03_boxplot_before_after.png")


def plot_hist_by_class(df, feats):
    feats = [f for f in feats if f in df.columns][:6]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, f in zip(axes.ravel(), feats):
        x0 = np.log1p(df.loc[df.label == 0, f].clip(lower=0))
        x1 = np.log1p(df.loc[df.label == 1, f].clip(lower=0))
        bins = np.linspace(0, max(x0.max(), x1.max(), 1), 40)
        ax.hist(x0, bins=bins, color=C_NORMAL, alpha=0.65, label="Normal", density=True)
        ax.hist(x1, bins=bins, color=C_PHISH, alpha=0.6, label="Phishing", density=True)
        ax.set_title(f, fontsize=9)
        ax.set_xlabel("log(1+x)")
    axes.ravel()[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Distribusi Fitur per Kelas (skala log)", fontsize=12)
    _save(fig, "04_hist_by_class.png")


def plot_corr_heatmap(df, feats):
    feats = [f for f in feats if pd.api.types.is_numeric_dtype(df[f])]
    corr = df[feats].corr().values
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=7)
    ax.set_yticks(range(len(feats)))
    ax.set_yticklabels(feats, fontsize=7)
    ax.set_title("Korelasi Antar Fitur (Pearson)", fontsize=12)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.grid(False)
    _save(fig, "05_correlation_heatmap.png")


def main():
    clean = pd.read_csv(PROCESSED / "hybrid_dataset_clean.csv")
    capped = pd.read_csv(PROCESSED / "hybrid_dataset_capped.csv")
    feats = [c for c in clean.columns if c not in META]
    log.info("Loaded clean=%d capped=%d | %d fitur", len(clean), len(capped), len(feats))

    plot_class_distribution(clean)
    plot_outlier_pct(clean, feats)
    plot_box_before_after(clean, capped, KEY_FEATS)
    plot_hist_by_class(clean, KEY_FEATS)
    plot_corr_heatmap(clean, feats)
    log.info("SELESAI. Figur ada di %s", OUTDIR)


if __name__ == "__main__":
    main()
