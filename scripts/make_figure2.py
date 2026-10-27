"""
scripts/make_figure2.py
-----------------------
Figure 2 of the revised manuscript: F1 (mean ± std over 10 seeds) per feature scenario and
model, from outputs/revision/multiseed_summary.csv.

The y-axis starts at 0.60 (not 0) so the differences are visible; the caption must say so.
Bars carry hatching and value labels so the figure stays readable in grayscale print.

Run: .venv/Scripts/python.exe scripts/make_figure2.py
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "revision"

SCEN = [("muldigraph_only", "S1\nMulDiGraph (18)"), ("muldigraph_plus_etherscan", "S2\n+ Etherscan (25)"),
        ("muldigraph_plus_bigquery", "S3\n+ BigQuery (28)"), ("muldigraph_plus_etherscan_plus_bigquery", "S4\n+ both (35)")]
MODELS = [("LogisticRegression", "Logistic regression", "#2a78d6", ""),
          ("RandomForest", "Random forest", "#eb6834", "//"),
          ("XGBoost", "XGBoost", "#1baf7a", ".."),
          ("GraphSAGE", "GraphSAGE", "#eda100", "\\\\")]
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#e4e3df"
Y0 = 0.60


def main() -> None:
    s = pd.read_csv(OUT / "multiseed_summary.csv")
    plt.rcParams.update({"font.size": 9, "font.family": "DejaVu Sans", "hatch.linewidth": 0.6})
    fig, ax = plt.subplots(figsize=(7.0, 3.6), dpi=300)
    width, gap = 0.19, 0.012
    x = np.arange(len(SCEN))
    for i, (key, label, color, hatch) in enumerate(MODELS):
        rows = [s[(s.scenario == sc) & (s.model == key)].iloc[0] for sc, _ in SCEN]
        mean = np.array([r.f1_mean for r in rows])
        std = np.array([r.f1_std for r in rows])
        pos = x + (i - 1.5) * (width + gap)
        ax.bar(pos, mean - Y0, width, bottom=Y0, color=color, edgecolor="white", linewidth=0.8,
               hatch=hatch, label=label, zorder=2)
        ax.errorbar(pos, mean, yerr=std, fmt="none", ecolor=INK, elinewidth=0.8, capsize=2, zorder=3)
        for px, m, sd in zip(pos, mean, std):
            ax.text(px, m + sd + 0.006, f"{m:.3f}", ha="center", va="bottom", fontsize=6.2, color=INK, rotation=90)
    ax.set_xticks(x, [lab for _, lab in SCEN], color=INK)
    ax.set_ylim(Y0, 1.0)
    ax.set_ylabel("F1-score (mean ± std, 10 seeds)", color=INK)
    ax.yaxis.grid(True, color=GRID, linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.13), fontsize=8)
    ax.text(0.0, -0.22, f"Note: y-axis starts at {Y0:.2f}.", transform=ax.transAxes, fontsize=7, color=MUTED)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"figure2_f1_by_scenario.{ext}", bbox_inches="tight")
    print("saved", OUT / "figure2_f1_by_scenario.png")


if __name__ == "__main__":
    main()
