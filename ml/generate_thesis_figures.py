"""Generate thesis-quality figures from scenario comparison results."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

df = pd.read_csv(ARTIFACTS / "dataset_scenarios_comparison.csv")

# Clean up names for display
SCENARIO_LABELS = {
    "muldigraph_only": "MulDiGraph Only\n(19 fitur)",
    "muldigraph_plus_etherscan": "MulDiGraph +\nEtherscan (26 fitur)",
    "muldigraph_plus_bigquery": "MulDiGraph +\nBigQuery (29 fitur)",
    "muldigraph_plus_etherscan_plus_bigquery": "MulDiGraph + Etherscan\n+ BigQuery (36 fitur)",
}
MODEL_LABELS = {
    "LogisticRegression": "Logistic Regression",
    "RandomForest": "Random Forest",
    "XGBoost": "XGBoost",
    "GNN-GraphSAGE-V2": "GNN-GraphSAGE V2",
}
COLORS = {
    "Logistic Regression": "#5B8DB8",
    "Random Forest": "#4CAF50",
    "XGBoost": "#F4A923",
    "GNN-GraphSAGE V2": "#E05C5C",
}

df["scenario_label"] = df["scenario"].map(SCENARIO_LABELS)
df["model_label"] = df["model"].map(MODEL_LABELS)


# ── Figure 1: Grouped bar chart — F1-Score per Scenario ──────────────────────
def fig_f1_by_scenario():
    scenarios = list(SCENARIO_LABELS.values())
    models = list(MODEL_LABELS.values())
    n_scenarios = len(scenarios)
    n_models = len(models)
    x = np.arange(n_scenarios)
    width = 0.18
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, ax = plt.subplots(figsize=(13, 6))
    for i, model in enumerate(models):
        vals = []
        for sc in SCENARIO_LABELS.keys():
            row = df[(df["scenario"] == sc) & (df["model_label"] == model)]
            vals.append(row["f1"].values[0] * 100 if len(row) else 0)
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(scenarios, fontsize=9)
    ax.set_ylabel("F1-Score (%)", fontsize=11)
    ax.set_xlabel("Skenario Dataset", fontsize=11)
    ax.set_title("Perbandingan F1-Score per Skenario Dataset dan Model", fontsize=13, fontweight="bold")
    ax.set_ylim(70, 100)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = ARTIFACTS / "thesis_f1_by_scenario.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 2: Heatmap — F1-Score scenarios × models ─────────────────────────
def fig_heatmap_f1():
    pivot = df.pivot_table(index="model_label", columns="scenario", values="f1") * 100
    col_order = list(SCENARIO_LABELS.keys())
    row_order = list(MODEL_LABELS.values())
    pivot = pivot[col_order].reindex(row_order)

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(pivot.values, cmap="YlOrRd", aspect="auto", vmin=75, vmax=95)

    ax.set_xticks(range(len(col_order)))
    ax.set_xticklabels([SCENARIO_LABELS[s] for s in col_order], fontsize=9)
    ax.set_yticks(range(len(row_order)))
    ax.set_yticklabels(row_order, fontsize=10)

    for i in range(len(row_order)):
        for j in range(len(col_order)):
            val = pivot.values[i, j]
            color = "white" if val > 89 else "black"
            ax.text(j, i, f"{val:.2f}%", ha="center", va="center",
                    fontsize=10, fontweight="bold", color=color)

    plt.colorbar(im, ax=ax, shrink=0.8, label="F1-Score (%)")
    ax.set_title("Heatmap F1-Score: Model × Skenario Dataset", fontsize=13, fontweight="bold", pad=15)
    fig.tight_layout()
    out = ARTIFACTS / "thesis_heatmap_f1.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 3: Best scenario (Sc2) — Accuracy/Precision/Recall/F1 ─────────────
def fig_best_scenario_metrics():
    sc2 = df[df["scenario"] == "muldigraph_plus_etherscan"].copy()
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]
    models = list(MODEL_LABELS.values())
    x = np.arange(len(metrics))
    width = 0.18
    offsets = np.linspace(-(len(models) - 1) / 2, (len(models) - 1) / 2, len(models)) * width

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, model in enumerate(models):
        row = sc2[sc2["model_label"] == model]
        vals = [row[m].values[0] * 100 for m in metrics]
        bars = ax.bar(x + offsets[i], vals, width, label=model,
                      color=COLORS[model], edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                    f"{val:.1f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylabel("Score (%)", fontsize=11)
    ax.set_title("Performa Model pada Skenario Terbaik\n(MulDiGraph + Etherscan, 26 fitur)", fontsize=12, fontweight="bold")
    ax.set_ylim(78, 103)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.9)
    fig.tight_layout()
    out = ARTIFACTS / "thesis_best_scenario_metrics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


# ── Figure 4: Impact of Etherscan features (Sc1 → Sc2) ──────────────────────
def fig_etherscan_impact():
    sc1 = df[df["scenario"] == "muldigraph_only"].set_index("model_label")["f1"] * 100
    sc2 = df[df["scenario"] == "muldigraph_plus_etherscan"].set_index("model_label")["f1"] * 100
    models = list(MODEL_LABELS.values())
    sc1 = sc1.reindex(models)
    sc2 = sc2.reindex(models)
    delta = sc2 - sc1

    x = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: side-by-side bars Sc1 vs Sc2
    ax = axes[0]
    w = 0.35
    b1 = ax.bar(x - w / 2, sc1.values, w, label="MulDiGraph Only", color="#90CAF9", edgecolor="white")
    b2 = ax.bar(x + w / 2, sc2.values, w, label="+ Etherscan", color="#1565C0", edgecolor="white")
    for bar, val in zip(b1, sc1.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    for bar, val in zip(b2, sc2.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.2,
                f"{val:.1f}", ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("F1-Score (%)", fontsize=11)
    ax.set_title("F1-Score: Sebelum vs Sesudah Fitur Etherscan", fontsize=11, fontweight="bold")
    ax.set_ylim(75, 100)
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)
    ax.legend(fontsize=9)

    # Right: delta bars
    ax = axes[1]
    bar_colors = [COLORS[m] for m in models]
    bars = ax.bar(x, delta.values, color=bar_colors, edgecolor="white")
    for bar, val in zip(bars, delta.values):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.05 if val >= 0 else bar.get_height() - 0.4,
                f"+{val:.2f}" if val >= 0 else f"{val:.2f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_ylabel("Peningkatan F1-Score (poin %)", fontsize=11)
    ax.set_title("Kontribusi Fitur Etherscan\nterhadap F1-Score", fontsize=11, fontweight="bold")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")
    ax.yaxis.grid(True, alpha=0.4)
    ax.set_axisbelow(True)

    fig.tight_layout()
    out = ARTIFACTS / "thesis_etherscan_impact.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}")


if __name__ == "__main__":
    fig_f1_by_scenario()
    fig_heatmap_f1()
    fig_best_scenario_metrics()
    fig_etherscan_impact()
    print("\nDone. All thesis figures generated.")
