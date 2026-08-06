"""Generate thesis-quality confusion matrices for XGBoost and GNN (best scenario: MulDiGraph + Etherscan)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import ConfusionMatrixDisplay

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Numbers from the 1:10 run (12.815 alamat; skenario MulDiGraph + Etherscan, n=2563 test).
# XGBoost & GNN diambil dari ml/artifacts/dataset_scenarios_comparison.csv (run 6 Agustus 2026);
# LR & RF adalah baseline EPTransNet 1:10 (§5.3).
CMS = {
    "xgboost": {
        "cm": np.array([[2288, 42], [12, 221]]),
        "title": "XGBoost – Confusion Matrix\n(Skenario MulDiGraph + Etherscan, n=2.563)",
        "fname": "thesis_cm_xgboost.png",
    },
    "gnn": {
        "cm": np.array([[2271, 59], [13, 220]]),
        "title": "GNN-GraphSAGE V2 – Confusion Matrix\n(Skenario MulDiGraph + Etherscan, n=2.563)",
        "fname": "thesis_cm_gnn.png",
    },
    "rf": {
        "cm": np.array([[2303, 27], [28, 205]]),
        "title": "Random Forest – Confusion Matrix\n(EPTransNet 1:10, n=2.563)",
        "fname": "thesis_cm_rf.png",
    },
    "lr": {
        "cm": np.array([[2277, 53], [74, 159]]),
        "title": "Logistic Regression – Confusion Matrix\n(EPTransNet 1:10, n=2.563)",
        "fname": "thesis_cm_lr.png",
    },
}

for key, cfg in CMS.items():
    cm = cfg["cm"]
    tn, fp, fn, tp = cm[0,0], cm[0,1], cm[1,0], cm[1,1]
    total = tn + fp + fn + tp
    phishing = fn + tp

    fig, ax = plt.subplots(figsize=(6, 5))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Phishing"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(cfg["title"], fontsize=11, fontweight="bold", pad=12)

    # Annotations below
    fn_rate = fn / phishing * 100
    fp_rate = fp / (tn + fp) * 100
    ax.set_xlabel(
        f"Predicted label\n\nTP={tp}  FP={fp}  FN={fn}  TN={tn}  "
        f"| FN rate={fn_rate:.1f}%  FP rate={fp_rate:.1f}%",
        fontsize=9
    )

    fig.tight_layout()
    out = ARTIFACTS / cfg["fname"]
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out}  |  TP={tp} FP={fp} FN={fn} TN={tn}  FN_rate={fn_rate:.1f}%")

print("\nDone.")
