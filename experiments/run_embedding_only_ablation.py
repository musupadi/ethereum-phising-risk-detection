"""
experiments/run_embedding_only_ablation.py
-------------------------------------------
Final ablation: "GraphSAGE Embedding + XGBoost (No Etherscan Ablation)".

Purpose
    Isolate WHY late fusion V2 (F1 88.25%) beats the graph-only GraphSAGE baseline
    (F1 81.89%). Two things changed at once in V2: the linear classification head was
    replaced by XGBoost, AND the 7 Etherscan features were concatenated. This ablation
    removes the Etherscan branch and keeps everything else, so the two effects separate:

        row 3 (this ablation)  = embedding only        -> isolates the XGBoost head
        row 4 (V2 late fusion) = embedding + Etherscan -> adds the Etherscan branch
        row 4 - row 3          = the Etherscan contribution

Protocol
    The GraphSAGE encoder is NOT retrained. The saved V2 encoder weights are loaded
    from models/graphsage_encoder_for_fusion_v2.pt and run in eval mode, so the 128-d
    pre-head embeddings are bit-identical to V2's. The split, scalers, topology and
    leakage controls are reproduced exactly. As a proof that the encoder state really
    is V2's, the encoder's own head predictions are re-scored and asserted equal to the
    GraphSAGE metrics stored in outputs/late_fusion_metrics_v2.json.

    Rows 1 (XGBoost Scenario 2) and 4 (Late Fusion V2) are read from the V2 metrics
    file rather than retrained - nothing is recomputed that V2 already established.

All outputs carry an _ablation suffix. V2 and V1 artifacts are never modified.
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))
sys.path.insert(0, str(ROOT / "experiments"))

from gnn_model_v2 import GraphPhiV2, DEVICE  # noqa: E402
from run_graphsage_etherscan_late_fusion_v2 import (  # noqa: E402
    SEED,
    extract_hidden_embeddings,
    extract_muldigraph_edges_only,
    load_transaction_graph,
    metrics_row,
    plot_cm,
    set_seeds,
    train_xgb_on,
)

HYBRID_CSV = ROOT / "data" / "processed" / "hybrid_dataset.csv"
HYBRID_SCHEMA_JSON = ROOT / "ml" / "artifacts" / "hybrid_feature_schema.json"
OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"
V2_METRICS_JSON = OUTPUTS / "late_fusion_metrics_v2.json"
V2_ENCODER_PT = MODELS / "graphsage_encoder_for_fusion_v2.pt"

ABLATION_NAME = "GraphSAGE Embedding + XGBoost (No Etherscan Ablation)"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("ablation")


def main() -> None:
    set_seeds()
    checks: list[str] = []

    if not V2_ENCODER_PT.exists() or not V2_METRICS_JSON.exists():
        raise FileNotFoundError(
            "V2 artifacts missing. Run experiments/run_graphsage_etherscan_late_fusion_v2.py first."
        )
    with open(V2_METRICS_JSON, "r", encoding="utf-8") as f:
        v2 = json.load(f)

    # ── 1. Reproduce the V2 data setup exactly ────────────────────────────────
    df = pd.read_csv(HYBRID_CSV)
    with open(HYBRID_SCHEMA_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)
    graph_cols = [c for c in schema["scenarios"]["muldigraph_only"] if c in df.columns]

    df["wallet_address"] = df["node_id"].astype(str).str.lower()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    train_addrs = train_df["wallet_address"].tolist()
    test_addrs = test_df["wallet_address"].tolist()
    y_train = train_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)
    n_train, n_test = len(train_df), len(test_df)

    assert (n_train, n_test) == (v2["split"]["n_train"], v2["split"]["n_test"]), "split differs from V2"
    checks.append(f"split reproduced identically to V2 (train={n_train}, test={n_test}, random_state={SEED})")

    tx_graph = load_transaction_graph()
    edge_list, edge_audit = extract_muldigraph_edges_only(tx_graph, train_addrs, test_addrs)
    assert edge_audit["unique_directed_edges"] == v2["edge_audit"]["unique_directed_edges"], "topology differs from V2"
    checks.append(
        f"topology reproduced identically to V2 ({edge_audit['unique_directed_edges']} unique "
        f"MulDiGraph edges; etherscan_edges.csv not read)"
    )

    X_train_g = train_df[graph_cols].fillna(0).values.astype(np.float64)
    X_test_g = test_df[graph_cols].fillna(0).values.astype(np.float64)
    gscaler = StandardScaler()
    X_train_gs = gscaler.fit_transform(X_train_g)
    X_test_gs = gscaler.transform(X_test_g)
    checks.append("graph-feature StandardScaler fitted on TRAIN rows only (same as V2)")

    # ── 2. Load the saved V2 encoder - no retraining ──────────────────────────
    gnn = GraphPhiV2(model_type="sage", input_dim=len(graph_cols), hidden_dim=128, num_layers=2)
    graph, train_mask, test_mask = gnn.build_graph_with_real_edges(
        X_train_gs, y_train, X_test_gs, y_test, edge_list
    )
    gnn.model.load_state_dict(torch.load(V2_ENCODER_PT, map_location=DEVICE))
    gnn.model.eval()
    log.info("Loaded saved V2 encoder weights (no retraining) <- %s", V2_ENCODER_PT.name)
    checks.append("GraphSAGE encoder loaded from the saved V2 checkpoint; not retrained, eval mode (BatchNorm running stats frozen)")

    # Proof the loaded state is genuinely V2's: its own head must reproduce V2's metrics.
    gnn_pred, gnn_proba_mat = gnn.predict(graph, test_mask)
    gnn_row = metrics_row(
        "GraphSAGE graph-only (leakage-controlled)",
        v2["graphsage_graph_only_leakage_controlled"]["Input"],
        y_test, gnn_pred, gnn_proba_mat[:, 1],
    )
    for k in ("Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC", "TN", "FP", "FN", "TP"):
        ref = v2["graphsage_graph_only_leakage_controlled"][k]
        assert abs(gnn_row[k] - ref) < 1e-9, f"encoder mismatch on {k}: {gnn_row[k]} vs V2 {ref}"
    checks.append("loaded encoder reproduces V2's GraphSAGE test metrics exactly -> identical encoder state confirmed")
    log.info("Encoder identity verified against V2 metrics.")

    # ── 3. Embeddings (identical to V2 by construction) ───────────────────────
    emb_all = extract_hidden_embeddings(gnn.model, graph.x, graph.edge_index)
    assert emb_all.shape == (n_train + n_test, 128)
    emb_train, emb_test = emb_all[:n_train], emb_all[n_train:]
    checks.append("embedding = final SAGEConv->BatchNorm->ReLU output (pre-head, 128-d); logits/probabilities not used")
    checks.append("no Etherscan feature enters this model - embedding is the only input")

    # ── 4. XGBoost on the embedding alone ─────────────────────────────────────
    emb_names = [f"sage_emb_{i}" for i in range(128)]
    log.info("Training embedding-only XGBoost (128-d, no Etherscan) ...")
    abl_pred, abl_proba, abl_model = train_xgb_on(emb_train, y_train, emb_test, emb_names)
    with open(MODELS / "graphsage_embedding_only_xgboost_ablation.pkl", "wb") as f:
        pickle.dump(abl_model, f)
    checks.append("XGBoost trained on TRAIN embeddings only; inner validation split drawn from TRAIN; test scored once")

    abl_row = metrics_row(ABLATION_NAME, "128-d graph embedding only (no Etherscan)", y_test, abl_pred, abl_proba)

    # ── 5. Four-model comparison ──────────────────────────────────────────────
    xgb_row = dict(v2["xgboost_scenario2_reference"])
    fus_row = dict(v2["late_fusion_v2"])
    rows = [xgb_row, gnn_row, abl_row, fus_row]
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUTPUTS / "four_model_comparison_ablation.csv", index=False)

    print("\n=== Four-model comparison (held-out test set, n=%d) ===" % n_test)
    print(comp_df.drop(columns=["Input"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ── 6. Interpretation rule from the task spec ─────────────────────────────
    f1_abl = abl_row["F1"]
    f1_fus = fus_row["F1"]
    f1_gnn = gnn_row["F1"]
    f1_xgb = xgb_row["F1"]

    d_eth = f1_fus - f1_abl        # Etherscan branch contribution
    d_head = f1_abl - f1_gnn       # XGBoost-head contribution
    total = f1_fus - f1_gnn

    # "Close to 88.25%" is operationalised as within 1.00 pp of the V2 F1.
    CLOSE_PP = 1.00
    close = abs(d_eth) * 100 <= CLOSE_PP

    if close:
        verdict = (
            f"Embedding-only XGBoost reaches F1 {f1_abl*100:.2f}%, within {abs(d_eth)*100:.2f} pp of "
            f"late fusion V2 ({f1_fus*100:.2f}%). **The V2 improvement over the GraphSAGE baseline comes "
            f"mostly from replacing the GraphSAGE linear classification head with XGBoost**, not from the "
            f"Etherscan branch. Of the {total*100:+.2f} pp total gain over the graph-only baseline, "
            f"{d_head*100:+.2f} pp is attributable to the classifier swap and only {d_eth*100:+.2f} pp to "
            f"the seven Etherscan features."
        )
    else:
        verdict = (
            f"Embedding-only XGBoost reaches F1 {f1_abl*100:.2f}%, which is materially lower than late "
            f"fusion V2 ({f1_fus*100:.2f}%) by {abs(d_eth)*100:.2f} pp. **The Etherscan branch contributes "
            f"measurable additional value beyond the classifier swap.** Of the {total*100:+.2f} pp total gain "
            f"over the graph-only baseline, {d_head*100:+.2f} pp is attributable to replacing the linear head "
            f"with XGBoost and {d_eth*100:+.2f} pp to the seven Etherscan features."
        )
    log.info("Attribution: head swap %+.2f pp | Etherscan branch %+.2f pp | total %+.2f pp",
             d_head * 100, d_eth * 100, total * 100)

    # Gain audit for the embedding-only model (all gain is necessarily embedding gain).
    gain = abl_model.model.get_score(importance_type="gain")
    gain_total = sum(gain.values()) or 1.0
    gain_report = {
        "features_used_by_trees": len(gain),
        "top_features": [
            {"feature": k, "gain": float(v), "share": float(v / gain_total)}
            for k, v in sorted(gain.items(), key=lambda x: -x[1])[:15]
        ],
    }

    # ── 7. Artifacts ──────────────────────────────────────────────────────────
    with open(OUTPUTS / "embedding_only_ablation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": ABLATION_NAME,
                "purpose": "separate the XGBoost-head effect from the Etherscan-branch effect in late fusion V2",
                "encoder": {
                    "retrained": False,
                    "checkpoint": str(V2_ENCODER_PT.relative_to(ROOT)).replace("\\", "/"),
                    "identity_verified_against": "outputs/late_fusion_metrics_v2.json",
                    "embedding_dimension": 128,
                    "input_features": len(graph_cols),
                },
                "split": v2["split"],
                "edge_audit": edge_audit,
                "models": {
                    "xgboost_scenario2": xgb_row,
                    "graphsage_graph_only": gnn_row,
                    "embedding_only_xgboost_ablation": abl_row,
                    "late_fusion_v2": fus_row,
                },
                "attribution_pp": {
                    "xgboost_head_swap": d_head * 100,
                    "etherscan_branch": d_eth * 100,
                    "total_over_graphsage_baseline": total * 100,
                    "gap_to_xgboost_scenario2": (f1_abl - f1_xgb) * 100,
                },
                "interpretation_rule": {
                    "threshold_pp": CLOSE_PP,
                    "embedding_only_close_to_late_fusion": bool(close),
                    "verdict": verdict,
                },
                "gain_report": gain_report,
                "leakage_and_alignment_checks": checks,
                "seeds": {"numpy": SEED, "torch": SEED, "sklearn_random_state": SEED},
                "device": str(DEVICE),
            },
            f,
            indent=2,
        )

    cm_abl = confusion_matrix(y_test, abl_pred, labels=[0, 1])
    pd.DataFrame(
        cm_abl, index=["actual_normal", "actual_phishing"], columns=["pred_normal", "pred_phishing"]
    ).to_csv(OUTPUTS / "embedding_only_confusion_matrix_ablation.csv")

    pd.DataFrame(
        {
            "wallet_address": test_addrs,
            "y_true": y_test,
            "y_pred": abl_pred,
            "y_proba_phishing": abl_proba,
        }
    ).to_csv(OUTPUTS / "embedding_only_predictions_ablation.csv", index=False)

    plot_cm(cm_abl, f"{ABLATION_NAME}\nConfusion Matrix",
            OUTPUTS / "embedding_only_confusion_matrix_ablation.png")

    # Four-model bar chart.
    fig, ax = plt.subplots(figsize=(9.5, 5.0))
    metric_names = ["Precision", "Recall", "F1"]
    x = np.arange(len(metric_names))
    width = 0.2
    short = ["XGBoost (S2, 26 feat)", "GraphSAGE (19 feat)", "Emb-only XGB (128)", "Late Fusion V2 (135)"]
    for i, (m, name) in enumerate(zip(rows, short)):
        bars = ax.bar(x + (i - 1.5) * width, [m[k] for k in metric_names], width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7)
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Four-model comparison - held-out test set")
    ax.legend(fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "four_model_comparison_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ROC / PR need probabilities; V2 rows come from the stored predictions file.
    v2_preds = pd.read_csv(OUTPUTS / "late_fusion_predictions_v2.csv")
    assert v2_preds["wallet_address"].tolist() == test_addrs, "V2 prediction file wallet order mismatch"
    assert (v2_preds["y_true"].values == y_test).all()
    fus_proba = v2_preds["y_proba_phishing"].values
    gnn_proba = v2_preds["graphsage_proba_phishing"].values
    checks.append("V2 stored predictions re-aligned by wallet_address before plotting curves")

    curves = [
        ("GraphSAGE graph-only", gnn_proba, gnn_row),
        ("Emb-only XGBoost (ablation)", abl_proba, abl_row),
        ("Late Fusion V2", fus_proba, fus_row),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for label, proba, m in curves:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={m['ROC-AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC - ablation vs V2")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "roc_curve_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for label, proba, m in curves:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"{label} (AP={m['PR-AUC']:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall - ablation vs V2")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "pr_curve_ablation.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 8. Report ─────────────────────────────────────────────────────────────
    table_md = "| # | Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |\n"
    table_md += "|---|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for i, m in enumerate(rows, 1):
        table_md += (
            f"| {i} | {m['Model']} | {m['Input']} | {m['Accuracy']*100:.2f}% | {m['Precision']*100:.2f}% | "
            f"{m['Recall']*100:.2f}% | {m['F1']*100:.2f}% | {m['ROC-AUC']:.4f} | {m['PR-AUC']:.4f} | "
            f"{m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} |\n"
        )

    report = f"""# Ablation: {ABLATION_NAME}

## Why this experiment exists

Late fusion V2 (F1 {f1_fus*100:.2f}%) beat the graph-only GraphSAGE baseline (F1 {f1_gnn*100:.2f}%) by
{total*100:+.2f} pp, but **two things changed at once**: the GraphSAGE linear classification head was
replaced by XGBoost, and seven Etherscan features were concatenated. That confound made the
V2 gain impossible to attribute. This ablation removes the Etherscan branch and changes
nothing else, which separates the two effects.

## Protocol

- The GraphSAGE encoder is **not retrained**. Weights are loaded from
  `models/graphsage_encoder_for_fusion_v2.pt` and run in eval mode, so the 128-d pre-head
  embeddings are identical to V2's.
- Identical split (train {n_train} / test {n_test}, `random_state=42`), identical topology
  ({edge_audit['unique_directed_edges']} unique MulDiGraph edges), identical leakage-controlled protocol.
- **Encoder identity is proven, not assumed**: the loaded encoder's own head predictions are
  re-scored and asserted equal - to 1e-9 on every metric - to the GraphSAGE row stored in
  `outputs/late_fusion_metrics_v2.json`.
- XGBoost input is the 128-d embedding **only**. No Etherscan feature enters this model.
- Rows 1 and 4 are read from the V2 metrics file; nothing V2 already established is recomputed.

## Results (held-out test set, n = {n_test})

{table_md}
## Attribution

| Effect | Contribution to F1 |
|---|---|
| Replacing the GraphSAGE linear head with XGBoost (row 2 -> row 3) | **{d_head*100:+.2f} pp** |
| Adding the 7 Etherscan features (row 3 -> row 4) | **{d_eth*100:+.2f} pp** |
| Total gain of late fusion V2 over the graph-only baseline (row 2 -> row 4) | **{total*100:+.2f} pp** |
| Remaining gap of the embedding-only model to XGBoost Scenario 2 (row 3 -> row 1) | **{(f1_abl - f1_xgb)*100:+.2f} pp** |

## Verdict

{verdict}

Decision rule applied: "close to {f1_fus*100:.2f}%" was operationalised before looking at the
result as **within {CLOSE_PP:.2f} pp** of the late-fusion F1. Observed difference:
{abs(d_eth)*100:.2f} pp -> classified as **{"close" if close else "materially lower"}**.

## Leakage and alignment controls

""" + "\n".join(f"- {c}" for c in checks) + f"""

Seeds: `numpy = torch = random_state = 42`. No synthetic samples. The test set is scored once.

## Files

`outputs/`: `embedding_only_ablation_metrics.json`, `four_model_comparison_ablation.csv`,
`embedding_only_confusion_matrix_ablation.csv`, `embedding_only_predictions_ablation.csv`,
`embedding_only_confusion_matrix_ablation.png`, `four_model_comparison_ablation.png`,
`roc_curve_ablation.png`, `pr_curve_ablation.png`, `embedding_only_ablation_report.md`

`models/`: `graphsage_embedding_only_xgboost_ablation.pkl`

V1 and V2 artifacts are untouched.
"""
    with open(OUTPUTS / "embedding_only_ablation_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nHead swap   : {d_head*100:+.2f} pp")
    print(f"Etherscan   : {d_eth*100:+.2f} pp")
    print(f"Total vs GNN: {total*100:+.2f} pp")
    print("Report ->", OUTPUTS / "embedding_only_ablation_report.md")


if __name__ == "__main__":
    main()
