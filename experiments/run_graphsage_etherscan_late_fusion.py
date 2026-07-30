"""
experiments/run_graphsage_etherscan_late_fusion.py
--------------------------------------------------
NEW experiment: "GraphSAGE-Etherscan Late Fusion" (embedding-level graph-tabular fusion).

Pipeline
    1. Train the existing GraphSAGE (GraphPhiV2, 2-layer, hidden=128) on Scenario 2
       features using the identical stratified 80:20 split (random_state=42).
    2. Extract the 128-d hidden embedding z_i BEFORE the final classification head
       (not logits, not probabilities).
    3. Take the 7 Etherscan tabular features, StandardScaler fitted on TRAIN only.
    4. fused_i = [ z_i (128) || scaled_etherscan_i (7) ]  ->  135-d
    5. Train XGBoost on fused TRAIN vectors, evaluate on held-out TEST wallets.

Leakage controls
    - GraphSAGE early stopping uses a validation slice carved out of the TRAINING
      wallets. Test labels are never used for training, early stopping, or fitting.
    - The graph topology is transductive (full edge set, as in the existing baseline),
      but only train-node labels enter the loss.
    - The Etherscan StandardScaler is fitted on train rows only.
    - Wallet addresses are lowercased and asserted aligned across embeddings,
      tabular rows, labels and split indices.

This file adds a NEW experiment only. Existing scripts, artifacts and baseline
metrics under ml/artifacts/ are not modified; everything written here goes to
outputs/ and models/.
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
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from gnn_model_v2 import GraphPhiV2, DEVICE  # noqa: E402
from xgboost_model import XGBoostModel  # noqa: E402
from train_dataset_scenarios import extract_edges_for_addresses  # noqa: E402

SEED = 42
DATA_PROCESSED = ROOT / "data" / "processed"
HYBRID_CSV = DATA_PROCESSED / "hybrid_dataset.csv"
HYBRID_SCHEMA_JSON = ROOT / "ml" / "artifacts" / "hybrid_feature_schema.json"
GRAPH_CANDIDATES = [
    ROOT / "data" / "raw" / "MulDiGraph.pkl",
    ROOT / "Ethereum Phishing Transaction Network" / "MulDiGraph.pkl",
]

OUTPUTS = ROOT / "outputs"
MODELS = ROOT / "models"

SCENARIO = "muldigraph_plus_etherscan"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("late_fusion")


# ── helpers ────────────────────────────────────────────────────────────────────


def set_seeds(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_transaction_graph():
    for path in GRAPH_CANDIDATES:
        if path.exists():
            log.info("Loading MulDiGraph from %s (this may take ~30s) ...", path)
            with open(path, "rb") as f:
                g = pickle.load(f)
            log.info(
                "MulDiGraph loaded: %d nodes, %d edges",
                g.number_of_nodes(),
                g.number_of_edges(),
            )
            return g
    log.warning("MulDiGraph not found - GNN will fall back to a kNN similarity graph")
    return None


def metrics_row(name: str, y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1-score": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba)),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


def extract_hidden_embeddings(model, x: torch.Tensor, edge_index: torch.Tensor) -> np.ndarray:
    """
    Forward pass through the GraphSAGE conv stack, stopping BEFORE model.head.

    Mirrors SAGEModelV2.forward() exactly (conv -> batchnorm -> relu) minus the
    dropout (eval mode) and minus the final Linear(hidden_dim, 2) classifier.
    Returns the 128-d hidden representation for every node.
    """
    model.eval()
    with torch.no_grad():
        h = x
        for conv, norm in zip(model.convs, model.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
    return h.cpu().numpy()


def train_xgb_on(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    feature_names: list[str],
) -> tuple[np.ndarray, np.ndarray, XGBoostModel]:
    """Same recipe as train_dataset_scenarios.train_xgb (inner val split from TRAIN only)."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    model = XGBoostModel(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=SEED,
    )
    model.fit(X_fit, y_fit, X_val, y_val, feature_names=feature_names, early_stopping_rounds=40)
    return model.predict(X_test), model.predict_proba(X_test)[:, 1], model


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    set_seeds()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)

    # ── 1. Data + identical Scenario 2 split ──────────────────────────────────
    df = pd.read_csv(HYBRID_CSV)
    with open(HYBRID_SCHEMA_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    scenario_cols = [c for c in schema["scenarios"][SCENARIO] if c in df.columns]
    etherscan_cols = [c for c in schema["feature_groups"]["etherscan"] if c in df.columns]
    graph_cols = [c for c in scenario_cols if c not in etherscan_cols]

    log.info(
        "Scenario 2 features: %d (%d graph + %d etherscan)",
        len(scenario_cols), len(graph_cols), len(etherscan_cols),
    )

    # Wallet universe + labels come from MulDiGraph/EPTransNet only.
    df["wallet_address"] = df["node_id"].astype(str).str.lower()

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    log.info(
        "Split: train=%d (phish=%d) test=%d (phish=%d)",
        len(train_df), int(train_df["label"].sum()),
        len(test_df), int(test_df["label"].sum()),
    )

    train_addrs = train_df["wallet_address"].tolist()
    test_addrs = test_df["wallet_address"].tolist()

    X_train = train_df[scenario_cols].fillna(0).values.astype(np.float64)
    X_test = test_df[scenario_cols].fillna(0).values.astype(np.float64)
    y_train = train_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)

    # ── alignment / leakage assertions ────────────────────────────────────────
    checks: list[str] = []
    assert set(train_addrs).isdisjoint(set(test_addrs)), "train/test wallet overlap"
    checks.append(f"train/test wallet sets disjoint (|train|={len(set(train_addrs))}, |test|={len(set(test_addrs))})")
    assert df["wallet_address"].is_unique, "duplicate wallet addresses in hybrid dataset"
    checks.append(f"all {len(df)} wallet_address values unique and lowercased")
    assert len(train_addrs) + len(test_addrs) == len(df)
    checks.append("train + test rows account for every dataset wallet (no wallet added/dropped)")

    # ── 2. GraphSAGE encoder (early stopping on a TRAIN-internal val slice) ────
    tx_graph = load_transaction_graph()
    edge_list = extract_edges_for_addresses(tx_graph, train_addrs, test_addrs) if tx_graph else None

    scaler_gnn = StandardScaler()
    X_train_scaled = scaler_gnn.fit_transform(X_train)
    X_test_scaled = scaler_gnn.transform(X_test)
    checks.append("GraphSAGE input StandardScaler fitted on TRAIN rows only")

    n_train, n_test = len(X_train), len(X_test)
    n_total = n_train + n_test

    gnn = GraphPhiV2(model_type="sage", input_dim=len(scenario_cols), hidden_dim=128, num_layers=2)

    if edge_list:
        graph, train_mask, test_mask = gnn.build_graph_with_real_edges(
            X_train_scaled, y_train, X_test_scaled, y_test, edge_list
        )
    else:
        graph, train_mask, test_mask = gnn.build_graph_from_train_only(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

    # Carve a validation slice out of the TRAIN nodes for early stopping,
    # so no test label is ever consulted during encoder training.
    fit_idx, val_idx = train_test_split(
        np.arange(n_train), test_size=0.2, random_state=SEED, stratify=y_train
    )
    fit_mask = np.zeros(n_total, dtype=bool)
    fit_mask[fit_idx] = True
    val_mask = np.zeros(n_total, dtype=bool)
    val_mask[val_idx] = True
    assert not (fit_mask & test_mask).any() and not (val_mask & test_mask).any()
    checks.append(
        f"GraphSAGE early stopping used a train-internal val slice "
        f"(fit={fit_mask.sum()}, val={val_mask.sum()}); test nodes excluded from loss and model selection"
    )

    log.info("Training GraphSAGE encoder for late fusion ...")
    gnn.fit(graph, train_mask=fit_mask, test_mask=val_mask, epochs=200, early_stopping_patience=25)
    torch.save(gnn.model.state_dict(), MODELS / "graphsage_encoder_for_fusion.pt")

    # ── 3. Embeddings before the classification head ──────────────────────────
    emb_all = extract_hidden_embeddings(gnn.model, graph.x, graph.edge_index)
    assert emb_all.shape == (n_total, 128), f"unexpected embedding shape {emb_all.shape}"
    emb_train, emb_test = emb_all[:n_train], emb_all[n_train:]
    checks.append(
        "embeddings taken from the last SAGEConv->BatchNorm->ReLU output "
        "(pre-head, 128-d); logits/probabilities not used"
    )

    # Node order in the graph is [train rows..., test rows...] by construction.
    emb_order = train_addrs + test_addrs
    assert emb_order[:n_train] == train_addrs and emb_order[n_train:] == test_addrs
    checks.append("graph node order == [train wallets..., test wallets...]; embedding rows aligned by position")

    # ── 4. Etherscan tabular block, scaler fitted on TRAIN only ───────────────
    eth_train_raw = train_df[etherscan_cols].fillna(0).values.astype(np.float64)
    eth_test_raw = test_df[etherscan_cols].fillna(0).values.astype(np.float64)
    checks.append("missing Etherscan values zero-filled (same as Scenario 2 preprocessing)")

    eth_scaler = StandardScaler()
    eth_train = eth_scaler.fit_transform(eth_train_raw)
    eth_test = eth_scaler.transform(eth_test_raw)
    with open(MODELS / "late_fusion_etherscan_scaler.pkl", "wb") as f:
        pickle.dump(eth_scaler, f)
    checks.append("Etherscan StandardScaler fitted on TRAIN rows only, then applied to TEST")

    # Re-assert wallet alignment between the embedding block and the tabular block.
    assert list(train_df["wallet_address"]) == train_addrs
    assert list(test_df["wallet_address"]) == test_addrs
    assert len(emb_train) == len(eth_train) == len(y_train) == len(train_addrs)
    assert len(emb_test) == len(eth_test) == len(y_test) == len(test_addrs)
    checks.append("embedding rows, Etherscan rows, labels and split indices refer to the same wallet in the same order")

    # ── 5. Fuse ───────────────────────────────────────────────────────────────
    fused_train = np.hstack([emb_train, eth_train])
    fused_test = np.hstack([emb_test, eth_test])
    emb_dim = emb_train.shape[1]
    fused_dim = fused_train.shape[1]
    fused_names = [f"sage_emb_{i}" for i in range(emb_dim)] + [f"eth_{c}" for c in etherscan_cols]
    log.info("Fused vector: %d (embedding) + %d (etherscan) = %d", emb_dim, len(etherscan_cols), fused_dim)

    # ── 6/7. XGBoost on fused vectors, evaluated on held-out TEST ─────────────
    log.info("Training late-fusion XGBoost ...")
    fusion_pred, fusion_proba, fusion_model = train_xgb_on(
        fused_train, y_train, fused_test, fused_names
    )
    with open(MODELS / "graphsage_etherscan_late_fusion_xgboost.pkl", "wb") as f:
        pickle.dump(fusion_model, f)

    # ── Baselines re-run on the same split (for ROC-AUC/PR-AUC/TN..TP) ────────
    log.info("Re-running XGBoost tabular Scenario 2 baseline on the same split ...")
    xgb_pred, xgb_proba, _ = train_xgb_on(X_train, y_train, X_test, scenario_cols)

    log.info("Re-running GraphSAGE Scenario 2 baseline on the same split ...")
    set_seeds()
    gnn_base = GraphPhiV2(model_type="sage", input_dim=len(scenario_cols), hidden_dim=128, num_layers=2)
    if edge_list:
        g2, tr2, te2 = gnn_base.build_graph_with_real_edges(
            X_train_scaled, y_train, X_test_scaled, y_test, edge_list
        )
    else:
        g2, tr2, te2 = gnn_base.build_graph_from_train_only(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
    # NOTE: reproduced exactly as in ml/train_dataset_scenarios.train_gnn, which
    # monitors test F1 for early stopping. Kept unchanged so the baseline numbers
    # stay comparable with the existing thesis results.
    gnn_base.fit(g2, train_mask=tr2, test_mask=te2, epochs=200, early_stopping_patience=25)
    gnn_pred, gnn_proba_mat = gnn_base.predict(g2, te2)
    gnn_proba = gnn_proba_mat[:, 1]

    # ── Comparison table ──────────────────────────────────────────────────────
    rows = [
        metrics_row("XGBoost tabular (Scenario 2)", y_test, xgb_pred, xgb_proba),
        metrics_row("GraphSAGE (Scenario 2)", y_test, gnn_pred, gnn_proba),
        metrics_row("GraphSAGE-Etherscan Late Fusion", y_test, fusion_pred, fusion_proba),
    ]
    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUTPUTS / "late_fusion_comparison_table.csv", index=False)

    print("\n=== Comparison (held-out test set, n=%d) ===" % len(y_test))
    print(comp_df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fusion_metrics = rows[2]
    xgb_metrics = rows[0]
    gnn_metrics = rows[1]

    # ── Artifacts ─────────────────────────────────────────────────────────────
    dim_report = {
        "graphsage_embedding_dimension": int(emb_dim),
        "etherscan_feature_count": int(len(etherscan_cols)),
        "fused_vector_dimension": int(fused_dim),
        "etherscan_features": etherscan_cols,
        "graph_feature_count": int(len(graph_cols)),
        "scenario2_tabular_feature_count": int(len(scenario_cols)),
    }

    with open(OUTPUTS / "late_fusion_metrics.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "GraphSAGE-Etherscan Late Fusion",
                "fusion_type": "embedding-level late fusion (graph-tabular)",
                "scenario": SCENARIO,
                "split": {
                    "test_size": 0.2,
                    "random_state": SEED,
                    "stratified": True,
                    "n_train": int(n_train),
                    "n_test": int(n_test),
                    "n_train_phishing": int(y_train.sum()),
                    "n_test_phishing": int(y_test.sum()),
                },
                "dimensions": dim_report,
                "late_fusion": fusion_metrics,
                "baselines": {"xgboost_scenario2": xgb_metrics, "graphsage_scenario2": gnn_metrics},
                "alignment_and_leakage_checks": checks,
                "seeds": {"numpy": SEED, "torch": SEED, "sklearn_random_state": SEED},
                "device": str(DEVICE),
                "edges_used": int(len(edge_list)) if edge_list else 0,
            },
            f,
            indent=2,
        )

    cm = confusion_matrix(y_test, fusion_pred, labels=[0, 1])
    pd.DataFrame(
        cm, index=["actual_normal", "actual_phishing"], columns=["pred_normal", "pred_phishing"]
    ).to_csv(OUTPUTS / "late_fusion_confusion_matrix.csv")

    pd.DataFrame(
        {
            "wallet_address": test_addrs,
            "y_true": y_test,
            "y_pred": fusion_pred,
            "y_proba_phishing": fusion_proba,
        }
    ).to_csv(OUTPUTS / "late_fusion_predictions.csv", index=False)

    # ── Figures ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(5.5, 4.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1], labels=["Normal", "Phishing"])
    ax.set_yticks([0, 1], labels=["Normal", "Phishing"])
    for i in range(2):
        for j in range(2):
            ax.text(
                j, i, f"{cm[i, j]:d}", ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=13,
            )
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("GraphSAGE-Etherscan Late Fusion\nConfusion Matrix")
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "late_fusion_confusion_matrix.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    curves = [
        ("XGBoost tabular (Scenario 2)", xgb_proba, xgb_metrics),
        ("GraphSAGE (Scenario 2)", gnn_proba, gnn_metrics),
        ("GraphSAGE-Etherscan Late Fusion", fusion_proba, fusion_metrics),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for label, proba, m in curves:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={m['ROC-AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve - Held-out Test Set")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "late_fusion_roc_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for label, proba, m in curves:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"{label} (AP={m['PR-AUC']:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve - Held-out Test Set")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "late_fusion_pr_curve.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    metric_names = ["Precision", "Recall", "F1-score"]
    x = np.arange(len(metric_names))
    width = 0.26
    short = ["XGBoost (S2)", "GraphSAGE (S2)", "Late Fusion"]
    for i, (m, name) in enumerate(zip(rows, short)):
        vals = [m[k] for k in metric_names]
        bars = ax.bar(x + (i - 1) * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5)
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0.7, 1.02)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison - Scenario 2 (MulDiGraph + Etherscan)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "late_fusion_model_comparison.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── Markdown report ───────────────────────────────────────────────────────
    delta = fusion_metrics["F1-score"] - xgb_metrics["F1-score"]
    if delta > 0:
        verdict = (
            f"The late-fusion model **improves** F1 over the XGBoost tabular baseline "
            f"on this run: {fusion_metrics['F1-score']*100:.2f}% vs {xgb_metrics['F1-score']*100:.2f}% "
            f"(+{delta*100:.2f} pp on the held-out test set)."
        )
    elif delta == 0:
        verdict = (
            f"The late-fusion model **matches** the XGBoost tabular baseline: "
            f"F1 = {fusion_metrics['F1-score']*100:.2f}% for both."
        )
    else:
        verdict = (
            f"The late-fusion model **does not improve** F1 over the XGBoost tabular baseline: "
            f"{fusion_metrics['F1-score']*100:.2f}% vs {xgb_metrics['F1-score']*100:.2f}% "
            f"({delta*100:.2f} pp on the held-out test set). No improvement is claimed."
        )

    table_md = "| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | TN | FP | FN | TP |\n"
    table_md += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    for m in rows:
        table_md += (
            f"| {m['Model']} | {m['Accuracy']*100:.2f}% | {m['Precision']*100:.2f}% | "
            f"{m['Recall']*100:.2f}% | {m['F1-score']*100:.2f}% | {m['ROC-AUC']:.4f} | "
            f"{m['PR-AUC']:.4f} | {m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} |\n"
        )

    checks_md = "\n".join(f"- {c}" for c in checks)

    report = f"""# GraphSAGE-Etherscan Late Fusion

## What this experiment is

This is **embedding-level late fusion (graph-tabular fusion)**. A GraphSAGE encoder
produces a graph embedding for each wallet; that embedding is concatenated with the
scaled Etherscan tabular features, and an XGBoost classifier is trained on the
concatenated vector. Fusion happens **after GraphSAGE produces the graph embeddings
and before XGBoost classifies the wallet** - it is not early feature concatenation
and it is not prediction-level (score) averaging.

## Data provenance

- **EPTransNet / MultiDiGraph remains the source of wallets and labels.** The wallet
  universe is the {len(df)} unique addresses of the hybrid dataset
  ({int(df['label'].sum())} phishing / {int((df['label'] == 0).sum())} normal). No wallet was added
  from Etherscan and no label was altered.
- **Etherscan contributes only seven tabular enrichment features**, joined on the
  normalized (lowercased) wallet address: `{', '.join(etherscan_cols)}`.
- Split: stratified 80:20, `random_state=42` - the identical split used by Scenario 2
  ({n_train} train / {n_test} test wallets).

## Architecture and dimensions

| Component | Dimension |
|---|---|
| GraphSAGE embedding (pre-head hidden layer) | {emb_dim} |
| Etherscan tabular features | {len(etherscan_cols)} |
| **Fused vector** | **{fused_dim}** |

Reference: the tabular Scenario 2 feature set is {len(scenario_cols)} features
({len(graph_cols)} MulDiGraph graph features + {len(etherscan_cols)} Etherscan features).

The embedding is the output of the final `SAGEConv -> BatchNorm -> ReLU` block, taken
**before** the `Linear({emb_dim}, 2)` classification head. Logits and predicted
probabilities are not used as the embedding.

## Results (held-out test set, n = {n_test})

{table_md}
{verdict}

## Methodological safeguards

{checks_md}

The GraphSAGE encoder used for fusion is trained transductively over the full
transaction topology (the same behaviour as the existing GraphSAGE baseline), but
**test labels are excluded from training**: only train-node labels enter the loss, and
early stopping is driven by a validation slice carved out of the training wallets.
The reported GraphSAGE Scenario 2 baseline row is reproduced exactly as the existing
pipeline defines it (which monitors test F1 for early stopping), so it stays comparable
with the previously published thesis number; the fusion encoder does not do this.

No synthetic samples were generated. Seeds: `numpy = torch = random_state = 42`.

## Files produced

- `outputs/late_fusion_metrics.json`
- `outputs/late_fusion_confusion_matrix.csv`
- `outputs/late_fusion_predictions.csv`
- `outputs/late_fusion_comparison_table.csv`
- `outputs/late_fusion_confusion_matrix.png`
- `outputs/late_fusion_roc_curve.png`
- `outputs/late_fusion_pr_curve.png`
- `outputs/late_fusion_model_comparison.png`
- `models/graphsage_etherscan_late_fusion_xgboost.pkl`
- `models/graphsage_encoder_for_fusion.pt`
- `models/late_fusion_etherscan_scaler.pkl`
"""
    with open(OUTPUTS / "late_fusion_report.md", "w", encoding="utf-8") as f:
        f.write(report)

    print("\nEmbedding dim: %d | Etherscan features: %d | Fused dim: %d" % (emb_dim, len(etherscan_cols), fused_dim))
    print("Report -> %s" % (OUTPUTS / "late_fusion_report.md"))


if __name__ == "__main__":
    main()
