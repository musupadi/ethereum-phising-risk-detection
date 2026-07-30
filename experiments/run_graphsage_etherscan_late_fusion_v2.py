"""
experiments/run_graphsage_etherscan_late_fusion_v2.py
------------------------------------------------------
V2 experiment: "GraphSAGE-Etherscan Late Fusion (Leakage-Controlled, Non-Redundant)".

Corrections over V1 (experiments/run_graphsage_etherscan_late_fusion.py, untouched):

  1. NON-REDUNDANT. The GraphSAGE encoder is fed ONLY the 19 graph-derived
     (Scenario-1) features. The 7 Etherscan features enter the model exactly once -
     in the tabular branch that is concatenated after the embedding. In V1 the same
     7 Etherscan features were both GraphSAGE inputs and concatenated again.

  2. LEAKAGE-CONTROLLED. No test signal (F1, loss, ROC-AUC, PR-AUC) is ever used for
     early stopping or model selection - not for the fusion encoder and not for the
     GraphSAGE baseline. Early stopping is driven by an internal validation split
     carved out of the TRAINING wallets only. The test set is scored exactly once.

  3. EPTransNet EDGES ONLY. Graph topology uses only direct transaction edges read
     from MulDiGraph.pkl. data/raw/etherscan_edges.csv is deliberately NOT read.

Design
    Same 8,595-wallet population, labels and stratified 80:20 split (random_state=42).
    Wallets and labels come only from EPTransNet/MulDiGraph.
    fused_i = [ graphsage_embedding_128 || scaled_etherscan_7 ]  ->  135-d
    XGBoost is trained on fused TRAIN vectors only.

Outputs carry a _v2 suffix; V1 artifacts are never overwritten.
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

# Published reference (thesis, Scenario 2 tabular XGBoost) - asserted, never overwritten.
PUBLISHED_XGB_S2 = {
    "Accuracy": 0.9750,
    "Precision": 0.8862,
    "Recall": 0.9356,
    "F1": 0.9102,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("late_fusion_v2")


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
                g.number_of_nodes(), g.number_of_edges(),
            )
            return g
    raise FileNotFoundError(
        "MulDiGraph.pkl not found. V2 requires the real EPTransNet topology; "
        "the kNN similarity fallback is not an acceptable substitute here."
    )


def extract_muldigraph_edges_only(
    nx_graph, train_addresses: list[str], test_addresses: list[str]
) -> tuple[list[list[int]], dict]:
    """
    Extract direct transaction edges between dataset wallets from MulDiGraph ONLY.

    Deliberately does NOT read data/raw/etherscan_edges.csv - V2 topology must be
    EPTransNet-only. Returns (edges, audit) where audit records raw vs deduplicated
    counts so the edge provenance is verifiable.

    Node index convention (must match build_graph_with_real_edges):
        0 .. n_train-1        -> training wallets, in X_train row order
        n_train .. n_total-1  -> test wallets,     in X_test row order
    """
    all_addresses = list(train_addresses) + list(test_addresses)
    addr_to_idx = {a: i for i, a in enumerate(all_addresses)}
    addr_set = set(addr_to_idx)

    edges: list[list[int]] = []
    found = 0
    for addr, src_idx in addr_to_idx.items():
        if addr not in nx_graph:
            continue
        found += 1
        for neighbor in nx_graph.successors(addr):
            nbr = str(neighbor).lower()
            if nbr in addr_set:
                edges.append([src_idx, addr_to_idx[nbr]])
        for neighbor in nx_graph.predecessors(addr):
            nbr = str(neighbor).lower()
            if nbr in addr_set:
                edges.append([addr_to_idx[nbr], src_idx])

    unique = np.unique(np.array(edges, dtype=np.int64), axis=0) if edges else np.empty((0, 2))
    n_unique = int(len(unique))

    # Cross-partition breakdown (descriptive only - topology is transductive by design).
    n_train = len(train_addresses)
    tt = tr_te = te_te = 0
    for s, d in unique:
        a, b = int(s) < n_train, int(d) < n_train
        if a and b:
            tt += 1
        elif a != b:
            tr_te += 1
        else:
            te_te += 1

    audit = {
        "edge_sources_used": ["MulDiGraph.pkl (EPTransNet)"],
        "edge_sources_excluded": [
            "data/raw/etherscan_edges.csv (Etherscan-derived edges) - excluded by V2 design"
        ],
        "muldigraph_addresses_found": found,
        "dataset_addresses_total": len(all_addresses),
        "raw_directed_edge_entries": len(edges),
        "unique_directed_edges": n_unique,
        "edges_train_train": tt,
        "edges_train_test": tr_te,
        "edges_test_test": te_te,
        "self_loops_added_by_pyg_builder": len(all_addresses),
        "note": (
            "raw_directed_edge_entries counts every (successor/predecessor) hit and so "
            "double-counts each edge once from each endpoint; unique_directed_edges is the "
            "deduplicated set actually passed to the GNN. V1 reported 528 raw MulDiGraph "
            "entries + 126 Etherscan entries; V2 uses the MulDiGraph entries only."
        ),
    }
    log.info(
        "Edge audit (MulDiGraph only): %d raw entries -> %d unique directed edges "
        "(train-train=%d, train-test=%d, test-test=%d)",
        len(edges), n_unique, tt, tr_te, te_te,
    )
    return edges, audit


def metrics_row(name: str, inputs: str, y_true, y_pred, y_proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    return {
        "Model": name,
        "Input": inputs,
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_proba)),
        "PR-AUC": float(average_precision_score(y_true, y_proba)),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def extract_hidden_embeddings(model, x, edge_index) -> np.ndarray:
    """
    Forward through the SAGEConv stack, stopping BEFORE model.head.

    Mirrors SAGEModelV2.forward() (conv -> BatchNorm -> ReLU) in eval mode (dropout off),
    minus the final Linear(hidden_dim, 2). Returns the 128-d hidden representation.
    """
    model.eval()
    with torch.no_grad():
        h = x
        for conv, norm in zip(model.convs, model.norms):
            h = conv(h, edge_index)
            h = norm(h)
            h = F.relu(h)
    return h.cpu().numpy()


def train_xgb_on(X_train, y_train, X_test, feature_names):
    """Baseline XGBoost recipe; inner validation split comes from TRAIN only."""
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=SEED, stratify=y_train
    )
    model = XGBoostModel(
        n_estimators=250, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85, random_state=SEED,
    )
    model.fit(X_fit, y_fit, X_val, y_val, feature_names=feature_names, early_stopping_rounds=40)
    return model.predict(X_test), model.predict_proba(X_test)[:, 1], model


def plot_cm(cm: np.ndarray, title: str, path: Path) -> None:
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
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)


# ── main ───────────────────────────────────────────────────────────────────────


def main() -> None:
    set_seeds()
    OUTPUTS.mkdir(parents=True, exist_ok=True)
    MODELS.mkdir(parents=True, exist_ok=True)
    checks: list[str] = []

    # ── 1. Population, labels, split ──────────────────────────────────────────
    df = pd.read_csv(HYBRID_CSV)
    with open(HYBRID_SCHEMA_JSON, "r", encoding="utf-8") as f:
        schema = json.load(f)

    graph_cols = [c for c in schema["scenarios"]["muldigraph_only"] if c in df.columns]
    etherscan_cols = [c for c in schema["feature_groups"]["etherscan"] if c in df.columns]
    scenario2_cols = [c for c in schema["scenarios"]["muldigraph_plus_etherscan"] if c in df.columns]

    assert not set(graph_cols) & set(etherscan_cols), "graph/etherscan feature overlap"
    checks.append(
        f"non-redundancy: GraphSAGE input ({len(graph_cols)} graph features) and the fused "
        f"Etherscan block ({len(etherscan_cols)} features) are disjoint - no feature enters twice"
    )
    log.info(
        "Features: graph=%d (GraphSAGE input) | etherscan=%d (tabular branch) | scenario2 ref=%d",
        len(graph_cols), len(etherscan_cols), len(scenario2_cols),
    )

    df["wallet_address"] = df["node_id"].astype(str).str.lower()
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=SEED, stratify=df["label"]
    )
    train_addrs = train_df["wallet_address"].tolist()
    test_addrs = test_df["wallet_address"].tolist()
    y_train = train_df["label"].values.astype(int)
    y_test = test_df["label"].values.astype(int)
    n_train, n_test = len(train_df), len(test_df)
    n_total = n_train + n_test

    log.info(
        "Split: train=%d (phish=%d) test=%d (phish=%d)",
        n_train, int(y_train.sum()), n_test, int(y_test.sum()),
    )
    assert (n_train, n_test) == (6876, 1719), "split differs from the published Scenario 2 split"
    assert df["wallet_address"].is_unique
    assert set(train_addrs).isdisjoint(set(test_addrs))
    assert n_total == len(df) == 8595
    checks.append("wallet population = 8,595 unique lowercased EPTransNet addresses; no wallet added or dropped")
    checks.append(f"stratified 80:20 split, random_state=42 -> train={n_train}, test={n_test} (identical to Scenario 2)")

    # ── 2. Topology: MulDiGraph edges only ────────────────────────────────────
    tx_graph = load_transaction_graph()
    edge_list, edge_audit = extract_muldigraph_edges_only(tx_graph, train_addrs, test_addrs)
    assert edge_list, "no EPTransNet edges found between dataset wallets"
    checks.append(
        f"topology built from MulDiGraph only ({edge_audit['unique_directed_edges']} unique directed edges); "
        f"etherscan_edges.csv never read"
    )

    # ── 3. GraphSAGE encoder on the 19 graph features ─────────────────────────
    X_train_g = train_df[graph_cols].fillna(0).values.astype(np.float64)
    X_test_g = test_df[graph_cols].fillna(0).values.astype(np.float64)

    gscaler = StandardScaler()
    X_train_gs = gscaler.fit_transform(X_train_g)
    X_test_gs = gscaler.transform(X_test_g)
    checks.append("graph-feature StandardScaler fitted on TRAIN rows only")

    gnn = GraphPhiV2(model_type="sage", input_dim=len(graph_cols), hidden_dim=128, num_layers=2)
    graph, train_mask, test_mask = gnn.build_graph_with_real_edges(
        X_train_gs, y_train, X_test_gs, y_test, edge_list
    )

    # Internal validation slice, drawn from TRAINING wallets only.
    fit_idx, val_idx = train_test_split(
        np.arange(n_train), test_size=0.2, random_state=SEED, stratify=y_train
    )
    fit_mask = np.zeros(n_total, dtype=bool); fit_mask[fit_idx] = True
    val_mask = np.zeros(n_total, dtype=bool); val_mask[val_idx] = True
    assert not (fit_mask & val_mask).any()
    assert not (fit_mask & test_mask).any() and not (val_mask & test_mask).any()
    checks.append(
        f"early stopping driven by a train-internal validation slice (fit={int(fit_mask.sum())}, "
        f"val={int(val_mask.sum())}); test F1/loss/ROC-AUC/PR-AUC never monitored"
    )

    log.info("Training leakage-controlled GraphSAGE encoder (graph-only features) ...")
    # GraphPhiV2.fit names its monitoring mask `test_mask`; we pass the TRAIN-internal
    # validation mask, so the "test_f1" it logs is validation F1.
    gnn.fit(graph, train_mask=fit_mask, test_mask=val_mask, epochs=200, early_stopping_patience=25)
    torch.save(gnn.model.state_dict(), MODELS / "graphsage_encoder_for_fusion_v2.pt")

    # ── 4. Single scoring pass on the held-out test set ───────────────────────
    # The same encoder serves both roles: its own classification head is the
    # leakage-controlled GraphSAGE baseline, and its pre-head hidden layer is the
    # embedding fed to XGBoost. Only the classifier differs between the two rows.
    gnn_pred, gnn_proba_mat = gnn.predict(graph, test_mask)
    gnn_proba = gnn_proba_mat[:, 1]

    emb_all = extract_hidden_embeddings(gnn.model, graph.x, graph.edge_index)
    assert emb_all.shape == (n_total, 128), f"unexpected embedding shape {emb_all.shape}"
    emb_train, emb_test = emb_all[:n_train], emb_all[n_train:]
    checks.append("embedding = output of the final SAGEConv->BatchNorm->ReLU (pre-head, 128-d); logits/probabilities not used")

    assert (train_addrs + test_addrs)[:n_train] == train_addrs
    checks.append("graph node order == [train wallets..., test wallets...]; embedding rows aligned by position")

    # ── 5. Etherscan branch ───────────────────────────────────────────────────
    eth_train_raw = train_df[etherscan_cols].fillna(0).values.astype(np.float64)
    eth_test_raw = test_df[etherscan_cols].fillna(0).values.astype(np.float64)
    eth_scaler = StandardScaler()
    eth_train = eth_scaler.fit_transform(eth_train_raw)
    eth_test = eth_scaler.transform(eth_test_raw)
    with open(MODELS / "late_fusion_etherscan_scaler_v2.pkl", "wb") as f:
        pickle.dump(eth_scaler, f)
    checks.append("Etherscan values zero-filled (same as Scenario 2); StandardScaler fitted on TRAIN only")

    assert len(emb_train) == len(eth_train) == len(y_train) == len(train_addrs)
    assert len(emb_test) == len(eth_test) == len(y_test) == len(test_addrs)
    assert list(train_df["wallet_address"]) == train_addrs
    assert list(test_df["wallet_address"]) == test_addrs
    checks.append("embedding rows, Etherscan rows, labels and split indices refer to the same wallet in the same order")

    # ── 6. Fuse and classify ──────────────────────────────────────────────────
    fused_train = np.hstack([emb_train, eth_train])
    fused_test = np.hstack([emb_test, eth_test])
    emb_dim = emb_train.shape[1]
    fused_dim = fused_train.shape[1]
    assert fused_dim == 135, f"expected 135-d fused vector, got {fused_dim}"
    fused_names = [f"sage_emb_{i}" for i in range(emb_dim)] + [f"eth_{c}" for c in etherscan_cols]
    log.info("Fused vector: %d + %d = %d", emb_dim, len(etherscan_cols), fused_dim)

    log.info("Training late-fusion XGBoost V2 ...")
    fusion_pred, fusion_proba, fusion_model = train_xgb_on(fused_train, y_train, fused_test, fused_names)
    with open(MODELS / "graphsage_etherscan_late_fusion_xgboost_v2.pkl", "wb") as f:
        pickle.dump(fusion_model, f)

    # ── 7. Published Scenario 2 tabular reference ─────────────────────────────
    log.info("Reproducing the published XGBoost Scenario 2 tabular baseline ...")
    X_train_s2 = train_df[scenario2_cols].fillna(0).values.astype(np.float64)
    X_test_s2 = test_df[scenario2_cols].fillna(0).values.astype(np.float64)
    xgb_pred, xgb_proba, _ = train_xgb_on(X_train_s2, y_train, X_test_s2, scenario2_cols)

    rows = [
        metrics_row(
            "XGBoost tabular (Scenario 2)",
            f"{len(scenario2_cols)} tabular features (19 graph + 7 Etherscan)",
            y_test, xgb_pred, xgb_proba,
        ),
        metrics_row(
            "GraphSAGE graph-only (leakage-controlled)",
            f"{len(graph_cols)} graph features + {edge_audit['unique_directed_edges']} EPTransNet edges",
            y_test, gnn_pred, gnn_proba,
        ),
        metrics_row(
            "GraphSAGE-Etherscan Late Fusion V2",
            f"{emb_dim}-d graph embedding || {len(etherscan_cols)} scaled Etherscan features = {fused_dim}-d",
            y_test, fusion_pred, fusion_proba,
        ),
    ]
    xgb_m, gnn_m, fus_m = rows

    # The published reference must reproduce exactly; otherwise the split drifted.
    for k, v in PUBLISHED_XGB_S2.items():
        assert abs(xgb_m[k] - v) < 5e-4, f"Scenario 2 baseline drifted on {k}: {xgb_m[k]:.4f} vs published {v:.4f}"
    checks.append("published XGBoost Scenario 2 baseline reproduced exactly (Acc 97.50 / Prec 88.62 / Rec 93.56 / F1 91.02)")

    comp_df = pd.DataFrame(rows)
    comp_df.to_csv(OUTPUTS / "late_fusion_comparison_table_v2.csv", index=False)
    print("\n=== Comparison V2 (held-out test set, n=%d) ===" % n_test)
    print(comp_df.drop(columns=["Input"]).to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # ── 8. Audits ─────────────────────────────────────────────────────────────
    with open(OUTPUTS / "edge_source_audit_v2.json", "w", encoding="utf-8") as f:
        json.dump(edge_audit, f, indent=2)

    feature_audit_rows = (
        [{"feature": c, "source": "MulDiGraph (EPTransNet)", "branch": "GraphSAGE encoder input",
          "used_in_v2": True} for c in graph_cols]
        + [{"feature": c, "source": "Etherscan (enrichment)", "branch": "tabular branch (concatenated after embedding)",
            "used_in_v2": True} for c in etherscan_cols]
    )
    feature_audit = pd.DataFrame(feature_audit_rows)
    feature_audit.to_csv(OUTPUTS / "feature_source_audit_v2.csv", index=False)

    # Where does the fused model actually get its signal from?
    gain = fusion_model.model.get_score(importance_type="gain")
    gain_total = sum(gain.values()) or 1.0
    eth_gain = sum(v for k, v in gain.items() if k.startswith("eth_"))
    gain_report = {
        "features_used_by_trees": len(gain),
        "etherscan_gain_share": eth_gain / gain_total,
        "embedding_gain_share": 1.0 - eth_gain / gain_total,
        "top_features": [
            {"feature": k, "gain": float(v), "share": float(v / gain_total)}
            for k, v in sorted(gain.items(), key=lambda x: -x[1])[:15]
        ],
    }
    log.info(
        "Fused-model gain: etherscan=%.1f%% embedding=%.1f%% (%d features used)",
        100 * gain_report["etherscan_gain_share"],
        100 * gain_report["embedding_gain_share"],
        len(gain),
    )

    # ── 9. Metrics + predictions ──────────────────────────────────────────────
    with open(OUTPUTS / "late_fusion_metrics_v2.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "experiment": "GraphSAGE-Etherscan Late Fusion V2 (Leakage-Controlled, Non-Redundant)",
                "fusion_type": "embedding-level late fusion (graph-tabular)",
                "corrections_over_v1": [
                    "GraphSAGE input restricted to the 19 graph-derived features (no Etherscan double-use)",
                    "no test signal used for early stopping or model selection, for any model",
                    "topology restricted to EPTransNet/MulDiGraph edges (etherscan_edges.csv excluded)",
                ],
                "split": {
                    "test_size": 0.2, "random_state": SEED, "stratified": True,
                    "n_train": n_train, "n_test": n_test,
                    "n_train_phishing": int(y_train.sum()), "n_test_phishing": int(y_test.sum()),
                },
                "dimensions": {
                    "graphsage_input_features": len(graph_cols),
                    "graphsage_embedding_dimension": emb_dim,
                    "etherscan_feature_count": len(etherscan_cols),
                    "fused_vector_dimension": fused_dim,
                    "scenario2_tabular_feature_count": len(scenario2_cols),
                    "etherscan_features": etherscan_cols,
                    "graph_features": graph_cols,
                },
                "late_fusion_v2": fus_m,
                "graphsage_graph_only_leakage_controlled": gnn_m,
                "xgboost_scenario2_reference": xgb_m,
                "published_xgboost_scenario2_reference": PUBLISHED_XGB_S2,
                "edge_audit": edge_audit,
                "fused_model_gain_report": gain_report,
                "alignment_and_leakage_checks": checks,
                "seeds": {"numpy": SEED, "torch": SEED, "sklearn_random_state": SEED},
                "device": str(DEVICE),
            },
            f,
            indent=2,
        )

    cm_fus = confusion_matrix(y_test, fusion_pred, labels=[0, 1])
    cm_gnn = confusion_matrix(y_test, gnn_pred, labels=[0, 1])
    cm_xgb = confusion_matrix(y_test, xgb_pred, labels=[0, 1])
    idx = ["actual_normal", "actual_phishing"]
    cols = ["pred_normal", "pred_phishing"]
    pd.DataFrame(cm_fus, index=idx, columns=cols).to_csv(OUTPUTS / "late_fusion_confusion_matrix_v2.csv")
    pd.concat(
        {
            "XGBoost_Scenario2": pd.DataFrame(cm_xgb, index=idx, columns=cols),
            "GraphSAGE_graph_only": pd.DataFrame(cm_gnn, index=idx, columns=cols),
            "LateFusion_V2": pd.DataFrame(cm_fus, index=idx, columns=cols),
        }
    ).to_csv(OUTPUTS / "confusion_matrices_all_v2.csv")

    pd.DataFrame(
        {
            "wallet_address": test_addrs,
            "y_true": y_test,
            "y_pred": fusion_pred,
            "y_proba_phishing": fusion_proba,
            "graphsage_pred": gnn_pred,
            "graphsage_proba_phishing": gnn_proba,
        }
    ).to_csv(OUTPUTS / "late_fusion_predictions_v2.csv", index=False)

    # ── 10. Figures ───────────────────────────────────────────────────────────
    plot_cm(cm_fus, "Late Fusion V2 - Confusion Matrix", OUTPUTS / "late_fusion_confusion_matrix_v2.png")
    plot_cm(cm_gnn, "GraphSAGE graph-only (leakage-controlled)\nConfusion Matrix",
            OUTPUTS / "cm_graphsage_graph_only_v2.png")
    plot_cm(cm_xgb, "XGBoost tabular Scenario 2 - Confusion Matrix",
            OUTPUTS / "cm_xgboost_scenario2_v2.png")

    curves = [
        ("XGBoost tabular (S2)", xgb_proba, xgb_m),
        ("GraphSAGE graph-only", gnn_proba, gnn_m),
        ("Late Fusion V2", fusion_proba, fus_m),
    ]

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for label, proba, m in curves:
        fpr, tpr, _ = roc_curve(y_test, proba)
        ax.plot(fpr, tpr, label=f"{label} (AUC={m['ROC-AUC']:.4f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve V2 - Held-out Test Set")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "late_fusion_roc_curve_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    for label, proba, m in curves:
        prec, rec, _ = precision_recall_curve(y_test, proba)
        ax.plot(rec, prec, label=f"{label} (AP={m['PR-AUC']:.4f})")
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curve V2 - Held-out Test Set")
    ax.legend(loc="lower left", fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "late_fusion_pr_curve_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.0, 4.8))
    metric_names = ["Precision", "Recall", "F1"]
    x = np.arange(len(metric_names))
    width = 0.26
    short = ["XGBoost (S2)", "GraphSAGE graph-only", "Late Fusion V2"]
    for i, (m, name) in enumerate(zip(rows, short)):
        bars = ax.bar(x + (i - 1) * width, [m[k] for k in metric_names], width, label=name)
        ax.bar_label(bars, fmt="%.3f", fontsize=7.5)
    ax.set_xticks(x, metric_names)
    ax.set_ylim(0.5, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison V2 (leakage-controlled, non-redundant)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(OUTPUTS / "late_fusion_model_comparison_v2.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # ── 11. Report ────────────────────────────────────────────────────────────
    d_xgb = fus_m["F1"] - xgb_m["F1"]
    d_gnn = fus_m["F1"] - gnn_m["F1"]

    def verdict(delta: float, other: str, other_f1: float) -> str:
        if delta > 0:
            return (f"**improves** F1 over {other} ({fus_m['F1']*100:.2f}% vs {other_f1*100:.2f}%, "
                    f"+{delta*100:.2f} pp on the held-out test set)")
        if delta == 0:
            return f"**matches** {other} (F1 = {fus_m['F1']*100:.2f}% for both)"
        return (f"**does not improve** on {other} ({fus_m['F1']*100:.2f}% vs {other_f1*100:.2f}%, "
                f"{delta*100:.2f} pp on the held-out test set); no improvement is claimed")

    table_md = "| Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |\n"
    table_md += "|---|---|---|---|---|---|---|---|---|---|---|---|\n"
    for m in rows:
        table_md += (
            f"| {m['Model']} | {m['Input']} | {m['Accuracy']*100:.2f}% | {m['Precision']*100:.2f}% | "
            f"{m['Recall']*100:.2f}% | {m['F1']*100:.2f}% | {m['ROC-AUC']:.4f} | {m['PR-AUC']:.4f} | "
            f"{m['TN']} | {m['FP']} | {m['FN']} | {m['TP']} |\n"
        )

    report = f"""# GraphSAGE-Etherscan Late Fusion V2 (Leakage-Controlled, Non-Redundant)

## What changed relative to V1

| # | V1 problem | V2 correction |
|---|---|---|
| 1 | GraphSAGE received all 26 Scenario-2 features (including the 7 Etherscan ones), which were then concatenated again before XGBoost - redundant fusion | GraphSAGE input is restricted to the **{len(graph_cols)} graph-derived (Scenario-1) features**. The 7 Etherscan features enter the model exactly once, in the tabular branch. |
| 2 | The GraphSAGE baseline monitored test F1 for early stopping | **No test signal** (F1, loss, ROC-AUC, PR-AUC) is used for early stopping or model selection by any model. Early stopping uses a validation slice drawn from training wallets only. |
| 3 | Topology mixed MulDiGraph and Etherscan-derived edges | Topology uses **only direct EPTransNet/MulDiGraph transaction edges**; `data/raw/etherscan_edges.csv` is never read. |

## Design

- Population: **{len(df)} wallets** ({int(df['label'].sum())} phishing / {int((df['label'] == 0).sum())} normal). Wallets and labels come **only from EPTransNet/MulDiGraph**.
- Split: stratified 80:20, `random_state=42` -> {n_train} train / {n_test} test - identical to Scenario 2.
- Fusion: `fused_i = [ graphsage_embedding_{emb_dim} || scaled_etherscan_{len(etherscan_cols)} ]` = **{fused_dim}-d**.
- The embedding is the output of the final `SAGEConv -> BatchNorm -> ReLU`, taken before `Linear({emb_dim}, 2)`. Logits and probabilities are not used.
- XGBoost is trained on fused **training** vectors only; the test set is scored once.

The GraphSAGE baseline row and the fusion row share **the same trained encoder**: the baseline is that encoder's own classification head, the fusion model is XGBoost over its pre-head embedding. The only variable that changes between the two rows is the classifier (plus the Etherscan block) - which is what makes the comparison a clean ablation of the fusion step itself.

## Edge-source audit

| Item | Value |
|---|---|
| Sources used | MulDiGraph.pkl (EPTransNet) |
| Sources excluded | `data/raw/etherscan_edges.csv` (excluded by V2 design) |
| Dataset addresses present in MulDiGraph | {edge_audit['muldigraph_addresses_found']} / {edge_audit['dataset_addresses_total']} |
| Raw directed edge entries | {edge_audit['raw_directed_edge_entries']} |
| Unique directed edges passed to the GNN | **{edge_audit['unique_directed_edges']}** |
| ├─ train-train | {edge_audit['edges_train_train']} |
| ├─ train-test | {edge_audit['edges_train_test']} |
| └─ test-test | {edge_audit['edges_test_test']} |
| Self-loops added by the PyG builder | {edge_audit['self_loops_added_by_pyg_builder']} |

The raw entry count double-counts each edge (once from each endpoint's adjacency scan); the deduplicated figure is what the model actually sees. V1's "528 MulDiGraph + 126 Etherscan" figures were raw entry counts of the merged sources.

## Feature-source audit

| Branch | Source | Count | Features |
|---|---|---|---|
| GraphSAGE encoder input | MulDiGraph (EPTransNet) | {len(graph_cols)} | {', '.join(f'`{c}`' for c in graph_cols)} |
| Tabular branch (post-embedding) | Etherscan (enrichment) | {len(etherscan_cols)} | {', '.join(f'`{c}`' for c in etherscan_cols)} |

The two sets are disjoint - asserted at runtime - so no feature is consumed twice. Full listing: `outputs/feature_source_audit_v2.csv`.

## Results (held-out test set, n = {n_test})

{table_md}
### Does fusion improve on XGBoost?

The late-fusion V2 model {verdict(d_xgb, 'the XGBoost Scenario 2 tabular baseline', xgb_m['F1'])}.

### Does fusion improve on GraphSAGE?

The late-fusion V2 model {verdict(d_gnn, 'the leakage-controlled graph-only GraphSAGE baseline', gnn_m['F1'])}.

### Where does the fused model get its signal?

Of the {gain_report['features_used_by_trees']} fused dimensions the trees actually split on, the Etherscan block accounts for **{gain_report['etherscan_gain_share']*100:.1f}%** of total gain and the graph embedding for **{gain_report['embedding_gain_share']*100:.1f}%**.

| Feature | Gain share |
|---|---|
""" + "".join(
        f"| `{t['feature']}` | {t['share']*100:.1f}% |\n" for t in gain_report["top_features"][:10]
    ) + f"""
Because the encoder no longer sees the Etherscan features, this split is a genuine measure of what each modality contributes - unlike V1, where the Etherscan block was largely redundant with information already baked into the embedding.

## Leakage and alignment controls

""" + "\n".join(f"- {c}" for c in checks) + f"""

The encoder is trained transductively over the full topology (unchanged from the existing baseline), but **no test label ever enters training**: only training-node labels contribute to the loss, and model selection is driven by the train-internal validation slice. `GraphPhiV2.fit` names its monitoring mask `test_mask`; V2 passes the validation mask there, so the `test_f1` value in the training log is validation F1.

No synthetic samples were generated. Class imbalance is handled by weighted cross-entropy (GNN) and `scale_pos_weight` (XGBoost). Seeds: `numpy = torch = random_state = 42`.

## Files

`outputs/`: `late_fusion_metrics_v2.json`, `late_fusion_comparison_table_v2.csv`, `late_fusion_confusion_matrix_v2.csv`, `confusion_matrices_all_v2.csv`, `late_fusion_predictions_v2.csv`, `edge_source_audit_v2.json`, `feature_source_audit_v2.csv`, `late_fusion_confusion_matrix_v2.png`, `cm_graphsage_graph_only_v2.png`, `cm_xgboost_scenario2_v2.png`, `late_fusion_roc_curve_v2.png`, `late_fusion_pr_curve_v2.png`, `late_fusion_model_comparison_v2.png`, `late_fusion_report_v2.md`

`models/`: `graphsage_encoder_for_fusion_v2.pt`, `graphsage_etherscan_late_fusion_xgboost_v2.pkl`, `late_fusion_etherscan_scaler_v2.pkl`

V1 artifacts (no `_v2` suffix) are untouched.
"""
    with open(OUTPUTS / "late_fusion_report_v2.md", "w", encoding="utf-8") as f:
        f.write(report)

    print(f"\nEmbedding {emb_dim} + Etherscan {len(etherscan_cols)} = fused {fused_dim}")
    print(f"F1 vs XGBoost S2 : {d_xgb*100:+.2f} pp")
    print(f"F1 vs GraphSAGE  : {d_gnn*100:+.2f} pp")
    print("Report ->", OUTPUTS / "late_fusion_report_v2.md")


if __name__ == "__main__":
    main()
