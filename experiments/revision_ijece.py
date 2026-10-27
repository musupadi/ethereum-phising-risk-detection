"""
experiments/revision_ijece.py
-----------------------------
Experiments requested by the IJECE reviewers (Paper ID 42848), all on the full
12,815-wallet population (1,165 phishing : 11,650 normal, 1:10) with the
25-predictor Scenario-2 schema.

What it adds over ml/train_dataset_scenarios.py
  A-2.1  multi-seed stability: every model x scenario is re-run on 10 independent
         stratified 80:20 splits (seed = split seed = model seed).
  A-2.1  majority-class size: XGBoost S2 at phishing:normal = 1:2, 1:5, 1:10.
  A-2.2  structural leakage: GraphSAGE trained inductively (every edge touching a
         test wallet removed during training) vs the transductive default.
  A-2.4  GraphSAGE early stopping: decomposition of the old Table-3 protocol
         (early stopping on TEST F1 + Etherscan edges) vs the leakage-controlled one.
         All GraphSAGE numbers in the revision use a train-internal validation slice.
  Tab.4  late fusion [GraphSAGE embedding || Etherscan] and embedding-only ablation,
         re-run on the 12,815 population, 10 seeds.
  B-3.3  cost-sensitive threshold: threshold picked on the train-internal validation
         slice to minimise C_FN*FN + C_FP*FP, for C_FN/C_FP in {1, 2, 5, 10, 20}.
  B-2.5  probability calibration: Platt scaling and isotonic regression fitted on the
         validation slice; Brier, ECE and log-loss on test.
  B-2.2  isolated wallets: how many wallets have no 1-hop transactions / no edge in
         the induced GNN graph, per class.

The test set of every split is scored once; nothing is tuned on it.

Outputs: outputs/revision/
Run    : .venv/Scripts/python.exe experiments/revision_ijece.py
"""

from __future__ import annotations

import json
import logging
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
    f1_score, log_loss, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "ml"))

from gnn_model_v2 import GraphPhiV2, DEVICE  # noqa: E402
from xgboost_model import XGBoostModel  # noqa: E402

SEEDS = [42, 0, 1, 2, 3, 4, 5, 6, 7, 8]
COST_RATIOS = [1, 2, 5, 10, 20]
MAJORITY_RATIOS = [2, 5, 10]

HYBRID_CSV = ROOT / "data" / "processed" / "hybrid_dataset.csv"
SCHEMA_JSON = ROOT / "ml" / "artifacts" / "hybrid_feature_schema.json"
GRAPH_PKL = ROOT / "data" / "raw" / "MulDiGraph.pkl"
ETHERSCAN_EDGES_CSV = ROOT / "data" / "raw" / "etherscan_edges.csv"
OUT = ROOT / "outputs" / "revision"

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("revision")
for noisy in ("gnn_model_v2", "xgboost_model"):
    logging.getLogger(noisy).setLevel(logging.WARNING)


# ── helpers ────────────────────────────────────────────────────────────────────

def set_seeds(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def metrics(y, pred, proba) -> dict:
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1": f1_score(y, pred, zero_division=0),
        "roc_auc": roc_auc_score(y, proba),
        "pr_auc": average_precision_score(y, proba),
        "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
    }


def ece(y, p, bins: int = 10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(total)


def edges_between(nx_graph, addrs: list[str], use_etherscan: bool = False) -> np.ndarray:
    """Directed MulDiGraph edges whose both endpoints are dataset wallets (node index = position)."""
    idx = {a: i for i, a in enumerate(addrs)}
    out = []
    for a, i in idx.items():
        if a not in nx_graph:
            continue
        for nb in nx_graph.successors(a):
            j = idx.get(str(nb).lower())
            if j is not None:
                out.append((i, j))
    if use_etherscan and ETHERSCAN_EDGES_CSV.exists():
        ec = pd.read_csv(ETHERSCAN_EDGES_CSV)
        for s, d in zip(ec["src"].astype(str).str.lower(), ec["dst"].astype(str).str.lower()):
            if s in idx and d in idx:
                out.append((idx[s], idx[d]))
    if not out:
        return np.empty((0, 2), dtype=np.int64)
    return np.unique(np.array(out, dtype=np.int64), axis=0)


def make_graph(X: np.ndarray, y: np.ndarray, edges: np.ndarray):
    from torch_geometric.data import Data
    n = len(X)
    ei = torch.as_tensor(edges.T, dtype=torch.long) if len(edges) else torch.zeros((2, 0), dtype=torch.long)
    loops = torch.arange(n, dtype=torch.long).repeat(2, 1)
    return Data(
        x=torch.as_tensor(X, dtype=torch.float32).to(DEVICE),
        y=torch.as_tensor(y, dtype=torch.long).to(DEVICE),
        edge_index=torch.cat([ei, loops], dim=1).to(DEVICE),
    )


def sage_embeddings(model, x, edge_index) -> np.ndarray:
    """Pre-head hidden representation (conv -> BN -> ReLU per layer, eval mode)."""
    model.eval()
    with torch.no_grad():
        h = x
        for conv, norm in zip(model.convs, model.norms):
            h = F.relu(norm(conv(h, edge_index)))
    return h.cpu().numpy()


def train_sage(X_all, y_all, edges, fit_mask, monitor_mask, seed, train_edges=None):
    """2-layer GraphSAGE (hidden 128, mean aggr). `monitor_mask` drives early stopping."""
    set_seeds(seed)
    gnn = GraphPhiV2(model_type="sage", input_dim=X_all.shape[1], hidden_dim=128, num_layers=2)
    g_full = make_graph(X_all, y_all, edges)
    g_train = g_full if train_edges is None else make_graph(X_all, y_all, train_edges)
    # GraphPhiV2.fit calls its monitoring mask `test_mask`; what we pass decides the protocol.
    hist = gnn.fit(g_train, train_mask=fit_mask, test_mask=monitor_mask, epochs=200, early_stopping_patience=25)
    pred, proba = gnn.predict(g_full)
    return gnn, g_full, pred, proba[:, 1], len(hist["train_loss"])


def train_xgb(X_tr, y_tr, X_te, names, seed):
    """Same recipe as ml/train_dataset_scenarios.py; inner validation slice from TRAIN only."""
    X_fit, X_val, y_fit, y_val = train_test_split(X_tr, y_tr, test_size=0.2, random_state=seed, stratify=y_tr)
    m = XGBoostModel(n_estimators=250, max_depth=6, learning_rate=0.05, subsample=0.85,
                     colsample_bytree=0.85, random_state=seed)
    m.fit(X_fit, y_fit, X_val, y_val, feature_names=names, early_stopping_rounds=40)
    spw = float((y_fit == 0).sum() / (y_fit == 1).sum())
    return m, m.predict_proba(X_te)[:, 1], m.predict_proba(X_val)[:, 1], y_val, spw


# ── main ───────────────────────────────────────────────────────────────────────

def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    df = pd.read_csv(HYBRID_CSV)
    df["addr"] = df["node_id"].astype(str).str.lower()
    schema = json.loads(SCHEMA_JSON.read_text(encoding="utf-8"))
    scenarios = {k: [c for c in v if c in df.columns] for k, v in schema["scenarios"].items()}
    graph_cols = scenarios["muldigraph_only"]
    eth_cols = [c for c in schema["feature_groups"]["etherscan"] if c in df.columns]
    s2 = scenarios["muldigraph_plus_etherscan"]
    log.info("Population %d (phishing %d). Scenarios: %s", len(df), int(df.label.sum()),
             {k: len(v) for k, v in scenarios.items()})

    log.info("Loading MulDiGraph ...")
    with open(GRAPH_PKL, "rb") as f:
        G = pickle.load(f)

    # Edges are defined on the fixed population order; each split only permutes masks.
    addrs = df["addr"].tolist()
    edges = edges_between(G, addrs)
    edges_eth = edges_between(G, addrs, use_etherscan=True)
    y_all = df["label"].values.astype(int)

    # ── B-2.2 isolated wallets ────────────────────────────────────────────────
    deg = df["in_degree"].fillna(0) + df["out_degree"].fillna(0)
    touched = np.zeros(len(df), dtype=bool)
    touched[edges[:, 0]] = True
    touched[edges[:, 1]] = True
    iso = {
        "population": len(df),
        "induced_graph_unique_directed_edges": int(len(edges)),
        "induced_graph_edges_with_etherscan": int(len(edges_eth)),
        "no_1hop_tx_in_muldigraph": {
            "normal": int(((deg == 0) & (y_all == 0)).sum()),
            "phishing": int(((deg == 0) & (y_all == 1)).sum()),
        },
        "no_edge_in_induced_gnn_graph": {
            "normal": int((~touched & (y_all == 0)).sum()),
            "phishing": int((~touched & (y_all == 1)).sum()),
        },
        "etherscan_tx_count_zero": {
            "normal": int(((df["transaction_count"].fillna(0) == 0) & (y_all == 0)).sum()),
            "phishing": int(((df["transaction_count"].fillna(0) == 0) & (y_all == 1)).sum()),
        },
    }
    (OUT / "isolated_wallets.json").write_text(json.dumps(iso, indent=2))
    log.info("Isolation audit: %s", iso)

    rows, cost_rows, calib_rows, decomp_rows = [], [], [], []
    reliability = None

    for seed in SEEDS:
        ts = time.time()
        idx_tr, idx_te = train_test_split(np.arange(len(df)), test_size=0.2, random_state=seed, stratify=y_all)
        idx_fit, idx_val = train_test_split(idx_tr, test_size=0.2, random_state=seed, stratify=y_all[idx_tr])
        n = len(df)
        m_fit = np.zeros(n, bool); m_fit[idx_fit] = True
        m_val = np.zeros(n, bool); m_val[idx_val] = True
        m_tr = np.zeros(n, bool); m_tr[idx_tr] = True
        m_te = np.zeros(n, bool); m_te[idx_te] = True
        y_tr, y_te = y_all[idx_tr], y_all[idx_te]
        # Inductive topology: drop every edge that touches a test wallet.
        keep = m_tr[edges[:, 0]] & m_tr[edges[:, 1]]
        edges_inductive = edges[keep]

        def add(scenario, model, res, **extra):
            rows.append({"seed": seed, "scenario": scenario, "model": model, **res, **extra})

        sage_s1 = None
        for scen, cols in scenarios.items():
            X = df[cols].fillna(0).values.astype(np.float64)
            X_tr, X_te = X[idx_tr], X[idx_te]

            sc = StandardScaler().fit(X_tr)
            lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)
            lr.fit(sc.transform(X_tr), y_tr)
            p = lr.predict_proba(sc.transform(X_te))[:, 1]
            add(scen, "LogisticRegression", metrics(y_te, (p >= 0.5).astype(int), p))

            rf = RandomForestClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced",
                                        n_jobs=-1, random_state=seed).fit(X_tr, y_tr)
            p = rf.predict_proba(X_te)[:, 1]
            add(scen, "RandomForest", metrics(y_te, (p >= 0.5).astype(int), p))

            xgbm, p, p_val, y_val, spw = train_xgb(X_tr, y_tr, X_te, cols, seed)
            add(scen, "XGBoost", metrics(y_te, (p >= 0.5).astype(int), p), scale_pos_weight=spw)

            # GraphSAGE: scaler on TRAIN rows only, early stopping on train-internal validation.
            Xs = StandardScaler().fit(X[m_tr]).transform(X)
            gnn, g_full, pred, proba, ep = train_sage(Xs, y_all, edges, m_fit, m_val, seed)
            add(scen, "GraphSAGE", metrics(y_all[m_te], pred[m_te], proba[m_te]), epochs_run=ep)
            if scen == "muldigraph_only":
                sage_s1 = (gnn, g_full)

            if scen == "muldigraph_plus_etherscan":
                # A-2.2 inductive GraphSAGE
                _, _, pred_i, proba_i, ep = train_sage(Xs, y_all, edges, m_fit, m_val, seed,
                                                      train_edges=edges_inductive)
                add(scen, "GraphSAGE-inductive", metrics(y_all[m_te], pred_i[m_te], proba_i[m_te]), epochs_run=ep)

                # B-3.3 cost-sensitive threshold (chosen on validation, applied to test)
                grid = np.linspace(0.05, 0.95, 181)
                for c in COST_RATIOS:
                    costs = [c * ((p_val < t) & (y_val == 1)).sum() + ((p_val >= t) & (y_val == 0)).sum() for t in grid]
                    t_best = float(grid[int(np.argmin(costs))])
                    for label, t in (("default_0.5", 0.5), ("cost_optimal", t_best)):
                        pr = (p >= t).astype(int)
                        r = metrics(y_te, pr, p)
                        cost_rows.append({"seed": seed, "cost_ratio_FN_to_FP": c, "policy": label,
                                          "threshold": t, "FP": r["FP"], "FN": r["FN"],
                                          "total_cost": c * r["FN"] + r["FP"],
                                          "precision": r["precision"], "recall": r["recall"], "f1": r["f1"]})

                # B-2.5 calibration (fitted on validation slice)
                eps = 1e-6
                logit = lambda q: np.log(np.clip(q, eps, 1 - eps) / (1 - np.clip(q, eps, 1 - eps)))  # noqa: E731
                platt = LogisticRegression(C=1e6).fit(logit(p_val).reshape(-1, 1), y_val)
                iso_r = IsotonicRegression(out_of_bounds="clip", y_min=0, y_max=1).fit(p_val, y_val)
                variants = {
                    "uncalibrated": p,
                    "platt": platt.predict_proba(logit(p).reshape(-1, 1))[:, 1],
                    "isotonic": iso_r.predict(p),
                }
                for name, q in variants.items():
                    qc = np.clip(q, eps, 1 - eps)
                    calib_rows.append({"seed": seed, "method": name,
                                       "brier": brier_score_loss(y_te, q), "ece_10bin": ece(y_te, q),
                                       "log_loss": log_loss(y_te, qc), "roc_auc": roc_auc_score(y_te, q),
                                       "f1_at_0.5": f1_score(y_te, (q >= 0.5).astype(int))})
                if seed == 42:
                    reliability = (y_te, variants)

                # A-2.4 decomposition of the old Table-3 GraphSAGE protocol (seed 42 only)
                if seed == 42:
                    Xs1 = StandardScaler().fit(df.loc[m_tr, graph_cols].fillna(0)).transform(df[graph_cols].fillna(0))
                    cases = [
                        ("A: old Table-3 protocol (25 feats, ES on TEST F1, +Etherscan edges, fit on all train)",
                         Xs, edges_eth, m_tr, m_te),
                        ("B: 25 feats, ES on TEST F1, MulDiGraph edges", Xs, edges, m_tr, m_te),
                        ("C: 25 feats, ES on train-internal VALIDATION, MulDiGraph edges", Xs, edges, m_fit, m_val),
                        ("D: 18 graph feats, ES on VALIDATION (old Table-4 encoder)", Xs1, edges, m_fit, m_val),
                    ]
                    for name, Xc, ec, mf, mm in cases:
                        _, _, pr, pb, ep = train_sage(Xc, y_all, ec, mf, mm, seed)
                        decomp_rows.append({"protocol": name, "epochs_run": ep,
                                            **metrics(y_all[m_te], pr[m_te], pb[m_te])})

        # ── Table 4: late fusion + embedding-only ablation (encoder = S1 GraphSAGE) ──
        gnn1, g1 = sage_s1
        emb = sage_embeddings(gnn1.model, g1.x, g1.edge_index)
        eth = df[eth_cols].fillna(0).values.astype(np.float64)
        eth_s = StandardScaler().fit(eth[idx_tr]).transform(eth)
        fused = np.hstack([emb, eth_s])
        names_f = [f"emb_{i}" for i in range(emb.shape[1])] + [f"eth_{c}" for c in eth_cols]
        _, p, *_ = train_xgb(fused[idx_tr], y_tr, fused[idx_te], names_f, seed)
        add("late_fusion", "XGBoost[emb128||eth7]", metrics(y_te, (p >= 0.5).astype(int), p))
        _, p, *_ = train_xgb(emb[idx_tr], y_tr, emb[idx_te], names_f[:emb.shape[1]], seed)
        add("late_fusion", "XGBoost[emb128 only]", metrics(y_te, (p >= 0.5).astype(int), p))

        log.info("seed %d done in %.0fs", seed, time.time() - ts)

    # ── A-2.1 majority-class size (XGBoost S2) ────────────────────────────────
    ratio_rows = []
    phish = df[df.label == 1]
    normal = df[df.label == 0]
    for r in MAJORITY_RATIOS:
        for seed in SEEDS[:5]:
            sub = pd.concat([phish, normal.sample(n=min(len(normal), r * len(phish)), random_state=seed)])
            X = sub[s2].fillna(0).values.astype(np.float64)
            y = sub["label"].values.astype(int)
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
            _, p, *_ , spw = train_xgb(X_tr, y_tr, X_te, s2, seed)
            ratio_rows.append({"ratio": f"1:{r}", "seed": seed, "n_total": len(sub), "scale_pos_weight": spw,
                               "prevalence_test": float(y_te.mean()),
                               **metrics(y_te, (p >= 0.5).astype(int), p)})

    # ── save ──────────────────────────────────────────────────────────────────
    res = pd.DataFrame(rows)
    res.to_csv(OUT / "multiseed_all_runs.csv", index=False)
    num = ["accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "TN", "FP", "FN", "TP"]
    summ = res.groupby(["scenario", "model"], sort=False)[num].agg(["mean", "std"])
    summ.columns = [f"{a}_{b}" for a, b in summ.columns]
    summ.reset_index().to_csv(OUT / "multiseed_summary.csv", index=False)
    res[res.seed == 42].to_csv(OUT / "seed42_table.csv", index=False)

    cost = pd.DataFrame(cost_rows)
    cost.to_csv(OUT / "cost_threshold_all_runs.csv", index=False)
    cost.groupby(["cost_ratio_FN_to_FP", "policy"])[["threshold", "FP", "FN", "total_cost", "precision", "recall", "f1"]] \
        .agg(["mean", "std"]).round(4).to_csv(OUT / "cost_threshold_summary.csv")

    cal = pd.DataFrame(calib_rows)
    cal.to_csv(OUT / "calibration_all_runs.csv", index=False)
    cal.groupby("method", sort=False).agg(["mean", "std"]).drop(columns="seed").round(4).to_csv(OUT / "calibration_summary.csv")

    pd.DataFrame(decomp_rows).to_csv(OUT / "graphsage_protocol_decomposition.csv", index=False)

    rat = pd.DataFrame(ratio_rows)
    rat.to_csv(OUT / "majority_ratio_all_runs.csv", index=False)
    rat.groupby("ratio", sort=False)[["n_total", "scale_pos_weight", "prevalence_test", "precision", "recall", "f1", "roc_auc", "pr_auc"]] \
        .agg(["mean", "std"]).round(4).to_csv(OUT / "majority_ratio_summary.csv")

    # reliability diagram (seed 42)
    y_te, variants = reliability
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="grey", lw=1, label="perfect")
    edges_b = np.linspace(0, 1, 11)
    for name, q in variants.items():
        b = np.clip(np.digitize(q, edges_b[1:-1]), 0, 9)
        xs = [q[b == i].mean() for i in range(10) if (b == i).any()]
        ys = [y_te[b == i].mean() for i in range(10) if (b == i).any()]
        ax.plot(xs, ys, marker="o", label=name)
    ax.set_xlabel("Predicted probability (risk score / 100)")
    ax.set_ylabel("Observed phishing fraction")
    ax.set_title("Reliability diagram – XGBoost S2 (seed 42)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT / "reliability_diagram.png", dpi=200)
    plt.close(fig)

    meta = {"seeds": SEEDS, "population": len(df), "phishing": int(y_all.sum()),
            "scenarios": {k: len(v) for k, v in scenarios.items()},
            "graph_edges_muldigraph_only": int(len(edges)), "runtime_s": round(time.time() - t0, 1)}
    (OUT / "run_meta.json").write_text(json.dumps(meta, indent=2))

    pd.set_option("display.width", 220)
    print("\n=== multi-seed summary (mean) ===")
    print(summ[[c for c in summ.columns if c.endswith("_mean")]].round(4).to_string())
    print("\n=== decomposition ===\n", pd.DataFrame(decomp_rows).round(4).to_string())
    log.info("Done in %.0fs -> %s", time.time() - t0, OUT)


if __name__ == "__main__":
    main()
