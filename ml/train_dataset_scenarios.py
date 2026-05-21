"""
Train and compare four dataset scenarios on the same split:
1) MulDiGraph only
2) MulDiGraph + Etherscan
3) MulDiGraph + BigQuery
4) MulDiGraph + Etherscan + BigQuery

Models compared per scenario:
- Logistic Regression
- Random Forest
- XGBoost
- GNN-GCN-V2 (GraphSAGE, 2-layer)

GNN uses real Ethereum transaction edges from MulDiGraph when available,
falling back to kNN feature-similarity graph otherwise.
"""

from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gnn_model_v2 import GraphPhiV2
from xgboost_model import XGBoostModel

ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"

GRAPH_CANDIDATES = [
    ROOT / "data" / "raw" / "MulDiGraph.pkl",
    ROOT / "Ethereum Phishing Transaction Network" / "MulDiGraph.pkl",
]

# Loaded once in main() and shared across all scenario runs.
_tx_graph = None

TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
HYBRID_CSV = DATA_PROCESSED / "hybrid_dataset.csv"
HYBRID_SCHEMA_JSON = ARTIFACTS / "hybrid_feature_schema.json"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


def evaluate(name: str, scenario: str, y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    return {
        "scenario": scenario,
        "model": name,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def load_transaction_graph():
    """Load MulDiGraph once; returns None if file not found."""
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
    log.warning("MulDiGraph not found — GNN will use kNN graph fallback")
    return None


def extract_edges_for_addresses(
    nx_graph,
    train_addresses: list[str],
    test_addresses: list[str],
) -> list[list[int]] | None:
    """
    Extract Ethereum transaction edges between sampled addresses.

    Node index convention (matches build_graph_with_real_edges):
      0 .. len(train_addresses)-1            → training nodes
      len(train_addresses) .. n_total-1      → test nodes

    Returns None if nx_graph is None or no edges found.
    """
    if nx_graph is None:
        return None

    all_addresses = list(train_addresses) + list(test_addresses)
    addr_to_idx = {str(a).lower(): i for i, a in enumerate(all_addresses)}
    addr_set = set(addr_to_idx.keys())

    edges: list[list[int]] = []
    found_in_graph = 0

    for addr_lower, src_idx in addr_to_idx.items():
        if addr_lower not in nx_graph:
            continue
        found_in_graph += 1
        for neighbor in nx_graph.successors(addr_lower):
            nbr = str(neighbor).lower()
            if nbr in addr_set:
                edges.append([src_idx, addr_to_idx[nbr]])
        for neighbor in nx_graph.predecessors(addr_lower):
            nbr = str(neighbor).lower()
            if nbr in addr_set:
                edges.append([addr_to_idx[nbr], src_idx])

    log.info(
        "Edge extraction: %d/%d addresses found in MulDiGraph, %d transaction edges",
        found_in_graph, len(all_addresses), len(edges),
    )
    return edges if edges else None


def ensure_external_proxy_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create deterministic proxy features for Etherscan/BigQuery when unavailable."""
    out = df.copy()

    if "created_at_ts" not in out.columns:
        out["created_at_ts"] = (out["timestamp_spread_in"] * 0.01).round().astype(int)

    if "balance" not in out.columns:
        out["balance"] = (out["total_received"] - out["total_sent"]).astype(float)
    if "transaction_count" not in out.columns:
        out["transaction_count"] = (out["in_degree"] + out["out_degree"]).astype(float)
    if "activity_score" not in out.columns:
        out["activity_score"] = (out["unique_senders"] + out["unique_receivers"]).astype(float)
    if "avg_tx_value" not in out.columns:
        out["avg_tx_value"] = ((out["avg_in_amount"] + out["avg_out_amount"]) / 2.0).astype(float)
    if "tx_frequency" not in out.columns:
        out["tx_frequency"] = (out["transaction_count"] / 30.0).astype(float)
    if "unique_counterparties" not in out.columns:
        out["unique_counterparties"] = (out["unique_senders"] + out["unique_receivers"]).astype(float)

    if "bq_tx_count_out" not in out.columns:
        out["bq_tx_count_out"] = out["out_degree"].astype(float)
    if "bq_tx_count_in" not in out.columns:
        out["bq_tx_count_in"] = out["in_degree"].astype(float)
    if "bq_total_out_eth" not in out.columns:
        out["bq_total_out_eth"] = out["total_sent"].astype(float)
    if "bq_total_in_eth" not in out.columns:
        out["bq_total_in_eth"] = out["total_received"].astype(float)
    if "bq_unique_to" not in out.columns:
        out["bq_unique_to"] = out["unique_receivers"].astype(float)
    if "bq_unique_from" not in out.columns:
        out["bq_unique_from"] = out["unique_senders"].astype(float)
    if "bq_active_days_out" not in out.columns:
        out["bq_active_days_out"] = np.maximum(1.0, out["timestamp_spread_out"] / 86400.0)
    if "bq_active_days_in" not in out.columns:
        out["bq_active_days_in"] = np.maximum(1.0, out["timestamp_spread_in"] / 86400.0)
    if "bq_first_seen_ts" not in out.columns:
        out["bq_first_seen_ts"] = (out["created_at_ts"] - out["timestamp_spread_in"]).clip(lower=0)
    if "bq_last_seen_ts" not in out.columns:
        out["bq_last_seen_ts"] = out["created_at_ts"]

    return out


def derive_feature_groups(df: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    drop_meta = {"label", "node_id", "address", "class_name", "source_dataset", "error"}
    etherscan_features = [
        c for c in [
            "created_at_ts",
            "balance",
            "transaction_count",
            "activity_score",
            "avg_tx_value",
            "tx_frequency",
            "unique_counterparties",
        ]
        if c in df.columns
    ]
    bigquery_features = [
        c for c in [
            "bq_tx_count_out",
            "bq_tx_count_in",
            "bq_total_out_eth",
            "bq_total_in_eth",
            "bq_unique_to",
            "bq_unique_from",
            "bq_active_days_out",
            "bq_active_days_in",
            "bq_first_seen_ts",
            "bq_last_seen_ts",
        ]
        if c in df.columns
    ]
    graph_features = [c for c in df.columns if c not in drop_meta and c not in set(etherscan_features) | set(bigquery_features)]
    return graph_features, etherscan_features, bigquery_features


def load_scenarios(df: pd.DataFrame) -> tuple[dict[str, list[str]], str]:
    if HYBRID_SCHEMA_JSON.exists():
        with open(HYBRID_SCHEMA_JSON, "r", encoding="utf-8") as f:
            schema = json.load(f)
        schema_scenarios = schema.get("scenarios", {})
        scenario_features = {
            name: [feature for feature in features if feature in df.columns]
            for name, features in schema_scenarios.items()
        }
        scenario_features = {name: features for name, features in scenario_features.items() if features}
        if scenario_features:
            return scenario_features, "Scenario comparison uses schema-defined feature groups on a single shared split."

    graph_features, etherscan_features, bigquery_features = derive_feature_groups(df)
    scenario_features = {
        "muldigraph_only": graph_features,
        "muldigraph_plus_etherscan": graph_features + etherscan_features,
        "muldigraph_plus_bigquery": graph_features + bigquery_features,
        "muldigraph_plus_etherscan_plus_bigquery": graph_features + etherscan_features + bigquery_features,
    }
    scenario_features = {name: features for name, features in scenario_features.items() if features}
    return scenario_features, "Scenario comparison uses inferred feature groups on a single shared split."


def train_lr(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    return lr.predict(X_test_scaled)


def train_rf(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    return rf.predict(X_test)


def train_xgb(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, feature_names: list[str]) -> np.ndarray:
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )
    xgb = XGBoostModel(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.85,
    )
    xgb.fit(
        X_fit,
        y_fit,
        X_val,
        y_val,
        feature_names=feature_names,
        early_stopping_rounds=40,
    )
    return xgb.predict(X_test)


def train_gnn(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_cols: list[str],
    edge_list: list[list[int]] | None = None,
) -> np.ndarray:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 2 layers: 1-hop then 2-hop aggregation — more layers cause over-smoothing
    # on graphs of this size (~8K nodes).
    gnn = GraphPhiV2(
        model_type="sage",
        input_dim=len(feature_cols),
        hidden_dim=128,
        num_layers=2,
    )

    if edge_list is not None and len(edge_list) > 0:
        log.info("GNN: using %d real transaction edges", len(edge_list))
        graph, train_mask, test_mask = gnn.build_graph_with_real_edges(
            X_train_scaled, y_train, X_test_scaled, y_test, edge_list
        )
    else:
        log.info("GNN: real edges unavailable, falling back to kNN graph")
        graph, train_mask, test_mask = gnn.build_graph_from_train_only(
            X_train_scaled, y_train, X_test_scaled, y_test
        )

    gnn.fit(
        graph,
        train_mask=train_mask,
        test_mask=test_mask,
        epochs=200,
        early_stopping_patience=25,
    )
    pred, _ = gnn.predict(graph, test_mask)
    return pred


def run_scenario(name: str, train_df: pd.DataFrame, test_df: pd.DataFrame, feature_cols: list[str]) -> list[dict]:
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["label"].values

    log.info("Scenario %s | train=%d test=%d features=%d", name, len(X_train), len(X_test), len(feature_cols))

    # Try to get real Ethereum transaction edges for GNN.
    # node_id column is present in hybrid dataset (= Ethereum address).
    edge_list = None
    if _tx_graph is not None:
        id_col = next((c for c in ("node_id", "address") if c in train_df.columns), None)
        if id_col:
            train_addrs = train_df[id_col].astype(str).str.lower().tolist()
            test_addrs = test_df[id_col].astype(str).str.lower().tolist()
            edge_list = extract_edges_for_addresses(_tx_graph, train_addrs, test_addrs)

    results = []
    lr_pred = train_lr(X_train, y_train, X_test)
    results.append(evaluate("LogisticRegression", name, y_test, lr_pred))

    rf_pred = train_rf(X_train, y_train, X_test)
    results.append(evaluate("RandomForest", name, y_test, rf_pred))

    xgb_pred = train_xgb(X_train, y_train, X_test, feature_cols)
    results.append(evaluate("XGBoost", name, y_test, xgb_pred))

    gnn_pred = train_gnn(X_train, y_train, X_test, y_test, feature_cols, edge_list)
    results.append(evaluate("GNN-GraphSAGE-V2", name, y_test, gnn_pred))

    return results


def main() -> None:
    global _tx_graph
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    _tx_graph = load_transaction_graph()

    if HYBRID_CSV.exists():
        df = pd.read_csv(HYBRID_CSV)
        scenario_features, note = load_scenarios(df)
        train_df, test_df = train_test_split(
            df,
            test_size=0.2,
            random_state=42,
            stratify=df["label"],
        )
    else:
        train_df = pd.read_csv(TRAIN_CSV)
        test_df = pd.read_csv(TEST_CSV)
        train_df = ensure_external_proxy_features(train_df)
        test_df = ensure_external_proxy_features(test_df)
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        scenario_features, note = load_scenarios(combined_df)
        note = f"{note} External features were generated as deterministic proxies because the real hybrid dataset was not present."

    all_results = []
    for scenario_name, feature_cols in scenario_features.items():
        all_results.extend(run_scenario(scenario_name, train_df, test_df, feature_cols))

    result_df = pd.DataFrame(all_results)
    result_df.to_csv(ARTIFACTS / "dataset_scenarios_comparison.csv", index=False)

    best_row = result_df.sort_values("f1", ascending=False).iloc[0]
    summary = {
        "best_scenario": best_row["scenario"],
        "best_model": best_row["model"],
        "best_f1": float(best_row["f1"]),
        "note": note,
        "scenarios": {name: len(features) for name, features in scenario_features.items()},
    }

    with open(ARTIFACTS / "dataset_scenarios_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    log.info("Saved: %s", ARTIFACTS / "dataset_scenarios_comparison.csv")
    log.info("Saved: %s", ARTIFACTS / "dataset_scenarios_summary.json")
    print("\nScenario Comparison:")
    print(result_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print("\nBest:", summary)


if __name__ == "__main__":
    main()