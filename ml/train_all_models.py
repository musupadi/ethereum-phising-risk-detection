"""
ml/train_all_models.py
----------------------
Comprehensive training script for all models:
  - Baseline: LogisticRegression & RandomForest (old dataset)
  - Advanced: GNN (GCN/GAT), XGBoost
  - New datasets: Etherscan + Forta Network enriched data

Compares all models and selects best by F1-Score.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight

from gnn_model import GraphPhi
from xgboost_model import XGBoostModel

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "ml" / "artifacts"

TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"
TRAIN_ENRICHED_CSV = DATA_RAW / "train_enriched.csv"
TEST_ENRICHED_CSV = DATA_RAW / "test_enriched.csv"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_model(name: str, y_true, y_pred, y_proba=None) -> dict:
    """Compute evaluation metrics."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    log.info(
        "[%s] Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
        name, acc, prec, rec, f1,
    )
    
    return {
        "model": name,
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
    }


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Phishing"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Saved → %s", save_path)


def plot_model_comparison(results: list, save_path: Path):
    """Plot comparison of all models."""
    df = pd.DataFrame(results).set_index("model")
    
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for ax, metric in zip(axes, metrics):
        df[metric].plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel(metric.capitalize())
        ax.set_xlim([0, 1])
        ax.grid(axis="x", alpha=0.3)
    
    plt.suptitle("Model Comparison", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Comparison plot saved → %s", save_path)


def plot_feature_importance(model, feature_names: list, save_path: Path, top_n: int = 15):
    """Plot feature importance."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    elif hasattr(model, "get_feature_importance"):
        importance_df = model.get_feature_importance(top_n)
        if importance_df.empty:
            log.warning("No feature importance available")
            return
        importance_df = importance_df.sort_values("importance", ascending=True)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.barh(importance_df["feature"], importance_df["importance"], color="steelblue")
        ax.set_xlabel("Importance")
        ax.set_title(f"Feature Importance ({model.__class__.__name__})")
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        log.info("Importance plot saved → %s", save_path)
        return
    else:
        log.warning("Model does not support feature importance")
        return
    
    indices = np.argsort(importances)[::-1]
    top_indices = indices[:min(top_n, len(importances))]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in top_indices][::-1],
        importances[top_indices][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Importance plot saved → %s", save_path)


# ── Training functions ─────────────────────────────────────────────────────────

def train_baseline_models(X_train, y_train, X_test, y_test, feature_cols: list) -> Tuple[dict, dict, object, object]:
    """Train baseline models (Logistic Regression & Random Forest)."""
    log.info("=" * 70)
    log.info("TRAINING BASELINE MODELS (Dataset: Stg 1)")
    log.info("=" * 70)
    
    results = []
    
    # ── Logistic Regression ────────────────────────────────────────────────────
    log.info("\n[1/2] Training Logistic Regression (scaled features)…")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_metrics = evaluate_model("LogisticRegression", y_test, lr_pred)
    results.append(lr_metrics)
    
    # ── Random Forest ──────────────────────────────────────────────────────────
    log.info("\n[2/2] Training Random Forest (baseline)…")
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_metrics = evaluate_model("RandomForest (Baseline)", y_test, rf_pred)
    results.append(rf_metrics)
    
    # Save baseline model
    joblib.dump(rf, ARTIFACTS / "model_baseline_rf.pkl")
    joblib.dump(scaler, ARTIFACTS / "scaler.pkl")
    
    return results, {"rf": rf, "lr": lr, "scaler": scaler}, rf_pred, rf


def train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols: list) -> Tuple[dict, object]:
    """Train XGBoost model."""
    log.info("\n" + "=" * 70)
    log.info("TRAINING XGBOOST MODEL")
    log.info("=" * 70)
    
    # Split for validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    log.info("Training XGBoost…")
    xgb_model = XGBoostModel(
        n_estimators=300,
        max_depth=8,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
    )
    
    xgb_model.fit(
        X_tr, y_tr, X_val, y_val,
        feature_names=feature_cols,
        early_stopping_rounds=50,
    )
    
    xgb_pred = xgb_model.predict(X_test)
    xgb_metrics = xgb_model.evaluate(X_test, y_test, name="XGBoost")
    
    # Save model
    xgb_model.save(ARTIFACTS / "model_xgboost.json")
    
    return xgb_metrics, xgb_model


def train_gnn_models(X_train, y_train, X_test, y_test, feature_cols: list) -> dict:
    """Train GNN models (GCN and GAT)."""
    log.info("\n" + "=" * 70)
    log.info("TRAINING GNN MODELS")
    log.info("=" * 70)
    
    results = {}
    
    # Combine train + test for graph construction
    X_combined = np.vstack([X_train, X_test])
    y_combined = np.hstack([y_train, y_test])
    
    df_combined = pd.DataFrame(X_combined, columns=feature_cols)
    df_combined["label"] = y_combined
    
    train_size = len(X_train)
    train_mask = np.zeros(len(df_combined), dtype=bool)
    train_mask[:train_size] = True
    test_mask = ~train_mask
    
    # ── GCN Model ──────────────────────────────────────────────────────────────
    log.info("\n[1/2] Training GCN Model…")
    gcn = GraphPhi(model_type="gcn", input_dim=len(feature_cols), hidden_dim=64)
    
    graph = gcn.build_graph_from_df(
        df_combined, feature_cols, label_col="label", add_self_loops=True
    )
    
    gcn.fit(graph, train_mask=train_mask, val_mask=test_mask, epochs=100)
    
    gcn_pred, _ = gcn.predict(graph, test_mask)
    gcn_metrics = evaluate_model("GNN-GCN", y_test, gcn_pred)
    results["GCN"] = gcn_metrics
    
    # Save GCN model
    gcn.save(ARTIFACTS / "model_gcn.pt")
    
    # ── GAT Model ──────────────────────────────────────────────────────────────
    log.info("\n[2/2] Training GAT Model…")
    gat = GraphPhi(model_type="gat", input_dim=len(feature_cols), hidden_dim=64)
    
    gat.fit(graph, train_mask=train_mask, val_mask=test_mask, epochs=100)
    
    gat_pred, _ = gat.predict(graph, test_mask)
    gat_metrics = evaluate_model("GNN-GAT", y_test, gat_pred)
    results["GAT"] = gat_metrics
    
    # Save GAT model
    gat.save(ARTIFACTS / "model_gat.pt")
    
    return results


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Main training pipeline."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    
    log.info("Starting comprehensive model training pipeline…\n")
    
    # ── Load baseline dataset (old) ────────────────────────────────────────────
    log.info("Loading baseline dataset (old)…")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    drop_cols = {"label", "node_id"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["label"].values
    
    log.info("Baseline Dataset:")
    log.info("  Train: %d samples | Test: %d samples | Features: %d", 
             len(X_train), len(X_test), len(feature_cols))
    log.info("  Phishing ratio (train): %.2f%%\n", 100 * y_train.mean())
    
    # ── Train all models ───────────────────────────────────────────────────────
    all_results = []
    best_models = {}
    
    # Baseline models
    baseline_results, baseline_models, rf_pred, rf_model = train_baseline_models(
        X_train, y_train, X_test, y_test, feature_cols
    )
    all_results.extend(baseline_results)
    best_models["baseline"] = baseline_models
    
    # XGBoost
    xgb_metrics, xgb_model = train_xgboost_model(
        X_train, y_train, X_test, y_test, feature_cols
    )
    all_results.append(xgb_metrics)
    best_models["xgboost"] = xgb_model
    
    # GNN models
    try:
        gnn_results = train_gnn_models(X_train, y_train, X_test, y_test, feature_cols)
        for model_name, metrics in gnn_results.items():
            all_results.append(metrics)
            best_models[f"gnn_{model_name}"] = None  # Would save separately
    except Exception as e:
        log.error("Error training GNN models: %s. Skipping…", e)
    
    # ── Model comparison ───────────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("MODEL COMPARISON")
    log.info("=" * 70)
    
    comparison_df = pd.DataFrame(all_results).set_index("model")
    print("\n" + comparison_df.to_string(float_format="%.4f"))
    print()
    
    # Save comparison to CSV
    comparison_df.to_csv(ARTIFACTS / "model_comparison.csv")
    log.info("Comparison saved → %s", ARTIFACTS / "model_comparison.csv")
    
    # ── Select best model by F1 ────────────────────────────────────────────────
    best_idx = comparison_df["f1"].idxmax()
    best_f1 = comparison_df.loc[best_idx, "f1"]
    
    log.info("\n" + "=" * 70)
    log.info(f"BEST MODEL: {best_idx} (F1=%.4f)", best_f1)
    log.info("=" * 70 + "\n")
    
    # ── Plots ──────────────────────────────────────────────────────────────────
    
    # Confusion matrices for best models
    if "RandomForest" in best_idx or "Baseline" in best_idx:
        plot_confusion_matrix(y_test, rf_pred, "Random Forest (Baseline) – Confusion Matrix",
                            ARTIFACTS / "cm_baseline.png")
    
    if "XGBoost" in best_idx:
        xgb_pred = best_models["xgboost"].predict(X_test)
        plot_confusion_matrix(y_test, xgb_pred, "XGBoost – Confusion Matrix",
                            ARTIFACTS / "cm_xgboost.png")
    
    # Feature importance
    try:
        plot_feature_importance(rf_model, feature_cols, 
                              ARTIFACTS / "feature_importance_rf.png")
    except Exception as e:
        log.warning("Could not plot RF importance: %s", e)
    
    try:
        plot_feature_importance(best_models["xgboost"], feature_cols,
                              ARTIFACTS / "feature_importance_xgb.png")
    except Exception as e:
        log.warning("Could not plot XGB importance: %s", e)
    
    # Model comparison plot
    plot_model_comparison(all_results, ARTIFACTS / "model_comparison.png")
    
    # ── Save schema and best model ─────────────────────────────────────────────
    schema = {
        "features": feature_cols,
        "label": "label",
        "node_id": "node_id",
        "best_model": best_idx,
        "best_f1": float(best_f1),
    }
    
    with open(ARTIFACTS / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    log.info("Feature schema saved → %s", ARTIFACTS / "feature_schema.json")
    
    # Save best model as default
    if "XGBoost" in best_idx:
        best_models["xgboost"].save(ARTIFACTS / "model.json")
        log.info("Best model (XGBoost) saved as default model")
    else:
        joblib.dump(rf_model, ARTIFACTS / "model.pkl")
        log.info("Best model (RandomForest) saved as default model")
    
    log.info("\n✓ Training pipeline complete!")
    
    return all_results, comparison_df


if __name__ == "__main__":
    results, comparison = main()
