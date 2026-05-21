"""
ml/train_all_models_v2.py
------------------------
Updated training script using GNN V2 (fixed overfitting):
  - GNN V2: Train-only graph, class weighting, smaller model
  - RandomForest: Baseline (already good)
  - XGBoost: Advanced (already good)
  - GNN V1: (removed due to overfitting)

Compares Random Forest, XGBoost, and GNN V2.
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
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from gnn_model_v2 import GraphPhiV2
from xgboost_model import XGBoostModel

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "ml" / "artifacts"

TRAIN_CSV = DATA_PROCESSED / "train.csv"
TEST_CSV = DATA_PROCESSED / "test.csv"

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
    """Plot comparison of models."""
    df = pd.DataFrame(results).set_index("model")
    
    fig, axes = plt.subplots(1, 4, figsize=(14, 4))
    metrics = ["accuracy", "precision", "recall", "f1"]
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for ax, metric in zip(axes, metrics):
        df[metric].plot(kind="barh", ax=ax, color=colors)
        ax.set_xlabel(metric.capitalize())
        ax.set_xlim([0, 1])
        ax.grid(axis="x", alpha=0.3)
    
    plt.suptitle("Model Comparison (RandomForest, XGBoost, GNN V2)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Comparison plot saved → %s", save_path)


# ── Training functions ─────────────────────────────────────────────────────────

def train_baseline_models(X_train, y_train, X_test, y_test, feature_cols: list) -> Tuple[dict, dict]:
    """Train baseline RandomForest model."""
    log.info("=" * 70)
    log.info("TRAINING BASELINE MODEL (RandomForest)")
    log.info("=" * 70)
    
    log.info("\nTraining Random Forest…")
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
    rf_metrics = evaluate_model("RandomForest", y_test, rf_pred)
    
    # Save baseline
    joblib.dump(rf, ARTIFACTS / "model_baseline_rf.pkl")
    
    return rf_metrics, {"rf": rf, "rf_pred": rf_pred}


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


def train_gnn_v2_model(X_train, y_train, X_test, y_test, feature_cols: list) -> dict:
    """Train GNN V2 model — GraphSAGE, 2 layers."""
    log.info("\n" + "=" * 70)
    log.info("TRAINING GNN MODEL (V2 - GraphSAGE, 2-layer)")
    log.info("=" * 70)

    try:
        log.info("\nTraining GNN-GraphSAGE V2 (train-only graph, class weighting)…")

        gnn = GraphPhiV2(
            model_type="sage",
            input_dim=len(feature_cols),
            hidden_dim=128,
            num_layers=2,
        )
        
        # Build graph from training data only (prevents data leakage)
        graph, train_mask, test_mask = gnn.build_graph_from_train_only(
            X_train, y_train, X_test, y_test
        )
        
        # Train with class weighting
        gnn.fit(
            graph,
            train_mask=train_mask,
            test_mask=test_mask,
            epochs=200,
            early_stopping_patience=25,
        )

        gnn_pred, _ = gnn.predict(graph, test_mask)
        gnn_metrics = evaluate_model("GNN-GraphSAGE V2", y_test, gnn_pred)
        
        # Save model
        gnn.save(ARTIFACTS / "model_gcn_v2.pt")
        
        return gnn_metrics
    except Exception as e:
        log.error("Error training GNN V2: %s", e)
        return {}


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Main training pipeline."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    
    log.info("Starting comprehensive model training pipeline (V2)…\n")
    
    # ── Load dataset ───────────────────────────────────────────────────────
    log.info("Loading dataset…")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
    
    drop_cols = {"label", "node_id"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]
    
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["label"].values
    
    log.info("Dataset loaded:")
    log.info("  Train: %d samples | Test: %d samples | Features: %d", 
             len(X_train), len(X_test), len(feature_cols))
    log.info("  Phishing ratio (train): %.2f%%\n", 100 * y_train.mean())
    
    # ── Train models ───────────────────────────────────────────────────────
    all_results = []
    best_models = {}
    
    # RandomForest
    rf_metrics, rf_outputs = train_baseline_models(X_train, y_train, X_test, y_test, feature_cols)
    all_results.append(rf_metrics)
    best_models["rf"] = rf_outputs["rf"]
    
    # XGBoost
    xgb_metrics, xgb_model = train_xgboost_model(X_train, y_train, X_test, y_test, feature_cols)
    all_results.append(xgb_metrics)
    best_models["xgboost"] = xgb_model
    
    # GNN V2
    try:
        gnn_metrics = train_gnn_v2_model(X_train, y_train, X_test, y_test, feature_cols)
        if gnn_metrics:
            all_results.append(gnn_metrics)
    except Exception as e:
        log.error("Skipping GNN V2: %s", e)
    
    # ── Model comparison ───────────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("MODEL COMPARISON")
    log.info("=" * 70)
    
    comparison_df = pd.DataFrame(all_results).set_index("model")
    print("\n" + comparison_df.to_string(float_format="%.4f"))
    print()
    
    # Save results
    comparison_df.to_csv(ARTIFACTS / "model_comparison_v2.csv")
    log.info("Comparison saved → %s", ARTIFACTS / "model_comparison_v2.csv")
    
    # Select best model
    best_idx = comparison_df["f1"].idxmax()
    best_f1 = comparison_df.loc[best_idx, "f1"]
    
    log.info("\n" + "=" * 70)
    log.info(f"BEST MODEL: {best_idx} (F1=%.4f)", best_f1)
    log.info("=" * 70 + "\n")
    
    # ── Plots ──────────────────────────────────────────────────────────
    
    # Confusion matrices
    if "RandomForest" in best_idx:
        plot_confusion_matrix(y_test, rf_outputs["rf_pred"], "RandomForest – Confusion Matrix",
                            ARTIFACTS / "cm_v2_rf.png")
    
    if "XGBoost" in best_idx:
        xgb_pred = best_models["xgboost"].predict(X_test)
        plot_confusion_matrix(y_test, xgb_pred, "XGBoost – Confusion Matrix",
                            ARTIFACTS / "cm_v2_xgb.png")
    
    # Comparison plot
    plot_model_comparison(all_results, ARTIFACTS / "model_comparison_v2.png")
    
    # ── Save results ───────────────────────────────────────────────────
    schema = {
        "features": feature_cols,
        "label": "label",
        "node_id": "node_id",
        "best_model": best_idx,
        "best_f1": float(best_f1),
        "note": "V2 with improved GNN (train-only graph, class weighting)",
    }
    
    with open(ARTIFACTS / "feature_schema_v2.json", "w") as f:
        json.dump(schema, f, indent=2)
    log.info("Schema saved → %s", ARTIFACTS / "feature_schema_v2.json")
    
    # Save best model
    if "RandomForest" in best_idx:
        joblib.dump(best_models["rf"], ARTIFACTS / "model_v2_best.pkl")
        log.info("Best model (RandomForest) saved as model_v2_best.pkl")
    elif "XGBoost" in best_idx:
        best_models["xgboost"].save(ARTIFACTS / "model_v2_best.json")
        log.info("Best model (XGBoost) saved as model_v2_best.json")
    
    log.info("\n✓ Training pipeline V2 complete!")
    
    return all_results, comparison_df


if __name__ == "__main__":
    results, comparison = main()
