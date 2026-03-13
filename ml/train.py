"""
ml/train.py
-----------
Trains Logistic Regression (baseline) and Random Forest (main model),
prints a comparison table, and saves the best model + scaler to ml/artifacts/.

Outputs:
  ml/artifacts/model.pkl         – RandomForestClassifier (joblib)
  ml/artifacts/scaler.pkl        – StandardScaler used for LR baseline
  ml/artifacts/feature_schema.json – (re-written with verified column order)
  ml/artifacts/confusion_matrix.png
  ml/artifacts/feature_importance.png
"""

import json
import logging
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report,
    confusion_matrix, f1_score, precision_score, recall_score,
    ConfusionMatrixDisplay,
)
from sklearn.preprocessing import StandardScaler

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


# ── Helpers ────────────────────────────────────────────────────────────────────

def evaluate(name: str, y_true, y_pred, y_proba=None) -> dict:
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    log.info(
        "[%s] Accuracy=%.4f  Precision=%.4f  Recall=%.4f  F1=%.4f",
        name, acc, prec, rec, f1,
    )
    return {"model": name, "accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Phishing"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Confusion matrix saved → %s", save_path)


def plot_feature_importance(model, feature_names: list, save_path: Path):
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    top_n = min(15, len(feature_names))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(
        [feature_names[i] for i in indices[:top_n]][::-1],
        importances[indices[:top_n]][::-1],
        color="steelblue",
    )
    ax.set_xlabel("Importance")
    ax.set_title("Top Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close(fig)
    log.info("Feature importance plot saved → %s", save_path)


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    ARTIFACTS.mkdir(parents=True, exist_ok=True)

    # ── Load data ──────────────────────────────────────────────────────────────
    log.info("Loading processed datasets…")
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    drop_cols = {"label", "node_id"}
    feature_cols = [c for c in train_df.columns if c not in drop_cols]

    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["label"].values

    log.info("Train: %d  |  Test: %d  |  Features: %d", len(X_train), len(X_test), len(feature_cols))
    log.info("Train phishing ratio: %.2f%%", 100 * y_train.mean())

    # ── Baseline: Logistic Regression (with scaling) ───────────────────────────
    log.info("Training Logistic Regression baseline…")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    lr_pred = lr.predict(X_test_scaled)
    lr_metrics = evaluate("LogisticRegression", y_test, lr_pred)

    # ── Main model: Random Forest ──────────────────────────────────────────────
    log.info("Training Random Forest…")
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
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = evaluate("RandomForest", y_test, rf_pred, rf_proba)

    # ── Comparison table ───────────────────────────────────────────────────────
    comparison = pd.DataFrame([lr_metrics, rf_metrics]).set_index("model")
    print("\n" + "=" * 60)
    print("Model Comparison")
    print("=" * 60)
    print(comparison.to_string(float_format="%.4f"))
    print("=" * 60 + "\n")

    print("Random Forest Classification Report:")
    print(classification_report(y_test, rf_pred, target_names=["Normal", "Phishing"]))

    # ── Plots ──────────────────────────────────────────────────────────────────
    plot_confusion_matrix(y_test, rf_pred, "Random Forest – Confusion Matrix",
                          ARTIFACTS / "confusion_matrix.png")
    plot_feature_importance(rf, feature_cols, ARTIFACTS / "feature_importance.png")

    # ── Save artifacts ─────────────────────────────────────────────────────────
    joblib.dump(rf, ARTIFACTS / "model.pkl")
    log.info("Model saved → %s", ARTIFACTS / "model.pkl")

    joblib.dump(scaler, ARTIFACTS / "scaler.pkl")
    log.info("Scaler saved → %s", ARTIFACTS / "scaler.pkl")

    schema = {"features": feature_cols, "label": "label", "node_id": "node_id"}
    with open(ARTIFACTS / "feature_schema.json", "w") as f:
        json.dump(schema, f, indent=2)
    log.info("Feature schema saved → %s", ARTIFACTS / "feature_schema.json")

    log.info("Training complete. RandomForest F1=%.4f", rf_metrics["f1"])


if __name__ == "__main__":
    main()
