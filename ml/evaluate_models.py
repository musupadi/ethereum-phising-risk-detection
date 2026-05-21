"""
ml/evaluate_models.py
---------------------
Comprehensive evaluation script comparing:
  - Baseline (old dataset): LR + RF
  - Advanced: GNN (GCN/GAT) + XGBoost
  - New data: Etherscan + Forta enriched datasets

Produces detailed reports and visualizations.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
    roc_curve, precision_recall_curve, average_precision_score,
    ConfusionMatrixDisplay, confusion_matrix, classification_report,
)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA_PROCESSED = ROOT / "data" / "processed"
DATA_RAW = ROOT / "data" / "raw"
ARTIFACTS = ROOT / "ml" / "artifacts"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class ModelEvaluator:
    """Comprehensive model evaluation and comparison."""

    def __init__(self):
        """Initialize evaluator."""
        self.results = {}
        self.models = {}

    def load_baseline_model(self) -> Tuple[object, np.ndarray, np.ndarray]:
        """Load baseline RandomForest model."""
        try:
            model = joblib.load(ARTIFACTS / "model_baseline_rf.pkl")
            test_df = pd.read_csv(DATA_PROCESSED / "test.csv")
            
            drop_cols = {"label", "node_id"}
            feature_cols = [c for c in test_df.columns if c not in drop_cols]
            
            X_test = test_df[feature_cols].fillna(0).values
            y_test = test_df["label"].values
            
            log.info("Loaded baseline RandomForest model")
            return model, X_test, y_test
        except Exception as e:
            log.error("Could not load baseline model: %s", e)
            return None, None, None

    def load_xgboost_model(self) -> object:
        """Load XGBoost model."""
        try:
            from xgboost_model import XGBoostModel
            model = XGBoostModel()
            model.load(ARTIFACTS / "model_xgboost.json")
            log.info("Loaded XGBoost model")
            return model
        except Exception as e:
            log.error("Could not load XGBoost model: %s", e)
            return None

    def load_schema(self) -> Dict:
        """Load feature schema."""
        try:
            with open(ARTIFACTS / "feature_schema.json", "r") as f:
                return json.load(f)
        except Exception as e:
            log.error("Could not load schema: %s", e)
            return {}

    def evaluate_on_baseline(self) -> Dict:
        """Evaluate best model on baseline test set."""
        log.info("\n" + "=" * 70)
        log.info("EVALUATING ON BASELINE TEST SET")
        log.info("=" * 70)
        
        model, X_test, y_test = self.load_baseline_model()
        if model is None or X_test is None:
            return {}
        
        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Compute metrics
        metrics = {
            "dataset": "Baseline (Old)",
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred),
            "recall": recall_score(y_test, y_pred),
            "f1": f1_score(y_test, y_pred),
            "roc_auc": roc_auc_score(y_test, y_proba),
            "avg_precision": average_precision_score(y_test, y_proba),
            "samples": len(y_test),
        }
        
        log.info("Baseline Results:")
        log.info(json.dumps(metrics, indent=2))
        
        # Plot confusion matrix
        self._plot_confusion_matrix(y_test, y_pred, "Baseline (RandomForest)",
                                   ARTIFACTS / "cm_baseline_detailed.png")
        
        # Plot ROC curve
        self._plot_roc_curve(y_test, y_proba, "Baseline (RandomForest)",
                            ARTIFACTS / "roc_baseline.png")
        
        # Plot precision-recall curve
        self._plot_pr_curve(y_test, y_proba, "Baseline (RandomForest)",
                           ARTIFACTS / "pr_baseline.png")
        
        self.results["baseline"] = metrics
        return metrics

    def evaluate_on_enriched_etherscan(self) -> Dict:
        """Evaluate on Etherscan-enriched dataset."""
        log.info("\n" + "=" * 70)
        log.info("EVALUATING ON ETHERSCAN-ENRICHED DATASET")
        log.info("=" * 70)
        
        if not (DATA_RAW / "test_etherscan_enriched.csv").exists():
            log.warning("Etherscan enriched test set not found, skipping…")
            return {}
        
        model, _, _ = self.load_baseline_model()
        if model is None:
            return {}
        
        try:
            test_df = pd.read_csv(DATA_RAW / "test_etherscan_enriched.csv")
            
            # Extract features that match baseline model
            schema = self.load_schema()
            feature_cols = schema.get("features", [])
            
            # Only use features that exist in enriched dataset
            available_features = [f for f in feature_cols if f in test_df.columns]
            
            if not available_features:
                log.warning("No overlapping features found")
                return {}
            
            X_test = test_df[available_features].fillna(0).values
            y_test = test_df.get("label", test_df.get("is_phishing", []))
            
            if len(y_test) == 0:
                log.warning("No labels found in enriched dataset")
                return {}
            
            y_test = np.array(y_test).astype(int)
            
            # Only evaluate on available features
            # Note: This is approximation; ideally reprocess with same feature pipeline
            if len(available_features) < len(feature_cols):
                log.warning("Using only %d/%d features for evaluation",
                           len(available_features), len(feature_cols))
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "dataset": "Etherscan Enriched",
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "avg_precision": average_precision_score(y_test, y_proba),
                "samples": len(y_test),
                "note": f"Evaluated on {len(available_features)} shared features",
            }
            
            log.info("Etherscan Results:")
            log.info(json.dumps(metrics, indent=2))
            
            self._plot_confusion_matrix(y_test, y_pred, "Etherscan Enriched",
                                       ARTIFACTS / "cm_etherscan.png")
            
            self.results["etherscan"] = metrics
            return metrics
        except Exception as e:
            log.error("Error evaluating on Etherscan data: %s", e)
            return {}

    def evaluate_on_forta(self) -> Dict:
        """Evaluate on Forta Network enriched dataset."""
        log.info("\n" + "=" * 70)
        log.info("EVALUATING ON FORTA NETWORK DATASET")
        log.info("=" * 70)
        
        if not (DATA_RAW / "test_forta_enriched.csv").exists():
            log.warning("Forta enriched test set not found, skipping…")
            return {}
        
        model, _, _ = self.load_baseline_model()
        if model is None:
            return {}
        
        try:
            test_df = pd.read_csv(DATA_RAW / "test_forta_enriched.csv")
            
            schema = self.load_schema()
            feature_cols = schema.get("features", [])
            available_features = [f for f in feature_cols if f in test_df.columns]
            
            if not available_features:
                log.warning("No overlapping features found")
                return {}
            
            X_test = test_df[available_features].fillna(0).values
            
            # Forta labels: phishing, scam, exploit
            # Combine into single binary label
            is_malicious = (
                test_df.get("is_phishing", 0).astype(int) |
                test_df.get("is_scam", 0).astype(int) |
                test_df.get("is_exploit", 0).astype(int)
            )
            y_test = np.array(is_malicious).astype(int)
            
            if len(y_test) == 0:
                log.warning("No labels found in Forta dataset")
                return {}
            
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1]
            
            metrics = {
                "dataset": "Forta Network",
                "accuracy": accuracy_score(y_test, y_pred),
                "precision": precision_score(y_test, y_pred),
                "recall": recall_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred),
                "roc_auc": roc_auc_score(y_test, y_proba),
                "avg_precision": average_precision_score(y_test, y_proba),
                "samples": len(y_test),
            }
            
            log.info("Forta Results:")
            log.info(json.dumps(metrics, indent=2))
            
            self._plot_confusion_matrix(y_test, y_pred, "Forta Network",
                                       ARTIFACTS / "cm_forta.png")
            
            self.results["forta"] = metrics
            return metrics
        except Exception as e:
            log.error("Error evaluating on Forta data: %s", e)
            return {}

    def compare_all_results(self) -> pd.DataFrame:
        """Summarize and compare all evaluation results."""
        log.info("\n" + "=" * 70)
        log.info("COMPREHENSIVE COMPARISON")
        log.info("=" * 70)
        
        if not self.results:
            log.warning("No results to compare")
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results).T
        df = df[["dataset", "accuracy", "precision", "recall", "f1", "roc_auc", "samples"]]
        
        print("\n" + df.to_string())
        print()
        
        # Save to CSV
        df.to_csv(ARTIFACTS / "evaluation_comprehensive.csv")
        log.info("Saved comprehensive evaluation → %s", ARTIFACTS / "evaluation_comprehensive.csv")
        
        # Plot comparison
        self._plot_comparison(df)
        
        return df

    # ── Helper methods ─────────────────────────────────────────────────────────

    @staticmethod
    def _plot_confusion_matrix(y_true, y_pred, title: str, save_path: Path):
        """Plot confusion matrix."""
        cm = confusion_matrix(y_true, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Normal", "Phishing"])
        fig, ax = plt.subplots(figsize=(5, 4))
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(title)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        log.info("Saved → %s", save_path)

    @staticmethod
    def _plot_roc_curve(y_true, y_proba, title: str, save_path: Path):
        """Plot ROC curve."""
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.3f})", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", label="Random", linewidth=1)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve – {title}")
        ax.legend(loc="lower right")
        ax.grid(alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        log.info("Saved → %s", save_path)

    @staticmethod
    def _plot_pr_curve(y_true, y_proba, title: str, save_path: Path):
        """Plot precision-recall curve."""
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        ap = average_precision_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(recall, precision, label=f"AP = {ap:.3f}", linewidth=2)
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Precision-Recall Curve – {title}")
        ax.legend(loc="best")
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1])
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close(fig)
        log.info("Saved → %s", save_path)

    @staticmethod
    def _plot_comparison(df: pd.DataFrame):
        """Plot comparison across all datasets."""
        metrics = ["accuracy", "precision", "recall", "f1", "roc_auc"]
        fig, axes = plt.subplots(1, len(metrics), figsize=(18, 4))
        
        for ax, metric in zip(axes, metrics):
            df[metric].plot(kind="barh", ax=ax, color=plt.cm.Set2(np.linspace(0, 1, len(df))))
            ax.set_xlabel(metric.capitalize())
            ax.set_xlim([0, 1])
            ax.grid(axis="x", alpha=0.3)
        
        plt.suptitle("Model Evaluation Across Datasets", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(ARTIFACTS / "evaluation_comparison.png", dpi=150)
        plt.close(fig)
        log.info("Saved comparison plot → %s", ARTIFACTS / "evaluation_comparison.png")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    """Run comprehensive evaluation."""
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    
    evaluator = ModelEvaluator()
    
    # Evaluate on all datasets
    evaluator.evaluate_on_baseline()
    evaluator.evaluate_on_enriched_etherscan()
    evaluator.evaluate_on_forta()
    
    # Comparison report
    comparison_df = evaluator.compare_all_results()
    
    log.info("\n✓ Evaluation complete!")
    
    return comparison_df


if __name__ == "__main__":
    main()
