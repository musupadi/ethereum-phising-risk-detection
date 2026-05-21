"""
ml/xgboost_model.py
-------------------
XGBoost model for phishing detection with hyperparameter tuning.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

# ── Settings ───────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost model for binary classification with class balancing."""

    def __init__(
        self,
        n_estimators: int = 300,
        max_depth: int = 8,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        random_state: int = 42,
    ):
        """
        Initialize XGBoost model.
        
        Args:
            n_estimators: Number of boosting rounds
            max_depth: Maximum depth of trees
            learning_rate: Learning rate (shrinkage)
            subsample: Subsample ratio of training instances
            colsample_bytree: Subsample ratio of features
            random_state: Random seed
        """
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.random_state = random_state
        
        self.model = None
        self.feature_names = None
        self.feature_importance = None
        
        log.info(
            "Initialized XGBoost: estimators=%d, depth=%d, lr=%.3f",
            n_estimators, max_depth, learning_rate,
        )

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        feature_names: list = None,
        early_stopping_rounds: int = 50,
    ) -> Dict:
        """
        Train XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            feature_names: Feature names (optional)
            early_stopping_rounds: Early stopping patience
            
        Returns:
            Training results dictionary
        """
        self.feature_names = feature_names
        
        # Compute scale_pos_weight for imbalanced data
        scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
        
        # Prepare DMatrix
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
        
        eval_set = []
        if X_val is not None and y_val is not None:
            dval = xgb.DMatrix(X_val, label=y_val, feature_names=feature_names)
            eval_set = [(dtrain, "train"), (dval, "val")]
        else:
            eval_set = [(dtrain, "train")]
        
        # Training parameters
        params = {
            "objective": "binary:logistic",
            "eval_metric": "logloss",
            "max_depth": self.max_depth,
            "learning_rate": self.learning_rate,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "scale_pos_weight": scale_pos_weight,
            "random_state": self.random_state,
            "tree_method": "hist",  # Fast histogram-based learning
            "device": "cuda" if _has_gpu() else "cpu",
        }
        
        log.info("Training XGBoost with scale_pos_weight=%.2f", scale_pos_weight)
        
        # Train
        evals_result = {}
        self.model = xgb.train(
            params,
            dtrain,
            num_boost_round=self.n_estimators,
            evals=eval_set,
            evals_result=evals_result,
            early_stopping_rounds=early_stopping_rounds if eval_set else None,
            verbose_eval=50,
        )
        
        # Store feature importance
        self.feature_importance = self.model.get_score(importance_type="weight")
        
        log.info(f"Training complete. Best iteration: {self.model.best_iteration}")
        
        return evals_result

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.
        
        Args:
            X: Input features
            
        Returns:
            Predicted labels (0 or 1)
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        proba = self.model.predict(dmatrix)
        return (proba > 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict probability of phishing class.
        
        Args:
            X: Input features
            
        Returns:
            Probability scores (shape: [n_samples, 2])
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        dmatrix = xgb.DMatrix(X, feature_names=self.feature_names)
        proba_pos = self.model.predict(dmatrix)
        proba_neg = 1 - proba_pos
        
        return np.column_stack([proba_neg, proba_pos])

    def evaluate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        name: str = "XGBoost",
    ) -> Dict:
        """
        Evaluate model on test set.
        
        Args:
            X: Test features
            y: Test labels
            name: Name for logging
            
        Returns:
            Metrics dictionary
        """
        y_pred = self.predict(X)
        y_proba = self.predict_proba(X)[:, 1]
        
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred, zero_division=0)
        rec = recall_score(y, y_pred, zero_division=0)
        f1 = f1_score(y, y_pred, zero_division=0)
        
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

    def get_feature_importance(self, top_n: int = 15) -> pd.DataFrame:
        """
        Get top N most important features.
        
        Args:
            top_n: Number of top features to return
            
        Returns:
            DataFrame with feature importances
        """
        if self.feature_importance is None:
            log.warning("No feature importance available")
            return pd.DataFrame()
        
        importance_list = [
            {"feature": k, "importance": v}
            for k, v in sorted(self.feature_importance.items(), 
                             key=lambda x: x[1], reverse=True)[:top_n]
        ]
        
        return pd.DataFrame(importance_list)

    def save(self, path: Path):
        """Save model to disk."""
        if self.model is None:
            raise ValueError("No model to save. Train first.")
        
        self.model.save_model(str(path))
        log.info("Model saved → %s", path)

    def load(self, path: Path):
        """Load model from disk."""
        self.model = xgb.Booster()
        self.model.load_model(str(path))
        self.feature_importance = self.model.get_score(importance_type="weight")
        log.info("Model loaded ← %s", path)


def _has_gpu() -> bool:
    """Check if GPU is available for XGBoost."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


# ── Hyperparameter tuning ──────────────────────────────────────────────────────

def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: list = None,
    n_trials: int = 20,
) -> Dict:
    """
    Tune XGBoost hyperparameters using simple grid search.
    
    Args:
        X_train: Training features
        y_train: Training labels
        X_val: Validation features
        y_val: Validation labels
        feature_names: Feature names
        n_trials: Number of trials (note: this is simplified)
        
    Returns:
        Best parameters and results
    """
    try:
        import optuna
    except ImportError:
        log.warning("Optuna not installed. Using default parameters.")
        model = XGBoostModel()
        model.fit(X_train, y_train, X_val, y_val, feature_names)
        metrics = model.evaluate(X_val, y_val)
        return {"params": vars(model), "best_f1": metrics["f1"]}
    
    def objective(trial):
        params = {
            "max_depth": trial.suggest_int("max_depth", 4, 12),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
        }
        
        model = XGBoostModel(**params)
        model.fit(X_train, y_train, X_val, y_val, feature_names, early_stopping_rounds=30)
        
        metrics = model.evaluate(X_val, y_val, name=f"Trial_{trial.number}")
        return metrics["f1"]
    
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=n_trials)
    
    log.info("Best F1: %.4f", study.best_value)
    log.info("Best params: %s", study.best_params)
    
    return {
        "best_params": study.best_params,
        "best_f1": study.best_value,
        "trials": len(study.trials),
    }
