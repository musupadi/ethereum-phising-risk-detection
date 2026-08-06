"""
ml/train_production_model.py
----------------------------
Train and persist the *production* model served by the API / Streamlit demo:
XGBoost on the best thesis scenario = MulDiGraph + Etherscan (18 graph-derived +
7 Etherscan-derived = 25 predictors; see hasil_implementasi.md §5.5, scenario
`muldigraph_plus_etherscan`). `graph_present` is a join-quality audit flag, not
a predictor — it is excluded from the feature list by the hybrid schema.

It reproduces the exact split/config used in ml/train_dataset_scenarios.py so the
served model matches the reported numbers, then writes:
  - ml/artifacts/model_prod_xgb.json   – XGBoost booster (servable)
  - ml/artifacts/prod_feature_schema.json – 25 feature names (ordered) + metadata
  - data/processed/prod_demo.csv       – held-out test rows (node_id + label + 25 feats)
                                         used for Streamlit "Demo (row_id)" mode

Run:  python ml/train_production_model.py
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from xgboost_model import XGBoostModel

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
ARTIFACTS = ROOT / "ml" / "artifacts"
PROCESSED = ROOT / "data" / "processed"
HYBRID_CSV = PROCESSED / "hybrid_dataset.csv"
HYBRID_SCHEMA = ARTIFACTS / "hybrid_feature_schema.json"

SCENARIO = "muldigraph_plus_etherscan"  # best scenario by F1
MODEL_OUT = ARTIFACTS / "model_prod_xgb.json"
SCHEMA_OUT = ARTIFACTS / "prod_feature_schema.json"
DEMO_OUT = PROCESSED / "prod_demo.csv"


def main() -> None:
    df = pd.read_csv(HYBRID_CSV)
    schema = json.loads(HYBRID_SCHEMA.read_text(encoding="utf-8"))
    feature_cols = schema["scenarios"][SCENARIO]
    log.info("Loaded %d rows; scenario '%s' with %d features", len(df), SCENARIO, len(feature_cols))

    # Same split as ml/train_dataset_scenarios.py so metrics match the thesis.
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["label"],
    )
    X_train = train_df[feature_cols].fillna(0).values
    y_train = train_df["label"].values
    X_test = test_df[feature_cols].fillna(0).values
    y_test = test_df["label"].values

    # Identical XGBoost config + internal validation split (train_xgb in scenarios).
    X_fit, X_val, y_fit, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train,
    )
    model = XGBoostModel(
        n_estimators=250, max_depth=6, learning_rate=0.05,
        subsample=0.85, colsample_bytree=0.85,
    )
    model.fit(X_fit, y_fit, X_val, y_val, feature_names=feature_cols, early_stopping_rounds=40)

    metrics = model.evaluate(X_test, y_test, name="XGBoost-prod (MulDiGraph+Etherscan)")
    log.info("Test metrics: %s", {k: round(v, 4) for k, v in metrics.items() if k != "model"})

    # ── Persist servable artifacts ────────────────────────────────────────────
    model.save(MODEL_OUT)
    SCHEMA_OUT.write_text(json.dumps({
        "features": feature_cols,
        "audit_columns": schema.get("audit_columns", ["graph_present"]),
        "label": "label",
        "node_id": "node_id",
        "model_type": "xgboost",
        "scenario": SCENARIO,
        "metrics": {k: round(float(v), 4) for k, v in metrics.items() if k != "model"},
    }, indent=2), encoding="utf-8")
    log.info("Schema saved → %s", SCHEMA_OUT)

    # Held-out test rows for the Streamlit demo (so demo scores unseen wallets).
    demo_cols = ["node_id", "label"] + feature_cols
    test_df[demo_cols].reset_index(drop=True).to_csv(DEMO_OUT, index=False)
    log.info("Demo set saved → %s (%d rows, %d phishing)", DEMO_OUT, len(test_df), int(y_test.sum()))


if __name__ == "__main__":
    main()
