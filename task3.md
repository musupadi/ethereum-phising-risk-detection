Add one final ablation experiment without retraining the GraphSAGE encoder.

Use the exact saved V2 GraphSAGE encoder, the same 128-dimensional pre-head embeddings, the same train/test split, and the same leakage-controlled protocol.

Train a new XGBoost classifier using ONLY:
    graphsage_embedding_128

Do not include any Etherscan feature.

Name the model:
    GraphSAGE Embedding + XGBoost (No Etherscan Ablation)

Compare exactly four models:
1. XGBoost tabular Scenario 2: 26 features
2. GraphSAGE graph-only leakage-controlled: 19 graph features + 264 unique MultiDiGraph edges
3. GraphSAGE Embedding + XGBoost: 128-dimensional embedding only
4. GraphSAGE–Etherscan Late Fusion V2: 128 embedding + 7 Etherscan features

Report Accuracy, Precision, Recall, F1, ROC-AUC, PR-AUC, TN, FP, FN, TP.

Interpretation rule:
- If embedding-only XGBoost is close to 88.25%, state that the V2 improvement mostly comes from replacing the GraphSAGE linear head with XGBoost.
- If embedding-only XGBoost is materially lower than 88.25%, state that the Etherscan branch contributes measurable additional value.
- Do not modify existing V2 artifacts; save new files with suffix _ablation.