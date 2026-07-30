Please create a corrected V2 experiment named:
GraphSAGE–Etherscan Late Fusion (Leakage-Controlled, Non-Redundant)

Do not overwrite the existing V1 late-fusion experiment. Create new output files with a v2 suffix.

Problems to correct from V1:
1. GraphSAGE currently receives all 26 Scenario-2 features, including the seven Etherscan features, and then the same seven Etherscan features are concatenated again before XGBoost. This creates redundant fusion.
2. The old GraphSAGE baseline monitored test F1 for early stopping. Do not do that.
3. The report mentions 528 MultiDiGraph edges plus 126 Etherscan edges. For V2, use ONLY the original 528 direct transaction edges from EPTransNet/MultiDiGraph. Do not add Etherscan-derived edges.

Required V2 design:
- Same 8,595-wallet population, labels, train/test split, and random_state=42.
- Wallets and labels must still come only from EPTransNet.
- GraphSAGE input: only the 19 graph-derived Scenario-1 input features.
- Graph topology: only the original 528 direct EPTransNet/MultiDiGraph edges.
- Train GraphSAGE with an internal validation split taken only from training data.
- Never monitor test F1, test loss, test ROC-AUC, or test PR-AUC for early stopping or model selection.
- Extract the final 128-dimensional hidden GraphSAGE embedding before the classification head.
- Etherscan branch: only the seven Etherscan features, zero-filled consistently; fit StandardScaler on training data only.
- Fusion vector:
  fused_i = [graphsage_embedding_128 || scaled_etherscan_features_7]
  final dimension = 135.
- Train XGBoost only on fused training vectors.
- Test only once on the held-out 1,719-wallet test set.

Also retrain a leakage-controlled GraphSAGE baseline:
- same graph-only 19 features,
- same 528 MultiDiGraph edges,
- same train/internal-validation protocol,
- report its held-out test metrics.

Keep the original published XGBoost Scenario-2 baseline as reference:
Accuracy 97.50%, Precision 88.62%, Recall 93.56%, F1 91.02%.

Save a comparison table:
Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP

Compare:
1. XGBoost Scenario 2 tabular baseline
2. Leakage-controlled GraphSAGE graph-only baseline
3. GraphSAGE–Etherscan Late Fusion V2

Generate:
- confusion matrices
- ROC curve
- PR curve
- F1/precision/recall comparison chart
- edge-source audit
- feature-source audit
- detailed markdown report explaining whether fusion improves XGBoost and/or GraphSAGE.

Do not claim improvement unless the held-out test metrics support it.