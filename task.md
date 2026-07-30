Task: Add a new experimental model called “GraphSAGE–Etherscan Late Fusion” to the existing Ethereum phishing-wallet detection project.

Important context:
- The existing main hybrid dataset has 8,595 unique wallet addresses:
  - 1,165 phishing
  - 7,430 normal
- The authoritative wallet universe and labels come from EPTransNet/MultiDiGraph.
- Etherscan is feature enrichment only, joined by normalized wallet_address.
- Do NOT add wallets from Etherscan and do NOT alter labels.
- The current best tabular model is XGBoost using MultiDiGraph + Etherscan features:
  Accuracy 97.50%, Precision 88.62%, Recall 93.56%, F1 91.02%.
- Existing GraphSAGE result on the same Scenario 2:
  Accuracy 96.97%, Precision 83.90%, Recall 96.14%, F1 89.60%.
- Use the identical stratified 80:20 split with random_state=42 used by Scenario 2.
- Keep all existing experiments unchanged. Implement this as a NEW experiment and new output files only.

Goal:
Implement embedding-level / late graph-tabular fusion:
1. Train the existing GraphSAGE model to learn a hidden graph embedding z_i for every wallet.
2. Extract the hidden embedding BEFORE the final GraphSAGE classification layer.
   - Do not use final logits or predicted probabilities as the embedding.
   - Prefer 128-dimensional hidden embedding if this is the existing hidden dimension.
3. Take the 7 Etherscan tabular features for the same wallet.
4. Fit StandardScaler ONLY on training-set Etherscan features, then transform validation/test features.
5. Concatenate:
   fused_vector_i = [GraphSAGE_embedding_i || scaled_Etherscan_features_i]
6. Train a new XGBoost classifier using fused_vector_i on TRAINING wallets only.
7. Evaluate it on the held-out TEST wallets only.

Methodological safeguards:
- Wallet addresses must be normalized to lowercase before alignment.
- Assert that graph embedding rows, Etherscan rows, labels, and train/test indices refer to the same wallet_address in the same order.
- Do not use test labels during GraphSAGE training, early stopping, hyperparameter tuning, or XGBoost fitting.
- If the existing GraphSAGE pipeline is transductive and uses the complete graph topology, preserve this behavior only if it is already used by the baseline GraphSAGE. Clearly document that test labels are excluded from training.
- Do not generate synthetic samples.
- Handle missing Etherscan values exactly like the existing Scenario 2 preprocessing, preferably zero-fill according to the current pipeline.
- Use reproducible seeds: random_state / torch seed = 42.

Deliverables:
A. New executable experiment file, for example:
   experiments/run_graphsage_etherscan_late_fusion.py

B. Save these artifacts:
   outputs/late_fusion_metrics.json
   outputs/late_fusion_confusion_matrix.csv
   outputs/late_fusion_predictions.csv
   models/graphsage_etherscan_late_fusion_xgboost.pkl
   models/graphsage_encoder_for_fusion.pt
   models/late_fusion_etherscan_scaler.pkl

C. Print and save a comparison table with:
   Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | TN | FP | FN | TP

Include at least:
   1. XGBoost tabular Scenario 2 baseline
   2. GraphSAGE Scenario 2 baseline
   3. New GraphSAGE–Etherscan Late Fusion model

D. Produce:
   - confusion matrix PNG
   - ROC curve PNG
   - precision-recall curve PNG
   - bar chart comparing F1-score, precision, and recall across the three models
   - feature/embedding dimension report:
     GraphSAGE embedding dimension
     Etherscan feature count
     final fused-vector dimension

E. Add a concise Markdown report:
   outputs/late_fusion_report.md

The report must state:
- This is embedding-level late fusion / graph-tabular fusion.
- EPTransNet/MultiDiGraph remains the source of wallets and labels.
- Etherscan contributes only seven tabular enrichment features.
- Fusion happens after GraphSAGE produces graph embeddings and before XGBoost classifies the wallet.
- Whether the late-fusion model improves or does not improve F1 relative to XGBoost 91.02%.
- Do not claim improvement unless supported by the held-out test result.

Before coding:
1. Inspect the existing project structure and identify the current Scenario 2 dataset builder, GraphSAGE training code, XGBoost training code, feature list, and split logic.
2. Reuse existing preprocessing and split logic rather than creating a second incompatible pipeline.
3. Briefly report the implementation plan and exact input/output shapes.

After coding:
1. Run the experiment.
2. Report all saved file paths.
3. Report any data-alignment or leakage checks performed.
4. Do not modify thesis, journal, or existing baseline metrics.