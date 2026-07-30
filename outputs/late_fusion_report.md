# GraphSAGE-Etherscan Late Fusion

## What this experiment is

This is **embedding-level late fusion (graph-tabular fusion)**. A GraphSAGE encoder
produces a graph embedding for each wallet; that embedding is concatenated with the
scaled Etherscan tabular features, and an XGBoost classifier is trained on the
concatenated vector. Fusion happens **after GraphSAGE produces the graph embeddings
and before XGBoost classifies the wallet** - it is not early feature concatenation
and it is not prediction-level (score) averaging.

## Data provenance

- **EPTransNet / MultiDiGraph remains the source of wallets and labels.** The wallet
  universe is the 8595 unique addresses of the hybrid dataset
  (1165 phishing / 7430 normal). No wallet was added
  from Etherscan and no label was altered.
- **Etherscan contributes only seven tabular enrichment features**, joined on the
  normalized (lowercased) wallet address: `created_at_ts, balance, transaction_count, activity_score, avg_tx_value, tx_frequency, unique_counterparties`.
- Split: stratified 80:20, `random_state=42` - the identical split used by Scenario 2
  (6876 train / 1719 test wallets).

## Architecture and dimensions

| Component | Dimension |
|---|---|
| GraphSAGE embedding (pre-head hidden layer) | 128 |
| Etherscan tabular features | 7 |
| **Fused vector** | **135** |

Reference: the tabular Scenario 2 feature set is 26 features
(19 MulDiGraph graph features + 7 Etherscan features).

The embedding is the output of the final `SAGEConv -> BatchNorm -> ReLU` block, taken
**before** the `Linear(128, 2)` classification head. Logits and predicted
probabilities are not used as the embedding.

## Results (held-out test set, n = 1719)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost tabular (Scenario 2) | 97.50% | 88.62% | 93.56% | 91.02% | 0.9948 | 0.9701 | 1458 | 28 | 15 | 218 |
| GraphSAGE (Scenario 2) | 96.97% | 83.90% | 96.14% | 89.60% | 0.9861 | 0.9225 | 1443 | 43 | 9 | 224 |
| GraphSAGE-Etherscan Late Fusion | 97.27% | 87.20% | 93.56% | 90.27% | 0.9921 | 0.9592 | 1454 | 32 | 15 | 218 |

The late-fusion model **does not improve** F1 over the XGBoost tabular baseline: 90.27% vs 91.02% (-0.75 pp on the held-out test set). No improvement is claimed.

## Methodological safeguards

- train/test wallet sets disjoint (|train|=6876, |test|=1719)
- all 8595 wallet_address values unique and lowercased
- train + test rows account for every dataset wallet (no wallet added/dropped)
- GraphSAGE input StandardScaler fitted on TRAIN rows only
- GraphSAGE early stopping used a train-internal val slice (fit=5500, val=1376); test nodes excluded from loss and model selection
- embeddings taken from the last SAGEConv->BatchNorm->ReLU output (pre-head, 128-d); logits/probabilities not used
- graph node order == [train wallets..., test wallets...]; embedding rows aligned by position
- missing Etherscan values zero-filled (same as Scenario 2 preprocessing)
- Etherscan StandardScaler fitted on TRAIN rows only, then applied to TEST
- embedding rows, Etherscan rows, labels and split indices refer to the same wallet in the same order

The GraphSAGE encoder used for fusion is trained transductively over the full
transaction topology (the same behaviour as the existing GraphSAGE baseline), but
**test labels are excluded from training**: only train-node labels enter the loss, and
early stopping is driven by a validation slice carved out of the training wallets.
The reported GraphSAGE Scenario 2 baseline row is reproduced exactly as the existing
pipeline defines it (which monitors test F1 for early stopping), so it stays comparable
with the previously published thesis number; the fusion encoder does not do this.

No synthetic samples were generated. Seeds: `numpy = torch = random_state = 42`.

## Files produced

- `outputs/late_fusion_metrics.json`
- `outputs/late_fusion_confusion_matrix.csv`
- `outputs/late_fusion_predictions.csv`
- `outputs/late_fusion_comparison_table.csv`
- `outputs/late_fusion_confusion_matrix.png`
- `outputs/late_fusion_roc_curve.png`
- `outputs/late_fusion_pr_curve.png`
- `outputs/late_fusion_model_comparison.png`
- `models/graphsage_etherscan_late_fusion_xgboost.pkl`
- `models/graphsage_encoder_for_fusion.pt`
- `models/late_fusion_etherscan_scaler.pkl`
