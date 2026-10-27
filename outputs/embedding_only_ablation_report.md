# Ablation: GraphSAGE Embedding + XGBoost (No Etherscan Ablation)

## Why this experiment exists

Late fusion V2 (F1 86.35%) beat the graph-only GraphSAGE baseline (F1 76.69%) by
+9.65 pp, but **two things changed at once**: the GraphSAGE linear classification head was
replaced by XGBoost, and seven Etherscan features were concatenated. That confound made the
V2 gain impossible to attribute. This ablation removes the Etherscan branch and changes
nothing else, which separates the two effects.

## Protocol

- The GraphSAGE encoder is **not retrained**. Weights are loaded from
  `models/graphsage_encoder_for_fusion_v2.pt` and run in eval mode, so the 128-d pre-head
  embeddings are identical to V2's.
- Identical split (train 10252 / test 2563, `random_state=42`), identical topology
  (354 unique MulDiGraph edges), identical leakage-controlled protocol.
- **Encoder identity is proven, not assumed**: the loaded encoder's own head predictions are
  re-scored and asserted equal - to 1e-9 on every metric - to the GraphSAGE row stored in
  `outputs/late_fusion_metrics_v2.json`.
- XGBoost input is the 128-d embedding **only**. No Etherscan feature enters this model.
- Rows 1 and 4 are read from the V2 metrics file; nothing V2 already established is recomputed.

## Results (held-out test set, n = 2563)

| # | Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost tabular (Scenario 2) | 25 tabular features (19 graph + 7 Etherscan) | 97.89% | 84.03% | 94.85% | 89.11% | 0.9938 | 0.9582 | 2288 | 42 | 12 | 221 |
| 2 | GraphSAGE graph-only (leakage-controlled) | 18 graph features + 354 EPTransNet edges | 95.16% | 68.23% | 87.55% | 76.69% | 0.9714 | 0.8161 | 2235 | 95 | 29 | 204 |
| 3 | GraphSAGE Embedding + XGBoost (No Etherscan Ablation) | 128-d graph embedding only (no Etherscan) | 95.59% | 68.99% | 93.56% | 79.42% | 0.9814 | 0.8931 | 2232 | 98 | 15 | 218 |
| 4 | GraphSAGE-Etherscan Late Fusion V2 | 128-d graph embedding || 7 scaled Etherscan features = 135-d | 97.35% | 81.13% | 92.27% | 86.35% | 0.9912 | 0.9375 | 2280 | 50 | 18 | 215 |

## Attribution

| Effect | Contribution to F1 |
|---|---|
| Replacing the GraphSAGE linear head with XGBoost (row 2 -> row 3) | **+2.73 pp** |
| Adding the 7 Etherscan features (row 3 -> row 4) | **+6.93 pp** |
| Total gain of late fusion V2 over the graph-only baseline (row 2 -> row 4) | **+9.65 pp** |
| Remaining gap of the embedding-only model to XGBoost Scenario 2 (row 3 -> row 1) | **-9.70 pp** |

## Verdict

Embedding-only XGBoost reaches F1 79.42%, which is materially lower than late fusion V2 (86.35%) by 6.93 pp. **The Etherscan branch contributes measurable additional value beyond the classifier swap.** Of the +9.65 pp total gain over the graph-only baseline, +2.73 pp is attributable to replacing the linear head with XGBoost and +6.93 pp to the seven Etherscan features.

Decision rule applied: "close to 86.35%" was operationalised before looking at the
result as **within 1.00 pp** of the late-fusion F1. Observed difference:
6.93 pp -> classified as **materially lower**.

## Leakage and alignment controls

- split reproduced identically to V2 (train=10252, test=2563, random_state=42)
- topology reproduced identically to V2 (354 unique MulDiGraph edges; etherscan_edges.csv not read)
- graph-feature StandardScaler fitted on TRAIN rows only (same as V2)
- GraphSAGE encoder loaded from the saved V2 checkpoint; not retrained, eval mode (BatchNorm running stats frozen)
- loaded encoder reproduces V2's GraphSAGE test metrics exactly -> identical encoder state confirmed
- embedding = final SAGEConv->BatchNorm->ReLU output (pre-head, 128-d); logits/probabilities not used
- no Etherscan feature enters this model - embedding is the only input
- XGBoost trained on TRAIN embeddings only; inner validation split drawn from TRAIN; test scored once
- V2 stored predictions re-aligned by wallet_address before plotting curves

Seeds: `numpy = torch = random_state = 42`. No synthetic samples. The test set is scored once.

## Files

`outputs/`: `embedding_only_ablation_metrics.json`, `four_model_comparison_ablation.csv`,
`embedding_only_confusion_matrix_ablation.csv`, `embedding_only_predictions_ablation.csv`,
`embedding_only_confusion_matrix_ablation.png`, `four_model_comparison_ablation.png`,
`roc_curve_ablation.png`, `pr_curve_ablation.png`, `embedding_only_ablation_report.md`

`models/`: `graphsage_embedding_only_xgboost_ablation.pkl`

V1 and V2 artifacts are untouched.
