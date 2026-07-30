# Ablation: GraphSAGE Embedding + XGBoost (No Etherscan Ablation)

## Why this experiment exists

Late fusion V2 (F1 88.25%) beat the graph-only GraphSAGE baseline (F1 81.89%) by
+6.35 pp, but **two things changed at once**: the GraphSAGE linear classification head was
replaced by XGBoost, and seven Etherscan features were concatenated. That confound made the
V2 gain impossible to attribute. This ablation removes the Etherscan branch and changes
nothing else, which separates the two effects.

## Protocol

- The GraphSAGE encoder is **not retrained**. Weights are loaded from
  `models/graphsage_encoder_for_fusion_v2.pt` and run in eval mode, so the 128-d pre-head
  embeddings are identical to V2's.
- Identical split (train 6876 / test 1719, `random_state=42`), identical topology
  (264 unique MulDiGraph edges), identical leakage-controlled protocol.
- **Encoder identity is proven, not assumed**: the loaded encoder's own head predictions are
  re-scored and asserted equal - to 1e-9 on every metric - to the GraphSAGE row stored in
  `outputs/late_fusion_metrics_v2.json`.
- XGBoost input is the 128-d embedding **only**. No Etherscan feature enters this model.
- Rows 1 and 4 are read from the V2 metrics file; nothing V2 already established is recomputed.

## Results (held-out test set, n = 1719)

| # | Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost tabular (Scenario 2) | 26 tabular features (19 graph + 7 Etherscan) | 97.50% | 88.62% | 93.56% | 91.02% | 0.9948 | 0.9701 | 1458 | 28 | 15 | 218 |
| 2 | GraphSAGE graph-only (leakage-controlled) | 19 graph features + 264 EPTransNet edges | 94.88% | 78.66% | 85.41% | 81.89% | 0.9714 | 0.8880 | 1432 | 54 | 34 | 199 |
| 3 | GraphSAGE Embedding + XGBoost (No Etherscan Ablation) | 128-d graph embedding only (no Etherscan) | 94.71% | 76.49% | 87.98% | 81.84% | 0.9820 | 0.9222 | 1423 | 63 | 28 | 205 |
| 4 | GraphSAGE-Etherscan Late Fusion V2 | 128-d graph embedding || 7 scaled Etherscan features = 135-d | 96.68% | 84.92% | 91.85% | 88.25% | 0.9916 | 0.9529 | 1448 | 38 | 19 | 214 |

## Attribution

| Effect | Contribution to F1 |
|---|---|
| Replacing the GraphSAGE linear head with XGBoost (row 2 -> row 3) | **-0.06 pp** |
| Adding the 7 Etherscan features (row 3 -> row 4) | **+6.41 pp** |
| Total gain of late fusion V2 over the graph-only baseline (row 2 -> row 4) | **+6.35 pp** |
| Remaining gap of the embedding-only model to XGBoost Scenario 2 (row 3 -> row 1) | **-9.19 pp** |

## Verdict

Embedding-only XGBoost reaches F1 81.84%, which is materially lower than late fusion V2 (88.25%) by 6.41 pp. **The Etherscan branch contributes measurable additional value beyond the classifier swap.** Of the +6.35 pp total gain over the graph-only baseline, -0.06 pp is attributable to replacing the linear head with XGBoost and +6.41 pp to the seven Etherscan features.

Decision rule applied: "close to 88.25%" was operationalised before looking at the
result as **within 1.00 pp** of the late-fusion F1. Observed difference:
6.41 pp -> classified as **materially lower**.

## Leakage and alignment controls

- split reproduced identically to V2 (train=6876, test=1719, random_state=42)
- topology reproduced identically to V2 (264 unique MulDiGraph edges; etherscan_edges.csv not read)
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
