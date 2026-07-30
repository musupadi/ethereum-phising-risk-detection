# GraphSAGE-Etherscan Late Fusion V2 (Leakage-Controlled, Non-Redundant)

## What changed relative to V1

| # | V1 problem | V2 correction |
|---|---|---|
| 1 | GraphSAGE received all 26 Scenario-2 features (including the 7 Etherscan ones), which were then concatenated again before XGBoost - redundant fusion | GraphSAGE input is restricted to the **19 graph-derived (Scenario-1) features**. The 7 Etherscan features enter the model exactly once, in the tabular branch. |
| 2 | The GraphSAGE baseline monitored test F1 for early stopping | **No test signal** (F1, loss, ROC-AUC, PR-AUC) is used for early stopping or model selection by any model. Early stopping uses a validation slice drawn from training wallets only. |
| 3 | Topology mixed MulDiGraph and Etherscan-derived edges | Topology uses **only direct EPTransNet/MulDiGraph transaction edges**; `data/raw/etherscan_edges.csv` is never read. |

## Design

- Population: **8595 wallets** (1165 phishing / 7430 normal). Wallets and labels come **only from EPTransNet/MulDiGraph**.
- Split: stratified 80:20, `random_state=42` -> 6876 train / 1719 test - identical to Scenario 2.
- Fusion: `fused_i = [ graphsage_embedding_128 || scaled_etherscan_7 ]` = **135-d**.
- The embedding is the output of the final `SAGEConv -> BatchNorm -> ReLU`, taken before `Linear(128, 2)`. Logits and probabilities are not used.
- XGBoost is trained on fused **training** vectors only; the test set is scored once.

The GraphSAGE baseline row and the fusion row share **the same trained encoder**: the baseline is that encoder's own classification head, the fusion model is XGBoost over its pre-head embedding. The only variable that changes between the two rows is the classifier (plus the Etherscan block) - which is what makes the comparison a clean ablation of the fusion step itself.

## Edge-source audit

| Item | Value |
|---|---|
| Sources used | MulDiGraph.pkl (EPTransNet) |
| Sources excluded | `data/raw/etherscan_edges.csv` (excluded by V2 design) |
| Dataset addresses present in MulDiGraph | 8595 / 8595 |
| Raw directed edge entries | 528 |
| Unique directed edges passed to the GNN | **264** |
| ├─ train-train | 150 |
| ├─ train-test | 95 |
| └─ test-test | 19 |
| Self-loops added by the PyG builder | 8595 |

The raw entry count double-counts each edge (once from each endpoint's adjacency scan); the deduplicated figure is what the model actually sees. V1's "528 MulDiGraph + 126 Etherscan" figures were raw entry counts of the merged sources.

## Feature-source audit

| Branch | Source | Count | Features |
|---|---|---|---|
| GraphSAGE encoder input | MulDiGraph (EPTransNet) | 19 | `in_degree`, `out_degree`, `unique_senders`, `unique_receivers`, `total_received`, `total_sent`, `avg_in_amount`, `avg_out_amount`, `max_in_amount`, `max_out_amount`, `min_in_amount`, `min_out_amount`, `std_in_amount`, `std_out_amount`, `in_out_ratio`, `sent_received_ratio`, `timestamp_spread_in`, `timestamp_spread_out`, `graph_present` |
| Tabular branch (post-embedding) | Etherscan (enrichment) | 7 | `created_at_ts`, `balance`, `transaction_count`, `activity_score`, `avg_tx_value`, `tx_frequency`, `unique_counterparties` |

The two sets are disjoint - asserted at runtime - so no feature is consumed twice. Full listing: `outputs/feature_source_audit_v2.csv`.

## Results (held-out test set, n = 1719)

| Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost tabular (Scenario 2) | 26 tabular features (19 graph + 7 Etherscan) | 97.50% | 88.62% | 93.56% | 91.02% | 0.9948 | 0.9701 | 1458 | 28 | 15 | 218 |
| GraphSAGE graph-only (leakage-controlled) | 19 graph features + 264 EPTransNet edges | 94.88% | 78.66% | 85.41% | 81.89% | 0.9714 | 0.8880 | 1432 | 54 | 34 | 199 |
| GraphSAGE-Etherscan Late Fusion V2 | 128-d graph embedding || 7 scaled Etherscan features = 135-d | 96.68% | 84.92% | 91.85% | 88.25% | 0.9916 | 0.9529 | 1448 | 38 | 19 | 214 |

### Does fusion improve on XGBoost?

The late-fusion V2 model **does not improve** on the XGBoost Scenario 2 tabular baseline (88.25% vs 91.02%, -2.78 pp on the held-out test set); no improvement is claimed.

### Does fusion improve on GraphSAGE?

The late-fusion V2 model **improves** F1 over the leakage-controlled graph-only GraphSAGE baseline (88.25% vs 81.89%, +6.35 pp on the held-out test set).

### Where does the fused model get its signal?

Of the 134 fused dimensions the trees actually split on, the Etherscan block accounts for **1.8%** of total gain and the graph embedding for **98.2%**.

| Feature | Gain share |
|---|---|
| `sage_emb_89` | 39.8% |
| `sage_emb_107` | 11.1% |
| `sage_emb_127` | 10.8% |
| `sage_emb_19` | 4.5% |
| `sage_emb_4` | 1.9% |
| `sage_emb_65` | 1.3% |
| `sage_emb_41` | 1.1% |
| `sage_emb_125` | 0.9% |
| `sage_emb_98` | 0.8% |
| `sage_emb_60` | 0.8% |

Because the encoder no longer sees the Etherscan features, this split is a genuine measure of what each modality contributes - unlike V1, where the Etherscan block was largely redundant with information already baked into the embedding.

## Leakage and alignment controls

- non-redundancy: GraphSAGE input (19 graph features) and the fused Etherscan block (7 features) are disjoint - no feature enters twice
- wallet population = 8,595 unique lowercased EPTransNet addresses; no wallet added or dropped
- stratified 80:20 split, random_state=42 -> train=6876, test=1719 (identical to Scenario 2)
- topology built from MulDiGraph only (264 unique directed edges); etherscan_edges.csv never read
- graph-feature StandardScaler fitted on TRAIN rows only
- early stopping driven by a train-internal validation slice (fit=5500, val=1376); test F1/loss/ROC-AUC/PR-AUC never monitored
- embedding = output of the final SAGEConv->BatchNorm->ReLU (pre-head, 128-d); logits/probabilities not used
- graph node order == [train wallets..., test wallets...]; embedding rows aligned by position
- Etherscan values zero-filled (same as Scenario 2); StandardScaler fitted on TRAIN only
- embedding rows, Etherscan rows, labels and split indices refer to the same wallet in the same order
- published XGBoost Scenario 2 baseline reproduced exactly (Acc 97.50 / Prec 88.62 / Rec 93.56 / F1 91.02)

The encoder is trained transductively over the full topology (unchanged from the existing baseline), but **no test label ever enters training**: only training-node labels contribute to the loss, and model selection is driven by the train-internal validation slice. `GraphPhiV2.fit` names its monitoring mask `test_mask`; V2 passes the validation mask there, so the `test_f1` value in the training log is validation F1.

No synthetic samples were generated. Class imbalance is handled by weighted cross-entropy (GNN) and `scale_pos_weight` (XGBoost). Seeds: `numpy = torch = random_state = 42`.

## Files

`outputs/`: `late_fusion_metrics_v2.json`, `late_fusion_comparison_table_v2.csv`, `late_fusion_confusion_matrix_v2.csv`, `confusion_matrices_all_v2.csv`, `late_fusion_predictions_v2.csv`, `edge_source_audit_v2.json`, `feature_source_audit_v2.csv`, `late_fusion_confusion_matrix_v2.png`, `cm_graphsage_graph_only_v2.png`, `cm_xgboost_scenario2_v2.png`, `late_fusion_roc_curve_v2.png`, `late_fusion_pr_curve_v2.png`, `late_fusion_model_comparison_v2.png`, `late_fusion_report_v2.md`

`models/`: `graphsage_encoder_for_fusion_v2.pt`, `graphsage_etherscan_late_fusion_xgboost_v2.pkl`, `late_fusion_etherscan_scaler_v2.pkl`

V1 artifacts (no `_v2` suffix) are untouched.
