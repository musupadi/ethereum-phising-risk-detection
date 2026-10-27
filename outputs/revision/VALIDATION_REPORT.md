# Validation report — revision v2 (IJECE Paper ID 42848)

| Item | Value |
|---|---|
| Code commit at run time | `990a8a0`, plus the uncommitted seed fix in `ml/train_dataset_scenarios.py` (see below) |
| `experiments/revision_ijece.py` | 27 Oct 2026, 12:38 → 12:54 (921 s), exit 0 |
| `experiments/revision_extra.py` | 27 Oct 2026, 12:54 → 12:58 (211 s), exit 0 |
| `ml/train_dataset_scenarios.py`, late fusion V2, embedding ablation | 13:00 → 13:09, exit 0; scenario script run twice, identical output |
| `experiments/revision_ops.py --no-live --eth-usd 2313.76` | 13:09; live Etherscan results kept from the earlier same-day measurement (no new API calls) |
| `scripts/build_final_metrics.py`, `experiments/ijece_final_audit.py` | 13:09–13:10 |
| `scripts/validate_revision.py` | **22 PASS, 0 FAIL, 0 BLOCKED** (`validation_results.json`) |
| Environment | Windows 11, CPU, Python 3.13.5, numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.11.0+cpu, torch_geometric 2.7.0 |

## Validator checks

| Check | Result |
|---|---|
| dataset rows = 12,815; labels 1,165 / 11,650; no duplicate addresses; required columns; no NaN in predictors | PASS |
| scenario sizes 18 / 25 / 28 / 35; feature groups 18 / 7 / 10; `graph_present` never a predictor | PASS |
| every per-seed confusion matrix sums to 2,563 with 233 phishing | PASS |
| precision / recall / F1 recomputed from TN / FP / FN / TP | PASS |
| 10 seeds (42, 0–8) for every scenario × model; summary = mean of per-seed rows | PASS |
| test-monitored GraphSAGE decomposition rows flagged invalid; main GraphSAGE monitors the validation mask | PASS |
| fit / early-stopping / calibration slices partition the training split (6,151 + 2,050 + 2,051 = 10,252) | PASS |
| production booster feature order = prod schema = S2 order | PASS |
| production F1 recomputed on the seed-42 test split = schema metric (0.8911) = seed-42 XGBoost S2 in the 10-seed table | PASS |
| production booster `best_iteration` = last round (so the H-1 fix does not change it) | PASS |

No check was BLOCKED: `MulDiGraph.pkl` is present on this machine, so its hash was recorded. On a
machine without it, the hash check reports BLOCKED (third-party file, not redistributed).

## Artifact hashes (SHA-256)

| File | SHA-256 |
|---|---|
| `ml/artifacts/model_prod_xgb.json` | `33962304efd5ede75ab5c25815311e07bfa99cb1c4932f2a6c2decc4dcc35c11` |
| `ml/artifacts/prod_feature_schema.json` | `1963e8c74bb2735edc971067b4bbf7831df842c5d99d9c7642ac0b7f0c669c2f` |
| `ml/artifacts/hybrid_feature_schema.json` | `e8e8d09a6e3de7b49fd9ac01364148fda52f778b41e8c7fe4542b4b6522d3a4b` |
| `data/processed/hybrid_dataset.csv` | `5a6085a367a42c03a05aa27596c4ea7e76ea47d173102fcff6929639107dcbe3` |
| `data/raw/MulDiGraph.pkl` | `155fbbd738a7855a194a3c975711412b48d6d483f6387e47638832d999dc8d14` |

## Production model identity

`model_prod_xgb.json` (unchanged since commit `ac57cb7`, 6 Aug 2026) is produced by
`ml/train_production_model.py`:
- data: `hybrid_dataset.csv`, S2 (25 features, order as in `prod_feature_schema.json`);
- split: stratified 80:20 with `random_state=42`;
- fitted on the 80% inner slice of the training split (8,201 wallets), with early stopping on the remaining
  2,051 wallets;
- `objective=binary:logistic`, 250 rounds, `max_depth=6`, `learning_rate=0.05`, `subsample=0.85`,
  `colsample_bytree=0.85`, `scale_pos_weight=9.993`, `tree_method=hist`, `random_state=42`;
- `best_iteration = 249`, so all 250 trees are used;
- held-out metrics on 2,563 wallets: accuracy 0.9789, precision 0.8403, recall 0.9485, F1 0.8911.

It is the seed-42 member of the 10-seed XGBoost S2 evaluation. The 10-seed mean (F1 0.9007 ± 0.0132)
describes the *training procedure*, not this artifact. Scores are uncalibrated (`round(p × 100)`). Streamlit
calls the FastAPI service, which loads only this artifact.

## Outputs produced by v2 (`outputs/revision/`)

| Group | Files |
|---|---|
| Main 4 × 4 + fusion, 10 seeds | `multiseed_all_runs.csv`, `multiseed_summary.csv`, `seed42_table.csv` |
| GraphSAGE protocol | `graphsage_protocol_decomposition.csv` (rows A/B flagged invalid) |
| Calibration | `calibration_all_runs.csv`, `calibration_summary.csv`, `calibration_split_sizes.csv`, `reliability_diagram.png` |
| Cost-sensitive threshold | `cost_threshold_all_runs.csv`, `cost_threshold_summary.csv` |
| Majority ratio | `majority_ratio_all_runs.csv`, `majority_ratio_summary.csv` |
| Temporal split | `temporal_split_all_runs.csv`, `temporal_split_summary.csv` |
| Prevalence | `prevalence_sensitivity_all_runs.csv`, `prevalence_sensitivity_summary.csv`, `prevalence_prior_shift_estimate.csv` |
| Feature selection (train-only CV) | `feature_selection_cv_all_runs.csv`, `feature_selection_cv_summary.csv`, `feature_selection_cv_selected.json` |
| Operations | `ops_metrics.json`; `isolated_wallets.json` (unchanged from v1, identical) |
| Consolidated | `FINAL_VERIFIED_METRICS.csv` (468 rows, each with source file and validity flag), `validation_results.json`, `run_meta.json` |
| Logs | `run.log`, `run_extra.log`, `train_dataset_scenarios.log`, `late_fusion_v2.log`, `ablation.log` |

Also regenerated: `outputs/late_fusion_*_v2.*`, `outputs/embedding_only_*`, `outputs/four_model_comparison_ablation.*`,
`ml/artifacts/dataset_scenarios_comparison.csv` and its confusion matrices, and the evidence tables in
`outputs/ijece_final_audit/`.

v1 outputs are archived unchanged in `outputs/archive/revision_v1_2026-10-27/` and
`outputs/archive/ijece_final_audit_v1_2026-10-27/`.

## Changed results v1 → v2 (10-seed means)

| Item | v1 | v2 | Cause |
|---|---|---|---|
| XGBoost S1 F1 | 0.8809 ± 0.0147 | 0.8823 ± 0.0138 | H-1 |
| XGBoost S2 F1 | 0.9003 ± 0.0128 | 0.9007 ± 0.0132 | H-1 |
| XGBoost S2 recall / FN | 0.9373 / 14.6 | 0.9391 / 14.2 | H-1 |
| XGBoost S3 / S4 F1 | 0.8794 / 0.9006 | 0.8797 / 0.9004 | H-1 |
| Late fusion F1 | 0.8769 ± 0.0175 | 0.8743 ± 0.0181 | H-1 |
| Embedding-only F1 | 0.8079 ± 0.0225 | 0.8064 ± 0.0234 | H-1 |
| GraphSAGE-inductive-strict | — | 0.8655 ± 0.0203 | new |
| S2 > S1 for XGBoost | 10 / 10 seeds | **8 / 10** seeds | H-1 |
| RF higher F1 than XGBoost on S2 | 7 / 10, Wilcoxon p = 0.16 | 6 / 10, p = 0.43 | H-1 |
| Calibration, uncalibrated ECE → Platt ECE | 0.0136 → 0.0065 | 0.0146 → 0.0069 | H-2 (new model and slice) |
| Cost threshold, mean test-cost change vs 0.5 (c = 1/2/5/10/20) | −5.2 / +1.8 / +1.8 / −8.5 / −30.5% | −8.6 / **+1.8** / −1.0 / −9.4 / −24.5% | H-2 |
| Seeds where the selected threshold is worse on test (c = 1/2/5/10/20) | 2/4/5/1/0 | 2/4/3/4/1 | H-2 |
| Majority ratio F1 1:2 / 1:5 / 1:10 | 0.948 / 0.929 / 0.902 (5 seeds) | 0.948 / 0.930 / 0.898 (10 seeds) | H-1 + 10 seeds |
| `train_dataset_scenarios.py` GraphSAGE S1 / S2 (seed 42, single split) | 0.7649 / 0.8503 (unseeded) | 0.7669 / 0.8617 (seeded, reproducible) | M-3 |
| Late fusion V2 single split F1 | 0.8635 | 0.8617 | H-1 |
| Production model metrics | 0.8911 | 0.8911 | unchanged |

Unchanged: LR, RF and every GraphSAGE row of the 10-seed table, the GraphSAGE decomposition, the isolation
audit, the dataset audit, BigQuery coverage.
