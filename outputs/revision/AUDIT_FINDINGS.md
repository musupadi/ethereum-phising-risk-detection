# Audit findings — IJECE revision (Paper ID 42848)

Audit of 27 Oct 2026. Code base: commit `990a8a0` plus the uncommitted changes listed in
`VALIDATION_REPORT.md`. Status: **FIXED** = code changed and affected experiments re-run (v2);
**DOCUMENTED** = cannot be removed and is reported as a limitation; **NOT REPORTABLE** = output
must not be used in the paper.

## Critical

| ID | Finding | Evidence | Status |
|---|---|---|---|
| C-1 | The submitted Table 3 GraphSAGE (F1 0.8960) stopped training on **test-set F1**. On the revised data this inflates GraphSAGE F1 by 2.1 points (0.853 → 0.874, seed 42). | `graphsage_protocol_decomposition.csv` (rows A/B flagged `valid_performance_estimate = False`) | FIXED in v1; every reported GraphSAGE now stops on a train-internal validation slice |
| C-2 | The submitted 8,595-wallet population was an uncommitted intermediate enrichment state and **cannot be re-run**. All submitted numbers are therefore superseded, not reproduced. | `reconciliation_submitted_vs_revised.csv`; git `6f98c6c`, `b632480` | DOCUMENTED (revised population = 12,815, exactly reproducible) |

## High

| ID | Finding | Evidence | Status |
|---|---|---|---|
| H-1 | **XGBoost early stopping had no effect on predictions.** `xgb.train` returns the last-round booster (best + 40) and `Booster.predict` uses all trees by default. | toy reproduction in this audit; `ml/xgboost_model.py` | FIXED (v2): `iteration_range=(0, best_iteration+1)`. The production artifact is unaffected (best_iteration = 249 = last round; predictions verified identical). XGBoost S2 F1 0.9003 → 0.9007; fusion 0.8769 → 0.8743 |
| H-2 | Calibration and threshold selection reused the **same validation slice** that drove early stopping. | v1 `revision_ijece.py` | FIXED (v2): dedicated XGBoost, fit 60% / early stopping 20% / calibration-threshold 20% of train (6,151 / 2,050 / 2,051) |
| H-3 | `ml/tune_gnn.py` used test F1 for early stopping **and** for choosing the configuration. | `ml/artifacts/gnn_tuning_results.json` (`"protocol": "... early stopping pada test_mask"`) | NOT REPORTABLE. Its best configuration (3 layers, lr 0.02) was never adopted; the defaults used (128 / 2 / 0.01 / 0.5) exist since commit `a71c3dc` (21 May 2026), before the script (`2d5a437`, 30 Jul 2026). How those defaults were originally chosen is not documented |
| H-4 | `ml/feature_selection.py` and `ml/smote_experiment.py` pick the "best" variant by **test F1**. | source; `feature_selection_report.json → best_by_f1` | NOT REPORTABLE as selection. Replaced by train-only 5-fold CV selection (`feature_selection_cv_*`): CV picks `full` 5/10, `corr_pruned` 3/10, `top_20` 2/10; test F1 of the CV-selected subset 0.9005 ± 0.0132 vs all features 0.9020 ± 0.0122, so selection gives no gain |
| H-5 | **Prevalence dependence.** At 1:10 XGBoost and RF tie on F1, but with more normals (S1 only) XGBoost precision collapses: F1 at 1:100 XGBoost 0.593 vs RF 0.741 (precision 0.445 vs 0.762). The analytic prior-shift estimate puts precision at the real MulDiGraph prevalence (0.039%) at about 2–3%. | `prevalence_sensitivity_summary.csv`, `prevalence_prior_shift_estimate.csv` | DOCUMENTED. The 1:10 results must not be read as real-world precision. The choice of XGBoost (driven by `scale_pos_weight = N_neg/N_pos`) is not robust to prevalence |
| H-6 | **Temporal mismatch between label/graph and enrichment.** Graph features and labels describe 2015-08 … 2019-01; Etherscan attributes were collected in 2026 and may contain post-label information (e.g., accounts emptied after being flagged). | `dataset_audit.json` (`graph_time_range_utc`), BigQuery 2026 window | DOCUMENTED. Not testable without historical snapshots. The temporal split (below) addresses distribution shift only |
| H-7 | **Sampling bias.** Phishing and normal wallets differ strongly in degree (median 13 vs 1), an artefact of how EPTransNet was crawled; part of the performance may reflect activity level rather than phishing behaviour. | `ml/artifacts/graph_profile/graph_stats.json`; `hybrid_dataset.csv` | DOCUMENTED |

## Medium

| ID | Finding | Status |
|---|---|---|
| M-1 | The 18 graph features are built on the **full** MulDiGraph before splitting (transductive construction). They are label-free (labels = `isp`, used only for the target). GraphSAGE inductive variants show no advantage of the transductive graph: test-edge removal +0.004 ± 0.029 F1 (Wilcoxon p = 0.77); strict (fit-only edges) +0.008 ± 0.030 (p = 0.63). Low power: 354 edges, 96.6% isolated wallets | DOCUMENTED; strict variant added (v2) |
| M-2 | v1 "inductive" GraphSAGE still let validation wallets pass messages to fit wallets during training | FIXED: `GraphSAGE-inductive-strict` added; original variant kept |
| M-3 | `ml/train_dataset_scenarios.py` did not seed PyTorch, so its single-split GraphSAGE F1 changed between runs (S1 0.7649 vs 0.7871) | FIXED: seeded; two consecutive runs now byte-identical |
| M-4 | The API scores dataset wallets from rows that include ~80% training wallets: those scores are in-sample | DOCUMENTED (do not present as evaluation) |
| M-5 | `ml/predict.py` silently fills missing schema features with 0.0 | DOCUMENTED (not changed: production behaviour) |
| M-6 | Two S2 predictors carry no or redundant information: `created_at_ts` (constant 0) and `tx_frequency` (= `activity_score`/30, r = 1.000). Both are still in the production model | DOCUMENTED |
| M-7 | BigQuery enrichment uses a window relative to the query date (`CURRENT_TIMESTAMP() − 90 days`), so it is not re-runnable. Coverage 126 / 12,815 (0.98%) | DOCUMENTED |
| M-8 | The Etherscan free-tier 1,000-record page cap (announced for July 2026) is untested against the code's `offset=10000` pagination | DOCUMENTED |
| M-9 | Live latency was measured on only 11 wallets, while another job was using the CPU | DOCUMENTED |
| M-10 | `REVISI_IJECE.md` contained claims beyond the evidence (leakage-free, 1:10 rationale, `>` vs `≥ 0.95`, threshold optimality, mainnet costs, 28-study protocol) | FIXED in the audit pass; v2 numbers applied in this pass |

## Low

| ID | Finding | Status |
|---|---|---|
| L-1 | Threshold operator differs: `XGBoostModel.predict` / `predict.py` use `p > 0.5`, the revision scripts use `p >= 0.5` (no tie observed) | DOCUMENTED |
| L-2 | Stale docstrings in `api/services/ml_service.py` ("~8.595 sampled wallets", "model B") | FIXED (comments only; no behaviour change) |
| L-3 | Legacy scripts and outputs on the 8,595 population (`ml/train.py`, late-fusion V1, two notebooks) | DOCUMENTED (`EXPERIMENT_CHANGELOG.md`) |
| L-4 | `ml/clean_hybrid_dataset.py` fits p1/p99 caps on the full dataset; its capped CSV is used for EDA only, never by a reported model | DOCUMENTED |

## Robustness results added in v2 (not bugs)

- **Temporal split** (train = first-seen ≤ 2018-06-05; test = later 2,563 wallets, 294 phishing): XGBoost S2
  F1 0.9165 ± 0.0060, RF S2 0.9007 ± 0.0026. There is no degradation, but test prevalence is 11.5%
  (vs 9.1%), so values are not directly comparable with the random-split results.
- **Majority ratio** (10 seeds): XGBoost S2 F1 0.9478 (1:2), 0.9300 (1:5), 0.8976 (1:10); ROC-AUC
  0.993–0.995.
