# Experiment changelog — IJECE revision (Paper ID 42848)

Every change to the experimental protocol, newest first. Earlier outputs are preserved:
`outputs/archive/revision_v1_2026-10-27/` and `outputs/archive/ijece_final_audit_v1_2026-10-27/`
(revision v1), and git history for the submitted 8,595-wallet results (`6f98c6c`, `b632480`).

## v2 — 27 Oct 2026 (this audit)

| # | Change | Reason | Files | Affected outputs |
|---|---|---|---|---|
| 1 | `XGBoostModel.predict/predict_proba` now predict with `iteration_range=(0, best_iteration + 1)` | `xgb.train` returns the booster at the last round (best + 40), and `Booster.predict` uses all trees, so early stopping had no effect on predictions | `ml/xgboost_model.py` | every XGBoost and fusion row; production artifact **unchanged** (its best_iteration = 249 = last round; predictions verified identical) |
| 2 | Calibration and cost-sensitive thresholds use a **dedicated** XGBoost with three disjoint training-side slices: fit 60% / early stopping 20% / calibration + threshold 20% | v1 reused the early-stopping validation slice for calibration and threshold selection | `experiments/revision_ijece.py` | `calibration_*`, `cost_threshold_*`, `reliability_diagram.png`, new `calibration_split_sizes.csv` |
| 3 | New GraphSAGE variant `GraphSAGE-inductive-strict`: during training and early stopping only edges among FIT wallets exist (validation and test wallets pass no messages) | v1 "inductive" removed test edges only; validation wallets still exchanged messages with fit wallets | `experiments/revision_ijece.py` | `multiseed_*` |
| 4 | GraphSAGE decomposition rows that stop on test F1 are flagged `valid_performance_estimate = False` | kept only to show the bias of the old protocol | `experiments/revision_ijece.py` | `graphsage_protocol_decomposition.csv` |
| 5 | Majority-ratio sensitivity uses 10 seeds (was 5) | consistency with the main protocol | `experiments/revision_ijece.py` | `majority_ratio_*` |
| 6 | New temporal split (earliest 80% by first MulDiGraph transaction → train) | requested temporal-leakage / distribution-shift check | `experiments/revision_extra.py` | `temporal_split_*` |
| 7 | New lower-prevalence sensitivity (S1, 1:10 … 1:100 with extra normals featurised from MulDiGraph) and an analytic prior-shift estimate of precision | 1:10 does not represent the real prevalence | `experiments/revision_extra.py` | `prevalence_*` |
| 8 | New feature selection with the subset chosen by 5-fold CV inside the training split | `ml/feature_selection.py` selected the best subset by **test** F1 | `experiments/revision_extra.py` | `feature_selection_cv_*` |
| 9 | Consistency validator | requested reproducibility check | `scripts/validate_revision.py` | `validation_results.json` |
| 10 | `ml/train_dataset_scenarios.py` seeds NumPy and PyTorch before building GraphSAGE | the single-split GraphSAGE F1 changed between runs (S1 0.7649 vs 0.7871); two runs are now identical | `ml/train_dataset_scenarios.py` | `ml/artifacts/dataset_scenarios_comparison.csv` and its GNN confusion matrices |
| 11 | Late-fusion V2 labels report the actual feature counts (18 graph + 7 Etherscan) instead of the hard-coded "19 graph" | stale text from the 8,595 schema | `experiments/run_graphsage_etherscan_late_fusion_v2.py` | `outputs/late_fusion_report_v2.md`, `outputs/embedding_only_ablation_report.md` |
| 12 | Consolidated metrics table | single traceable source of every number | `scripts/build_final_metrics.py` | `FINAL_VERIFIED_METRICS.csv` |
| 13 | Model benchmarks re-measured (live Etherscan runs kept) | consistent v2 timestamp; values vary run to run (load 75.5 → 60.0 ms, service path 3.54 → 2.87 ms) | `experiments/revision_ops.py` | `ops_metrics.json` |
| 14 | v1 markdown in `outputs/ijece_final_audit/` replaced by pointers to the v2 documents | avoid v1 numbers being read as final | — | `outputs/ijece_final_audit/*.md` |
| 15 | Figure 2 regenerated from v2 10-seed results | Reviewer C6 | `scripts/make_figure2.py` | `figure2_f1_by_scenario.png/.pdf` |
| 16 | Stale docstrings fixed (comments only) | audit L-2 | `api/services/ml_service.py` | — |

Run log of v2: `revision_ijece.py` 12:38–12:54, `revision_extra.py` 12:54–12:58, follow-up scripts
13:00–13:10 (27 Oct 2026). Results: `VALIDATION_REPORT.md`.

Not changed (deliberately): the production artifact `ml/artifacts/model_prod_xgb.json`, `ml/predict.py`,
the API, the RiskRegistry contract, raw data and labels.

## v1 — 27 Oct 2026 (first revision pass, commits `8b5267f`, `ec8eed3`, `3d159b5`)

| # | Change | Reason |
|---|---|---|
| 1 | Population 8,595 → 12,815 wallets (1:10), 26 → 25 predictors (`graph_present` → audit only) | the 8,595 snapshot was an intermediate, uncommitted enrichment state and cannot be regenerated |
| 2 | GraphSAGE early stopping moved from the test mask to a train-internal validation mask | test-set early stopping inflated GraphSAGE F1 |
| 3 | Etherscan-derived edges disabled (`USE_ETHERSCAN_EDGES = False`) | crawled for the 8,595 population |
| 4 | 10 stratified splits (seeds 42, 0–8) replace the single seed-42 split | seed stability (Reviewer A1) |
| 5 | Added: inductive GraphSAGE, late fusion and embedding ablation on 12,815, ratio sensitivity, cost thresholds, calibration, operational benchmarks | reviewer requests |
| 6 | Late-fusion V2 script asserts updated from the 8,595 split to the 12,815 split | the script could no longer run |
| 7 | A bug in the first v1 run (GraphSAGE labels and predictions in different order, F1 ≈ 0.09) was found and fixed before any number was reported | — |

## Legacy outputs that must not be reported as revised results

| File | Why |
|---|---|
| `ml/artifacts/gnn_tuning_results.json` (`ml/tune_gnn.py`) | early stopping **and** configuration choice on test F1. Its "best" configuration was never adopted; the GraphSAGE defaults date from 21 May 2026, before this script (30 Jul 2026) |
| `ml/artifacts/feature_selection_report.json` → `best_by_f1` | subset chosen by test F1 (ranking itself is train-only) |
| `ml/artifacts/smote_experiment.json` → `best_by_f1` | comparison scored on test; "best" chosen on test |
| `ml/artifacts/xgb_tuning_results.json` | CV search on train and threshold on validation (valid), but its parameters are not used by any reported model |
| `outputs/late_fusion_*` without `_v2`, `notebooks/experiments.ipynb`, `notebooks/results_testing.ipynb` | 8,595-wallet population |
| `outputs/archive/revision_v1_*` | XGBoost rows used all trees; calibration/threshold shared the early-stopping slice |
