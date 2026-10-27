# Manuscript changes required — IJECE Paper ID 42848

Every value below comes from a file in `outputs/ijece_final_audit/` or `outputs/revision/`.
"Submitted" values are those of the original manuscript (8,595-wallet population); their sources
are listed in `reconciliation_submitted_vs_revised.csv`.

## 1. Numbers that must be replaced everywhere

| Item | Submitted | Revised | Evidence |
|---|---|---|---|
| Wallet population | 8,595 (1,165 : 7,430) | **12,815 (1,165 : 11,650, 1:10)** | `dataset_audit.json` |
| S2 predictors | 26 | **25** (`graph_present` = audit only) | `feature_dictionary.csv` |
| S1 / S3 / S4 predictors | 19 / 29 / 36 | **18 / 28 / 35** | `dataset_audit.json` |
| XGBoost S2 F1 | 0.9102 | **0.9003 ± 0.0128** (seed 42: 0.8911) | `model_results_summary.csv` |
| RandomForest S2 F1 | 0.8884 | **0.9040 ± 0.0109** (seed 42: 0.8921) | same |
| GraphSAGE S2 F1 | 0.8960 | **0.8580 ± 0.0398** (validation stopping) | same |
| GraphSAGE graph-only (S1 encoder) | 0.8189 | **0.7802 ± 0.0320** | same |
| Embedding-only ablation | 0.8184 | **0.8079 ± 0.0225** | same |
| Late fusion | 0.8825 | **0.8769 ± 0.0175** | same |
| LogisticRegression S2 F1 | 0.8540 | **0.7621 ± 0.0368** | same |
| BigQuery coverage | 0.93% | **0.98%** (126 / 12,815) | `bigquery_coverage_audit.json` |
| Production model F1 | 0.9102 | **0.8911** (seed-42 split) | `ml/artifacts/prod_feature_schema.json` |

Complete per-scenario values: `model_results_summary.csv`. Figure 2 data: `figure2_f1_mean_std.csv`.

## 2. Conclusions that change

1. **"XGBoost is the best model" is no longer supported for F1.** RandomForest S2 has the highest mean
   F1 (0.9040 vs 0.9003; higher in 7/10 seeds), accuracy and precision. The difference is not
   significant (paired Wilcoxon p = 0.16). XGBoost has the highest recall, ROC-AUC and PR-AUC and the
   fewest FN; on S2 it beats RF in recall and PR-AUC in 10/10 seeds and has fewer FN in 10/10 seeds
   (`best_model_by_metric.json`). Write: "RF and XGBoost are statistically tied on F1; XGBoost is
   selected for deployment because it misses fewer phishing wallets."
2. **GraphSAGE drops from near-best (0.896) to clearly below the tree models (0.780–0.858).** The old
   value was inflated by test-set early stopping (+2.1 F1 points on the revised data, Table B).
3. **The data-centric conclusion holds and is now multi-seed.** Etherscan enrichment: XGBoost
   0.8809 → 0.9003, RF 0.8816 → 0.9040; S2 > S1 in 10/10 seeds for both. BigQuery adds nothing.
4. **Late fusion does not beat the tabular model** (0.8769 vs 0.9003), as before.

## 3. Claims to remove or qualify

| Claim pattern | Problem | Replace with |
|---|---|---|
| "leakage-free pipeline" / "the topology does not leak" | Features are built transductively on the full graph; the inductive check has low power (354 edges) | "Features are label-free; scaling, stopping, thresholds and calibration use training wallets only; an inductive GraphSAGE variant shows no difference (p = 0.77)." |
| Any F1 tied to the deployed dashboard / demo scores | API scores dataset wallets from rows that include ~80% training wallets (in-sample) | Report only held-out test metrics |
| Mainnet gas cost or latency as measured | Only Hardhat local gas and latency were measured | "Estimated from the measured 35,462 gas at 0.1–30 gwei" |
| Etherscan throughput as measured | Limits are from documentation; no load test | "Upper bound derived from the documented free-tier limits" |
| "Calibrated risk score" | Calibration was evaluated, not deployed | "Platt scaling halves ECE and is planned for the next release" |
| Feature selection affects the deployed model | Production uses all 25 S2 features | Present the ranking as analysis |
| "|r| > 0.95" | The code uses ≥ 0.95 | "|r| ≥ 0.95" |
| Rationale "1:10 = largest ratio we could enrich" | Not documented in the repository | Author confirms or removes |
| Novelty "first" / "real-time" without qualification | Live scoring takes 8–17 s per wallet, dominated by Etherscan | "near-real-time (≈ 9 s median per new wallet; 3.5 ms per pre-featurized wallet)" |

## 4. Known data or model issues to mention as limitations

- `created_at_ts` is a constant 0 and is still one of the 25 production predictors (zero gain).
- `tx_frequency = activity_score / 30` (r = 1.000), so these two features are redundant.
- Etherscan "activity" features use only the latest 100 transactions; `transaction_count` is capped at
  10,000 (8 wallets).
- Etherscan attributes describe the wallet in 2026, while the graph features describe 2015–2019.
- The phishing and normal groups differ strongly in degree (median 13 vs 1), which is a sampling
  property of how EPTransNet was crawled. Part of the performance may reflect activity level rather
  than phishing behaviour.
- BigQuery features only cover a 90-day window in 2026 (2026-01-22 … 2026-04-20).

## 5. New tables and figures

| New element | Source file |
|---|---|
| Table 2: feature dictionary (36 attributes) | `feature_dictionary.csv` |
| Table 3: F1 mean ± std (4 scenarios × 4 models + inductive GraphSAGE) | `model_results_summary.csv` |
| Table 4: fusion and embedding ablation | `graphsage_fusion_results.csv` |
| Table 6(a): prediction panel | `model_results_summary.csv` (S2 XGBoost), `model_results_per_seed.csv` (seed 42) |
| Table 6(b): deployment panel (measured vs estimated) | `operational_benchmarks.json` |
| Table A: majority-class ratio | `sampling_ratio_sensitivity.csv` |
| Table B: GraphSAGE protocol decomposition | `graphsage_protocol_audit.csv` (`row_type = decomposition`) |
| Table C: calibration + reliability figure | `calibration_results.csv`, `outputs/revision/reliability_diagram.png` |
| Table D: cost-sensitive threshold (including the cases where it is worse) | `cost_sensitive_thresholds.csv` |
| Figure 2: F1 with ± std error bars | `figure2_f1_mean_std.csv` (axis must start below 0.70 if LR and GraphSAGE are shown) |

## 6. Pending author input (cannot come from the repository)

- E4: Funding ("research grant or contract"), Acknowledgement, CRediT, Data Availability, CoI.
- C2: the real literature-selection protocol for the 28 studies (no evidence in the repo).
- C1, C3, C6: figure/table ordering, Figure 1 S1–S4 labels, regenerated Figure 2.
- C8: author biographies.
- A1: the original reason for choosing 1:10, if any beyond the evidence above.
- Any AI-use disclosure required by the journal template.
- The revision deadline (8 weeks from 10 Aug 2026 = 5 Oct 2026) has passed. Confirm an extension
  with the editor.
