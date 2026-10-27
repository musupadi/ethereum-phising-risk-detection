# IJECE reviewer evidence — revision v2 (Paper ID 42848)

All numbers come from revision **v2** outputs (27 Oct 2026; validator 22/22 PASS). Population 12,815
wallets (1,165 phishing / 11,650 normal); 10 stratified 80:20 splits (seeds 42, 0–8); mean ± std.
Every value can be found in `FINAL_VERIFIED_METRICS.csv` together with its source file.

Legend for "status": **VERIFIED** = measured and re-checked; **PARTIAL** = measured in part, rest
derived or documented; **PENDING-AUTHOR** = cannot come from the repository.

---

## Editor

### E1 — Emphasize the approach
- **Code:** `ml/preprocess.py:extract_node_features`, `ml/build_hybrid_dataset.py`, `experiments/revision_ijece.py`
- **Evidence:** `outputs/ijece_final_audit/feature_dictionary.csv`
- **Claim allowed:** Each wallet's one-hop neighbourhood in MulDiGraph is converted into 18 label-free
  statistics, enriched with 7 Etherscan attributes, and classified by tabular models. A 4 × 4 design
  (S1–S4 × LR / RF / XGBoost / GraphSAGE) separates the effect of the data from the effect of the model.
- **Pending:** wording in the manuscript (PENDING-AUTHOR).

### E2 — Purpose, findings, implications
- **Evidence:** `multiseed_summary.csv`
- **Verified:** Etherscan enrichment raises F1 for XGBoost from 0.8823 ± 0.0138 to 0.9007 ± 0.0132 (S2 > S1
  in 8/10 seeds) and for RF from 0.8816 ± 0.0141 to 0.9040 ± 0.0109 (10/10). BigQuery adds nothing
  (S3 ≈ S1). GraphSAGE (0.7802–0.8580) and late fusion (0.8743 ± 0.0181) stay below the tree models.
- **Claim allowed:** "On this benchmark, feature enrichment contributed more than model complexity."
- **Limitation:** results hold at 1:10 prevalence; see B7 / H-5 for lower prevalence.

### E3 — Novel contributions, future impact
- **Claim allowed:** a reproducible graph-to-tabular pipeline usable both offline (MulDiGraph) and live
  (Etherscan) with the same model; a leakage-controlled 4 × 4 comparison with multi-seed, temporal and
  prevalence analyses; a deployed API with an on-chain audit trail (local Hardhat measurements).
- **Do not claim:** state-of-the-art accuracy, real-world precision, mainnet deployment, a calibrated
  deployed score, or "real-time" scoring without qualification (live ≈ 9 s per new wallet).
- **Status:** PENDING-AUTHOR (wording).

### E4 — Funding, Acknowledgement, CRediT, Data Availability, CoI
- **Status:** PENDING-AUTHOR. Data Availability can state: MulDiGraph from XBlock (not redistributed);
  derived dataset, enrichment CSVs, models and scripts in the GitHub repository.

---

## Reviewer A

### A1 — Majority-class size; seed stability
- **Code:** `ml/extract_addresses_from_graph.py`, `experiments/revision_ijece.py`
- **Evidence:** `majority_ratio_summary.csv`, `multiseed_*`, `prevalence_sensitivity_summary.csv`, `outputs/ijece_final_audit/dataset_audit.json`
- **Verified:**
  - All 1,165 phishing nodes are kept; 11,650 normals are drawn at 1:10 with `random.seed(42)`, exactly
    reproducible.
  - XGBoost S2, 10 seeds: F1 0.9478 ± 0.0088 (1:2), 0.9300 ± 0.0065 (1:5), 0.8976 ± 0.0134 (1:10);
    ROC-AUC 0.9931 / 0.9945 / 0.9943.
  - Seed stability: XGBoost S2 F1 std 0.0132, accuracy 0.9812 ± 0.0026.
  - Lower prevalence (S1, extra normals from MulDiGraph, 5 seeds): XGBoost F1 0.8833 → 0.8033 → 0.7183 →
    0.5930 and RF 0.8837 → 0.8372 → 0.7828 → 0.7411 for 1:10 / 1:20 / 1:50 / 1:100.
- **Claim allowed:** "F1 decreases as the normal class grows while ROC-AUC stays ≈ 0.99, i.e. the loss is in
  precision. Results at 1:10 do not represent real-world prevalence."
- **Pending:** the original reason for choosing 1:10 is not documented (PENDING-AUTHOR).
- **Status:** VERIFIED

### A2 — Is one-hop extraction restricted to the training split?
- **Code:** `ml/preprocess.py`, `ml/build_hybrid_dataset.py`, `revision_ijece.py:train_sage`
- **Evidence:** `multiseed_summary.csv` (GraphSAGE, -inductive, -inductive-strict), `temporal_split_summary.csv`, `dataset_audit.json`
- **Verified:**
  1. *Label leakage:* none. Features use only edge counts, amounts and timestamps; `isp` is used only as
     the target (label = `isp` for 12,815 / 12,815).
  2. *Transductive feature construction:* yes. Features are computed on the full graph before splitting.
  3. *Structural leakage via GraphSAGE:* no evidence. Transductive 0.8580 ± 0.0398; test edges removed
     0.8619 ± 0.0195 (p = 0.77); fit-only edges, i.e. validation and test isolated during training,
     0.8655 ± 0.0203 (p = 0.63). The check has low power: 354 edges, 96.6% of wallets isolated.
  4. *Fitted preprocessing:* scalers, early stopping, thresholds and calibration are fitted on training
     wallets only.
  5. *Temporal:* random splits mix 2015–2019 wallets. A temporal split (first-seen ≤ 2018-06-05 → train)
     gives XGBoost S2 F1 0.9165 ± 0.0060 (test prevalence 11.5%, so not directly comparable). Etherscan
     attributes are from 2026 and may contain post-label information; this is not testable.
- **Claim allowed:** "Features are label-free but constructed transductively; inductive GraphSAGE variants
  show no advantage of the transductive graph." Do **not** claim "leakage-free".
- **Status:** VERIFIED (temporal post-label risk documented)

### A3 — Cause of low BigQuery coverage
- **Code:** `ml/enrich_real_addresses_bigquery.py` (`block_timestamp >= CURRENT_TIMESTAMP() − 90 days`)
- **Evidence:** `outputs/ijece_final_audit/bigquery_coverage_audit.json`
- **Verified:** 126 / 12,815 (0.98%; 125 normal, 1 phishing). Covered activity lies in 2026-01-22 … 2026-04-20;
  MulDiGraph covers 2015-08-07 … 2019-01-19. The cause is the query window, not the public dataset. The
  submitted 0.93% (8,595 population) cannot be re-checked.
- **Status:** VERIFIED

### A4 — Class weights
- **Code:** `ml/xgboost_model.py:fit`, `ml/gnn_model_v2.py:fit`, `revision_ijece.py`
- **Verified:**
  - XGBoost: `scale_pos_weight = N_neg/N_pos` on the inner fit slice = 7,455 / 746 = 9.993.
  - LR and RF: `class_weight="balanced"` → w0 = 0.550, w1 = 5.500 on 10,252 training rows.
  - GraphSAGE: w_c = N/(2N_c) on 8,201 fit nodes → w0 = 0.550, w1 = 5.497.
- **Status:** VERIFIED

### A5 — 0.8184 vs 0.8960
- **Evidence:** `graphsage_protocol_decomposition.csv`, `outputs/ijece_final_audit/reconciliation_submitted_vs_revised.csv`
- **Verified:**
  - In the submission, 0.8184 = embedding-only XGBoost ablation and 0.8189 = GraphSAGE graph-only. Both
    used validation stopping, 19 inputs and MulDiGraph edges. 0.8960 = GraphSAGE S2 with **test-F1**
    stopping, 26 inputs, plus Etherscan edges.
  - Revised data, seed 42: test-F1 stopping 0.8738 vs validation stopping 0.8526 (bias +2.1 points; rows
    flagged invalid); 25 vs 18 inputs 0.8526 vs 0.7880; Etherscan edges 0.8733 vs 0.8738 (none).
- **Claim allowed:** "The difference stemmed from the input set and from test-based early stopping, which
  we corrected."
- **Status:** VERIFIED

### A6 — Ranking formula; FP in Table 6
- **Code:** `ml/feature_selection.py`; train-only re-run in `experiments/revision_extra.py`
- **Evidence:** `outputs/ijece_final_audit/feature_ranking_audit.csv`, `feature_selection_cv_summary.csv`
- **Verified:**
  - combined(f) = ½[gain_XGB/Σ + MDI_RF/Σ] on the training split after train-fitted p1/p99 capping.
  - Correlation filter is applied after ranking, |r| ≥ 0.95: drops `in_degree` (0.954 with
    `unique_senders`) and `tx_frequency` (1.000 with `activity_score`). `in_degree` keeps rank 2 because
    ranking precedes pruning.
  - The original subset choice used test F1. With train-only CV, the selected subset reaches F1
    0.9005 ± 0.0132 vs 0.9020 ± 0.0122 for all features, so there is no gain.
  - The production model uses all 25 S2 features.
  - Table 6: seed 42 TN 2,288 / FP 42 / FN 12 / TP 221; 10-seed mean FP 34.1, FN 14.2.
- **Status:** VERIFIED

---

## Reviewer B

### B1 — Live feature complexity and latency
- **Code:** `ml/live_features.py`; `experiments/revision_ops.py`
- **Evidence:** `ops_metrics.json → live`
- **Verified (measured, n = 11):** O(k) in the number of transactions; 5 Etherscan calls per wallet;
  8.0–16.6 s end-to-end (median 9.15 s); computation ≈ 18 ms.
- **Claim allowed:** "near-real-time; latency dominated by the external API."
- **Status:** VERIFIED (small sample)

### B2 — Isolated nodes
- **Evidence:** `isolated_wallets.json`, `dataset_audit.json`
- **Verified:** Every dataset wallet has ≥ 1 MulDiGraph edge. In the induced GNN graph, 98.4% of normal and
  78.8% of phishing wallets have no neighbour. Live mode: empty `txlist` → 404 and no score; fewer than 5
  transactions → scored with an `insufficient_data` flag; all live scores are `out_of_distribution`.
- **Status:** VERIFIED

### B3 — Etherscan rate limits and throughput
- **Verified:** 5 calls per wallet (measured).
- **Derived (not load-tested):** 3 calls/s and 100k/day (Etherscan documentation) → ≤ 0.6 wallet/s,
  ≈ 20,000 wallets/day per key. The model itself: 2.9 ms per wallet (service path), 1.3 µs per wallet in
  batch.
- **Not tested:** the 1,000-record page cap for free keys.
- **Status:** PARTIAL

### B4 — Calibration
- **Code:** `revision_ijece.py` (dedicated XGBoost; fit 60% / early stopping 20% / calibration 20%)
- **Evidence:** `calibration_summary.csv`, `reliability_diagram.png`
- **Verified (10 seeds):**

  | Method | Brier | ECE | Log loss | ROC-AUC | PR-AUC | F1@0.5 |
  |---|---|---|---|---|---|---|
  | Uncalibrated | 0.0166 ± 0.0020 | 0.0146 ± 0.0029 | 0.0639 | 0.9938 | 0.9588 | 0.8924 |
  | Platt | 0.0142 ± 0.0017 | 0.0069 ± 0.0018 | 0.0522 | 0.9938 | 0.9588 | 0.8977 |
  | Isotonic | 0.0147 ± 0.0019 | 0.0072 ± 0.0026 | 0.0586 | 0.9924 | 0.9422 | 0.8970 |

- **Claim allowed:** "Platt scaling halves ECE without changing ranking; planned for the next release."
  **The deployed score is uncalibrated.**
- **Status:** VERIFIED (not deployed)

### B5 — Model size and memory
- **Evidence:** `ops_metrics.json → model`; `VALIDATION_REPORT.md` (identity and hash)
- **Verified:**
  - Model: 878.3 KB, 250 trees, 25 features.
  - Load: 60 ms, +4.0 MB RSS; process 162.6 MB after inference.
  - Latency: pure booster 0.60 ms (p95 0.73); full service path 2.87 ms (p95 3.40); batch 1.26 µs per wallet.
  - The production artifact is the seed-42 XGBoost S2 (F1 0.8911). The 10-seed mean describes the
    procedure, not the artifact.
- **Status:** VERIFIED

### B6 — Gas and fiat cost
- **Measured (Hardhat local, 20 calls):** 35,462 gas (35,452–35,464); commit latency 13.57 ± 5.93 ms.
- **Derived:** 0.1–30 gwei → 3.5 × 10⁻⁶ – 1.06 × 10⁻³ ETH → USD 0.008–2.46 at USD 2,313.76/ETH
  (CoinDesk, 26 Oct 2026). Mainnet latency is about 1 block, not measured. No public-network transaction
  was made.
- **Status:** PARTIAL

### B7 — Threshold against a cost matrix
- **Evidence:** `cost_threshold_summary.csv`; per-seed comparison in `outputs/ijece_final_audit/cost_sensitive_thresholds.csv`
- **Verified (threshold chosen on the calibration slice, test once):**

  | C_FN/C_FP | Threshold (mean) | Test cost at 0.5 | Test cost at selected | Change | Seeds worse than 0.5 |
  |---|---|---|---|---|---|
  | 1 | 0.69 | 52.5 | 48.0 | −8.6% | 2 / 10 |
  | 2 | 0.54 | 68.0 | 69.2 | **+1.8%** | 4 / 10 |
  | 5 | 0.40 | 114.5 | 113.4 | −1.0% | 3 / 10 |
  | 10 | 0.27 | 192.0 | 173.9 | −9.4% | 4 / 10 |
  | 20 | 0.14 | 347.0 | 261.9 | −24.5% | 1 / 10 |

- **Claim allowed:** "Validation-selected thresholds reduce expected cost for high miss costs (≥ 10×). For
  C = 2 they are slightly worse than 0.5 on test, and the selected threshold varies by seed (std up to
  0.19)." Do not call any threshold universally optimal.
- **Status:** VERIFIED

---

## Reviewer C

### C1 — Figures and tables cited before they appear
- **Status:** PENDING-AUTHOR.

### C2 — Selection criteria for the 28 studies
- **Status:** PENDING-AUTHOR. The repository contains **no** evidence of the search protocol. Do not invent
  databases, queries or dates.

### C3 — S1–S4 labels in Figure 1
- **Verified:** S1 = 18, S2 = 25, S3 = 28, S4 = 35 predictors.
- **Status:** PENDING-AUTHOR (figure edit).

### C4 — Feature dictionary
- **Evidence:** `outputs/ijece_final_audit/feature_dictionary.csv` (36 rows: 18 graph, 7 Etherscan,
  10 BigQuery predictors and `graph_present` audit only).
- **Note:** `created_at_ts` is constant 0; `tx_frequency` = `activity_score`/30.
- **Status:** VERIFIED

### C5 — GraphSAGE in Table 3 vs Table 4
- **Verified:** Same architecture in both: 2 × SAGEConv (mean), hidden 128, BatchNorm, ReLU, dropout 0.5,
  Adam lr 0.01, ≤ 200 epochs, patience 25, validation stopping. Table 3 uses its own linear head on the
  scenario features. Table 4 uses the S1 encoder: 128-d pre-head embedding ‖ 7 standardized Etherscan →
  XGBoost (0.8743 ± 0.0181); embedding only 0.8064 ± 0.0234.
- **Status:** VERIFIED

### C6 — Y-axis note in Figure 2
- **Evidence:** `outputs/ijece_final_audit/figure2_f1_mean_std.csv`
- **Evidence:** `outputs/revision/figure2_f1_by_scenario.png` / `.pdf` (`scripts/make_figure2.py`): F1 mean ± std for 4 scenarios × 4 models, y-axis from 0.60, hatching and value labels for grayscale print.
- **Caption text:** "F1-score (mean ± standard deviation over 10 stratified splits) by feature scenario and model. Note: the y-axis starts at 0.60 to make differences visible; bar heights are not proportional to absolute performance."
- **Status:** VERIFIED (figure ready; inserting it into the manuscript is PENDING-AUTHOR)

### C7 — Separate Table 6 panels
- **Evidence:** `seed42_table.csv`, `multiseed_summary.csv` (panel a); `ops_metrics.json` (panel b, measured
  vs derived marked as above).
- **Status:** VERIFIED (data)

### C8 — Author biographies
- **Status:** PENDING-AUTHOR.

---

## Model selection statement (for the paper)

On S2, RandomForest has the highest mean F1 (0.9040 ± 0.0109 vs 0.9007 ± 0.0132; higher in 6/10 seeds;
paired Wilcoxon p = 0.43, not significant), precision and accuracy. XGBoost has higher recall and PR-AUC
in 10/10 seeds and fewer false negatives in 10/10 seeds (14.2 vs 19.8 per split), and is deployed for
that reason. At lower prevalence (1:50, 1:100, S1) RF clearly outperforms XGBoost in F1, so this choice is
specific to the 1:10 setting.
