# IJECE Revision — Paper ID 42848

**"Data-centric Ethereum phishing detection through graph-to-tabular transformation"**

> **Cara pakai dokumen ini**
> - **Bagian 1** = *Revision declaration* + *Response to reviewers*. Taruh sebelum naskah (sesuai permintaan editor), dan kirim juga ke ijece@iaesjournal.com.
> - **Bagian 2** = teks/tabel baru yang ditempel ke naskah, diurutkan per subbab.
> - **Bagian 3** = yang harus kamu isi atau kerjakan sendiri (data pribadi, gambar, biografi).
> - Semua angka berasal dari `experiments/revision_ijece.py` dan `experiments/revision_ops.py`. Output mentahnya ada di `outputs/revision/`.
> - Nomor tabel dan gambar mengikuti naskah lama. Sesuaikan kalau penomorannya bergeser.

---

## BAGIAN 1 — Revision declaration & response to reviewers

### Revision declaration

We thank the Editor and the three reviewers for their careful reading and constructive comments. All comments have been addressed. The main changes are:

1. **All experiments were re-run on the complete, reproducible population of 12,815 wallets** (1,165 phishing : 11,650 normal, ratio 1:10) with the final 25-predictor schema. The earlier submission reported results on an intermediate 8,595-wallet snapshot that was produced while Etherscan enrichment was still in progress. That snapshot could not be regenerated exactly, so it was replaced. All tables and figures have been updated.
2. **All results are now reported as mean ± standard deviation over 10 independent stratified 80:20 splits** (seeds 42, 0–8).
3. **We corrected a protocol inconsistency in the GraphSAGE baseline of Table 3.** Its early stopping previously monitored test-set F1. All GraphSAGE models now stop on a validation slice taken from the training wallets only, so the test set is scored exactly once.
4. **New analyses were added:** majority-class ratio sensitivity, inductive versus transductive GraphSAGE (structural leakage check), cost-sensitive decision thresholds, probability calibration (Platt and isotonic), live-path latency and Etherscan rate-limit analysis, model memory footprint, and public-network gas cost of RiskRegistry.
5. **New presentation elements:** a centralized feature dictionary (36 attributes), explicit S1–S4 labels in Figure 1, the y-axis note in Figure 2, a restructured Table 6, the Table 1 study-selection criteria, author biographies, and the Funding, Acknowledgement, CRediT, Data Availability and Conflict of Interest sections.

Changes in the manuscript are highlighted in yellow.

---

### Response to the Editor

**E1. Emphasize the approach.**
The Introduction and Abstract now state the approach explicitly. The method is a *data-centric graph-to-tabular transformation*: each wallet's one-hop neighbourhood in the EPTransNet MulDiGraph is converted into 18 label-free behavioural statistics, enriched with 7 Etherscan account attributes, and classified by a gradient-boosted tree. A 4 × 4 design (four feature scenarios × four model families) isolates the contribution of the *data* from the contribution of *model complexity*.

**E2. Purpose, main findings, implications.**
A summary paragraph was added at the end of the Introduction and in the Abstract (text in Part 2, §Abstract/§1).

**E3. Novel contributions and future impact.**
The contribution list in Section 1 was rewritten (text in Part 2, §1), and a future-impact paragraph was added to the Conclusion.

**E4. Funding, Acknowledgement, CRediT, Data Availability, Conflict of Interest.**
All five sections were added according to the template.

---

### Response to Reviewer A

**A1 (Sec. 2.1). Rationale for the majority-class sample size; stability across seeds.**
*Rationale.* All 1,165 labelled phishing wallets were kept. Normal wallets were sampled at 1:10 because this is the largest ratio we could fully enrich within the Etherscan quota, and because it is the most conservative of the ratios we tested: it is closest to the real (extremely low) phishing prevalence and makes precision hardest to achieve.

*Ratio sensitivity (new Table A).* For XGBoost S2 over 5 seeds, F1 falls from 0.9476 (1:2) to 0.9287 (1:5) and 0.9018 (1:10). ROC-AUC stays constant at about 0.994. The F1 drop is therefore an effect of prevalence on precision, not a loss of ranking ability.

*Seed stability (new Table 3).* Every model × scenario was re-run on 10 independent splits. XGBoost S2 reaches F1 = 0.9003 ± 0.0128 and ROC-AUC = 0.9943 ± 0.0020. For both XGBoost and random forest, the Etherscan-enriched scenarios (S2, S4) beat S1 and S3 in 10 out of 10 seeds.

**A2 (Sec. 2.2). Is one-hop extraction restricted to the training split?**
The one-hop features are computed once from the full MulDiGraph, before the split. They are *label-free* aggregates of edge counts, amounts and timestamps, and the label of a neighbour is never read. Feature extraction therefore cannot transmit label information from test to training wallets. All fitted transformations (scalers, capping thresholds, early stopping, threshold selection, calibration) are fitted on training wallets only.

For GraphSAGE, which uses the topology directly, we added an explicit check. An *inductive* variant was trained with every edge touching a test wallet removed, and was then applied to the full graph. Its F1 (0.862 ± 0.020 over 10 seeds) is indistinguishable from the transductive model (0.858 ± 0.040), so the topology does not leak structural information. One reason is that the induced graph is very sparse: there are only 354 directed edges among the 12,815 wallets.

**A3 (Secs. 2.2, 3.1). Cause of the low BigQuery coverage.**
The low coverage comes from the query timeframe, not from the public dataset. `bigquery-public-data.crypto_ethereum.transactions` contains the full chain history. Our enrichment query, however, only aggregated a rolling window of the 90 days before the query date (2026), while the EPTransNet wallets were active between August 2015 and January 2019. Only 126 of 12,815 wallets (0.98%) transacted in that window (125 normal, 1 phishing). The manuscript now states this explicitly. A full-history BigQuery enrichment is listed as future work.

**A4 (Sec. 2.3). Exact `scale_pos_weight`.**
We use inverse class frequency computed on the training portion of each split: `scale_pos_weight = N_negative / N_positive`. At 1:10 this gives 9.993 for every seed (7,455 / 746 on the inner fitting slice). Logistic regression and random forest use `class_weight="balanced"`. GraphSAGE uses a weighted cross-entropy with `w_c = N / (2 · N_c)`.

**A5 (Sec. 2.4). Why the GraphSAGE F1 differed (0.8184 vs 0.8960).**
The two numbers came from different protocols:
(i) early stopping on test F1 (Table 3) versus a training-internal validation slice (Table 4);
(ii) 26 versus 19 input features;
(iii) MulDiGraph + Etherscan edges versus MulDiGraph edges only.
Monitoring test F1 selects the best test epoch and therefore inflates the score. We have corrected this: every GraphSAGE model in the revised manuscript uses validation-based early stopping. New Table B isolates each factor on the same split. Monitoring test F1 adds 2.1 F1 points (0.853 → 0.874). Feeding the 7 Etherscan features to GraphSAGE adds 6.5 points (0.788 → 0.853). The Etherscan-derived edges add nothing (0.873 vs 0.874). The original gap is therefore mainly due to the input feature set and, secondarily, to the stopping protocol. Over 10 seeds, the corrected GraphSAGE reaches F1 = 0.858 ± 0.040 (S2) and 0.780 ± 0.032 (S1), below XGBoost in every scenario.

**A6 (Sec. 3.3). Formula of the combined feature ranking; FP in Table 6.**
The ranking is `combined(f) = ½ · [ gain_XGB(f) / Σ gain_XGB + MDI_RF(f) / Σ MDI_RF ]`. Correlation is *not* part of the score. It is applied afterwards as a filter: for every feature pair with |Pearson r| > 0.95, the feature with the lower combined score is dropped (this removed `in_degree` and `tx_frequency`). Table 6 now reports TN, FP, FN and TP together.

---

### Response to Reviewer B

**B1 (Sec. 2.2). Complexity and latency of on-the-fly feature generation.**
For a wallet with *k* normal transactions, the 18 one-hop statistics need one pass over the transactions, so computation is O(k) time and O(k) memory. We measured the live path on 11 wallets:
- Computing the features takes about 18 ms per wallet.
- The end-to-end latency is 8.0–16.6 s (median 9.1 s).
- Each wallet needs 5 Etherscan requests, so the latency is dominated by network round-trips (about 1.5 s per request) and the 0.25 s client-side throttle.

A paged `txlist` (10,000 records per page, at most 5 pages) adds one request per extra page for very active wallets.

**B2 (Sec. 2.2). Handling of isolated nodes.**
There are two cases.
- *Dataset wallets:* every wallet has at least one transaction in MulDiGraph by construction, so no dataset wallet has an empty one-hop neighbourhood. Wallets whose counterparties lie outside the sampled population receive their features normally, but they are isolated in the induced GNN graph: 11,467 normal (98.4%) and 918 phishing (78.8%) wallets. For these wallets GraphSAGE reduces to a self-loop MLP.
- *Live wallets:* if Etherscan `txlist` returns no transactions, the service returns "wallet not found" with no risk score instead of scoring an all-zero vector. Wallets with fewer than the minimum number of transactions are scored but flagged `insufficient_data`. All live scores are flagged `out_of_distribution`.

**B3 (Sec. 2.2). Etherscan rate limits and FastAPI throughput.**
The Etherscan free tier allows 3 calls/s and 100,000 calls/day per account. At 5 calls per wallet, live scanning is bounded by about 0.6 wallets/s and about 20,000 wallets/day per API key, whatever the FastAPI concurrency. The model itself is not the bottleneck (3.5 ms per wallet; 2 µs per wallet in batch). Dataset wallets are served from a lookup without any external call. For production we propose response caching per address, a request queue, and a paid API tier or a self-hosted node.

**B4 (Sec. 2.5). Probability calibration.**
We evaluated calibration in this revision (new Table C). Both calibrators were fitted on the training-internal validation slice:

| Method | Brier | ECE (10 bins) |
|---|---|---|
| Uncalibrated | 0.0158 ± 0.0018 | 0.0136 ± 0.0021 |
| Platt scaling | 0.0133 ± 0.0016 | 0.0065 ± 0.0013 |
| Isotonic regression | 0.0138 ± 0.0016 | 0.0070 ± 0.0018 |

Platt scaling halves the expected calibration error and leaves ROC-AUC unchanged. It will be adopted for the risk index in the next model iteration.

**B5 (Sec. 2.5). Model file size and memory footprint.**
- File: XGBoost S2 booster, JSON format, 878 KB, 250 trees, 25 features.
- Loading: deserialisation takes 76 ms and adds 3.5 MB of resident memory. The whole Python inference process (pandas + xgboost) uses about 163 MB RSS.
- Inference latency: 0.7 ms for a single booster call; 3.5 ms median (6.4 ms p95) for the full service path (schema check, probability, score, top-5 explanation).

**B6 (Sec. 2.5). Gas and fiat cost outside Hardhat.**
`setRisk()` consumed 35,452–35,464 gas (mean 35,462) over 20 transactions. Its public-network cost is gas × gas price:

| Gas price (gwei) | Cost (ETH) | Cost (USD)* |
|---|---|---|
| 0.1 | 0.0000035 | 0.008 |
| 1 | 0.000035 | 0.082 |
| 5 | 0.00018 | 0.41 |
| 10 | 0.00035 | 0.82 |
| 30 | 0.00106 | 2.46 |

\*ETH = USD 2,313.76 (CoinDesk, 26 Oct 2026). Ethereum mainnet base fees in late 2026 were mostly below 1 gwei. The local commit latency of 13.6 ms becomes one block time (about 12 s) on mainnet.

**B7 (Sec. 3.3). Threshold against a cost matrix.**
We added a cost-sensitive analysis (new Table D). For each cost ratio *c* = C_FN / C_FP, the threshold minimising *c*·FN + FP was selected on the validation slice and applied once to the test set:

| C_FN : C_FP | Optimal threshold | Cost at 0.5 | Cost at optimal | FN / FP at optimal |
|---|---|---|---|---|
| 1 | 0.71 | 48.4 | 45.9 | 19.5 / 26.4 |
| 2 | 0.58 | 63.0 | 64.1 | 16.7 / 30.7 |
| 5 | 0.39 | 106.8 | 108.7 | 12.7 / 45.2 |
| 10 | 0.17 | 179.8 | 164.5 | 8.8 / 76.5 |
| 20 | 0.08 | 325.8 | 226.4 | 6.8 / 90.4 |

The default threshold of 0.5 is effectively cost-optimal when a missed phishing wallet costs up to 5× a false alarm. When a miss costs 10× or 20× more, lowering the threshold cuts total cost by 8.5% and 30.5% respectively. The threshold is therefore exposed as a deployment parameter.

---

### Response to Reviewer C

**C1. Figures and tables cited before they appear.** Every figure and table is now introduced and explained in the text before it appears.

**C2. Selection criteria for the 28 verified studies (Table 1).** A selection protocol paragraph was added to Section 1 (text in Part 2).

**C3. S1–S4 labels in Figure 1.** The "Hybrid feature construction" block now labels S1 (MulDiGraph, 18), S2 (+Etherscan, 25), S3 (+BigQuery, 28) and S4 (+Etherscan +BigQuery, 35).

**C4. Centralized feature dictionary.** New Table 2 lists all 36 attributes with source, definition and scenario membership.

**C5. GraphSAGE architecture in Table 3 vs Table 4.** Both use the same architecture: 2 SAGEConv layers with mean aggregation, hidden size 128, BatchNorm, ReLU, dropout 0.5, Adam (lr 0.01), weighted cross-entropy, at most 200 epochs, patience 25, and validation-based early stopping. They differ only in input and use:
- *Table 3:* GraphSAGE is trained on each scenario's full feature set and classifies with its own linear head.
- *Table 4:* GraphSAGE receives only the 18 graph features. Its 128-d pre-head representation is concatenated with the 7 standardized Etherscan features (135-d) and classified by XGBoost. An embedding-only variant is added as an ablation.

**C6. Y-axis truncation in Figure 2.** The caption now states: "Note: the y-axis is truncated (starts at 0.875) to make small differences visible; bar heights should not be read as proportional to performance. Error bars show ± one standard deviation over 10 splits." (If the regenerated figure includes GraphSAGE or LR, whose F1 is below 0.875, adjust the axis start and the note accordingly.)

**C7. Table 6 separated.** Table 6 is split into panel (a), *prediction performance*, and panel (b), *system and deployment performance*.

**C8. Author biographies.** Biographies of all authors were added at the end of the article.

---

## BAGIAN 2 — Teks dan tabel untuk naskah

### §Abstract (ganti seluruhnya — sesuaikan panjang dengan batas template)

> Phishing accounts drain Ethereum users through ordinary-looking transfers, and most detectors answer this with increasingly complex graph neural networks. This study asks whether the same accuracy can be reached by improving the data instead of the model. We propose a data-centric graph-to-tabular transformation that converts each wallet's one-hop neighbourhood in the EPTransNet MulDiGraph into 18 label-free behavioural statistics and enriches them with 7 Etherscan account attributes. A 4 × 4 design (four feature scenarios × logistic regression, random forest, XGBoost and GraphSAGE) on 12,815 wallets (1:10), repeated over 10 stratified splits, shows that Etherscan enrichment raises XGBoost F1 from 0.881 to 0.900 ± 0.013 (ROC-AUC 0.994), while GraphSAGE (0.858 ± 0.040) and late fusion of GraphSAGE embeddings (0.877 ± 0.018) do not exceed the tabular model. The model is served through a FastAPI service with 3.5 ms inference latency and writes risk scores to an EVM RiskRegistry contract at 35,462 gas per record. The results indicate that, for sparse transaction graphs, feature engineering contributes more than model complexity, offering a lightweight and auditable alternative for wallet risk screening.


### §1 Introduction — paragraf penutup + kontribusi (ganti daftar kontribusi lama)

> **Research gap and purpose.** Existing Ethereum phishing detectors largely compete on model architecture, while the quality and provenance of the input features receive little systematic attention, and results are seldom checked for split-level variance or test-set leakage. This study isolates the data from the model: it holds the wallet population, labels and splits fixed, and varies only the feature sources and the classifier.
>
> **Contributions.**
> 1. A *graph-to-tabular transformation* that turns a wallet's one-hop transaction neighbourhood into 18 label-free statistics, which can be computed identically from the offline MulDiGraph and from a live Etherscan transaction list, so the same model can score wallets outside the dataset.
> 2. A *controlled 4 × 4 evaluation* (4 feature scenarios × 4 model families, 10 seeds) that separates the gain from data enrichment (+1.9 F1 points from Etherscan) from the gain from model complexity (none from GraphSAGE or embedding fusion), with explicit leakage controls.
> 3. A *deployable risk pipeline*: FastAPI inference (3.5 ms per wallet, 878 KB model), calibrated risk scores and an on-chain RiskRegistry audit trail with measured gas cost.
>
> **Implications.** For sparse, label-seeded transaction graphs, feature provenance and enrichment matter more than message passing; practitioners can obtain near-best accuracy with an interpretable tree model whose features are computable in real time.

### §1 — Kriteria studi Tabel 1 (tambahkan sebelum Tabel 1)

> The 28 studies in Table 1 were identified by searching IEEE Xplore, ScienceDirect, SpringerLink, ACM Digital Library and Google Scholar (2019–2025) with the query ("Ethereum" OR "blockchain") AND ("phishing" OR "fraud" OR "scam") AND ("detection" OR "classification"). A study was retained if it (i) addressed account- or transaction-level phishing/fraud detection on Ethereum, (ii) reported a quantitative evaluation (at least F1, precision/recall or AUC) on a named dataset, and (iii) was peer-reviewed. Duplicates, surveys without experiments and non-Ethereum chains were excluded. Each study was categorized by *feature type* (handcrafted/tabular, graph embedding, GNN end-to-end, hybrid) and *data source* (EPTransNet/XBlock, Etherscan, other), which are the two axes of Table 1.

⚠️ *Sesuaikan database, rentang tahun dan query dengan yang benar-benar kamu lakukan — jangan ditempel kalau tidak sesuai.*

### §2.1 Dataset — tambahan

> The population contains all 1,165 phishing-labelled nodes of EPTransNet and 11,650 normal nodes sampled uniformly at random (ratio 1:10, seed 42), for 12,815 wallets in total. The ratio 1:10 was chosen as the most conservative setting we could fully enrich: Table A shows that increasing the share of normal wallets lowers precision and F1 while ROC-AUC stays constant (≈ 0.994), so 1:10 gives the least optimistic estimate. Every experiment is repeated over 10 independent stratified 80:20 splits (seeds 42, 0–8), and results are reported as mean ± standard deviation.

**Table A. Majority-class ratio sensitivity (XGBoost, S2, 5 seeds)**

| Ratio | N | `scale_pos_weight` | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|---|
| 1:2 | 3,495 | 2.00 | 0.944 ± 0.024 | 0.952 ± 0.010 | 0.948 ± 0.010 | 0.994 ± 0.002 | 0.988 ± 0.003 |
| 1:5 | 6,990 | 5.00 | 0.912 ± 0.008 | 0.946 ± 0.023 | 0.929 ± 0.010 | 0.994 ± 0.002 | 0.976 ± 0.006 |
| 1:10 | 12,815 | 9.99 | 0.863 ± 0.023 | 0.945 ± 0.008 | 0.902 ± 0.015 | 0.995 ± 0.001 | 0.966 ± 0.006 |

### §2.2 Feature transformation — tambahan

> **Leakage control.** The 18 one-hop statistics are computed from the full MulDiGraph before splitting; they aggregate edge counts, amounts and timestamps only and never read a neighbour's label. All fitted steps — scaling, capping, early stopping, threshold selection and calibration — are fitted on training wallets only. GraphSAGE additionally uses the induced topology among the 12,815 wallets (354 directed edges); an inductive variant trained without any edge incident to a test wallet performs the same as the transductive model (Table 3), confirming that the topology does not leak test information.
>
> **Isolated wallets.** Every dataset wallet has at least one transaction in MulDiGraph, so all 18 statistics are defined. In the induced GNN graph, however, 96.6% of wallets (98.4% of normal, 78.8% of phishing) have no neighbour inside the population; for them GraphSAGE reduces to a self-loop MLP. In live mode, a wallet with an empty Etherscan transaction list is reported as "not found" and is not scored.
>
> **BigQuery coverage.** The BigQuery enrichment aggregated a rolling 90-day window before the query date. Because EPTransNet wallets were active between August 2015 and January 2019, only 126 of 12,815 wallets (0.98%) had any activity in that window; the low coverage is therefore caused by the query timeframe rather than by the public dataset, which holds the complete chain history.
>
> **Live computation cost.** For a wallet with *k* transactions the 18 statistics are computed in O(k) time and memory (≈ 18 ms in practice). End-to-end live scoring requires five Etherscan requests and takes 8–17 s (median 9.1 s), dominated by network latency; under the Etherscan free tier (3 calls/s, 100,000 calls/day) this bounds live throughput to ≈ 0.6 wallets/s or ≈ 20,000 wallets/day per API key, independent of FastAPI concurrency.

**Table 2. Feature dictionary (36 attributes)** — *(tabel lengkap di bawah, letakkan di §2.2)*

| # | Attribute | Source | Definition (per wallet *v*) | Scenarios |
|---|---|---|---|---|
| 1 | `in_degree` | MulDiGraph | Number of incoming transaction edges | S1–S4 |
| 2 | `out_degree` | MulDiGraph | Number of outgoing transaction edges | S1–S4 |
| 3 | `unique_senders` | MulDiGraph | Distinct predecessor addresses | S1–S4 |
| 4 | `unique_receivers` | MulDiGraph | Distinct successor addresses | S1–S4 |
| 5 | `total_received` | MulDiGraph | Sum of incoming ETH | S1–S4 |
| 6 | `total_sent` | MulDiGraph | Sum of outgoing ETH | S1–S4 |
| 7 | `avg_in_amount` | MulDiGraph | Mean incoming amount | S1–S4 |
| 8 | `avg_out_amount` | MulDiGraph | Mean outgoing amount | S1–S4 |
| 9 | `max_in_amount` | MulDiGraph | Maximum incoming amount | S1–S4 |
| 10 | `max_out_amount` | MulDiGraph | Maximum outgoing amount | S1–S4 |
| 11 | `min_in_amount` | MulDiGraph | Minimum incoming amount | S1–S4 |
| 12 | `min_out_amount` | MulDiGraph | Minimum outgoing amount | S1–S4 |
| 13 | `std_in_amount` | MulDiGraph | Std. dev. of incoming amounts | S1–S4 |
| 14 | `std_out_amount` | MulDiGraph | Std. dev. of outgoing amounts | S1–S4 |
| 15 | `in_out_ratio` | MulDiGraph | in_degree / (out_degree + 1) | S1–S4 |
| 16 | `sent_received_ratio` | MulDiGraph | total_sent / (total_received + 10⁻⁹) | S1–S4 |
| 17 | `timestamp_spread_in` | MulDiGraph | max − min timestamp of incoming edges (s) | S1–S4 |
| 18 | `timestamp_spread_out` | MulDiGraph | max − min timestamp of outgoing edges (s) | S1–S4 |
| 19 | `graph_present` | MulDiGraph | 1 if the wallet is a MulDiGraph node (join audit flag) | audit only |
| 20 | `created_at_ts` | Etherscan | First-transaction timestamp (constant 0 in collected data) | S2, S4 |
| 21 | `balance` | Etherscan | Current ETH balance | S2, S4 |
| 22 | `transaction_count` | Etherscan | Number of normal transactions (≤ 10,000, API paging) | S2, S4 |
| 23 | `activity_score` | Etherscan | Transactions in the latest 100-transaction window | S2, S4 |
| 24 | `avg_tx_value` | Etherscan | Mean ETH value in that window | S2, S4 |
| 25 | `tx_frequency` | Etherscan | activity_score / 30 | S2, S4 |
| 26 | `unique_counterparties` | Etherscan | Distinct counterparties in that window | S2, S4 |
| 27 | `bq_tx_count_out` | BigQuery | Outgoing transactions in the query window | S3, S4 |
| 28 | `bq_tx_count_in` | BigQuery | Incoming transactions in the query window | S3, S4 |
| 29 | `bq_total_out_eth` | BigQuery | ETH sent in the query window | S3, S4 |
| 30 | `bq_total_in_eth` | BigQuery | ETH received in the query window | S3, S4 |
| 31 | `bq_unique_to` | BigQuery | Distinct recipients in the window | S3, S4 |
| 32 | `bq_unique_from` | BigQuery | Distinct senders in the window | S3, S4 |
| 33 | `bq_active_days_out` | BigQuery | Days with outgoing activity | S3, S4 |
| 34 | `bq_active_days_in` | BigQuery | Days with incoming activity | S3, S4 |
| 35 | `bq_first_seen_ts` | BigQuery | Earliest transaction in the window | S3, S4 |
| 36 | `bq_last_seen_ts` | BigQuery | Latest transaction in the window | S3, S4 |

*Predictors: S1 = 18, S2 = 25, S3 = 28, S4 = 35. `graph_present` is collected but not used as a predictor.*

### §2.3 Models — tambahan

> Class imbalance is handled without resampling. XGBoost uses `scale_pos_weight = N_neg / N_pos` computed on the training portion of each split (9.993 at 1:10); logistic regression and random forest use balanced class weights; GraphSAGE uses weighted cross-entropy with *w_c = N / (2N_c)*. XGBoost: 250 rounds, max depth 6, learning rate 0.05, subsample 0.85, column subsample 0.85, early stopping (40 rounds) on a validation slice (20% of training). GraphSAGE: two SAGEConv layers (mean aggregation), hidden size 128, BatchNorm, ReLU, dropout 0.5, Adam (lr = 0.01), ≤ 200 epochs, early stopping with patience 25 on F1 of a validation slice (20% of training wallets). No model ever observes the test set before its single final evaluation.

### §2.4 Late fusion — tambahan (jawaban C5 / A5)

> The fusion encoder is the S1 GraphSAGE model above (18 graph features, same architecture and stopping rule). Its 128-dimensional pre-head representation (output of the last SAGEConv→BatchNorm→ReLU block) is concatenated with the 7 standardized Etherscan features into a 135-dimensional vector and classified by XGBoost with the same recipe as the tabular baseline. An embedding-only variant (128-d) isolates the contribution of the embedding.

### §2.5 System and RiskRegistry — tambahan

> The production model is the XGBoost S2 booster (JSON, 878 KB, 250 trees, 25 features); loading takes 76 ms and adds 3.5 MB of resident memory, and the full service path (schema validation, probability, risk score, top-5 explanation) takes 3.5 ms median (6.4 ms p95) per wallet. The risk score is currently the uncalibrated probability × 100; Platt scaling fitted on the validation slice halves the expected calibration error (0.0136 → 0.0065; Table C) and will be applied in the next model release. A `setRisk()` write consumes 35,462 gas; at 0.1–30 gwei this is 3.5 × 10⁻⁶ – 1.06 × 10⁻³ ETH, i.e. USD 0.008–2.46 at an ETH price of USD 2,313.76 (26 Oct 2026). The 13.6 ms local commit latency becomes one block interval (≈ 12 s) on Ethereum mainnet.

### §3 Results — Table 3 baru (mean ± std, 10 seeds)

**Table 3. F1 by scenario and model (mean ± std over 10 stratified splits)**

| Scenario | LR | RF | XGBoost | GraphSAGE |
|---|---|---|---|---|
| S1 MulDiGraph (18) | 0.698 ± 0.031 | 0.882 ± 0.014 | 0.881 ± 0.015 | 0.780 ± 0.032 |
| S2 + Etherscan (25) | 0.762 ± 0.037 | 0.904 ± 0.011 | **0.900 ± 0.013** | 0.858 ± 0.040 |
| S3 + BigQuery (28) | 0.707 ± 0.042 | 0.882 ± 0.014 | 0.879 ± 0.013 | 0.783 ± 0.032 |
| S4 + Etherscan + BigQuery (35) | 0.777 ± 0.036 | 0.901 ± 0.014 | 0.901 ± 0.012 | 0.847 ± 0.040 |
| S2, GraphSAGE inductive | — | — | — | 0.862 ± 0.020 |

**Table 3b. Secondary metrics, S2 (mean ± std)**

| Model | Precision | Recall | ROC-AUC | PR-AUC | FP | FN |
|---|---|---|---|---|---|---|
| LR | 0.774 ± 0.033 | 0.751 ± 0.045 | 0.968 ± 0.011 | 0.777 ± 0.042 | 51.1 | 58.0 |
| RF | 0.894 ± 0.019 | 0.915 ± 0.016 | 0.993 ± 0.004 | 0.955 ± 0.010 | 25.5 | 19.8 |
| XGBoost | 0.867 ± 0.021 | **0.937 ± 0.015** | **0.994 ± 0.002** | **0.963 ± 0.009** | 33.8 | **14.6** |
| GraphSAGE | 0.817 ± 0.030 | 0.904 ± 0.060 | 0.986 ± 0.007 | 0.886 ± 0.053 | 47.1 | 22.3 |

> Discussion paragraph: Etherscan enrichment adds 1.9 F1 points to XGBoost (0.881 → 0.900) and 2.2 to random forest, consistently across all 10 splits, whereas BigQuery adds nothing (S3 ≈ S1, S4 ≈ S2) because of its 0.98% coverage. Random forest and XGBoost are statistically indistinguishable in F1 (0.904 ± 0.011 vs 0.900 ± 0.013); XGBoost is retained for deployment because it gives the highest recall, ROC-AUC and PR-AUC and the fewest missed phishing wallets (14.6 vs 19.8 per test split), which is the costlier error (Table D). GraphSAGE trails both tree models in every scenario (0.780–0.858 F1) and shows the largest variance across splits (std up to 0.040). This is consistent with the sparsity of the induced graph: 96.6% of wallets have no neighbour inside the population, so message passing has little to aggregate, and the model effectively becomes a less regularized MLP on the same tabular features.

### §3 — Table 4 baru (fusion, 10 seeds)

**Table 4. Late fusion and embedding ablation (mean ± std, 10 seeds)**

| Model | Input | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|---|
| XGBoost tabular (S2) | 25 features | 0.867 ± 0.021 | 0.937 ± 0.015 | **0.900 ± 0.013** | 0.994 ± 0.002 | 0.963 ± 0.009 |
| GraphSAGE (S1 encoder) | 18 graph features + topology | 0.717 ± 0.020 | 0.859 ± 0.070 | 0.780 ± 0.032 | 0.966 ± 0.016 | 0.839 ± 0.061 |
| Late fusion | 128-d emb ‖ 7 Etherscan | 0.836 ± 0.024 | 0.922 ± 0.017 | 0.877 ± 0.018 | 0.989 ± 0.004 | 0.941 ± 0.015 |
| Embedding only | 128-d emb | 0.738 ± 0.035 | 0.894 ± 0.024 | 0.808 ± 0.023 | 0.979 ± 0.006 | 0.895 ± 0.025 |

**Table B. GraphSAGE protocol decomposition (S2 split, seed 42)**

| Protocol | Early stopping on | Input | Edges | F1 | ROC-AUC |
|---|---|---|---|---|---|
| A — old Table 3 | **test** F1 | 25 features | MulDiGraph + Etherscan | 0.873 | 0.988 |
| B | **test** F1 | 25 features | MulDiGraph | 0.874 | 0.989 |
| C — revised Table 3 | validation F1 | 25 features | MulDiGraph | 0.853 | 0.985 |
| D — revised Table 4 encoder | validation F1 | 18 graph features | MulDiGraph | 0.788 | 0.972 |

> On the same split, early stopping on test F1 inflates GraphSAGE F1 by 2.1 points (B → C), the 7 Etherscan inputs account for 6.5 points (D → C), and the Etherscan-derived edges have no measurable effect (A ≈ B). The original 0.8960 vs 0.8184 gap is therefore mainly explained by the input feature set and, secondarily, by the stopping protocol.

### §3.3 — Table 6 baru (dipisah dua panel)

**Table 6(a). Prediction performance of the deployed model (XGBoost S2)**

| Metric | Seed 42 split | Mean ± std (10 splits) |
|---|---|---|
| Accuracy | 0.9789 | 0.9811 ± 0.0025 |
| Precision | 0.8403 | 0.8665 ± 0.0213 |
| Recall | 0.9485 | 0.9373 ± 0.0152 |
| F1 | 0.8911 | 0.9003 ± 0.0128 |
| ROC-AUC | 0.9938 | 0.9943 ± 0.0020 |
| PR-AUC | 0.9582 | 0.9628 ± 0.0087 |
| TN / FP | 2,288 / 42 | 2,296.2 / 33.8 |
| FN / TP | 12 / 221 | 14.6 / 218.4 |

**Table 6(b). System and deployment performance**

| Metric | Value |
|---|---|
| Model file size | 878 KB (250 trees, 25 features) |
| Model load time / memory increment | 76 ms / 3.5 MB |
| Inference latency, single wallet (service path) | 3.5 ms median, 6.4 ms p95 |
| Inference throughput, batch | 2.0 µs per wallet (2,563 wallets in 5.1 ms) |
| Live feature extraction (Etherscan) | 8.0–16.6 s end-to-end, 5 API calls, ≈ 18 ms compute |
| RiskRegistry `setRisk()` gas | 35,462 (range 35,452–35,464) |
| Commit latency, Hardhat local | 13.6 ms mean (11.2–38.4) |
| Est. public-network cost | USD 0.008–2.46 (0.1–30 gwei) |

**Table C. Probability calibration (XGBoost S2, 10 seeds)** — lihat jawaban B4. Tambah gambar `outputs/revision/reliability_diagram.png` sebagai Figure baru.

**Table D. Cost-sensitive threshold** — lihat jawaban B7.

**Feature ranking (tambahkan rumus di §3.3):**

> The combined importance is *I(f) = ½ [G(f)/Σ_j G(j) + M(f)/Σ_j M(j)]*, where *G* is XGBoost total gain and *M* is random-forest mean decrease in impurity, both computed on the training split. Correlation is used as a post-filter: for every pair with |r| > 0.95, the feature with lower *I* is removed (`in_degree`, `tx_frequency`). The top features are `unique_senders` (I = 0.311), `in_degree` (0.101), `timestamp_spread_in` (0.080), `sent_received_ratio` (0.058) and `total_sent` (0.051).

### §4 Conclusion — paragraf future impact

> Beyond the benchmark, the results suggest that wallet risk screening can be built from features that are cheap, interpretable and computable in real time from any block explorer, rather than from large graph models that require the full transaction graph. Future work will (i) extend the BigQuery enrichment to the full chain history to remove the coverage gap, (ii) apply Platt-calibrated scores and cost-sensitive thresholds to the on-chain risk index, (iii) re-label recent phishing modalities (token approvals, drainer contracts) absent from the 2015–2019 EPTransNet snapshot, and (iv) evaluate on Layer-2 networks where the RiskRegistry write costs fractions of a cent.

### Bagian wajib template (isi data aslimu)

> **Funding.** This research was supported by [nama pemberi dana] under research grant/contract no. [nomor kontrak]. — *atau, kalau memang tidak ada:* "This research received no specific research grant or contract from any funding agency in the public, commercial, or not-for-profit sectors."
>
> **Acknowledgement.** The authors thank [institusi/pembimbing] for … and the XBlock team for releasing the EPTransNet dataset.
>
> **CRediT author statement.** **Muhammad Supriyadi:** Conceptualization, Methodology, Software, Validation, Formal analysis, Investigation, Data curation, Writing – original draft, Visualization. **Kursehi [nama lengkap]:** Conceptualization, Supervision, Writing – review & editing, Project administration. *(sesuaikan)*
>
> **Data availability.** The EPTransNet MulDiGraph is publicly available from XBlock (http://xblock.pro/). The derived 12,815-wallet dataset, feature schema, trained model and all experiment scripts are available at [URL repo GitHub/Zenodo] or from the corresponding author upon reasonable request.
>
> **Conflict of interest.** The authors declare no conflict of interest.

---

## BAGIAN 3 — Yang harus kamu kerjakan sendiri

1. **Ganti semua angka lama (8.595 / F1 0,9102 / 0,8960 / 0,8184 / 0,93%)** di Abstract, Results, Discussion, dan Conclusion dengan angka di atas.
2. **Gambar 1:** tambahkan label S1–S4 di blok "Hybrid feature construction".
3. **Gambar 2:** buat ulang dari Table 3 (mean ± error bar std), lalu tambahkan catatan sumbu-y di caption.
4. **Urutan rujukan:** pastikan setiap "Table X / Figure X" disebut di teks *sebelum* tabel/gambarnya muncul.
5. **Kriteria Tabel 1:** sesuaikan dengan cara kamu benar-benar memilih 28 studi itu.
6. **Funding, CRediT, Acknowledgement:** isi data yang benar. Editor minta frasa "research grant or contract", jadi kalau tidak ada hibah, pakai kalimat "received no specific research grant or contract".
7. **Biografi penulis:** sekitar 100 kata per penulis + foto, sesuai format IAES (pendidikan, afiliasi, minat riset, email, ORCID/Scopus ID).
8. **Highlight kuning** setiap bagian yang berubah. Upload ke sistem IJECE dengan **ID yang sama (42848)**, bukan sebagai submission baru, paling lambat **5 Oktober 2026** (8 minggu dari 10 Agustus). ⚠️ *Hari ini 27 Oktober, jadi tenggat ini sudah lewat. Segera minta perpanjangan ke editor.*
