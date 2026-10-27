# Reproducibility report — IJECE Paper ID 42848

## 1. Environment

| Item | Value |
|---|---|
| Input commit | `ec8eed308396b913d3bad6a54652104b27e74693` (the audit commit is listed in the final report) |
| OS | Windows 11 (10.0.26200), CPU only, 12 logical cores |
| Python | 3.13.5 |
| Libraries | numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.11.0+cpu, torch_geometric 2.7.0, psutil 7.2.2 |
| Seeds | split seed = model seed ∈ {42, 0, 1, 2, 3, 4, 5, 6, 7, 8}; population sample `random.seed(42)` |
| External data | `data/raw/MulDiGraph.pkl` (XBlock EPTransNet, 2,973,489 nodes, 13,551,303 edges, 2015-08-07 … 2019-01-19), not redistributed |

Full version list: `environment.json`.

## 2. Commands

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
# place MulDiGraph.pkl in data/raw/
python ml/build_hybrid_dataset.py                 # 12,815-wallet dataset (no API calls; uses committed enrichment CSVs)
python ml/train_production_model.py               # served XGBoost model + schema
cd ml; python train_dataset_scenarios.py; cd ..   # 4 x 4, seed 42
python experiments/revision_ijece.py              # all revision tables, 10 seeds (~13 min CPU)
python experiments/revision_ops.py --no-live --eth-usd 2313.76   # footprint/latency/gas (keeps stored live results)
python experiments/run_graphsage_etherscan_late_fusion_v2.py     # single-split fusion
python experiments/run_embedding_only_ablation.py
python experiments/ijece_final_audit.py           # this evidence package (no retraining)
```

Optional, needs your own credentials: `python experiments/revision_ops.py` (live Etherscan
measurement, needs `ETHERSCAN_API_KEY`), and `ml/enrich_real_addresses_*.py` (re-enrichment; the
BigQuery query also needs a GCP service account).

## 3. What was re-executed and how it matched

Smoke test on 27 Oct 2026, run from a fresh `git clone` plus the revision changes in a separate
directory, without `.env`, with `MulDiGraph.pkl` copied in. **Same machine and same virtual
environment** as the original runs (not a second computer, and not a fresh `pip install`).

| Step | Result |
|---|---|
| `build_hybrid_dataset.py` | `hybrid_dataset.csv` **byte-identical** to the committed file |
| `train_production_model.py` | `model_prod_xgb.json` **byte-identical**; F1 0.8911 |
| `train_dataset_scenarios.py` | LR / RF / XGBoost identical to the seed-42 rows of `revision_ijece.py`; GraphSAGE 0.8503 vs 0.8526 (the two scripts draw the validation slice slightly differently) |
| `revision_ijece.py` | all 190 per-seed F1 values identical to the first run |
| fusion v2, ablation, `revision_ops.py --no-live`, API `/health` + `/analyze` | exit code 0 / HTTP 200 |
| Extracted reviewer ZIP | `predict.py`, `train_production_model.py` (identical model), `revision_ops.py` OK |

Integrity checks run by `ijece_final_audit.py` (`dataset_audit.json`):
- 12,815 unique lowercase addresses, 0 duplicates, all `0x` + 40 hex.
- Labels equal MulDiGraph `isp` for 12,815 / 12,815 wallets; all 1,165 `isp = 1` nodes are included.
- The 11,650 normal wallets are reproduced exactly from `random.seed(42)` over the 2,972,324 valid
  normal nodes.
- No NaN in the dataset; every model input uses `fillna(0)`.
- Scenario sizes are 18 / 25 / 28 / 35; `graph_present` (constant 1) is in no scenario.
- The served booster's feature names equal the schema order and the S2 order. The served model is the
  evaluated seed-42 XGBoost S2.

## 4. Test-set hygiene (per fitted component)

| Component | Fitted on | Test used? |
|---|---|---|
| Train/test split | stratified 80:20 per seed | — |
| StandardScaler (LR, GraphSAGE, Etherscan branch of fusion) | training rows | no |
| XGBoost early stopping (40 rounds) | 20% validation slice of training | no |
| GraphSAGE early stopping (patience 25) | 20% validation slice of training wallets | no (**fixed** in this revision; old Table 3 used test F1) |
| Fusion encoder / XGBoost | training rows (encoder = S1 GraphSAGE, validation stopping) | no |
| Cost-sensitive threshold | XGBoost validation slice | no |
| Platt / isotonic calibration | XGBoost validation slice | no |
| Feature-ranking capping and importances | seed-42 training split | no (analysis only; not used by the deployed model) |
| 1-hop features | full MulDiGraph, label-free | transductive by construction (see A2) |

Caveat: the XGBoost validation slice is used for early stopping *and* for threshold and calibration
fitting. This does not touch the test set, but it is the same slice.

## 5. Unresolved reproducibility problems

1. **The submitted 8,595-wallet results cannot be re-run.** That intermediate population was never
   committed. Old metrics are preserved in git history (`6f98c6c`, `b632480`) and summarized in
   `reconciliation_submitted_vs_revised.csv`.
2. **The BigQuery enrichment cannot be reproduced.** Its window is relative to `CURRENT_TIMESTAMP()`,
   so re-running it returns different wallets. The committed CSV is the reference.
3. **The Etherscan enrichment and live features depend on the current chain state.** Re-enrichment
   changes balances and activity. The committed CSV is the reference.
4. **The Etherscan free-tier page cap (1,000 records, announced for July 2026) is untested** against
   the code's `offset=10000` pagination, for both enrichment and live mode.
5. `data/raw/etherscan_edges.csv` was crawled for the 8,595 population. It is disabled
   (`USE_ETHERSCAN_EDGES = False`) and not used in any reported number.
6. Not re-verified in the smoke test: Hardhat deployment (`npm install`), Streamlit UI, a fresh
   `pip install` on another machine or OS, GPU execution (XGBoost and torch switch to CUDA if present,
   which may change numbers slightly).
7. Legacy items still in the repository: `ml/train.py` (RF baseline), late-fusion V1 script and outputs
   (8,595 population), and notebooks `experiments.ipynb` / `results_testing.ipynb` with old outputs.

## 6. Measured vs derived

| Quantity | Type |
|---|---|
| Model size, load time, RSS, inference latency, batch throughput | measured (`revision_ops.py`) |
| Live latency, 5 calls per wallet | measured, 11 wallets, while another job was running |
| `setRisk()` gas, local commit latency | measured on Hardhat (20 calls) |
| Public-network ETH/USD cost | derived: gas × assumed gas price × ETH price (CoinDesk, 26 Oct 2026) |
| Mainnet confirmation latency | assumption (~1 block); not measured |
| Etherscan throughput bounds | derived from documented limits; not load-tested |
| Calibration benefit | measured offline; not deployed |
