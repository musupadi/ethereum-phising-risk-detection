# Reproducibility — revision v2 (IJECE Paper ID 42848)

## Environment used

| Item | Value |
|---|---|
| OS / hardware | Windows 11 (10.0.26200), 12 logical CPU cores, no GPU |
| Python | 3.13.5 |
| Packages | numpy 2.4.4, pandas 3.0.2, scikit-learn 1.8.0, xgboost 3.2.0, torch 2.11.0+cpu, torch_geometric 2.7.0, networkx (requirements.txt), psutil 7.2.2, scipy (sklearn dependency) |
| Device | CPU. `XGBoostModel` switches to CUDA if a GPU is detected, which may change results slightly |

`requirements.txt` gives lower bounds only. Pin the versions above for bit-identical results.

## Data provenance

| Data | Source | Reproducible? |
|---|---|---|
| `data/raw/MulDiGraph.pkl` | XBlock EPTransNet (Kaggle mirror), 2,973,489 nodes, 13,551,303 edges, 2015-08-07 … 2019-01-19; SHA-256 `155fbbd7…8d14` | yes (download; not redistributed) |
| `data/raw/addresses*.csv` | `ml/extract_addresses_from_graph.py`: all 1,165 `isp = 1` nodes + 11,650 normals drawn with `random.seed(42)` from the address-sorted valid normal nodes | **yes, exactly** (verified) |
| `data/raw/addresses_etherscan_enriched.csv` | `ml/enrich_real_addresses_etherscan.py`, collected April 2026 | no (current chain state). The committed CSV is the reference |
| `data/raw/addresses_bigquery_enriched.csv` | `ml/enrich_real_addresses_bigquery.py`, 90-day window before the query date (observed activity 2026-01-22 … 2026-04-20) | no (window relative to the query date). The committed CSV is the reference |
| `data/processed/hybrid_dataset.csv` | `ml/build_hybrid_dataset.py` from the above; SHA-256 `5a6085a3…cbe3` | yes, byte-identical (verified from a clean clone) |
| Submitted 8,595-wallet population | intermediate enrichment state, never committed | **BLOCKED**: cannot be regenerated |

## Commands, runtimes and expected outputs

Run from the repository root with the virtual environment active.

| # | Command | Runtime* | Expected output |
|---|---|---|---|
| 1 | `python ml/build_hybrid_dataset.py` | ~1 min | `data/processed/hybrid_dataset.csv` (12,815 rows, hash above) |
| 2 | `python ml/train_production_model.py` | ~10 s | `ml/artifacts/model_prod_xgb.json` (SHA-256 `33962304…5c11`), F1 0.8911 |
| 3 | `python experiments/revision_ijece.py` | ~15 min | `outputs/revision/multiseed_*`, `calibration_*`, `cost_threshold_*`, `majority_ratio_*`, `graphsage_protocol_decomposition.csv`, `isolated_wallets.json`, `reliability_diagram.png`, `run_meta.json` |
| 4 | `python experiments/revision_extra.py` (after 3) | ~4 min | `temporal_split_*`, `prevalence_*`, `feature_selection_cv_*` |
| 5 | `python experiments/revision_ops.py --no-live --eth-usd 2313.76` | ~10 s | `ops_metrics.json` (model benchmarks; keeps stored live runs) |
| 6 | `cd ml; python train_dataset_scenarios.py; cd ..` | ~2 min | `ml/artifacts/dataset_scenarios_comparison.csv` (single split, seed 42) |
| 7 | `python experiments/run_graphsage_etherscan_late_fusion_v2.py` | ~1 min | `outputs/late_fusion_*_v2.*` |
| 8 | `python experiments/run_embedding_only_ablation.py` | ~1 min | `outputs/embedding_only_*`, `outputs/four_model_comparison_ablation.*` |
| 9 | `python scripts/build_final_metrics.py` | <5 s | `outputs/revision/FINAL_VERIFIED_METRICS.csv` |
| 10 | `python experiments/ijece_final_audit.py` | ~1 min | `outputs/ijece_final_audit/*.csv/json` |
| 11 | `python scripts/validate_revision.py` | ~10 s | `validation_results.json`; exit 0 when all checks pass |

\*12-core CPU, no GPU.

Optional, needs own credentials and costs API quota: `python experiments/revision_ops.py` (live Etherscan
latency, `ETHERSCAN_API_KEY`), re-enrichment scripts (Etherscan key; GCP service account for BigQuery).

## Seeds

| Use | Seed |
|---|---|
| Population sample | `random.seed(42)` |
| Outer splits, inner splits, model seeds (LR, RF, XGBoost, GraphSAGE `torch.manual_seed`) | 42, 0, 1, 2, 3, 4, 5, 6, 7, 8 |
| Majority-ratio subsampling | same 10 seeds |
| Prevalence pool of extra normals | `numpy.random.default_rng(2026)`; 5 model seeds (42, 0–3) |
| Feature-selection CV folds | `StratifiedKFold(5, shuffle=True, random_state=seed)` |

## Determinism verified

- `hybrid_dataset.csv` and `model_prod_xgb.json` rebuilt byte-identical from a clean clone (v1 smoke test,
  same machine and virtual environment).
- `revision_ijece.py` v1: two runs produced identical per-seed F1 for all 190 runs. v2 was run once.
- `train_dataset_scenarios.py`: two consecutive v2 runs produced an identical comparison CSV (after the
  M-3 seed fix).
- Not verified: a different machine, OS or GPU; a fresh `pip install`; Hardhat (`npm install`); Streamlit.

## Known non-reproducible or blocked items

1. The submitted 8,595-wallet results (BLOCKED, see above).
2. Etherscan and BigQuery enrichment and live features (depend on the date and chain state).
3. Live latency (network-dependent; 11 wallets measured).
4. Etherscan free-tier pagination after the announced 1,000-record cap (untested).
