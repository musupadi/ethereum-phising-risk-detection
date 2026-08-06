# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A thesis project: AI + blockchain phishing-risk scoring for Ethereum wallets. A model
scores a wallet 0–100, and the score is optionally anchored on-chain in a Solidity
`RiskRegistry` contract. Three tiers run as separate processes:

**Streamlit UI (`:8502`) → FastAPI (`:8000`) → XGBoost model + Hardhat node (`:8545`)**

The backend is required; the blockchain layer is optional (toggle with `CHAIN_ENABLED`).

## Two model tracks — don't conflate them

There are two distinct model pipelines. This is the single most important thing to
understand before touching `ml/`:

1. **Thesis comparison** (`ml/train_dataset_scenarios.py`) — trains 4 models
   (LogisticRegression, RandomForest, XGBoost, GNN-GraphSAGE V2) × 4 dataset scenarios
   (`muldigraph_only`, `muldigraph_plus_etherscan`, `muldigraph_plus_bigquery`,
   `muldigraph_plus_etherscan_plus_bigquery`). Produces the tables/figures/confusion
   matrices under `ml/artifacts/`. This is *evaluation*, not what gets served.
2. **Production/served model** (`ml/train_production_model.py`) — trains the single model
   the API actually loads: **XGBoost on MulDiGraph+Etherscan, 25 features (18 graph +
   7 Etherscan), F1 ≈ 91%**. `graph_present` is a join-coverage audit column, listed
   separately under `audit_columns` in the schema — it is never fed to the model.
   Writes `ml/artifacts/model_prod_xgb.json` + `prod_feature_schema.json` and the
   held-out demo set `data/processed/prod_demo.csv`.

`ml/train.py` is a legacy baseline (LR + RF, 18 graph features → `model.pkl`). The API
falls back to it only if the production XGBoost artifacts are missing. Prefer the
production track for anything user-facing.

Artifacts are committed, so the app runs without any training.

## Inference path (spans several files)

`api/main.py` `/analyze` → `api/services/ml_service.py` → `ml/predict.py:predict_full`.

- `ml/predict.py` lazy-loads the production XGBoost booster (or legacy sklearn model),
  orders incoming features by the schema's `features` list, and **defaults any missing
  feature to 0.0**. Score = `round(proba * 100)`; levels: LOW <40, MEDIUM 40–69, HIGH ≥70.
- `ml_service.py` supplies the features three ways, matching the UI's three input modes:
  - **Demo (`row_id`)** → row from `prod_demo.csv` (held-out, has ground-truth label).
  - **Scan by address** (`wallet` only) → looked up in `hybrid_dataset.csv` by lowercased
    `node_id`. Only dataset wallets (MulDiGraph 2019 snapshot) resolve; others 404.
  - **Manual** (`features` dict) → used as-is; Etherscan features you omit become 0.
- `_META_COLS` in `ml_service.py` lists string/label columns that must never reach the
  model. Add any new non-feature column there.
- `api/config.py` picks CSVs with fallbacks: `TEST_CSV` = `prod_demo.csv` else `test.csv`;
  `LOOKUP_CSV` = `hybrid_dataset.csv` else `TEST_CSV`. So the served feature dimension
  depends on which CSVs exist on disk — keep `prod_demo.csv`/`hybrid_dataset.csv` in sync
  with `prod_feature_schema.json`.

## Chain write

On `/analyze`, if `CHAIN_ENABLED=true`, `chain_service.set_risk` writes the score via
web3.py using the ABI + address in `chain/artifacts/deployment.json` (produced by
`deploy.ts`). Chain-write failures are caught and non-fatal (`tx_hash` → null). The wallet
stored on-chain is the dataset `node_id` when it's a valid address, not the free-typed
label. `/records` reconstructs the audit table from `RiskUpdated` event logs.

## Commands (Windows PowerShell; venv at `.venv`)

```powershell
.\.venv\Scripts\Activate.ps1          # activate once
pip install -r requirements.txt       # first time only
```

Run the app — each block is a separate terminal:

```powershell
# Mode A: no blockchain (2 terminals)
$env:CHAIN_ENABLED = "false"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
streamlit run app/streamlit_app.py --server.port 8502

# Mode B: with blockchain (adds, before the API terminal)
cd chain ; npx hardhat node                                   # terminal 1
cd chain ; npx hardhat run scripts/deploy.ts --network localhost  # terminal 2
```

UI → http://localhost:8502 · API docs → http://localhost:8000/docs

Health checks: `curl http://localhost:8000/health` (`chain_connected` true only in Mode B),
`curl http://localhost:8502/_stcore/health`.

Regenerate models:

```powershell
python ml\train_production_model.py    # the served XGBoost model
python ml\train_dataset_scenarios.py   # full thesis comparison tables/figures
```

There is no test suite or linter configured. `ml/predict.py` has a `__main__` smoke test
(`python ml/predict.py`).

## Data enrichment scripts (`ml/`, all resumable)

`enrich_real_addresses_etherscan.py`, `fetch_etherscan_edges.py` (GNN edges from txlist +
internal + tokentx), and `enrich_real_addresses_bigquery.py` fetch real on-chain features
for the 8,595 hybrid-dataset addresses. All skip already-processed addresses and are paced
via `ETHERSCAN_*` / `BIGQUERY_*` env vars (see README Steps 6–7). `build_hybrid_dataset.py`
merges the enriched CSVs into the hybrid dataset. **The enriched CSVs already exist** —
only re-run if the address list changes.

## Config / secrets

Runtime config is env-based (`api/config.py`, plus `.env` for enrichment keys). Key vars:
`CHAIN_ENABLED`, `RPC_URL`, `SIGNER_PRIVATE_KEY` (defaults to Hardhat account #0 — demo
only), `MODEL_VERSION` (`xgb-eth-v1`, stored on-chain), `ETHERSCAN_API_KEY`,
`GOOGLE_APPLICATION_CREDENTIALS`, `BIGQUERY_PROJECT_ID`. The repo currently contains a
`.env` and a GCP service-account JSON at root — keep those out of any commit.

## Layout

`api/` FastAPI (routes in `main.py`, feature plumbing in `services/`) · `ml/` all training,
preprocessing, inference, enrichment + `artifacts/` (committed models & figures) ·
`chain/` Hardhat/Solidity (`contracts/RiskRegistry.sol`, `scripts/deploy.ts`) ·
`app/streamlit_app.py` UI · `data/raw` (incl. `MulDiGraph.pkl`) + `data/processed` ·
`notebooks/`, `scripts/` (daily enrichment launchers). README.md is the exhaustive
reference; `hasil_implementasi.md` and the `Tesis *.docx` are thesis writeups.
