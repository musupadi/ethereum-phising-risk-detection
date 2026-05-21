# Ethereum Phishing Risk Registry

AI + Blockchain risk scoring system that:
1. Trains a Random Forest classifier on the **Ethereum Phishing Transaction Network** (XBlock / InPlusLab) dataset.
2. Produces a `risk_score` (0–100) per wallet.
3. Anchors results to a **Solidity smart contract** running on a local Hardhat node.
4. Exposes a **FastAPI** REST API and a **Streamlit** demo UI.

---

## Repository Structure

```
.
├── data/
│   ├── raw/                  # MulDiGraph.pkl (NetworkX graph)
│   └── processed/            # train.csv, test.csv (generated)
├── notebooks/
│   └── eda.ipynb             # Exploratory data analysis
├── ml/
│   ├── preprocess.py         # Graph → tabular feature extraction
│   ├── train.py              # Train LR baseline + Random Forest
│   ├── predict.py            # Inference helper
│   └── artifacts/            # model.pkl, scaler.pkl, feature_schema.json, plots
├── api/
│   ├── main.py               # FastAPI app
│   ├── schemas.py            # Pydantic models
│   ├── config.py             # Env-based config
│   └── services/
│       ├── ml_service.py
│       └── chain_service.py  # web3.py bridge
├── chain/
│   ├── contracts/
│   │   └── RiskRegistry.sol  # Solidity contract
│   ├── scripts/
│   │   └── deploy.ts         # Hardhat deploy script
│   ├── hardhat.config.ts
│   └── package.json
├── app/
│   └── streamlit_app.py      # Streamlit UI
├── requirements.txt
└── README.md
```

---

## Dataset

| Property | Value |
|---|---|
| Source | [Kaggle - Ethereum Phishing Transaction Network](https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network/data) (mirror of [XBlock](http://xblock.pro/ethereum/)) |
| Format | NetworkX `MulDiGraph` (pickle) |
| Nodes | 2,973,489 |
| Edges | 13,551,303 |
| Labeled nodes | 1,165 phishing (`isp=1`) |
| Edge features | `amount`, `timestamp` |

Place `MulDiGraph.pkl` in `data/raw/` (or leave it under `Ethereum Phishing Transaction Network/`; the scripts check both locations).

---

## Setup

### Python environment

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Blockchain (Hardhat)

```bash
cd chain
npm install
```

---

## Workflow

### Step 1 – Feature extraction

```bash
python ml/preprocess.py
```

Extracts 18 graph-topology features per node (degree, amount stats, timestamp spread, etc.),
samples 10× non-phishing nodes, splits 80/20, and writes:
- `data/processed/train.csv`
- `data/processed/test.csv`
- `ml/artifacts/feature_schema.json`

### Step 2 – Train models

```bash
python ml/train.py
```

Trains Logistic Regression (baseline) and Random Forest (main).
Prints a comparison table, saves:
- `ml/artifacts/model.pkl`
- `ml/artifacts/scaler.pkl`
- `ml/artifacts/confusion_matrix.png`
- `ml/artifacts/feature_importance.png`

### Step 3 – Start Hardhat node + deploy contract

```bash
# Terminal 1
cd chain
npx hardhat node

# Terminal 2
cd chain
npx hardhat run scripts/deploy.ts --network localhost
```

Deployment info written to `chain/artifacts/deployment.json` and ABI to `chain/artifacts/RiskRegistry.abi.json`.

### Step 4 – Start FastAPI

```bash
# from repo root
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

API docs: http://localhost:8000/docs

### Step 5 – Start Streamlit

```bash
streamlit run app/streamlit_app.py
```

UI: http://localhost:8501

### Step 6 - Etherscan enrichment (real wallet metrics)

This step appends real on-chain summary features (balance, tx count, activity proxies)
for addresses in `data/raw/addresses.csv` to:
- `data/raw/addresses_etherscan_enriched.csv`

Run one batch manually:

```bash
python ml/enrich_real_addresses_etherscan.py
```

Recommended (simplest) workflow on Windows:
1. Activate venv once.
2. Put `ETHERSCAN_API_KEY`, `ETHERSCAN_REQUEST_DELAY`, and `ETHERSCAN_PER_RUN_LIMIT` in `.env`.
3. Run `python ml/enrich_real_addresses_etherscan.py`.

PowerShell activation example:

```powershell
cd /d C:\Kuliah\ethereum-phising-risk-detection
.\.venv\Scripts\Activate.ps1
python ml\enrich_real_addresses_etherscan.py
```

Control batch size and request pacing (recommended for free Etherscan key):

```powershell
# Windows PowerShell example
cd /d C:\Kuliah\ethereum-phising-risk-detection
$env:ETHERSCAN_PER_RUN_LIMIT='1000'
$env:ETHERSCAN_REQUEST_DELAY='0.4'
.\.venv\Scripts\python.exe ml\enrich_real_addresses_etherscan.py
```

Notes:
- `ETHERSCAN_PER_RUN_LIMIT=0` means process all remaining addresses.
- Script is resumable: it skips addresses already present in `addresses_etherscan_enriched.csv`.

### Step 7 - BigQuery enrichment (additional transaction features)

This optional step adds aggregate transaction features from BigQuery public Ethereum data
for addresses in `data/raw/addresses.csv` to:
- `data/raw/addresses_bigquery_enriched.csv`

Required setup:
- Create a Google Cloud project and enable BigQuery API.
- Create service account credentials JSON and set `GOOGLE_APPLICATION_CREDENTIALS`.
- Set your GCP project id in `BIGQUERY_PROJECT_ID`.
- Grant IAM roles to the service account: `BigQuery Job User` (required for `bigquery.jobs.create`) and `BigQuery Data Viewer`.

Run one batch manually:

```powershell
# Windows PowerShell example (use PowerShell, not Command Prompt)
cd /d C:\Kuliah\ethereum-phising-risk-detection
$env:GOOGLE_APPLICATION_CREDENTIALS='C:\path\to\service-account.json'
$env:BIGQUERY_PROJECT_ID='your-gcp-project-id'
$env:BIGQUERY_DAYS_WINDOW='90'
$env:BIGQUERY_PER_RUN_LIMIT='1000'
.\.venv\Scripts\python.exe ml\enrich_real_addresses_bigquery.py
```

Quick run (if venv is already active):

```powershell
python ml\enrich_real_addresses_bigquery.py
```

Quick run (without activating venv first):

```powershell
.\.venv\Scripts\python.exe ml\enrich_real_addresses_bigquery.py
```

Recommended (simplest) workflow on Windows:
1. Activate venv once.
2. Put `GOOGLE_APPLICATION_CREDENTIALS`, `BIGQUERY_PROJECT_ID`, `BIGQUERY_DAYS_WINDOW`, `BIGQUERY_PER_RUN_LIMIT`, and `BIGQUERY_CHUNK_SIZE` in `.env`.
3. Run `python ml\enrich_real_addresses_bigquery.py`.

```powershell
cd /d C:\Kuliah\ethereum-phising-risk-detection
.\.venv\Scripts\Activate.ps1
python ml\enrich_real_addresses_bigquery.py
```

If you prefer Command Prompt (`cmd`), activate venv like this first:

```bat
cd /d C:\Kuliah\ethereum-phising-risk-detection
.venv\Scripts\activate.bat
python ml\enrich_real_addresses_bigquery.py
```

If you use Command Prompt (`cmd`), use this syntax instead:

```bat
cd /d C:\Kuliah\ethereum-phising-risk-detection
set GOOGLE_APPLICATION_CREDENTIALS=C:\Kuliah\ethereum-phising-risk-detection\primal-library-239611-b6941382db9f.json
set BIGQUERY_PROJECT_ID=primal-library-239611
set BIGQUERY_DAYS_WINDOW=90
set BIGQUERY_PER_RUN_LIMIT=1000
.venv\Scripts\python.exe ml\enrich_real_addresses_bigquery.py
```

Command Prompt quick run (if venv already active):

```bat
python ml\enrich_real_addresses_bigquery.py
```

Notes:
- `BIGQUERY_PER_RUN_LIMIT=0` means process all remaining addresses.
- Script is resumable: it skips addresses already present in `addresses_bigquery_enriched.csv`.
- `ml/build_hybrid_dataset.py` will auto-merge BigQuery features when this file exists.
- Keep the service-account JSON private and never commit it to git.
- In `cmd`, do not use `$env:...`; that syntax only works in PowerShell.

### Step 8 - Build combined dataset and compare scenarios

After Etherscan and BigQuery enrichment are available, rebuild the combined dataset:

```bash
python ml/build_hybrid_dataset.py
```

Then compare the four scenarios on the same split:

```bash
python ml/train_dataset_scenarios.py
```

Scenarios compared:
- `muldigraph_only`
- `muldigraph_plus_etherscan`
- `muldigraph_plus_bigquery`
- `muldigraph_plus_etherscan_plus_bigquery`

Models compared in each scenario:
- `LogisticRegression`
- `RandomForest`
- `XGBoost`
- `GNN-GCN-V2`

Outputs:
- `ml/artifacts/dataset_scenarios_comparison.csv`
- `ml/artifacts/dataset_scenarios_summary.json`

### Step 9 - Daily one-click run (Windows)

Use the provided launcher from repository root:

```bat
run_daily_etherscan.bat
```

For BigQuery enrichment:

```bat
run_daily_bigquery.bat
```

Launchers call `scripts/run_daily_etherscan_enrichment.ps1` or `scripts/run_daily_bigquery_enrichment.ps1` and write logs to:
- `data/raw/logs/`

---

## API Reference

### `POST /analyze`

```json
// Demo mode (fastest)
{
  "wallet": "0xAbCd...1234",
  "row_id": 42
}

// Manual mode
{
  "wallet": "0xAbCd...1234",
  "features": { "in_degree": 5, "out_degree": 2, ... }
}
```

Response:
```json
{
  "wallet": "0xAbCd...1234",
  "probability": 0.87,
  "risk_score": 87,
  "risk_level": "HIGH",
  "tx_hash": "0x..."
}
```

### `GET /risk/{wallet}`

Returns on-chain stored record:
```json
{
  "wallet": "0x...",
  "risk_score": 87,
  "timestamp": 1719999999,
  "model_version": "rf-v1"
}
```

---

## Risk Levels

| Score Range | Level |
|---|---|
| 0 – 39 | 🟢 LOW |
| 40 – 69 | 🟡 MEDIUM |
| 70 – 100 | 🔴 HIGH |

---

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `RPC_URL` | `http://127.0.0.1:8545` | Hardhat RPC endpoint |
| `CHAIN_ENABLED` | `true` | Set to `false` to skip chain writes |
| `SIGNER_PRIVATE_KEY` | Hardhat account #0 | Tx signer private key |
| `MODEL_VERSION` | `rf-v1` | Version tag stored on-chain |
| `API_HOST` | `0.0.0.0` | FastAPI host |
| `API_PORT` | `8000` | FastAPI port |
| `BIGQUERY_PROJECT_ID` | `` | GCP project used by BigQuery enrichment script |
| `BIGQUERY_DAYS_WINDOW` | `90` | Lookback window in days for BigQuery tx aggregates |
| `BIGQUERY_PER_RUN_LIMIT` | `0` | Max addresses per run (`0` = all pending) |
| `BIGQUERY_CHUNK_SIZE` | `300` | Addresses per BigQuery query batch |
| `GOOGLE_APPLICATION_CREDENTIALS` | `` | Path to GCP service-account JSON key |

---

## Smart Contract

`RiskRegistry.sol` (Solidity 0.8.20):

```
setRisk(address wallet, uint16 score, string modelVersion)  → emits RiskUpdated
getRisk(address wallet) → (uint16 score, uint64 timestamp, string modelVersion)
```

Only the deployer (owner) can write records.

---

## Citation

```bibtex
@misc{xblockEthereum,
  author = {Chen, Liang and Peng, Jiaying and Liu, Yang and Li, Jintang and Xie, Fenfang and Zheng, Zibin},
  title  = {{XBLOCK Blockchain Datasets}: {InPlusLab} Ethereum Phishing Detection Datasets},
  howpublished = {\url{http://xblock.pro/ethereum/}},
  year   = 2019
}
```
