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
| Source | [XBlock – Ethereum Phishing Transaction Network](http://xblock.pro/ethereum/) |
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
