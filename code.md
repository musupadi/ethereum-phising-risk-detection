# AI + Blockchain Risk Registry (Ethereum Phishing) — App Blueprint

Tujuan:
Membangun aplikasi yang:
1) Melatih model ML untuk klasifikasi address/wallet (phishing vs normal) dari dataset Kaggle "Ethereum Phishing Transaction Network".
2) Menghasilkan `risk_score` (0–100).
3) Menyimpan hasil skor ke smart contract (EVM / Hardhat local).
4) Menyediakan API (FastAPI) dan UI (Streamlit) untuk demo.

Dataset:
- Source: Kaggle "Ethereum Phishing Transaction Network" (XBlock) — phishing nodes dikumpulkan dari laporan (Etherscan) lalu dicrawl menjadi transaction network. :contentReference[oaicite:1]{index=1}


---

## 1) Tech Stack (disarankan biar cepat)
- Python 3.10+
- ML: pandas, scikit-learn, joblib
- API: FastAPI + uvicorn
- UI: Streamlit
- Blockchain: Hardhat (local) + Solidity
- Web3 bridge:
  - Python: web3.py (recommended)
  - (opsional) Node service: ethers.js

---

## 2) Repo Structure (recommended)

.
├── data/
│   ├── raw/                 # file dataset dari Kaggle (CSV/TSV dsb)
│   └── processed/           # hasil clean / feature set
├── notebooks/               # EDA + eksperimen (opsional)
├── ml/
│   ├── train.py             # training pipeline
│   ├── preprocess.py        # cleaning & feature prep
│   ├── predict.py           # inference helper
│   └── artifacts/
│       ├── model.pkl
│       ├── scaler.pkl       # kalau dipakai
│       └── feature_schema.json
├── api/
│   ├── main.py              # FastAPI
│   ├── schemas.py           # pydantic models
│   ├── services/
│   │   ├── ml_service.py
│   │   └── chain_service.py
│   └── config.py
├── chain/
│   ├── contracts/
│   │   └── RiskRegistry.sol
│   ├── scripts/
│   │   └── deploy.ts
│   └── hardhat.config.ts
├── app/
│   └── streamlit_app.py
├── requirements.txt
└── README.md


---

## 3) Data Setup

### 3.1 Download dataset from Kaggle
Cara cepat:
1) Download manual dari page dataset Kaggle (yang kamu sudah pilih) dan taruh ke `data/raw/`.
2) Catat nama file dan kolom-kolomnya.

Cara via Kaggle API (opsional):
- Install kaggle CLI dan set `~/.kaggle/kaggle.json`
- Jalankan:
  kaggle datasets download -d xblock/ethereum-phishing-transaction-network -p data/raw --unzip

Catatan:
- Setelah unzip, pastikan file ada di `data/raw/`.
- Buat `DATA_FILE` di config supaya gampang ganti.

---

## 4) ML Plan (simple but respectable)

### 4.1 Target
Binary classification:
- 1 = phishing
- 0 = non-phishing/normal

### 4.2 Baseline + Main Model
- Baseline: Logistic Regression
- Main: RandomForestClassifier

### 4.3 Evaluasi minimal (masuk thesis)
- Accuracy
- Precision, Recall, F1-score
- Confusion Matrix

### 4.4 Risk Score
- Jika model output probability `p` untuk phishing:
  risk_score = round(p * 100)

Risk label:
- 0–39  : Low
- 40–69 : Medium
- 70–100: High

---

## 5) Smart Contract Plan (Hardhat local)

### 5.1 Data yang disimpan
Minimal:
- wallet (address)
- riskScore (uint16)
- timestamp (uint64)

Opsional (nice-to-have, tapi tetap ringan):
- modelVersion (string pendek)

### 5.2 API Smart contract
- setRisk(address wallet, uint16 score)
- getRisk(address wallet) returns (uint16 score, uint64 timestamp)

---

## 6) API Plan (FastAPI)

### 6.1 Endpoints minimal
1) POST /analyze
Request:
{
  "wallet": "0x....",
  "features": { ... }  # atau id row dari dataset (untuk demo cepat)
}

Response:
{
  "wallet": "0x....",
  "probability": 0.87,
  "risk_score": 87,
  "risk_level": "HIGH",
  "tx_hash": "0x...."  # kalau disimpan ke chain
}

2) GET /risk/{wallet}
Response:
{
  "wallet": "0x....",
  "risk_score": 87,
  "timestamp": 1719999999
}

### 6.2 Flow /analyze
1) Validasi input
2) ML predict -> probability
3) risk_score = p*100
4) call smart contract setRisk(...)
5) return response

---

## 7) Streamlit App Plan (demo)

UI minimal:
- Text input: wallet address
- Button: "Analyze"
- Output cards:
  - risk_score + risk_level
  - probability
  - tx hash
- Section: "On-chain record"
  - tombol "Refresh from chain"
  - tampilkan score & timestamp

Optional (bagus untuk thesis BAB 4):
- Confusion matrix image (static)
- Feature importance (Random Forest)

---

## 8) Copilot Task Prompts (copy-paste)

### A) Preprocessing script
"Create ml/preprocess.py:
- load csv from data/raw/<FILENAME>
- identify target column (phishing label) and feature columns
- handle missing values (drop or impute)
- split train/test 80/20 stratify
- save processed datasets to data/processed/train.csv and test.csv
- output feature_schema.json listing column order"

### B) Training script
"Create ml/train.py:
- load processed train/test
- train LogisticRegression baseline + RandomForest main model
- compute metrics: accuracy, precision, recall, f1, confusion matrix
- print a model comparison table
- choose RandomForest as final
- save model to ml/artifacts/model.pkl (joblib)
- save feature_schema.json"

### C) Inference helper
"Create ml/predict.py:
- load model.pkl and feature_schema.json
- function predict_proba(features_dict) -> float
- ensure feature order matches schema
- return probability phishing"

### D) Solidity contract
"Create chain/contracts/RiskRegistry.sol:
- mapping(address => Record{uint16 score; uint64 timestamp})
- setRisk(address,uint16) public
- getRisk(address) view returns (uint16,uint64)
- emit event RiskUpdated(address wallet, uint16 score, uint64 timestamp)"

### E) Hardhat deploy script
"Create chain/scripts/deploy.ts to deploy RiskRegistry and print deployed address."

### F) Web3 service (Python)
"Create api/services/chain_service.py using web3.py:
- connect to local hardhat RPC
- load ABI + contract address from env
- function set_risk(wallet, score) -> tx_hash
- function get_risk(wallet) -> {score,timestamp}"

### G) FastAPI service
"Create api/main.py:
- POST /analyze: accepts wallet + feature payload (or row id)
- uses ml_service to predict probability
- computes risk_score
- calls chain_service.set_risk
- returns JSON response
- GET /risk/{wallet}: calls chain_service.get_risk"

### H) Streamlit UI
"Create app/streamlit_app.py:
- text input wallet
- inputs for features (or dropdown choose row id from test.csv)
- on Analyze: call FastAPI /analyze
- show risk_score, level, probability, tx hash
- add Refresh button calling /risk/{wallet}"

---

## 9) “Fastest Demo Mode” (biar cepat jadi)
Kalau input feature manual terlalu banyak:
- di Streamlit cukup pilih `row_id` dari `data/processed/test.csv`
- backend akan ambil features dari row itu
- wallet address bisa dummy (atau ambil kolom address jika ada)

Dengan begitu demo jalan cepat tanpa bikin form panjang.

---

## 10) Thesis Notes (apa yang ditulis)
BAB 3 (Metodologi):
- sumber dataset Kaggle (XBlock) + deskripsi singkat dataset
- preprocessing & split
- model baseline vs model utama
- evaluasi metrik
- definisi risk score
- desain smart contract + alasan (immutability)

BAB 4 (Implementasi):
- arsitektur sistem (diagram)
- hasil metrik model
- screenshot Streamlit
- contoh tx hash penyimpanan skor ke chain

Done.