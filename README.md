# Terminal 1 — Hardhat node
cd chain ; npx hardhat node

# Terminal 2 — deploy kontrak (setelah node siap)
cd chain ; npx hardhat run scripts/deploy.ts --network localhost

# Terminal 3 — API (chain aktif)
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 4 — UI
streamlit run app/streamlit_app.py --server.port 8502

# Ethereum Phishing Risk Registry

AI + Blockchain risk scoring system that:
1. Trains a Random Forest classifier on the **Ethereum Phishing Transaction Network** (XBlock / InPlusLab) dataset.
2. Produces a `risk_score` (0–100) per wallet.
3. Anchors results to a **Solidity smart contract** running on a local Hardhat node.
4. Exposes a **FastAPI** REST API and a **Streamlit** demo UI.

For thesis results, there are two different training paths:
- `python ml/train.py` = baseline only (Logistic Regression + Random Forest) on the processed graph features.
- `python ml/train_dataset_scenarios.py` = full thesis comparison (LR + RF + XGBoost + GNN) across the hybrid dataset scenarios.

---

## ⚡ Cara Menjalankan Aplikasi (TL;DR)

Prasyarat sekali saja: `pip install -r requirements.txt`. Artefak model (`ml/artifacts/model_prod_xgb.json`) sudah ada, jadi langsung jalan tanpa training. Tiap perintah = terminal terpisah (Windows PowerShell).

### Komponen sistem (apa menjalankan apa)

| Komponen | Teknologi | Perintah | Port | Fungsi |
|---|---|---|---|---|
| **Blockchain** (on-chain) | Hardhat (Solidity) | `npx hardhat node` lalu `npx hardhat run scripts/deploy.ts --network localhost` | `8545` | Node Ethereum lokal + deploy kontrak `RiskRegistry` (simpan skor permanen) |
| **Backend** (API) | FastAPI + web3.py | `python -m uvicorn api.main:app --port 8000` | `8000` | ML inference (XGBoost) + baca/tulis skor ke smart contract |
| **Frontend** (UI) | Streamlit | `streamlit run app/streamlit_app.py --server.port 8502` | `8502` | Antarmuka: scan wallet, lihat skor, lihat riwayat on-chain |

> Alur: **Streamlit (8502)** → **FastAPI (8000)** → model XGBoost + **Hardhat (8545)**. Backend wajib jalan; blockchain opsional (`CHAIN_ENABLED=false` untuk skip).

**Mode cepat — tanpa blockchain (2 terminal):**

```powershell
# Terminal 1 — API
$env:CHAIN_ENABLED = "false"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — UI
streamlit run app/streamlit_app.py --server.port 8502
```

**Mode lengkap — dengan blockchain (3 terminal):**

```powershell
# Terminal 1 — node lokal
cd chain ; npx hardhat node

# Terminal 2 — deploy kontrak (setelah node siap)
cd chain ; npx hardhat run scripts/deploy.ts --network localhost

# Terminal 3 — API (chain aktif) + UI
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# di terminal lain:
streamlit run app/streamlit_app.py --server.port 8502
```

➡️ Buka **http://localhost:8502** · API docs: **http://localhost:8000/docs**

> Detail lengkap + cek kesehatan ada di bagian [Menjalankan Aplikasi (Quick Start)](#menjalankan-aplikasi-quick-start--windows-powershell) di bawah. Aplikasi live memakai **model terbaik tesis: XGBoost (MulDiGraph + Etherscan, 26 fitur, F1 91,02%)** — buat ulang artefaknya dengan `python ml/train_production_model.py`.

### 🧪 Contoh wallet untuk membuktikan sistem jalan

Aplikasi punya **3 mode input** (pilih di sidebar):
- **🎲 Demo (row_id)** — pilih nomor baris held-out (`prod_demo.csv`, 1.719 wallet), fitur & alamat terisi otomatis.
- **🔎 Scan by Address** — **tempel alamat wallet dataset**, sistem mencari fitur 26-dim-nya (graf + Etherscan) lalu memprediksi. Hasil **tervalidasi** (ada ground-truth). Mode utama yang direkomendasikan.
- **✍️ Manual** — isi 18 fitur graf sendiri (fitur Etherscan = 0; untuk wallet di luar dataset).

> Catatan: model menilai dari **fitur perilaku transaksi**, bukan dari teks alamat. Karena dataset MulDiGraph adalah snapshot 2019, scan otomatis hanya untuk alamat yang ada di dataset; alamat live di luar itu perlu ekstraksi fitur Etherscan.

Contoh berikut sudah diverifikasi langsung pada API (model XGBoost + Etherscan), sengaja dipilih **bergradasi** (skor 12 → 100) supaya tingkat risiko terlihat bertingkat, bukan cuma 0/100. Untuk **Scan by Address** cukup tempel kolom **Wallet**; untuk **Demo** masukkan `row_id`:

| Wallet (mode Scan by Address) | Label asli | Hasil sistem | Skor |
|---|---|---|---|
| `0x4dad23689bef624af5d1757a56135901c915fcfd` | phishing | 🔴 **HIGH** | 100 |
| `0x8c155c3f4c79485dbb0eef5b4076fd426b3f8c65` | phishing | 🔴 **HIGH** | 81 |
| `0x35919ff0a88a0bdfd1b0ef3be574bcf6c1e90e01` | phishing | 🟡 **MEDIUM** | 62 |
| `0x5141ffe7d645818996515e7336174a614def6380` | normal | 🟡 **MEDIUM** | 64 |
| `0x2a7cf914a7825bb067703042eff254d54963234c` | normal | 🟢 **LOW** | 25 |
| `0x16bccf9ea461adbc202d5f0d7f740b43e2da514c` | normal | 🟢 **LOW** | 12 |

> Catatan: model cukup yakin sehingga mayoritas wallet jatuh dekat 0 atau 100; zona **MEDIUM (40–69)** memang langka (≈43 wallet) — itu justru kasus "abu-abu" yang membuat model ragu. Contoh `row_id` untuk mode Demo: `4` (HIGH, phishing), `0` (LOW, normal).

Atau uji langsung lewat API (tanpa UI):

```powershell
# Scan by address — wallet phishing -> HIGH
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{\"wallet\":\"0x4dad23689bef624af5d1757a56135901c915fcfd\"}'

# Scan by address — wallet normal -> LOW
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{\"wallet\":\"0x4f66f7e2de0a1f9848ac61278f61a69c2bd276be\"}'

# Mode blockchain: baca balik skor yang tersimpan di smart contract
curl http://localhost:8000/risk/0x4dad23689bef624af5d1757a56135901c915fcfd
```

> Phishing terdeteksi **HIGH (100)** dan normal **LOW (0)** → sistem bekerja sesuai harapan. Mode Demo `row_id` valid 0–1718 (1.719 baris held-out).

**Mode Manual (isi fitur sendiri):** default form memakai profil **wallet normal nyata** sehingga sekali klik Analyze hasilnya **LOW**. Untuk melihat **HIGH**, ubah pola jadi mirip phishing (mis. turunkan `in_degree`/amount, naikkan `in_out_ratio`). Skor manual juga **tetap ditulis ke smart contract** selama API berjalan pada Mode B (`CHAIN_ENABLED=true`) — kalau hasil tidak masuk on-chain, pastikan API tidak dijalankan dengan `CHAIN_ENABLED=false`.

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
│   ├── train.py              # Baseline only: LR + Random Forest
│   ├── train_dataset_scenarios.py # Full thesis comparison: LR + RF + XGBoost + GNN
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

## Menjalankan Aplikasi (Quick Start — Windows PowerShell)

Frontend **Streamlit** memanggil **FastAPI** (`:8000`), yang menjalankan model produksi **XGBoost (MulDiGraph + Etherscan, 26 fitur, F1 91,02%)** — `ml/artifacts/model_prod_xgb.json` — dan opsional menulis skor ke **smart contract**. Artefak model sudah tersedia, jadi tidak perlu training ulang untuk demo (regenerasi: `python ml/train_production_model.py`).

> Prasyarat: `pip install -r requirements.txt` sudah dijalankan. Tiap blok di bawah dijalankan di terminal terpisah.

### Mode A — Cepat, tanpa blockchain (2 terminal)

Untuk sekadar melihat UI / inferensi model. Skor tidak ditulis ke chain (`tx_hash` akan `null`).

```powershell
# Terminal 1 — FastAPI (chain dimatikan)
$env:CHAIN_ENABLED = "false"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Streamlit
streamlit run app/streamlit_app.py --server.port 8502
```

Buka **http://localhost:8502**

### Mode B — Lengkap, dengan blockchain (3 terminal)

Demo end-to-end: skor risiko tercatat permanen ke smart contract `RiskRegistry`.

```powershell
# Terminal 1 — Hardhat node lokal (biarkan jalan)
cd chain
npx hardhat node

# Terminal 2 — deploy kontrak (sekali, setelah node siap)
cd chain
npx hardhat run scripts/deploy.ts --network localhost
#   -> menulis chain/artifacts/deployment.json + RiskRegistry.abi.json
#   -> kontrak: 0x5FbDB2315678afecb367f032d93F642f64180aa3

# Terminal 3 — FastAPI (chain aktif = default) lalu Streamlit
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# di terminal lain:
streamlit run app/streamlit_app.py --server.port 8502
```

Buka **http://localhost:8502**

### Cek cepat aplikasi sudah jalan

```powershell
# API sehat? (chain_connected true hanya pada Mode B)
curl http://localhost:8000/health
# Streamlit sehat?
curl http://localhost:8502/_stcore/health
```

| Komponen | URL |
|---|---|
| Streamlit UI | http://localhost:8502 |
| FastAPI (docs) | http://localhost:8000/docs |
| Hardhat RPC (Mode B) | http://127.0.0.1:8545 |

> Catatan: aplikasi live memakai **XGBoost (MulDiGraph + Etherscan)** — model terbaik di tabel perbandingan tesis. GNN dan Random Forest tetap dilatih sebagai pembanding di `ml/train_dataset_scenarios.py`, tapi yang di-serve adalah XGBoost.

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

If you want the hybrid thesis dataset, build it after enrichment:

```bash
python ml/build_hybrid_dataset.py
```

This writes:
- `data/processed/hybrid_dataset_full.csv`
- `data/processed/hybrid_dataset.csv`
- `data/processed/hybrid_train.csv`
- `data/processed/hybrid_test.csv`
- `ml/artifacts/hybrid_feature_schema.json`

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

Use this step only for the baseline graph-feature experiment.

### Step 2b – Full thesis comparison

```bash
python ml/train_dataset_scenarios.py
```

This is the script that produces the thesis comparison tables and the final model ranking. It automatically:
- loads `data/raw/MulDiGraph.pkl`
- extracts real transaction edges for GNN
- trains 4 models × 4 scenarios
- saves `ml/artifacts/dataset_scenarios_comparison.csv`
- saves `ml/artifacts/dataset_scenarios_summary.json`
- saves scenario-level confusion matrix plots for XGBoost and GNN V2

Generated plot files follow this pattern:
- `ml/artifacts/cm_<scenario>_xgboost.png`
- `ml/artifacts/cm_<scenario>_gnn_v2.png`

Recommended final thesis result:
- use the `muldigraph_plus_etherscan` scenario
- report the best model as `XGBoost`
- report the GNN result as `GNN-GraphSAGE-V2` in the same scenario

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

> **Status:** All 8,595 hybrid-dataset addresses are already enriched. You only need to re-run this if you change the address list.

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

### Step 6b - Etherscan edge extraction (GNN graph enrichment)

This step fetches cross-transaction edges between the 8,595 hybrid-dataset addresses.
For each address it queries **three** Etherscan sources — normal transactions
(`txlist`), internal transactions (`txlistinternal`), and ERC-20 token transfers
(`tokentx`) — and keeps every counterparty that is also in the dataset. Using all
three sources (instead of normal ETH transfers only) captures far more real
relationships, so the resulting edge list better supplements the sparse MulDiGraph
edges used by GNN-GraphSAGE V2.

Output: `data/raw/etherscan_edges.csv` (columns: `src`, `dst`)

```bash
python ml/fetch_etherscan_edges.py
```

**Runtime estimate (free Etherscan key, 0.4s delay):**
- Each address now queries up to 3 sources, so expect **~3×** the time of a
  single-source run (~3 hours for all 8,595 addresses in 1 run).
- Disable internal/token sources to go faster (see env vars below).
- Script is **fully resumable** — safe to stop and restart.

PowerShell (with custom pacing):

```powershell
$env:ETHERSCAN_API_KEY='YOUR_KEY'
$env:ETHERSCAN_REQUEST_DELAY='0.4'
$env:ETHERSCAN_MAX_PAGES='3'          # 3 pages × 10k records per source
$env:ETHERSCAN_PER_RUN_LIMIT='1000'   # 0 = process all pending
$env:ETHERSCAN_INCLUDE_INTERNAL='1'   # 0 = skip internal transactions
$env:ETHERSCAN_INCLUDE_TOKENS='1'     # 0 = skip ERC-20 token transfers
python ml\fetch_etherscan_edges.py
```

**How it works:**
- For each address, queries up to `ETHERSCAN_MAX_PAGES` pages from each enabled
  source (normal + internal + token transfers).
- Keeps only edges where the counterparty is ALSO in the 8,595-node dataset.
- The edge list is automatically loaded by `train_dataset_scenarios.py` if the file exists.
- Merges with MulDiGraph edges before GNN training (no code changes needed).

**Expected improvement:**
The current MulDiGraph provides 528 cross-edges for 8,595 nodes (~0.06 edges/node).
Etherscan edges should add several hundred to a few thousand additional connections,
potentially improving GNN F1 by 0.5–2 points depending on transaction overlap.

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

### Step 8 - Recommended end-to-end run order for thesis outputs

If you want all figures and tables to match the thesis narrative, run in this order:

1. `python ml\preprocess.py`
2. `python ml\enrich_real_addresses_etherscan.py` and `python ml\enrich_real_addresses_bigquery.py` if you need the enriched datasets
3. `python ml\build_hybrid_dataset.py`
4. `python ml\train_dataset_scenarios.py`
5. Start Hardhat node and deploy the contract
6. Start FastAPI
7. Start Streamlit

Notes:
- `ml/train.py` is still useful for the simple baseline, but it does not generate the full XGBoost/GNN thesis comparison.
- `ml/train_dataset_scenarios.py` is the script to run if you want the complete thesis results in `ml/artifacts/dataset_scenarios_comparison.csv`.

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

Ringkasan endpoint (dokumentasi interaktif: **http://localhost:8000/docs**):

| Method | Endpoint | Fungsi |
|---|---|---|
| `GET` | `/health` | Status API + apakah chain terhubung |
| `GET` | `/demo/rows` | Jumlah baris demo (`prod_demo.csv`) |
| `POST` | `/analyze` | ML inference + tulis skor ke smart contract |
| `GET` | `/risk/{wallet}` | Baca record on-chain untuk **satu** wallet |
| `GET` | `/records` | **Daftar semua** wallet yang tercatat on-chain (audit trail) |

### `POST /analyze`

```json
// Scan by Address (hanya wallet — fitur diambil dari dataset)
{ "wallet": "0xAbCd...1234" }

// Demo mode (pakai row_id)
{ "wallet": "0xAbCd...1234", "row_id": 42 }

// Manual mode (isi fitur sendiri)
{ "wallet": "0xAbCd...1234", "features": { "in_degree": 5, "out_degree": 2 } }
```

Response:
```json
{
  "wallet": "0xAbCd...1234",
  "probability": 0.87,
  "risk_score": 87,
  "risk_level": "HIGH",
  "tx_hash": "0x...",
  "true_label": 1,
  "dataset_split": "test"
}
```

### `GET /risk/{wallet}`

Baca record on-chain untuk satu wallet:
```json
{
  "wallet": "0x...",
  "risk_score": 87,
  "timestamp": 1782462913,
  "model_version": "xgb-eth-v1"
}
```

### `GET /records`

Daftar semua wallet yang pernah dicatat ke smart contract (dari event `RiskUpdated`, terbaru per wallet) — sumber tabel **📜 Riwayat On-chain** di UI:
```json
{
  "count": 30,
  "records": [
    { "wallet": "0x8c155c3f...", "risk_score": 81, "timestamp": 1782462913, "model_version": "xgb-eth-v1" }
  ]
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
