# Ethereum Phishing Risk Registry

AI + Blockchain risk scoring system that:
1. Transforms each wallet's one-hop neighbourhood in the **Ethereum Phishing Transaction Network** (XBlock / InPlusLab) `MulDiGraph` into 18 tabular features, enriched with 7 Etherscan attributes.
2. Scores wallets with an **XGBoost** model (25 predictors) and produces a `risk_score` (0–100).
3. Anchors results to a **Solidity smart contract** (`RiskRegistry`) running on a local Hardhat node.
4. Exposes a **FastAPI** REST API and a **Streamlit** demo UI.

Training paths:
- `python ml/train_production_model.py` = the deployed XGBoost model (`ml/artifacts/model_prod_xgb.json`).
- `python ml/train_dataset_scenarios.py` = 4 scenarios × 4 models (LR, RF, XGBoost, GraphSAGE) on one split (seed 42).
- `python experiments/revision_ijece.py` = the same comparison over 10 seeds plus all analyses reported in the paper.
- `python ml/train.py` = legacy baseline (LR + RF) on graph features only.

---

## 📄 Reproducing the IJECE paper

*"Data-centric Ethereum phishing detection through graph-to-tabular transformation"*. Tested from a clean `git clone` on Windows 11 with Python 3.13.

**1. Install**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**2. Get the graph.** Download `MulDiGraph.pkl` (about 1.3 GB) from [XBlock](http://xblock.pro/ethereum/) or the [Kaggle mirror](https://www.kaggle.com/datasets/xblock/ethereum-phishing-transaction-network/data) and place it at `data/raw/MulDiGraph.pkl`. It is not redistributed in this repository.

No API key is needed for reproduction: the Etherscan and BigQuery enrichment results for all 12,815 wallets are already committed in `data/raw/addresses_*_enriched.csv`.

**3. Run, in order**

| Step | Command | Time* | Produces |
|---|---|---|---|
| Build the 12,815-wallet dataset | `python ml/build_hybrid_dataset.py` | ~1 min | `data/processed/hybrid_dataset.csv` (byte-identical to the committed file) |
| Production model | `python ml/train_production_model.py` | ~10 s | `ml/artifacts/model_prod_xgb.json`; test F1 = 0.8911 |
| 4 × 4 comparison, seed 42 | `cd ml; python train_dataset_scenarios.py; cd ..` | ~2 min | `ml/artifacts/dataset_scenarios_comparison.csv` |
| **All paper tables (10 seeds)** | `python experiments/revision_ijece.py` | ~13 min | `outputs/revision/*.csv`, `reliability_diagram.png` |
| Model footprint, latency, gas cost | `python experiments/revision_ops.py --no-live --eth-usd 2313.76` | ~10 s | `outputs/revision/ops_metrics.json` |
| Late fusion (single split) | `python experiments/run_graphsage_etherscan_late_fusion_v2.py` | ~1 min | `outputs/late_fusion_*_v2.*` |
| Embedding-only ablation | `python experiments/run_embedding_only_ablation.py` | ~1 min | `outputs/embedding_only_*` |

\*Measured on a 12-thread CPU without a GPU.

**Mapping to the paper**

| Paper | File in `outputs/revision/` |
|---|---|
| Table 3 (F1 mean ± std, 10 seeds) | `multiseed_summary.csv` |
| Table 4 (late fusion, embedding ablation) | `multiseed_summary.csv`, rows with `scenario = late_fusion` |
| Table 6(a) (TN/FP/FN/TP) | `seed42_table.csv`, `multiseed_summary.csv` |
| Table 6(b) (latency, memory, gas) | `ops_metrics.json` |
| Table A (majority-class ratio) | `majority_ratio_summary.csv` |
| Table B (GraphSAGE protocol decomposition) | `graphsage_protocol_decomposition.csv` |
| Table C (calibration) + reliability figure | `calibration_summary.csv`, `reliability_diagram.png` |
| Table D (cost-sensitive threshold) | `cost_threshold_summary.csv` |
| Isolated-wallet statistics | `isolated_wallets.json` |

**Protocol.** Every split is a stratified 80:20 split. Scaling, early stopping (XGBoost and GraphSAGE), threshold selection and calibration are all fitted on a validation slice taken from the training wallets, and each test set is scored exactly once. Seeds: 42, 0–8.

**Optional (needs your own credentials):** live scanning of wallets outside the dataset (`ETHERSCAN_API_KEY` in `.env`, `LIVE_SCAN_ENABLED=true`), re-running the enrichment (`ml/enrich_real_addresses_*.py`, which also needs a Google Cloud service account for BigQuery), and the blockchain demo (Node.js 18+, `cd chain; npm install`). Copy `.env.example` to `.env` to configure these.

---

## ⚡ Cara Menjalankan Aplikasi (TL;DR)

Prasyarat sekali saja: `pip install -r requirements.txt`. Artefak model (`ml/artifacts/model_prod_xgb.json`) sudah ada, jadi langsung jalan tanpa training. Tiap perintah = terminal terpisah (Windows PowerShell).

### Komponen sistem (apa menjalankan apa)

| Komponen | Teknologi | Perintah | Port | Fungsi |
|---|---|---|---|---|
| **Blockchain** (on-chain) | Hardhat (Solidity) | `npx hardhat node` lalu `npx hardhat run scripts/deploy.ts --network localhost` | `8545` | Node Ethereum lokal + deploy kontrak `RiskRegistry` (simpan skor permanen) |
| **Backend** (API) | FastAPI + web3.py | `python -m uvicorn api.main:app --port 8000` | `8000` | ML inference (XGBoost) + baca/tulis skor ke smart contract |
| **Frontend** (UI) | Streamlit | `streamlit run app/streamlit_app.py --server.port 8502` | `8502` | Antarmuka: scan wallet, lihat skor, lihat riwayat on-chain |
| **DApp** (baca on-chain) | HTML + JSON-RPC | `.\chain\dapp\run_dapp.ps1` | `8503` | Baca skor **langsung dari smart contract**, tanpa lewat FastAPI |

> Alur tulis: **Streamlit (8502)** → **FastAPI (8000)** → model XGBoost → **Hardhat (8545)**.
> Alur baca DApp: **Browser (8503)** → **Hardhat (8545)** langsung, tanpa backend.
> Backend wajib jalan untuk *menulis* skor; blockchain opsional (`CHAIN_ENABLED=false` untuk skip).

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

> ⚠️ Urutan wajib **node → deploy → API**, dan **deploy diulang tiap kali node
> di-restart**. Kalau `chain_connected` masih `false`, lihat
> [Troubleshooting](#troubleshooting--chain_connected-false).

**Mode DApp — mode lengkap + halaman pembaca on-chain (4 terminal):**

```powershell
# Terminal 1–3: sama persis seperti mode lengkap di atas

# Terminal 4 — halaman DApp
.\chain\dapp\run_dapp.ps1
```

➡️ Buka **http://localhost:8503/dapp/** — analisis dulu 1 wallet lewat Streamlit supaya ada datanya.

➡️ Buka **http://localhost:8502** · API docs: **http://localhost:8000/docs**

> Detail lengkap + cek kesehatan ada di bagian [Menjalankan Aplikasi (Quick Start)](#menjalankan-aplikasi-quick-start--windows-powershell) di bawah. Aplikasi live memakai **model terbaik tesis: XGBoost (MulDiGraph + Etherscan, 25 fitur, F1 89,11%)** — buat ulang artefaknya dengan `python ml/train_production_model.py`.

### 🧪 Contoh wallet untuk membuktikan sistem jalan

Aplikasi punya **3 mode input** (pilih di sidebar):
- **🎲 Demo (row_id)** — pilih nomor baris held-out (`prod_demo.csv`, 2.563 wallet), fitur & alamat terisi otomatis.
- **🔎 Scan by Address** — **tempel alamat wallet dataset**, sistem mencari fitur 25-dim-nya (18 graf + 7 Etherscan) lalu memprediksi. Hasil **tervalidasi** (ada ground-truth). Mode utama yang direkomendasikan.
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

> Phishing terdeteksi **HIGH (100)** dan normal **LOW (0)** → sistem bekerja sesuai harapan. Mode Demo `row_id` valid 0–2562 (2.563 baris held-out).

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
│   ├── dapp/                 # Halaman DApp: baca on-chain tanpa backend
│   │   ├── index.html        #   eth_call getRisk() + eth_getLogs RiskUpdated
│   │   └── run_dapp.ps1      #   launcher → http://localhost:8503/dapp/
│   ├── hardhat.config.ts
│   └── package.json
├── app/
│   └── streamlit_app.py      # Streamlit UI
├── experiments/
│   ├── revision_ijece.py     # All paper tables: 10 seeds, fusion, calibration, cost, ratios
│   ├── revision_ops.py       # Model footprint, latency, live Etherscan path, gas cost
│   └── run_*.py              # Single-split late fusion / embedding ablation
├── outputs/
│   └── revision/             # Raw results behind every table of the paper
├── models/                   # Trained fusion / ablation models
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

Frontend **Streamlit** memanggil **FastAPI** (`:8000`), yang menjalankan model produksi **XGBoost (MulDiGraph + Etherscan, 25 fitur, F1 89,11%)** — `ml/artifacts/model_prod_xgb.json` — dan opsional menulis skor ke **smart contract**. Artefak model sudah tersedia, jadi tidak perlu training ulang untuk demo (regenerasi: `python ml/train_production_model.py`).

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

> ⚠️ **Urutannya wajib: node → deploy → API.** Kalau FastAPI dinyalakan sebelum node
> siap dan kontrak ter-deploy, koneksi chain-nya gagal. Kalau kamu terlanjur salah
> urutan, **restart uvicorn** setelah node siap — lihat
> [Troubleshooting](#troubleshooting--chain_connected-false) di bawah.

**Terminal 1 — node blockchain lokal** (biarkan terbuka, jangan ditutup):

```powershell
cd chain
npx hardhat node
```

Tunggu sampai muncul daftar 20 akun dan baris
`Started HTTP and WebSocket JSON-RPC server at http://127.0.0.1:8545/`.

**Terminal 2 — deploy kontrak** (setelah node siap):

```powershell
cd chain
npx hardhat run scripts/deploy.ts --network localhost
```

Berhasil kalau outputnya seperti ini:

```
RiskRegistry deployed to: 0x5FbDB2315678afecb367f032d93F642f64180aa3
Deployment info written to: ...\chain\artifacts\deployment.json
ABI written to: ...\chain\artifacts\RiskRegistry.abi.json
```

> 🔴 **Deploy WAJIB diulang setiap kali `npx hardhat node` dinyalakan ulang.** Bukan
> cuma karena blockchain-nya kosong lagi — `deploy.ts` menulis `deployment.json` dan
> `RiskRegistry.abi.json` ke dalam `chain/artifacts/`, yaitu folder milik Hardhat
> sendiri, dan **Hardhat menghapus isi folder itu setiap kali compile ulang**. Tanpa
> kedua file itu FastAPI tidak tahu alamat kontraknya, dan `chain_connected` akan
> `false`. Terminal 2 boleh ditutup setelah deploy selesai; Terminal 1 tidak.

**Terminal 3 — FastAPI** (jalankan dari root repo, bukan dari `chain/`):

```powershell
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

**Terminal 4 — Streamlit:**

```powershell
streamlit run app/streamlit_app.py --server.port 8502
```

Buka **http://localhost:8502**

**Pastikan chain benar-benar tersambung** sebelum lanjut:

```powershell
curl http://localhost:8000/health
# harus: {"status":"ok","chain_enabled":true,"chain_connected":true}
```

Lalu analisis satu wallet dan cek `tx_hash`-nya **tidak `null`** — itu bukti skornya
benar-benar tertulis ke smart contract:

```powershell
curl -X POST http://localhost:8000/analyze -H "Content-Type: application/json" -d '{\"wallet\":\"0x4dad23689bef624af5d1757a56135901c915fcfd\"}'
```

### Mode C — Mode B + halaman DApp pembaca on-chain (4 terminal)

Menambahkan **`chain/dapp/index.html`**: halaman yang membaca skor **langsung dari smart
contract** lewat JSON-RPC — tanpa melewati FastAPI sama sekali. Ini yang membuktikan sifat
*decentralized* sistem: matikan FastAPI, pembacaan skor tetap jalan.

> FastAPI **tidak digantikan**. Menulis skor tetap butuh model XGBoost (harus off-chain,
> karena inferensi ML tidak mungkin dijalankan di dalam smart contract). Yang
> terdesentralisasi adalah **pembacaan dan verifikasi** hasilnya.

```powershell
# Terminal 1 — Hardhat node lokal (biarkan jalan)
cd chain
npx hardhat node

# Terminal 2 — deploy kontrak (sekali, tiap kali node dinyalakan ulang)
cd chain
npx hardhat run scripts/deploy.ts --network localhost

# Terminal 3 — FastAPI + Streamlit (JALUR TULIS: mengisi data ke chain)
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
# di terminal lain:
streamlit run app/streamlit_app.py --server.port 8502

# Terminal 4 — halaman DApp (JALUR BACA: langsung ke blockchain)
.\chain\dapp\run_dapp.ps1
```

Buka **http://localhost:8503/dapp/**

**Urutan mencoba:**

1. Lewat Streamlit (`:8502`), analisis minimal satu wallet supaya ada data on-chain.
   Contoh wallet phishing yang skornya tinggi: `0x60afc5761212ed59608b0559b4ccbaad24fc74b1` → skor 100 (HIGH).
2. Buka halaman DApp (`:8503/dapp/`) → klik **Hubungkan**.
   Alamat kontrak terisi otomatis dari `chain/artifacts/deployment.json`.
3. Tempel alamat wallet tadi → **Baca on-chain** (memanggil `getRisk()`).
4. Klik **Muat riwayat** → tabel semua wallet yang pernah dinilai, direkonstruksi dari
   event log `RiskUpdated`.
5. **Uji pembuktiannya:** matikan FastAPI (Ctrl+C di terminal 3), lalu ulangi langkah 3–4.
   Halaman DApp tetap berfungsi — karena ia bicara langsung ke blockchain.

**Opsional — lewat MetaMask** (bukti DApp paling nyata):
tambahkan jaringan Hardhat di MetaMask (RPC `http://127.0.0.1:8545`, chainId `31337`),
lalu klik tombol **MetaMask** di halaman itu, bukan tombol **Hubungkan**.

Catatan penting:

- Halaman **harus** dibuka lewat `http://localhost:8503/dapp/`, jangan klik dua kali file
  HTML-nya (`file://`) — alamat kontrak tidak akan terisi otomatis dan sebagian browser
  memblokir permintaan jaringannya.
- Skrip menyajikan folder `chain/`, bukan `chain/dapp/`, supaya halaman bisa membaca
  `../artifacts/deployment.json`.
- Setiap kali `npx hardhat node` di-restart, blockchain-nya kosong lagi — **deploy ulang**
  dan analisis ulang wallet-nya, kalau tidak halaman akan bilang "belum pernah dinilai".
- Halaman ini **tanpa library eksternal** (tidak ada ethers.js/CDN) — encoding & decoding
  ABI ditulis manual, jadi tetap jalan tanpa internet.

### Mode scan wallet di luar dataset (opsional, mati secara bawaan)

> Dengan pengaturan bawaan (`LIVE_SCAN_ENABLED=false`), wallet di luar dataset
> menghasilkan HTTP 404 dan angka-angka paper tidak terpengaruh.

Bila dinyalakan, `POST /analyze` punya dua jalur. Keduanya memakai **model yang sama**
(`ml/artifacts/model_prod_xgb.json`, XGBoost 25 fitur); yang berbeda hanya sumber fiturnya:

| Wallet | Sumber 25 fitur | Dicatat on-chain? |
|---|---|---|
| Ada di `hybrid_dataset.csv` (12.815 alamat) | precomputed (MulDiGraph + Etherscan) | ✅ ya |
| Di luar dataset | dihitung live dari Etherscan `txlist` oleh `ml/live_features.py` | ❌ tidak pernah |

Ke-18 fitur graf bersifat lokal 1-hop, sehingga bisa dihitung dari daftar transaksi
Etherscan dengan rumus yang sama persis seperti dari MulDiGraph. Wallet tanpa riwayat
transaksi dibalas "wallet tidak ditemukan" (404) tanpa skor. Hasil mode live selalu
ditandai `out_of_distribution: true` karena model dilatih pada snapshot 2015–2019.
Butuh `ETHERSCAN_API_KEY` di `.env`.

```powershell
# Menyalakan (bawaan: mati)
$env:LIVE_SCAN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Uji satu alamat langsung dari terminal
python ml/live_features.py 0x<alamat>
```

Variabel env terkait (`api/config.py`):

| Variabel | Default | Fungsi |
|---|---|---|
| `LIVE_SCAN_ENABLED` | **`false`** | Nyalakan untuk mengaktifkan scan live |
| `LIVE_MAX_PAGES` | `5` | Batas halaman `txlist` per alamat |
| `LIVE_REQUEST_DELAY` | `0.25` | Jeda antar panggilan Etherscan (detik) |

Satu scan live butuh 5 panggilan Etherscan, sekitar 8–17 detik per wallet (lihat
`outputs/revision/ops_metrics.json`).

### Cek cepat aplikasi sudah jalan

```powershell
# API sehat? (chain_connected true hanya pada Mode B / C)
curl http://localhost:8000/health
# Streamlit sehat?
curl http://localhost:8502/_stcore/health
# Node blockchain sehat? (Mode B / C) — harus balas result = 0x7a69 (chainId 31337)
Invoke-RestMethod -Uri http://127.0.0.1:8545 -Method Post -ContentType "application/json" `
  -Body '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}'
# Halaman DApp tersaji? (Mode C) — harus StatusCode 200
(Invoke-WebRequest http://localhost:8503/dapp/ -UseBasicParsing).StatusCode
```

| Komponen | URL | Mode |
|---|---|---|
| Streamlit UI | http://localhost:8502 | A, B, C |
| FastAPI (docs) | http://localhost:8000/docs | A, B, C |
| Hardhat RPC | http://127.0.0.1:8545 | B, C |
| **Halaman DApp (baca on-chain)** | **http://localhost:8503/dapp/** | **C** |

> Catatan: aplikasi live memakai **XGBoost (MulDiGraph + Etherscan)** — model terbaik di tabel perbandingan tesis. GNN dan Random Forest tetap dilatih sebagai pembanding di `ml/train_dataset_scenarios.py`, tapi yang di-serve adalah XGBoost.

### Troubleshooting — `chain_connected: false`

Ini masalah paling sering di Mode B/C. Diagnosis berurutan, berhenti di langkah pertama
yang gagal:

**1. Node-nya hidup?**

```powershell
Invoke-RestMethod -Uri http://127.0.0.1:8545 -Method Post -ContentType "application/json" `
  -Body '{"jsonrpc":"2.0","id":1,"method":"eth_chainId","params":[]}'
```

`Unable to connect to the remote server` → Terminal 1 mati. Nyalakan `npx hardhat node`
lagi, lalu **deploy ulang** (langkah 2). Harus balas `result : 0x7a69`.

**2. Kontraknya sudah ter-deploy?**

```powershell
Test-Path chain\artifacts\deployment.json, chain\artifacts\RiskRegistry.abi.json
```

Kalau ada yang `False` → deploy ulang:
`cd chain ; npx hardhat run scripts/deploy.ts --network localhost`.
Ingat, dua file ini **terhapus tiap kali Hardhat compile ulang** (lihat peringatan di
Mode B), jadi hilangnya file ini normal — tinggal deploy lagi.

**3. `CHAIN_ENABLED` benar-benar `true`?**

```powershell
curl http://localhost:8000/health
```

Kalau `chain_enabled: false`, berarti uvicorn dijalankan dengan `CHAIN_ENABLED=false`.
Nilai ini dibaca dari `.env` (`api/config.py`) **atau** dari `$env:CHAIN_ENABLED` di
terminal tempat uvicorn dijalankan — variabel env di PowerShell hanya berlaku untuk
terminal itu saja, jadi set ulang di terminal yang sama sebelum menjalankan uvicorn.

**4. Sudah benar semua tapi tetap `false`, atau `tx_hash` selalu `null`?**

**Restart uvicorn (Ctrl+C lalu jalankan lagi).** Koneksi chain dibuat sekali saat
permintaan pertama dan disimpan di memori proses; kalau saat itu node belum siap,
proses uvicorn yang lama tetap memegang state gagal. Urutan yang benar selalu:
**node → deploy → baru nyalakan API.**

> Catatan: kegagalan tulis on-chain sengaja tidak mematikan request (`api/main.py`
> menangkap error-nya), jadi `/analyze` tetap balas 200 dengan `tx_hash: null`.
> **Selalu cek `tx_hash`**, jangan hanya lihat status HTTP-nya.

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
cd /d <path-to-repo>
.\.venv\Scripts\Activate.ps1
python ml\enrich_real_addresses_etherscan.py
```

Control batch size and request pacing (recommended for free Etherscan key):

```powershell
# Windows PowerShell example
cd /d <path-to-repo>
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
cd /d <path-to-repo>
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
cd /d <path-to-repo>
.\.venv\Scripts\Activate.ps1
python ml\enrich_real_addresses_bigquery.py
```

If you prefer Command Prompt (`cmd`), activate venv like this first:

```bat
cd /d <path-to-repo>
.venv\Scripts\activate.bat
python ml\enrich_real_addresses_bigquery.py
```

If you use Command Prompt (`cmd`), use this syntax instead:

```bat
cd /d <path-to-repo>
set GOOGLE_APPLICATION_CREDENTIALS=C:\path\to\your-service-account.json
set BIGQUERY_PROJECT_ID=your-gcp-project-id
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

Scenarios compared (predictor counts; `graph_present` is a MulDiGraph join-coverage
audit column written to the dataset for QA, not counted as a predictor):
- `muldigraph_only` — 18 graph-derived predictors
- `muldigraph_plus_etherscan` — 18 graph + 7 Etherscan = 25 predictors (**production model**)
- `muldigraph_plus_bigquery` — 18 graph + 10 BigQuery = 28 predictors
- `muldigraph_plus_etherscan_plus_bigquery` — 18 graph + 7 Etherscan + 10 BigQuery = 35 predictors

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
