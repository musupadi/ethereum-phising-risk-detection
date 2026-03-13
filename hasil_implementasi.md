# AI + Blockchain Risk Registry (Ethereum Phishing) — Hasil Implementasi

> Dokumen ini mencatat hasil nyata dari implementasi sistem, berdasarkan blueprint di `code.md`.
> Dibuat: 3 Maret 2026

---

## 0) Rumusan Masalah & Research Questions (BAB 1)

Penelitian ini menjawab tiga research question berikut:

**RQ1:** Seberapa baik model Machine Learning (Random Forest) dalam mendeteksi node phishing pada dataset transaksi Ethereum (EPTransNet) dibandingkan baseline Logistic Regression?

**RQ2:** Apakah pencatatan risk score ke smart contract EVM dapat dilakukan dengan overhead waktu dan gas yang wajar untuk skenario produksi?

**RQ3:** Fitur graf transaksi mana yang paling berpengaruh terhadap prediksi phishing, dan apakah rasio sampling (1:5 vs 1:10) berdampak signifikan terhadap performa model?

> Tanpa RQ eksplisit ini, penelitian berpotensi dinilai "hanya implementasi". Ketiga RQ di atas mengaitkan hasil eksperimen ke kontribusi ilmiah yang terukur.

---

## 1) Tech Stack yang Digunakan (Aktual)

| Komponen | Teknologi | Versi |
|---|---|---|
| Runtime | Python | 3.14.3 ¹ |
| ML | scikit-learn, pandas, joblib, networkx | 1.8.0 / 2.3.3 / 1.5.3 / 3.6.1 |
| API | FastAPI + uvicorn | 0.135.1 / 0.41.0 |
| UI | Streamlit | 1.55.0 |
| Blockchain | Hardhat (local) + Solidity | 2.22.x / 0.8.20 |
| Web3 bridge | web3.py | 7.14.1 |
| Node.js | Node.js + npm | 25.7.0 / 11.10.1 |

> ¹ **Catatan Python 3.14.3:** Versi ini adalah rilis terbaru via Homebrew per Maret 2026. Kode kompatibel dengan Python 3.10+ (tidak menggunakan fitur eksklusif 3.14). Untuk reproduksi di environment lain, **Python 3.10 / 3.11 / 3.12 juga dapat digunakan** tanpa modifikasi kode.

---

## 2) Repo Structure (Aktual)

```
ethereum-phising-risk-detection/
├── data/
│   ├── raw/
│   │   └── MulDiGraph.pkl              ← dataset utama (symlink)
│   └── processed/
│       ├── train.csv                   ← 10.252 baris (generated)
│       └── test.csv                    ← 2.563 baris (generated)
├── notebooks/
│   └── eda.ipynb                       ← EDA notebook
├── ml/
│   ├── preprocess.py                   ✅ selesai
│   ├── train.py                        ✅ selesai
│   ├── predict.py                      ✅ selesai
│   └── artifacts/
│       ├── model.pkl                   ✅ Random Forest (joblib)
│       ├── scaler.pkl                  ✅ StandardScaler (untuk LR)
│       ├── feature_schema.json         ✅ 18 fitur
│       ├── confusion_matrix.png        ✅ generated
│       └── feature_importance.png      ✅ generated
├── api/
│   ├── main.py                         ✅ FastAPI aktif di :8000
│   ├── schemas.py                      ✅
│   ├── config.py                       ✅
│   └── services/
│       ├── ml_service.py               ✅
│       └── chain_service.py            ✅ web3.py terhubung
├── chain/
│   ├── contracts/
│   │   └── RiskRegistry.sol            ✅ compiled + deployed
│   ├── scripts/
│   │   └── deploy.ts                   ✅
│   ├── artifacts/
│   │   ├── deployment.json             ✅ alamat kontrak tersimpan
│   │   └── RiskRegistry.abi.json       ✅ ABI tersimpan
│   ├── hardhat.config.ts               ✅
│   └── package.json                    ✅
├── app/
│   └── streamlit_app.py                ✅ Streamlit aktif di :8502
├── requirements.txt                    ✅
└── README.md                           ✅
```

---

## 3) Dataset

### 3.1 Sumber Dataset

| Properti | Nilai |
|---|---|
| Nama | Ethereum Phishing Transaction Network (EPTransNet) |
| Sumber | XBlock / InPlusLab, Sun Yat-sen University |
| Format | NetworkX `MultiDiGraph` (pickle, Python 3.7) |
| File | `MulDiGraph.pkl` |

### 3.2 Statistik Dataset Asli

| Properti | Nilai |
|---|---|
| Total node | 2.973.489 |
| Total edge | 13.551.303 |
| Rata-rata degree | 4,5574 |
| **Node phishing** (`isp=1`) | **1.165** |
| Node normal (`isp=0`) | 2.972.324 |
| Rasio ketidakseimbangan | 1 : 2.550 |

### 3.3 Fitur Node (Edge Features)
- `G.nodes[node]['isp']` → label (1 = phishing, 0 = normal)
- `G[u][v][0]['amount']` → jumlah transaksi (ETH)
- `G[u][v][0]['timestamp']` → waktu transaksi (unix epoch)

---

## 4) Preprocessing & Feature Engineering

### 4.1 Strategi Sampling

Karena dataset sangat tidak seimbang (1 : 2.550), digunakan **strategi undersampling**:
- Semua 1.165 node phishing diambil
- 11.650 node normal diambil secara acak (rasio 1:10)
- **Total dataset**: 12.815 node

### 4.2 Fitur yang Diekstrak (18 Fitur)

Setiap node diwakili oleh fitur agregat dari subgraph transaksinya:

| # | Nama Fitur | Deskripsi |
|---|---|---|
| 1 | `in_degree` | Jumlah transaksi masuk |
| 2 | `out_degree` | Jumlah transaksi keluar |
| 3 | `unique_senders` | Jumlah pengirim unik |
| 4 | `unique_receivers` | Jumlah penerima unik |
| 5 | `total_received` | Total ETH diterima |
| 6 | `total_sent` | Total ETH dikirim |
| 7 | `avg_in_amount` | Rata-rata nilai transaksi masuk |
| 8 | `avg_out_amount` | Rata-rata nilai transaksi keluar |
| 9 | `max_in_amount` | Nilai transaksi masuk terbesar |
| 10 | `max_out_amount` | Nilai transaksi keluar terbesar |
| 11 | `min_in_amount` | Nilai transaksi masuk terkecil |
| 12 | `min_out_amount` | Nilai transaksi keluar terkecil |
| 13 | `std_in_amount` | Std. deviasi transaksi masuk |
| 14 | `std_out_amount` | Std. deviasi transaksi keluar |
| 15 | `in_out_ratio` | Rasio `in_degree / (out_degree + 1)` |
| 16 | `sent_received_ratio` | Rasio total ETH keluar / masuk |
| 17 | `timestamp_spread_in` | Rentang waktu transaksi masuk (detik) |
| 18 | `timestamp_spread_out` | Rentang waktu transaksi keluar (detik) |

### 4.3 Split Dataset

| Set | Jumlah Baris | Phishing | Normal |
|---|---|---|---|
| Train (80%) | 10.252 | 932 | 9.320 |
| Test (20%) | 2.563 | 233 | 2.330 |

- Metode: `train_test_split` stratified, `random_state=42`
- Rasio phishing di train: **9.09%**

---

## 5) Hasil Training Model

### 5.1 Perbandingan Model

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression (baseline) | 95,04% | 75,00% | 68,24% | 71,46% |
| **Random Forest (main)** | **97,82%** | **88,31%** | **87,55%** | **87,93%** |

→ **Random Forest dipilih sebagai model utama** (F1 +16,47 poin di atas baseline).

### 5.2 Classification Report — Random Forest

```
              precision    recall  f1-score   support

      Normal       0.99      0.99      0.99      2330
    Phishing       0.88      0.88      0.88       233

    accuracy                           0.98      2563
   macro avg       0.94      0.93      0.93      2563
weighted avg       0.98      0.98      0.98      2563
```

### 5.3 Analisis
- Dari 233 node phishing di test set: **205 terdeteksi benar** (True Positive)
- False Negative (phishing tidak terdeteksi): **~28 node**
- False Positive (normal salah dikira phishing): **~27 node**
- Model sangat bagus untuk kelas mayoritas (Normal F1 = 0.99)

### 5.4 Konfigurasi Random Forest

```python
RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=2,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
```

---

## 6) Risk Score

### 6.1 Formula

```
risk_score = round(predict_proba(phishing) × 100)
```

### 6.2 Risk Labels

| Range | Label | Keterangan |
|---|---|---|
| 0 – 39 | 🟢 LOW | Kemungkinan phishing rendah |
| 40 – 69 | 🟡 MEDIUM | Perlu perhatian |
| 70 – 100 | 🔴 HIGH | Kemungkinan phishing tinggi |

---

## 7) Smart Contract

### 7.1 Kontrak: `RiskRegistry.sol`

```solidity
// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

struct Record {
    uint16  riskScore;    // 0–100
    uint64  timestamp;    // unix epoch
    string  modelVersion; // e.g. "rf-v1"
}

mapping(address => Record) private _records;

function setRisk(address wallet, uint16 score, string calldata modelVersion) external onlyOwner
function getRisk(address wallet) external view returns (uint16, uint64, string)
event RiskUpdated(address indexed wallet, uint16 riskScore, uint64 timestamp, string modelVersion)
```

### 7.2 Deployment (Hardhat Local)

| Properti | Nilai |
|---|---|
| Network | Hardhat localhost (chainId: 31337) |
| RPC URL | `http://127.0.0.1:8545` |
| Contract Address | `0x5FbDB2315678afecb367f032d93F642f64180aa3` |
| Deployer | `0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266` |
| Model Version | `rf-v1` |

### 7.3 Bukti Deployment

```json
{
  "address": "0x5FbDB2315678afecb367f032d93F642f64180aa3",
  "deployer": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
  "network": "localhost",
  "chainId": 31337
}
```

---

## 8) API (FastAPI) — Hasil Uji

### 8.1 Endpoints

| Method | Endpoint | Fungsi |
|---|---|---|
| GET | `/health` | Status API + chain |
| POST | `/analyze` | ML inference + simpan ke chain |
| GET | `/risk/{wallet}` | Baca record dari chain |
| GET | `/demo/rows` | Jumlah baris test.csv |

### 8.2 Uji Health Check

**Request:** `GET http://localhost:8000/health`

**Response:**
```json
{
  "status": "ok",
  "chain_enabled": true,
  "chain_connected": true
}
```

### 8.3 Uji Endpoint `/analyze`

**Request:**
```json
POST /analyze
{
  "wallet": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
  "row_id": 0
}
```

**Response:**
```json
{
  "wallet": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
  "probability": 0.000008,
  "risk_score": 0,
  "risk_level": "LOW",
  "tx_hash": "0x918381d74ebda99838c854d9bc54ff093fe546d6de2ba2bd10a0ff07c7dd8188"
}
```

→ **tx_hash nyata** yang tercatat di blockchain lokal Hardhat.

### 8.4 Uji Endpoint `GET /risk/{wallet}`

**Request:** `GET /risk/0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266`

**Response (data dibaca langsung dari smart contract):**
```json
{
  "wallet": "0xf39Fd6e51aad88F6F4ce6aB8827279cffFb92266",
  "risk_score": 0,
  "timestamp": 1772591377,
  "model_version": "rf-v1"
}
```

> **Catatan timestamp:** `1772591377` decode menjadi **2026-03-04 02:29:37 UTC** — valid dan sesuai tanggal eksperimen (3–4 Maret 2026). Hardhat menggunakan `block.timestamp = system clock` pada node lokal, sehingga timestamp benar-benar mencerminkan waktu nyata saat transaksi dikirim. Bukan timestamp masa depan.

---

## 9) Alur Sistem (End-to-End)

```
User (Streamlit)
     │
     │  row_id + wallet
     ▼
FastAPI /analyze
     │
     ├──► ml_service.predict_from_row(row_id)
     │         │
     │         ▼
     │    ml/artifacts/model.pkl
     │    → RandomForest.predict_proba()
     │    → probability = 0.000008
     │    → risk_score  = 0
     │    → risk_level  = "LOW"
     │
     └──► chain_service.set_risk(wallet, score)
               │
               ▼
          web3.py → Hardhat RPC (localhost:8545)
          → RiskRegistry.setRisk(wallet, 0, "rf-v1")
          → tx_hash: 0x918381d7...
     │
     ▼
Response JSON → Streamlit UI
```

---

## 10) Catatan untuk Thesis

### BAB 3 — Metodologi

| Poin | Detail |
|---|---|
| Dataset | XBlock EPTransNet — 2.97M node, 1.165 label phishing |
| Preprocessing | Ekstraksi 18 fitur graf, sampling 1:10, split 80/20 stratified |
| Baseline | Logistic Regression — F1: 71,46% |
| Model Utama | Random Forest (200 trees, balanced class weight) — F1: **87,93%** |
| Risk Score | `round(P(phishing) × 100)` → label LOW/MEDIUM/HIGH |
| Smart Contract | `RiskRegistry.sol` — `setRisk` + `getRisk` + event `RiskUpdated` |
| Alasan blockchain | Immutability: skor tersimpan permanen, timestamped, tidak bisa dimodifikasi |

### BAB 4 — Implementasi

| Poin | Detail |
|---|---|
| Arsitektur | Streamlit → FastAPI → ML model + web3.py → Hardhat smart contract |
| Metrik RF | Accuracy 97,82% · Precision 88,31% · Recall 87,55% · F1 87,93% |
| Contract addr | `0x5FbDB2315678afecb367f032d93F642f64180aa3` |
| Contoh tx hash | `0x918381d74ebda99838c854d9bc54ff093fe546d6de2ba2bd10a0ff07c7dd8188` |
| Demo mode | Pilih `row_id` dari 2.563 test rows → analisis instan tanpa isi form |
| Artifacts | `model.pkl`, `confusion_matrix.png`, `feature_importance.png` tersimpan di `ml/artifacts/` |

---

## 11) Eksperimen Validasi Sampling Ratio: 1:5 vs 1:10 (RQ3)

### 11.1 Latar Belakang

Dataset asli sangat tidak seimbang (**rasio 1 : 2.550**). Karena tidak mungkin mengambil 2.97M node, digunakan strategi *random undersampling* terhadap kelas mayoritas (normal). Dua rasio diuji untuk mengevaluasi tradeoff antara ukuran dataset, representasi distribusi, dan performa model.

### 11.2 Konfigurasi Eksperimen

| Parameter | Nilai |
|---|---|
| Phishing nodes | 1.165 (tetap, semua digunakan) |
| Rasio 1:5 | 5.825 normal → total 6.990 node |
| Rasio 1:10 | 11.650 normal → total 12.815 node |
| Split | 80% train / 20% test, stratified |
| Random seed | 42 |
| Model baseline | Logistic Regression (`class_weight="balanced"`) |
| Model utama | Random Forest (200 trees, `class_weight="balanced"`) |

### 11.3 Hasil Perbandingan (Data Aktual)

| Rasio | Model | Total Node | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **1:5** | Logistic Regression | 6.990 | 95,28% | 90,34% | 80,26% | 85,00% |
| **1:5** | **Random Forest** | 6.990 | **97,71%** | **91,36%** | **95,28%** | **93,28%** |
| 1:10 | Logistic Regression | 12.815 | 95,04% | 75,00% | 68,24% | 71,46% |
| 1:10 | **Random Forest** | 12.815 | **97,82%** | **88,31%** | **87,55%** | **87,93%** |

### 11.4 Analisis & Justifikasi Pilihan 1:10

| Aspek | 1:5 | 1:10 | Kesimpulan |
|---|---|---|---|
| RF F1 | **93,28%** | 87,93% | 1:5 lebih tinggi karena distribusi lebih seimbang (16,7% phishing) |
| Dataset size | 6.990 | **12.815** | 1:10 lebih representatif terhadap distribusi nyata |
| Rasio phishing | 16,7% | 9,1% → lebih mendekati "dunia nyata" (daripada 0,04%) | |
| Recall RF | 95,28% | 87,55% | Pada 1:5, model lebih agresif mendeteksi phishing |
| Risiko overfitting | Lebih tinggi (distribusi artifisial padat) | Lebih rendah | 1:10 lebih aman |

**Kesimpulan:** Rasio **1:10 dipilih** sebagai konfigurasi utama karena:
1. Menghasilkan dataset lebih besar → estimasi performa lebih stabil
2. Distribusi lebih mendekati kondisi nyata (walau masih jauh dari 1:2550)
3. Metrik pada 1:10 lebih *konservatif* → jujur dalam melaporkan performa
4. F1 87,93% pada kondisi lebih sulit lebih bermakna daripada F1 93,28% pada kondisi lebih mudah

> **Catatan validitas:** Kedua rasio masih merupakan distribusi artifisial. Metrik yang dilaporkan tidak merepresentasikan performa pada distribusi populasi asli (1:2550). Hal ini merupakan *limitation* yang perlu disebutkan di BAB 5.

---

## 12) Evaluasi Performa Blockchain (RQ2)

### 12.1 Setup Pengukuran

Dilakukan **20 transaksi** `setRisk()` ke kontrak `RiskRegistry` yang ter-deploy di Hardhat localhost, menggunakan wallet dan skor berbeda-beda.

| Parameter | Nilai |
|---|---|
| Jaringan | Hardhat localhost (chainId 31337) |
| Jumlah transaksi | 20 |
| Fungsi diuji | `setRisk(address, uint16, string)` |
| Gas price | default Hardhat (auto) |
| Pengukuran | Client-side wall-clock (Python `time.perf_counter`) |

### 12.2 Hasil Pengukuran (Data Aktual — 20 Transaksi)

**Latency Commit (ms):**

| Metrik | Nilai |
|---|---|
| Minimum | 11,2 ms |
| Maksimum | 38,4 ms |
| **Rata-rata** | **13,6 ms** |
| Median | 12,0 ms |
| Std. Dev | 5,9 ms |

**Gas Used per Transaksi `setRisk()`:**

| Metrik | Nilai |
|---|---|
| Minimum | 35.452 |
| Maksimum | 35.464 |
| **Rata-rata** | **35.462** |
| Median | 35.464 |
| Std. Dev | 5 |

### 12.3 Interpretasi

- **Latency 13,6 ms rata-rata** sangat rendah untuk konteks localhost. Pada jaringan publik (Ethereum mainnet), latency akan 12–15 detik (1 block). Untuk Polygon atau chain L2, latency bisa 1–3 detik.
- **Gas ~35.462 unit** sangat konsisten (stdev = 5). Konsistensi ini karena `SSTORE` pertama ke slot baru memiliki biaya tetap. Variasi kecil terjadi karena panjang string `modelVersion` dan kondisi slot (write vs overwrite).
- Untuk konteks **Ethereum mainnet** (harga gas ~10 Gwei, ETH ~$3.000 saat penulisan), gas cost per transaksi ≈ **35.462 × 10 × 10⁻⁹ ETH × $3.000 ≈ $0,001** — sangat terjangkau.
- **Kesimpulan RQ2:** Overhead blockchain untuk mencatat satu risk score adalah **~14ms (lokal) / ~12 detik (mainnet)** dengan biaya gas yang dapat diabaikan. Overhead ini **wajar** untuk use case audit trail keamanan.

---

## 13) Reproducibility Checklist (BAB 3 / Lampiran)

### 13.1 Hardware yang Digunakan

| Komponen | Spesifikasi |
|---|---|
| CPU | Apple M2 |
| Cores | 8 |
| RAM | 8 GB |
| OS | macOS 26.3 |

### 13.2 Software & Versi

| Software | Versi |
|---|---|
| Python | 3.14.3 (kompatibel dengan 3.10+) |
| scikit-learn | 1.8.0 |
| pandas | 2.3.3 |
| numpy | 2.4.2 |
| networkx | 3.6.1 |
| joblib | 1.5.3 |
| FastAPI | 0.135.1 |
| uvicorn | 0.41.0 |
| Streamlit | 1.55.0 |
| web3.py | 7.14.1 |
| Node.js | 25.7.0 |
| Hardhat | 2.22.x |
| Solidity | 0.8.20 |

### 13.3 Cara Menjalankan Ulang (End-to-End)

```bash
# 1. Clone & masuk ke folder
git clone <repo> && cd ethereum-phising-risk-detection

# 2. Install Python dependencies
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. Pastikan MulDiGraph.pkl ada di data/raw/

# 4. Feature extraction
python3 ml/preprocess.py

# 5. Training
python3 ml/train.py

# 6. Install Hardhat & deploy kontrak
cd chain && npm install
npx hardhat node &             # Terminal background
npx hardhat run scripts/deploy.ts --network localhost

# 7. Jalankan API
cd .. && source .venv/bin/activate
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

# 8. Jalankan UI
streamlit run app/streamlit_app.py --server.port 8502
```

**Random seed**: semua eksperimen menggunakan `random_state=42` dan `random.seed(42)`.

---

## 14) Threats to Validity & Limitations (BAB 5)

### 14.1 Internal Validity

| Ancaman | Penjelasan | Mitigasi |
|---|---|---|
| **Imbalance ekstrem** | Rasio asli 1:2.550 jauh lebih ekstrem dari yang diuji (1:5 / 1:10) | Gunakan `class_weight="balanced"`, laporkan keterbatasan eksplisit |
| **Data leakage** | Fitur diekstrak per node dari graph yang sama tempat label ada | Split dilakukan setelah ekstraksi; tidak ada fitur dari test masuk ke train |
| **Sampling bias** | Pemilihan normal nodes secara acak bisa menyertakan "abu-abu" (belum terlaporkan) | Konsisten dengan literatur; label hanya dari yang sudah terlaporkan Etherscan |

### 14.2 External Validity

| Ancaman | Penjelasan |
|---|---|
| **Generalisasi terbatas** | Model dilatih pada dataset 2019; pola phishing Ethereum terus berkembang |
| **Dataset statis** | EPTransNet adalah snapshot, bukan live feed — tidak ada konsep *concept drift* |
| **Labeling bias** | Label phishing berdasarkan laporan publik Etherscan; node phishing tak terlaporkan bisa salah dikategorikan normal |

### 14.3 Construct Validity

| Ancaman | Penjelasan |
|---|---|
| **Demo mode (row_id)** | Demo Streamlit menggunakan row dari test.csv — bukan crawling real-time dari blockchain Ethereum. Hanya untuk demontstrasi antarmuka. |
| **Blockchain lokal** | Hardhat localhost ≠ jaringan publik. Latency dan gas cost nyata akan berbeda. |
| **Risk score linear** | `round(p × 100)` adalah mapping sederhana; tidak mempertimbangkan calibration probabilistik model RF |

### 14.4 Ringkasan Kontribusi vs Keterbatasan

| ✅ Kontribusi | ⚠️ Keterbatasan |
|---|---|
| Pipeline ML end-to-end dari graph pkl ke prediksi | Dataset 2019, tidak real-time |
| Bukti integrasi blockchain (immutable audit trail) | Blockchain lokal (Hardhat), bukan mainnet |
| Perbandingan baseline vs model utama dengan metrik standar | Imbalance tidak sepenuhnya diatasi |
| Eksperimen empiris sampling ratio | Demo mode, bukan crawling on-chain |
| Gas + latency terukur (20 transaksi nyata) | Risk score belum di-calibrate |
