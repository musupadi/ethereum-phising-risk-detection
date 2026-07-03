# AI + Blockchain Risk Registry (Ethereum Phishing) — Hasil Implementasi

> Dokumen ini mencatat hasil nyata dari implementasi sistem, berdasarkan blueprint di `code.md`.
> Dibuat: 3 Maret 2026
> Update terakhir: 21 Mei 2026

---

## Executive Summary (Ringkas)

Penelitian ini berhasil membangun sistem end-to-end untuk deteksi risiko phishing wallet Ethereum: **ML inference + penyimpanan immutable risk score ke smart contract**.

### Hasil Utama

| Komponen | Hasil | Kesimpulan |
|---|---|---|
| **Model utama eksperimen EPTransNet 1:10** | Random Forest: Accuracy 97,82% · Precision 88,31% · Recall 87,55% · **F1 87,93%** | Baseline kuat; model **terbaik & yang di-deploy** adalah XGBoost (lihat baris skenario) |
| **Baseline** | Logistic Regression: **F1 71,46%** | RF unggul +16,47 poin F1 |
| **Eksperimen skenario dataset gabungan (8.595 alamat, 4 skenario)** | MulDiGraph only → XGBoost: **F1 88,57%**, RF: **86,85%**, GNN-GraphSAGE: **84,32%**. MulDiGraph + Etherscan → XGBoost: **91,02%**, GNN-GraphSAGE: **89,60%**, RF: **88,84%** | Fitur Etherscan meningkatkan semua model; GNN-GraphSAGE dengan edge transaksi nyata hampir menyamai XGBoost (+1,42 poin F1) |
| **Blockchain write performance** | Rata-rata latency commit **13,6 ms** (localhost), gas rata-rata **35.462** | Overhead on-chain wajar untuk audit trail |
| **Integrasi data eksternal** | Pipeline Etherscan + Forta sudah tersedia (fitur dan label siap pakai) | Siap dipakai untuk eksperimen lanjutan dengan data real |

### Jawaban Singkat Research Questions

1. **RQ1 terjawab:** Random Forest konsisten mengungguli baseline Logistic Regression pada setup utama.
2. **RQ2 terjawab:** Pencatatan risk score ke smart contract feasible secara waktu dan biaya.
3. **RQ3 terjawab:** Fitur graf transaksi efektif; rasio sampling memengaruhi metrik dan trade-off generalisasi.

### Keputusan Implementasi

- **Model produksi yang di-deploy (API + Streamlit):** XGBoost pada skenario gabungan **MulDiGraph + Etherscan** (26 fitur, **F1 91,02%**) — model terbaik pada eksperimen. Untuk wallet yang ada di dataset, fitur Etherscan sudah ter-precompute sehingga inferensi tidak perlu memanggil API eksternal. Artefak: `ml/artifacts/model_prod_xgb.json` (dibuat oleh `ml/train_production_model.py`).
- **Random Forest:** baseline kuat pada eksperimen utama EPTransNet 1:10 (F1 87,93%); tetap dilatih sebagai pembanding dan tersedia sebagai fallback inferensi.
- **GNN-GraphSAGE:** setelah diperbaiki dengan arsitektur 2-layer + edge transaksi nyata dari MulDiGraph, berhasil mencapai F1 89,60% — hampir menyamai XGBoost dan layak sebagai kontribusi ilmiah perbandingan arsitektur.

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
│       ├── model_prod_xgb.json         ✅ XGBoost MulDiGraph+Etherscan (DI-DEPLOY, 26 fitur)
│       ├── prod_feature_schema.json    ✅ 26 fitur (skenario terbaik)
│       ├── model.pkl                   ✅ Random Forest baseline (joblib, fallback)
│       ├── scaler.pkl                  ✅ StandardScaler (untuk LR)
│       ├── feature_schema.json         ✅ 18 fitur (baseline graf)
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

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_cm_lr.png`
> **Caption:** Gambar 4.X — Confusion Matrix Logistic Regression Baseline (EPTransNet 1:10, n=2.563)
> *(Insert di Word pada bagian pembahasan model baseline — tunjukkan FN=74 sebagai kelemahan LR)*

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_cm_xgboost.png`
> **Caption:** Gambar 4.X — Confusion Matrix XGBoost pada Skenario Terbaik (MulDiGraph + Etherscan, n=1.719)
> *(Insert di Word sebagai confusion matrix model terbaik berdasarkan F1)*

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_cm_gnn.png`
> **Caption:** Gambar 4.X — Confusion Matrix GNN-GraphSAGE V2 pada Skenario Terbaik (MulDiGraph + Etherscan, n=1.719)
> *(Insert di Word berdampingan atau setelah confusion matrix XGBoost — highlight FN hanya 9)*

### 5.2 Classification Report — Random Forest

```
              precision    recall  f1-score   support

      Normal       0.99      0.99      0.99      2330
    Phishing       0.88      0.88      0.88       233

    accuracy                           0.98      2563
   macro avg       0.94      0.93      0.93      2563
weighted avg       0.98      0.98      0.98      2563
```

### 5.3 Analisis Confusion Matrix

Perbandingan lengkap keempat model (skenario terbaik = MulDiGraph + Etherscan, kecuali LR & RF baseline dari EPTransNet 1:10):

| Model | TN | FP | FN | TP | FN rate | FP rate |
|---|---|---|---|---|---|---|
| Logistic Regression (baseline, n=2563) | 2277 | 53 | 74 | 159 | **31,8%** | 2,3% |
| Random Forest (baseline, n=2563) | 2303 | 27 | 28 | 205 | 12,0% | 1,2% |
| **XGBoost** (Sc2, n=1719) | 1458 | 28 | 15 | 218 | **6,4%** | 1,9% |
| **GNN-GraphSAGE V2** (Sc2, n=1719) | 1443 | 43 | **9** | 224 | **3,9%** | 2,9% |

**Interpretasi untuk konteks keamanan (phishing detection):**
- **FN (False Negative) lebih kritis** — phishing yang tidak terdeteksi = wallet korban tidak terlindungi
- **GNN memiliki FN terendah (9 node, 3,9%)** → paling aman dari sisi recall, hanya 9 dari 233 phishing lolos
- **XGBoost lebih seimbang** (FN=15, FP=28) → F1 tertinggi (91,02%), cocok sebagai model produksi
- **GNN lebih agresif mendeteksi phishing** (FP=43 lebih tinggi dari XGBoost=28) → lebih banyak false alarm tapi lebih sedikit yang lolos
- **Trade-off XGBoost vs GNN:** XGBoost unggul F1 overall, GNN unggul recall — untuk keamanan siber, keduanya layak dipertimbangkan tergantung toleransi false alarm sistem

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/feature_importance.png`
> **Caption:** Gambar 4.X — Feature Importance Random Forest (Top 15 Fitur Graf dari MulDiGraph)
> *(Insert gambar di Word pada bagian analisis fitur/pembahasan RQ3)*

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

### 5.5 Perbandingan Empat Skenario Dataset (Eksperimen Final)

Eksperimen ini membandingkan empat kombinasi sumber fitur pada **8.595 alamat wallet** yang sama, menggunakan split train/test yang identik (80/20, stratified, random seed 42), sehingga selisih performa murni mencerminkan kontribusi tiap kelompok fitur.

**Model yang dibandingkan:** Logistic Regression (baseline), Random Forest, XGBoost, GNN-GraphSAGE V2.

**GNN menggunakan edge transaksi nyata** dari MulDiGraph (528 edge antar alamat sampel, atau 264 edge unik setelah deduplication), bukan graph kNN sintetis seperti pada versi sebelumnya.

#### Skenario 1 — Hanya fitur graf dari MulDiGraph (19 fitur)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 94,65% | 86,15% | 72,10% | 78,50% |
| Random Forest | 96,34% | 84,55% | 89,27% | 86,85% |
| XGBoost | 96,80% | 85,89% | 91,42% | 88,57% |
| GNN-GraphSAGE V2 | 95,52% | 80,23% | 88,84% | **84,32%** |

#### Skenario 2 — Fitur graf + fitur Etherscan (26 fitur)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 96,16% | 88,13% | 82,83% | 85,40% |
| Random Forest | 96,92% | 87,19% | 90,56% | 88,84% |
| **XGBoost** | **97,50%** | **88,62%** | **93,56%** | **91,02%** |
| GNN-GraphSAGE V2 | 96,97% | 83,90% | 96,14% | **89,60%** |

#### Skenario 3 — Fitur graf + fitur BigQuery (29 fitur)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 94,71% | 86,22% | 72,53% | 78,79% |
| Random Forest | 96,39% | 84,62% | 89,70% | 87,08% |
| XGBoost | 96,74% | 85,54% | 91,42% | 88,38% |
| GNN-GraphSAGE V2 | 95,00% | 75,97% | 92,27% | 83,33% |

#### Skenario 4 — Fitur graf + Etherscan + BigQuery (36 fitur)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 96,16% | 87,78% | 83,26% | 85,46% |
| Random Forest | 97,15% | 87,70% | 91,85% | 89,73% |
| XGBoost | 97,32% | 88,16% | 92,70% | 90,38% |
| GNN-GraphSAGE V2 | 96,57% | 86,25% | 88,84% | 87,53% |

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_f1_by_scenario.png`
> **Caption:** Gambar 4.X — Perbandingan F1-Score Empat Model pada Empat Skenario Dataset
> *(Insert gambar di Word setelah keempat tabel skenario di atas — gambar ini merangkum semua tabel dalam satu visualisasi grouped bar chart)*

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_heatmap_f1.png`
> **Caption:** Gambar 4.X — Heatmap F1-Score: Perbandingan Model × Skenario Dataset
> *(Alternatif/pelengkap gambar di atas — heatmap lebih ringkas, cocok untuk versi singkat di BAB 4)*

#### Interpretasi Hasil

1. **Fitur Etherscan secara konsisten meningkatkan performa semua model** (Skenario 1 → 2): XGBoost naik +2,45 poin F1, GNN naik +5,28 poin F1, RF naik +1,99 poin F1.

2. **Fitur BigQuery alone (Skenario 3) tidak menambah nilai** dibanding fitur graf saja (Skenario 1). Hal ini konsisten karena fitur BigQuery yang tersedia banyak tumpang-tindih semantik dengan fitur graf (misalnya `bq_tx_count_out` ≈ `out_degree`).

3. **Fitur gabungan Etherscan + BigQuery (Skenario 4) tidak melampaui Etherscan saja (Skenario 2)**, mengindikasikan bahwa BigQuery tidak menambah informasi independen setelah Etherscan tersedia.

4. **XGBoost adalah model terbaik secara keseluruhan** (F1 91,02% pada Skenario 2), unggul tipis atas GNN-GraphSAGE V2 yang mencapai F1 89,60% pada skenario yang sama.

5. **GNN-GraphSAGE V2 kini kompetitif** setelah diperbaiki: menggunakan edge transaksi nyata dari MulDiGraph, arsitektur 2-layer (mencegah *over-smoothing*), dan SAGEConv yang lebih robust terhadap variasi degree. Peningkatan dari versi sebelumnya (GNN-GCN 4-layer kNN) sangat signifikan: **+42,50 poin F1** pada skenario tanpa Etherscan dan **+22,76 poin F1** pada skenario dengan Etherscan.

6. **GNN mendapat manfaat terbesar dari penambahan fitur Etherscan** (+5,28 poin F1), lebih besar dari XGBoost (+2,45) dan RF (+1,99), menunjukkan bahwa fitur atribut wallet memperkaya representasi node dalam message passing.

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_best_scenario_metrics.png`
> **Caption:** Gambar 4.X — Perbandingan Accuracy, Precision, Recall, dan F1-Score pada Skenario Terbaik (MulDiGraph + Etherscan, 26 Fitur)
> *(Insert di Word pada sub-bagian pembahasan skenario terbaik)*

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_etherscan_impact.png`
> **Caption:** Gambar 4.X — Dampak Penambahan Fitur Etherscan terhadap F1-Score Semua Model (Skenario 1 → Skenario 2)
> *(Insert di Word pada sub-bagian pembahasan kontribusi fitur Etherscan / jawaban RQ3)*

> **Catatan metodologi:** Semua empat skenario menggunakan baris dan split yang identik. GNN menggunakan 528 edge transaksi nyata yang diekstrak dari MulDiGraph untuk 8.595 alamat sampel (rata-rata ~0,06 edge per node dalam subgraph — sangat sparse), namun GraphSAGE dengan self-loop dan 2-layer aggregation tetap mampu memanfaatkan sinyal struktural yang ada.

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
    string  modelVersion; // e.g. "xgb-eth-v1"
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
| Model Version | `xgb-eth-v1` |

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

**Request (Scan by Address — fitur diambil dari dataset):**
```json
POST /analyze
{
  "wallet": "0x4dad23689bef624af5d1757a56135901c915fcfd"
}
```

**Response (model XGBoost MulDiGraph + Etherscan):**
```json
{
  "wallet": "0x4dad23689bef624af5d1757a56135901c915fcfd",
  "probability": 0.999801,
  "risk_score": 100,
  "risk_level": "HIGH",
  "tx_hash": "c7c8069430427b1e8f482c49b9d26d1b0eb359f5a18f210160b8257d27de099c",
  "true_label": 1,
  "dataset_split": "dataset"
}
```

→ **tx_hash nyata** yang tercatat di blockchain lokal Hardhat. `true_label: 1` mengonfirmasi wallet ini memang phishing menurut dataset (prediksi cocok).

### 8.4 Uji Endpoint `GET /risk/{wallet}`

**Request:** `GET /risk/0x4dad23689bef624af5d1757a56135901c915fcfd`

**Response (data dibaca langsung dari smart contract):**
```json
{
  "wallet": "0x4dad23689bef624af5d1757a56135901c915fcfd",
  "risk_score": 100,
  "timestamp": 1782465140,
  "model_version": "xgb-eth-v1"
}
```

> **Catatan:** `model_version: "xgb-eth-v1"` menandakan model produksi = XGBoost (MulDiGraph + Etherscan). `timestamp` mengikuti `block.timestamp` Hardhat (= system clock node lokal), jadi mencerminkan waktu nyata saat transaksi dikirim.

---

## 9) Alur Sistem (End-to-End)

```
User (Streamlit)
     │
     │  wallet (Scan by Address) / row_id (Demo)
     ▼
FastAPI /analyze
     │
     ├──► ml_service.predict_from_address(wallet)   # lookup fitur 26-dim dari dataset
     │         │
     │         ▼
     │    ml/artifacts/model_prod_xgb.json
     │    → XGBoost.predict() (MulDiGraph + Etherscan)
     │    → probability = 0.999801
     │    → risk_score  = 100
     │    → risk_level  = "HIGH"
     │
     └──► chain_service.set_risk(wallet, score)
               │
               ▼
          web3.py → Hardhat RPC (localhost:8545)
          → RiskRegistry.setRisk(wallet, 100, "xgb-eth-v1")
          → tx_hash: c7c80694...
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
| **Model produksi (di-deploy)** | **XGBoost (MulDiGraph + Etherscan, 26 fitur), F1 91,02%** — `model_prod_xgb.json`, model_version `xgb-eth-v1` |
| Metrik RF (baseline eksperimen EPTransNet 1:10) | Accuracy 97,82% · Precision 88,31% · Recall 87,55% · F1 **87,93%** |
| Hasil eksperimen skenario (8.595 alamat, 4 skenario) | XGBoost terbaik: **F1 91,02%** · GNN-GraphSAGE V2: **F1 89,60%** · RF: **F1 88,84%** (semua pada skenario MulDiGraph + Etherscan) |
| Model GNN | GraphSAGE V2, 2-layer, hidden=128, edge transaksi nyata dari MulDiGraph (528 edges) |
| Contract addr | `0x5FbDB2315678afecb367f032d93F642f64180aa3` |
| Contoh tx hash | `c7c8069430427b1e8f482c49b9d26d1b0eb359f5a18f210160b8257d27de099c` |
| Mode aplikasi | **Scan by Address** (tempel wallet dataset), **Demo** (`row_id` dari 1.719 held-out), **Manual** (isi fitur) |
| Artifacts | `model_prod_xgb.json`, `model.pkl`, `dataset_scenarios_comparison.csv`, `confusion_matrix.png`, `feature_importance.png` di `ml/artifacts/` |

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

## 12) Pipeline ML Final — Ringkasan Arsitektur & Hasil (Update: 3 Juni 2026)

### Pipeline Architecture

```
train_dataset_scenarios.py
├── Load hybrid_dataset.csv  (8.595 alamat, 41 kolom)
├── Load MulDiGraph.pkl       (2,97M node — untuk ekstraksi edge GNN)
├── Build 4 scenarios dari hybrid_feature_schema.json:
│   ├── muldigraph_only                    (19 fitur)
│   ├── muldigraph_plus_etherscan          (26 fitur)
│   ├── muldigraph_plus_bigquery           (29 fitur)
│   └── muldigraph_plus_etherscan_plus_bigquery (36 fitur)
├── Per skenario, train 4 model:
│   ├── Logistic Regression  (baseline)
│   ├── Random Forest
│   ├── XGBoost
│   └── GNN-GraphSAGE V2  ← edge transaksi nyata dari MulDiGraph
└── Simpan hasil ke ml/artifacts/dataset_scenarios_comparison.csv
```

### Hasil Final Eksperimen Skenario (Diverifikasi 3 Juni 2026)

| Skenario | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MulDiGraph only | LR | 94,65% | 86,15% | 72,10% | 78,50% |
| MulDiGraph only | RF | 96,34% | 84,55% | 89,27% | 86,85% |
| MulDiGraph only | XGBoost | 96,80% | 85,89% | 91,42% | 88,57% |
| MulDiGraph only | GNN-GraphSAGE V2 | 95,52% | 80,23% | 88,84% | 84,32% |
| MulDiGraph + Etherscan | LR | 96,16% | 88,13% | 82,83% | 85,40% |
| MulDiGraph + Etherscan | RF | 96,92% | 87,19% | 90,56% | 88,84% |
| **MulDiGraph + Etherscan** | **XGBoost** | **97,50%** | **88,62%** | **93,56%** | **91,02%** |
| MulDiGraph + Etherscan | GNN-GraphSAGE V2 | 96,97% | 83,90% | 96,14% | **89,60%** |
| MulDiGraph + BigQuery | LR | 94,71% | 86,22% | 72,53% | 78,79% |
| MulDiGraph + BigQuery | RF | 96,39% | 84,62% | 89,70% | 87,08% |
| MulDiGraph + BigQuery | XGBoost | 96,74% | 85,54% | 91,42% | 88,38% |
| MulDiGraph + BigQuery | GNN-GraphSAGE V2 | 95,00% | 75,97% | 92,27% | 83,33% |
| MulDiGraph + Etherscan + BigQuery | LR | 96,16% | 87,78% | 83,26% | 85,46% |
| MulDiGraph + Etherscan + BigQuery | RF | 97,15% | 87,70% | 91,85% | 89,73% |
| MulDiGraph + Etherscan + BigQuery | XGBoost | 97,32% | 88,16% | 92,70% | 90,38% |
| MulDiGraph + Etherscan + BigQuery | GNN-GraphSAGE V2 | 96,57% | 86,25% | 88,84% | 87,53% |

### Konfigurasi Model Final

**GNN-GraphSAGE V2:**
```
Architecture  : GraphSAGE (SAGEConv), 2 layers, hidden=128, dropout=0.5
Graph edges   : 528 edge transaksi nyata dari MulDiGraph (ekstraksi subgraph)
Training      : Adam lr=0.01, CrossEntropyLoss dengan class weights
                normal=0.578, phishing=3.689
Early stop    : patience=25, best-weight restoration
Epochs        : maks 200, berhenti di epoch 90–118 (per skenario)
Training loss : konvergen di 0.12–0.23 (CrossEntropyLoss, tergantung skenario fitur)
              Sc1 (MulDiGraph only)          : ~0.145 (ep 90)
              Sc2 (+ Etherscan)              : ~0.121 (ep 94)  ← loss terendah
              Sc3 (+ BigQuery)               : ~0.234 (ep 98)
              Sc4 (+ Etherscan + BigQuery)   : ~0.164 (ep 118)
```

**XGBoost:**
```
n_estimators  : 250, max_depth=6, learning_rate=0.05
subsample     : 0.85, colsample_bytree=0.85
scale_pos_weight: 6.38 (auto dari rasio kelas)
Early stop    : patience=40
```

**Random Forest:**
```
n_estimators  : 200, max_depth=None, min_samples_leaf=2
class_weight  : balanced, n_jobs=-1
```

### Artifacts Tersimpan

```
ml/artifacts/
├── dataset_scenarios_comparison.csv   ← semua 16 baris hasil eksperimen
├── dataset_scenarios_summary.json     ← best model per skenario
├── hybrid_feature_schema.json         ← daftar 36 fitur + 4 skenario
├── model_v2_best.pkl                  ← best model (RandomForest)
├── model_xgboost.json                 ← XGBoost tersimpan
├── model_gcn_v2.pt                    ← GNN-GraphSAGE tersimpan
└── [confusion matrices, feature importance plots]
```

### Next Steps

1️⃣ **Local Testing**: ✅ COMPLETE  
2️⃣ **Generate real enriched files** (`train_enriched.csv`, `test_enriched.csv`) dari API Etherscan/Forta  
3️⃣ **Re-run scenario script**: `python ml/train_dataset_scenarios.py`  
4️⃣ **Lock model final berdasarkan F1 tertinggi** pada data real gabungan  
5️⃣ **Production Deployment**: gunakan model final + versioning artifacts  

---

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
- Pengujian ini masih dilakukan pada **Hardhat localhost**. Biaya aktual di jaringan publik akan bergantung pada gas price, kondisi jaringan, dan biaya eksekusi saat transaksi dikirim.
- **Kesimpulan RQ2:** Overhead blockchain untuk mencatat satu risk score adalah **~14ms (lokal)** dengan gas yang stabil dan wajar untuk use case audit trail keamanan.

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
| **Subgraph sangat sparse** | Dari 8.595 alamat sampel, hanya 528 edge transaksi langsung ditemukan antar sesama alamat sampel (rata-rata ~0,06 edge/node). GNN beroperasi hampir tanpa koneksi semantik antar node — keunggulan strukturalnya belum optimal. |

### 14.3 Construct Validity

| Ancaman | Penjelasan |
|---|---|
| **Demo mode (row_id)** | Demo Streamlit menggunakan row dari test.csv — bukan crawling real-time dari blockchain Ethereum. Hanya untuk demontstrasi antarmuka. |
| **Blockchain lokal** | Hardhat localhost ≠ jaringan publik. Latency dan gas cost nyata akan berbeda. |
| **Risk score linear** | `round(p × 100)` adalah mapping sederhana; tidak mempertimbangkan calibration probabilistik model RF |

### 14.4 Kualitas Data Fitur Eksternal (Etherscan & BigQuery)

Audit kualitas data terhadap 8.595 alamat sampel mengungkap beberapa keterbatasan pada fitur eksternal yang perlu dicatat secara jujur:

**Fitur Etherscan (coverage 99,6% — layak dipakai, dengan catatan):**

| Catatan | Detail | Mitigasi / Status |
|---|---|---|
| `created_at_ts` konstan nol | Pada run eksperimen, kolom ini tidak terisi (selalu 0) sehingga menjadi *dead feature* tanpa kontribusi ke model. Karena varians nol, fitur ini **tidak memengaruhi metrik** RF/XGBoost/LR yang dilaporkan. | Kode enrichment (`ml/data_fetchers.py`) telah diperbaiki agar mengambil timestamp transaksi pertama untuk run berikutnya. |
| `unique_counterparties > transaction_count` pada ~32% baris | Bukan korupsi acak, melainkan **beda basis hitung**: `transaction_count` dihitung dari seluruh tx normal (cap 10.000), sedangkan `unique_counterparties`, `avg_tx_value`, `tx_frequency`, dan `activity_score` dihitung hanya dari **100 transaksi terbaru** (jendela sampling) dan sempat menyertakan alamat itu sendiri. | Inkonsistensi self-count diperbaiki di kode; keterbatasan jendela 100-tx didokumentasikan di sini. |
| `transaction_count` ter-cap 10.000 | 5 alamat sangat aktif nilainya terpotong oleh limit halaman API Etherscan. | Dampak minor; dicatat sebagai keterbatasan pengukuran. |
| 35 baris (0,41%) gagal fetch | Di-imputasi nol (missing menyamar sebagai nilai nol). | Dampak kecil terhadap distribusi; dicatat. |

**Fitur BigQuery (coverage 0,93% — keterbatasan signifikan):**

Hanya 80 dari 8.595 alamat (0,93%) yang benar-benar memiliki data BigQuery; 8.515 baris di-imputasi nol sehingga seluruh 10 fitur `bq_*` praktis konstan-nol. Karena itu Skenario 3 (graf + BigQuery) tidak menambah nilai dibanding Skenario 1 — penyebab utamanya adalah **coverage yang sangat rendah dan jendela waktu berbeda** (graf MulDiGraph 2019 vs sebagian data BigQuery yang lebih baru), bukan semata tumpang-tindih semantik. Selain itu, dari 80 baris ber-data tersebut 79 berlabel normal dan hanya 1 phishing, sehingga keberadaan data BigQuery sedikit terkorelasi dengan label (potensi bias yang, untungnya, berdampak kecil karena hampir semua baris bernilai nol). Fitur BigQuery tetap dipertahankan dalam pelaporan sebagai bagian dari eksperimen, namun **tidak direkomendasikan sebagai sumber fitur produksi** sampai coverage-nya diperbaiki.

### 14.5 Ringkasan Kontribusi vs Keterbatasan

| ✅ Kontribusi | ⚠️ Keterbatasan |
|---|---|
| Pipeline ML end-to-end dari graph pkl ke prediksi | Dataset 2019, tidak real-time |
| Bukti integrasi blockchain (immutable audit trail) | Blockchain lokal (Hardhat), bukan mainnet |
| Perbandingan 4 model × 4 skenario dengan metrik standar | Imbalance tidak sepenuhnya diatasi |
| Eksperimen empiris sampling ratio (RQ3) | Demo mode, bukan crawling on-chain |
| GNN-GraphSAGE dengan edge transaksi nyata (89,60% F1) | Subgraph sparse — hanya 528 edge dari 8.595 node |
| Gas + latency terukur (20 transaksi nyata) | Risk score belum di-calibrate |

---

## 15) Penguatan Akademik yang Disarankan

Bagian ini dirangkum sebagai penguatan tesis bila masih ada waktu sebelum sidang. Tujuannya bukan sekadar menaikkan skor, tetapi membuat hasil lebih stabil, fair, dan lebih sulit dibantah secara metodologis.

### 15.1 Prioritas Utama

1. **Tambahkan evaluasi imbalance-aware**: PR-AUC, ROC-AUC, confusion matrix, dan false negative rate untuk melengkapi accuracy, precision, recall, serta F1.
2. **Lakukan threshold tuning** untuk XGBoost dan GNN, misalnya pada threshold 0.3 sampai 0.7, agar F1 terbaik tidak bergantung pada ambang default 0.5.
3. **Validasi multi-random-seed** dengan beberapa split, misalnya 7, 21, 42, 99, dan 123, lalu laporkan rata-rata serta standar deviasi F1.
4. **Perkuat interpretasi XGBoost** lewat feature importance dan SHAP untuk menjelaskan mengapa fitur Etherscan konsisten meningkatkan performa.
5. **Rapikan narasi blockchain** sebagai audit trail immutable, bukan sebagai klaim biaya mainnet yang presisi, karena pengujian utama masih pada Hardhat localhost.

### 15.2 Penguatan Eksperimen GNN

Karena performa GNN-GraphSAGE V2 sudah kompetitif tetapi masih sedikit di bawah XGBoost pada skenario terbaik, fokus akademik yang paling kuat adalah menjelaskan batasannya secara empiris.

| Eksperimen | Tujuan |
|---|---|
| Edge asli antar sampled node | Menjadi baseline GNN saat ini |
| Edge 1-hop neighborhood tambahan | Menguji apakah graph yang lebih padat meningkatkan performa |
| Edge berarah transaksi | Membedakan sinyal incoming dan outgoing |
| Edge berbobot nilai transaksi | Menguji apakah transaksi besar lebih informatif |
| Edge berbobot frekuensi transaksi | Menguji pengaruh intensitas interaksi antar wallet |
| GAT dibanding GraphSAGE | Melihat apakah attention membantu memilih neighbor yang lebih relevan |

Narasi yang aman secara akademik adalah bahwa performa GNN sangat dipengaruhi oleh kepadatan dan kualitas struktur graph. Pada dataset ini subgraph yang dipakai masih sangat sparse, sehingga keunggulan struktural GNN belum sepenuhnya maksimal.

### 15.3 Kalibrasi Risk Score

Formula saat ini tetap sederhana dan mudah dioperasikan:

```text
risk_score = round(P(phishing) × 100)
```

Namun, untuk versi tesis yang lebih matang, risk score sebaiknya dipertimbangkan bersama metode kalibrasi probabilitas seperti Platt Scaling atau Isotonic Regression. Ini membuat skor lebih defensible ketika dipakai sebagai masukan smart contract.

### 15.4 Narasi Sidang yang Disarankan

Jika penguji menanyakan mengapa XGBoost masih unggul tipis dari GNN, jawaban yang paling kuat adalah:

- XGBoost bekerja sangat baik pada fitur tabular yang sudah diekstrak.
- GNN-GraphSAGE V2 sudah kompetitif setelah menggunakan edge transaksi nyata dan arsitektur 2-layer.
- Keterbatasan utama GNN bukan pada idenya, tetapi pada sparsity graph: subgraph yang digunakan masih sangat jarang, sehingga sinyal struktural belum optimal.
- Penambahan fitur Etherscan memberi kenaikan yang besar pada semua model, menunjukkan bahwa atribut wallet eksternal memang membawa informasi yang tidak sepenuhnya tercermin dari graph EPTransNet saja.
