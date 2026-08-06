# AI + Blockchain Risk Registry (Ethereum Phishing) — Hasil Implementasi

> Dokumen ini mencatat hasil nyata dari implementasi sistem, berdasarkan blueprint di `code.md`.
> Dibuat: 3 Maret 2026
> Update terakhir: 13 Juli 2026 (§17 pendalaman audit 18 catatan: merge audit, statistik deskriptif, audit fitur, missingness pra-zero-fill, salah input, SMOTE empiris, PR-AUC, tuning GNN, output biner utama)
>
> **Koreksi 6 Agustus 2026:** skenario `muldigraph_plus_etherscan` sebelumnya disebut "26 fitur"
> karena `graph_present` (flag audit cakupan join MulDiGraph, bukan perilaku wallet) ikut
> dihitung sebagai predictor. Skema fitur & kode (`build_hybrid_dataset.py`,
> `train_production_model.py`, `hybrid_feature_schema.json`, `prod_feature_schema.json`) sudah
> diperbaiki agar skenario ini benar **25 fitur** (18 graf + 7 Etherscan) dan `graph_present`
> hanya muncul di `audit_columns`. Angka F1/precision/recall/AUC di bawah ini masih berasal dari
> model 26-fitur lama — **harus dijalankan ulang** (`python ml/build_hybrid_dataset.py && python
> ml/train_production_model.py && python ml/train_dataset_scenarios.py`) untuk memvalidasi ulang
> angka pada skema 25-fitur yang benar sebelum dikutip di naskah final.

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

## 6) Output Model: Verdict Biner (utama) + Risk Score (pendukung)

> **Catatan #17 (arahan dosen):** output **utama** sistem adalah **klasifikasi biner
> Phishing / Bukan Phishing**. Risk score 0–100 dan level LOW/MEDIUM/HIGH bersifat
> **informasi tambahan** untuk membantu interpretasi, bukan keluaran utama.

### 6.1 Output Utama — Verdict Biner

```
p = predict_proba(phishing) ∈ [0,1]
verdict = "Phishing"        bila p ≥ 0,5
        = "Bukan Phishing"  bila p < 0,5
```

Inilah keputusan akhir model dan yang ditonjolkan di API (`is_phishing`, `verdict`) serta
UI (banner merah/hijau). Sesuai tugas klasifikasi biner (phishing vs bukan phishing) yang
divalidasi lewat ROC-AUC 0,9948 & PR-AUC 0,9701.

### 6.2 Output Pendukung — Risk Score & Level

Untuk membantu pengguna menilai *derajat* risiko (bukan sekadar ya/tidak), probabilitas
dipetakan menjadi skor dan level:

```
risk_score = round(p × 100)   ∈ [0,100]
```

| Range | Label | Keterangan |
|---|---|---|
| 0 – 39 | 🟢 LOW | Kemungkinan phishing rendah |
| 40 – 69 | 🟡 MEDIUM | Perlu perhatian |
| 70 – 100 | 🔴 HIGH | Kemungkinan phishing tinggi |

Skor & level bersifat **sekunder**: keputusan biner tetap ditentukan ambang p ≥ 0,5.

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

---

## 16) Tindak Lanjut Bimbingan — Data-Centric, Tuning, ROC/AUC, Penjelasan Skor & Deployment (Update: 6 Juli 2026)

Bagian ini merekap seluruh tindak lanjut catatan bimbingan (18 poin), dari pembersihan
data sampai deployment, sebagai satu kesatuan. Rencana kerja bertahap ada di
`RENCANA_BIMBINGAN.md`; seluruh angka dapat direproduksi lewat script di `ml/`
(lihat §16.8). Beberapa item yang di §15 masih berupa *saran* kini sudah **dikerjakan**:
data cleaning (§16.1), EDA/outlier (§16.2), seleksi fitur (§16.3), tuning + threshold
(§16.4), ROC/AUC (§16.5), dan SHAP (§16.6).

### 16.1 Data Quality & Cleaning (poin 1, 6, 7, 11)

**Script:** `ml/clean_hybrid_dataset.py` → `data/processed/hybrid_dataset_clean.csv`,
`hybrid_dataset_capped.csv`, `ml/artifacts/cleaning_report.json`.

| Langkah | Aksi | Hasil |
|---|---|---|
| Deduplikasi wallet | Cek `node_id`/`address` (lowercased) | **0 dihapus** (sudah unik) |
| Buang fitur mati | Hapus `created_at_ts` (konstan 0) | **26 → 25 fitur** inti |
| Cek NaN | Verifikasi missing value | **0 NaN** → tanpa imputasi |
| Cek salah input | Cari nilai negatif tak logis | **0 negatif** |
| Baris low-activity | Tinjau baris hampir kosong | **3 baris** dipertahankan |
| Outlier | Winsorize (capping) p1–p99 | **2.127 nilai** di-cap |

Total baris tetap **8.595** (tidak ada yang dibuang — pendekatan **data-centric**: data
dirapikan, bukan dibuang). Distribusi label tetap 7.430 normal / 1.165 phishing (13,55%).
Keputusan paper: (1) tidak ada duplikat; (2) `created_at_ts` dibuang karena nol informasi;
(3) nilai 0 bersifat **struktural** (bukan missing) → dipertahankan; (4) outlier banyak
(~15–24% per fitur, IQR) → **di-capping** bukan dibuang; (5) imbalance ditangani di modeling.
**Anti-leakage:** capping/scaling di-fit pada TRAIN saja; threshold p1/p99 disimpan di
`cleaning_report.json`.

### 16.2 EDA, Outlier & Statistik Deskriptif (poin 4, 8, data-centric)

**Script:** `ml/eda_visualize.py` → `ml/artifacts/eda/01..05_*.png`.

| Figur | Isi | Menjawab |
|---|---|---|
| `01_class_distribution.png` | Distribusi kelas (imbalance 1 : 6,4) | poin 8 |
| `02_outlier_percentage.png` | Proporsi outlier (IQR) per fitur | poin 4 |
| `03_boxplot_before_after.png` | Boxplot 8 fitur: asli vs setelah capping | poin 4 |
| `04_hist_by_class.png` | Distribusi fitur per kelas (skala log) | poin 5, 8 |
| `05_correlation_heatmap.png` | Korelasi antar fitur (Pearson) | poin 3, 5 |

Temuan: sebaran fitur sangat **skewed** (ekor kanan panjang); setelah capping p1–p99 ekor
ekstrem hilang tapi median/bentuk inti tetap. Beberapa fitur (mis. `in_degree` vs
`unique_senders`) berkorelasi tinggi → ada redundansi (ditindaklanjuti di §16.3).

### 16.3 Seleksi Fitur (poin 3, 5, 9, 10)

**Script:** `ml/feature_selection.py` → `ml/artifacts/feature_selection_report.json`,
`eda/06_feature_importance.png`. Metode: importance gabungan **XGBoost (gain) +
RandomForest** (ternormalisasi) + pemangkasan pasangan berkorelasi |r| ≥ 0,95;
`scale_pos_weight = 6,38`, capping di-fit pada train saja.

Peringkat teratas: `unique_senders` ≫ `in_degree` > `timestamp_spread_in` >
`sent_received_ratio` > `total_sent` … Fitur nyaris tak berguna: **`graph_present`**
(importance ~0). Redundan (dibuang): **`in_degree`**, **`tx_frequency`**.

| Skenario | Jumlah fitur | F1 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|
| full_25 | 25 | 0,8921 | 0,8635 | 0,9227 |
| corr_pruned | 23 | 0,8880 | 0,8594 | 0,9185 |
| **top_20** | **20** | **0,8953** | 0,8583 | 0,9356 |
| top_18 | 18 | 0,8948 | 0,8611 | 0,9313 |
| top_15 | 15 | 0,8957 | 0,8555 | 0,9399 |

**Kesimpulan:** memangkas fitur **tidak menurunkan** kinerja — `top_20` (0,8953) & `top_15`
(0,8957) sedikit **lebih baik** dari 25 fitur penuh (0,8921), membuktikan arahan dosen bahwa
banyak fitur bisa dipangkas. Rekomendasi: **`top_20`**. (F1 ~0,89 di sini di bawah produksi
karena `scale_pos_weight` menaikkan recall dengan sedikit korban precision — perbandingan
resmi strategi imbalance ada di §16.4.)

### 16.4 Hyperparameter Tuning & Penanganan Imbalance (poin 2, 8, 12)

**Script:** `ml/tune_xgboost.py` → `ml/artifacts/xgb_tuning_results.json`.
`RandomizedSearchCV` (60 iterasi × StratifiedKFold 5-fold, `scoring="f1"`) pada 20 fitur
hasil seleksi, capping p1–p99 di-fit pada train saja.

| Konfigurasi | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (parameter lama, thr 0.5) | 0,9732 | 0,8725 | 0,9399 | **0,9050** |
| Tuned (thr 0.5) | 0,9732 | 0,8610 | 0,9571 | **0,9065** |
| Tuned (thr 0.741, optimasi F1) | 0,9721 | 0,8807 | 0,9185 | 0,8992 |
| CV F1 terbaik (5-fold) | — | — | — | 0,9295 |

Perbandingan strategi imbalance (poin dosen: kenapa undersampling? / XGBoost harusnya bisa
tanpa undersampling):

| Strategi | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Undersampling 1:5 | 0,9715 | 0,8770 | 0,9185 | 0,8973 |
| **`scale_pos_weight` (tanpa buang data)** | 0,9732 | 0,8610 | 0,9571 | **0,9065** |
| `sample_weight` per-baris | 0,9727 | 0,8577 | 0,9571 | 0,9047 |

**Kesimpulan:** `scale_pos_weight` menang → **XGBoost menangani imbalance tanpa membuang
satu baris pun**; undersampling tidak diperlukan. Tuning menaikkan F1 tipis (0,9050 →
0,9065); optimasi threshold tidak membantu, jadi **ambang default 0,5 dipertahankan**.

#### Uji empiris SMOTE / oversampling (catatan #8)

**Script:** `ml/smote_experiment.py` → `ml/artifacts/smote_experiment.json`. Menjawab arahan
dosen agar SMOTE/oversampling **diuji empiris**, bukan hanya alasan konseptual. Resampling
diterapkan **hanya pada data train** (anti-leakage; test dibiarkan asli), model resampling
memakai `scale_pos_weight=1` agar tidak dobel-koreksi. Semua memakai `best_params` yang sama.

| Strategi | Precision | Recall | F1 | PR-AUC | Ukuran train |
|---|---|---|---|---|---|
| `scale_pos_weight` (acuan, tanpa resampling) | 0,8610 | 0,9571 | 0,9065 | 0,9646 | 6.876 (asli) |
| SMOTE | 0,8667 | 0,9485 | 0,9057 | 0,9641 | 11.888 (5944:5944) |
| BorderlineSMOTE | 0,8651 | 0,9356 | 0,8990 | 0,9598 | 11.888 |
| **SMOTETomek** | 0,8615 | 0,9614 | **0,9087** | 0,9632 | 11.306 |
| RandomOverSampler | 0,8643 | 0,9571 | 0,9084 | 0,9655 | 11.888 |

**Kesimpulan #8:** semua metode oversampling menghasilkan F1 **setara** (0,899–0,909) dan
berada dalam rentang variasi acak terhadap `scale_pos_weight` (0,9065); SMOTETomek unggul
sangat tipis (0,9087) namun **tetap di bawah model produksi 26-fitur (0,9102)**. Karena tidak
ada peningkatan berarti dan oversampling **menghasilkan data sintetis** (menggandakan train
hampir 2×), penelitian **tetap memilih `scale_pos_weight`** yang lebih sederhana, tidak
membuang maupun mengarang data. SMOTE terbukti **bukan** solusi untuk menembus plafon ~0,91.

#### Mengapa XGBoost "tidak perlu takut" data tidak seimbang (penjelasan mekanisme)

Pernyataan bahwa XGBoost tidak perlu penanganan imbalance eksternal **bukan** berarti
algoritma ini kebal secara otomatis, melainkan karena XGBoost menangani ketidakseimbangan
**dari dalam fungsi loss-nya**, tanpa mengubah komposisi data. Mekanismenya:

XGBoost dilatih dengan meminimalkan *loss* (kesalahan prediksi). Secara default setiap baris
berbobot sama, sehingga pada data timpang (**7.430 normal : 1.165 phishing ≈ 1 : 6,38**)
model cenderung "bermain aman" dengan memprediksi mayoritas (normal) — menebak normal terus
saja sudah benar ±86% waktu, tetapi recall phishing menjadi rendah. Parameter
**`scale_pos_weight`** mengoreksi ini dengan **memperbesar bobot kesalahan pada kelas
positif (phishing)** di dalam gradien loss, sehingga kesalahan mendeteksi phishing "dihukum"
jauh lebih berat daripada kesalahan pada kelas normal. Nilainya dihitung langsung dari rasio
kelas:

```text
scale_pos_weight = jumlah_negatif / jumlah_positif = 7.430 / 1.165 ≈ 6,38
```

Angka **6,38** inilah yang dipakai di seluruh eksperimen (lihat `cleaning_report.json`,
`feature_selection_report.json`, `xgb_tuning_results.json`).

**Perbandingan tiga pendekatan imbalance:**

| Pendekatan | Cara kerja | Konsekuensi | Analogi |
|---|---|---|---|
| Undersampling | Buang sebagian baris kelas mayoritas | **Informasi hilang**, dataset mengecil | Murid mayoritas disuruh keluar kelas agar jumlah seimbang |
| Oversampling (SMOTE) | Duplikasi/sintesis baris minoritas | Data tiruan, risiko *overfitting* | Murid minoritas difotokopi jadi banyak |
| **`scale_pos_weight`** | Bobot error minoritas dinaikkan di loss | **Tidak buang & tidak mengarang data** | Semua murid tetap, nilai murid minoritas dikali 6,38 |

Terbukti empiris pada dataset ini `scale_pos_weight` (F1 **0,9065**) mengungguli undersampling
1:5 (0,8973) dan `sample_weight` (0,9047) — tabel di atas. Karena tidak ada baris yang dibuang,
estimasi performa lebih **stabil dan jujur** terhadap distribusi data yang tersedia.

**Trade-off yang perlu disebut saat sidang:** `scale_pos_weight` menaikkan **recall** (lebih
banyak phishing tertangkap) dengan sedikit korban **precision** (false alarm naik tipis). Untuk
konteks keamanan/phishing, arah trade-off ini **tepat** — membiarkan phishing lolos (false
negative) jauh lebih berbahaya daripada satu peringatan palsu (false positive).

> ✍️ **[UNTUK DITULIS DI TESIS]**
> **BAB 3 (Metodologi) — sub-bab Penanganan Data Tidak Seimbang:** tuliskan bahwa penelitian
> ini menangani imbalance kelas (1 : 6,38) menggunakan `scale_pos_weight = jumlah_negatif /
> jumlah_positif ≈ 6,38` pada XGBoost, **bukan** undersampling/oversampling, dengan alasan
> tidak membuang informasi dan tidak menghasilkan data sintetis.
> **BAB 4 (Hasil/Pembahasan):** sertakan tabel perbandingan tiga strategi imbalance
> (undersampling 0,8973 vs `scale_pos_weight` 0,9065 vs `sample_weight` 0,9047) sebagai bukti
> empiris bahwa penanganan internal XGBoost lebih unggul; tegaskan trade-off recall–precision
> yang sesuai untuk kasus keamanan.

### 16.5 ROC / AUC & Perbandingan XGBoost vs GNN (poin 13, 14)

**Script:** `ml/evaluate_roc.py` → `roc_xgb.png`, `roc_comparison.png`,
`pr_curve_comparison.png`, `roc_auc_report.json`. Skenario MulDiGraph + Etherscan, split
identik (1.719 baris uji, 233 phishing, base rate positif 0,1355).

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** | **PR-AUC** |
|---|---|---|---|---|---|---|
| **XGBoost** | 0,9750 | 0,8862 | 0,9356 | **0,9102** | **0,9948** | **0,9701** |
| GNN-GraphSAGE | 0,9663 | 0,8159 | 0,9700 | 0,8863 | 0,9880 | 0,9007 |

Selain **kurva ROC asli (FPR–TPR)** di `roc_xgb.png`/`roc_comparison.png`, kini tersedia pula
**kurva Precision-Recall** (`pr_curve_comparison.png`) beserta **PR-AUC** (*average precision*).
PR-AUC penting untuk data **imbalance**: baseline acak hanya 0,1355 (proporsi kelas phishing),
sementara XGBoost mencapai **PR-AUC 0,9701** — jauh di atas baseline, menegaskan model benar-benar
membedakan phishing, bukan sekadar unggul karena kelas mayoritas. XGBoost unggul pada ROC-AUC
maupun PR-AUC.

ROC/AUC **valid untuk klasifikasi biner** (phishing vs bukan phishing). AUC ~0,99 pada
kedua model berarti **kemampuan diskriminasi sangat baik** — plafon F1 ~0,91 murni karena
trade-off precision–recall di kelas minoritas pada ambang 0,5, bukan karena model lemah.
XGBoost unggul pada F1 dan AUC → dipilih sebagai model produksi.

> **Catatan non-determinisme GNN:** angka XGBoost bersifat **deterministik** (seed tetap,
> bit-per-bit sama tiap run). Angka **GNN-GraphSAGE sedikit bervariasi antar run** karena
> inisialisasi bobot & dropout PyTorch tidak sepenuhnya deterministik pada CPU — pada
> pengujian terpisah GNN berkisar **F1 ≈ 0,886–0,899 dan AUC ≈ 0,988–0,989**. Variasi ini
> **tidak mengubah kesimpulan**: XGBoost tetap unggul F1 & AUC pada setiap run, dan GNN tetap
> di kisaran kompetitif. Angka GNN yang dilaporkan di tabel adalah salah satu run representatif.
> GNN pada tabel ini memakai konfigurasi default; **hasil tuning GNN penuh (36 konfigurasi ×
> 2 seed) ada di §17.7** — konfigurasi terbaik menaikkan F1 GNN ke 0,9093 (nyaris = XGBoost).

### 16.6 Asal Skor Risiko & Penjelasan (Interpretability) (poin 15)

```text
26 fitur wallet ─► XGBoost ─► probabilitas p ∈ [0,1]
   risk_score = round(p × 100) ∈ [0,100]
   risk_level : LOW (<40) · MEDIUM (40–69) · HIGH (≥70)
   verdict    : Phishing bila p ≥ 0,5, selain itu Bukan Phishing
```

Penjelasan per-prediksi memakai **nilai SHAP bawaan XGBoost** (`pred_contribs=True`, tanpa
paket eksternal) di `ml/predict.py:explain()`. API `/analyze` mengembalikan
`top_contributors` (5 fitur paling berpengaruh + arah naik/turun risiko) dan UI Streamlit
menampilkannya sebagai tabel "Mengapa skor ini?". Skor tidak lagi *black box*.

### 16.7 Output Biner & Deployment (poin 16, 17, 18)

- **Output biner sebagai keluaran UTAMA (catatan #17):** API mengembalikan `is_phishing`
  (bool) dan `verdict` ("Phishing"/"Bukan Phishing") sebagai hasil utama; UI menampilkan
  banner merah/hijau. Risk score 0–100 & level LOW/MEDIUM/HIGH diposisikan sebagai
  **informasi tambahan** (lihat §6). File: `api/schemas.py`, `api/main.py`,
  `app/streamlit_app.py`.
- **Model terdeploy:** XGBoost 26 fitur (**F1 0,9102, AUC 0,9948**) dipertahankan sebagai
  model produksi karena mengungguli hasil tuning 20-fitur (F1 0,9065).
- **Target F1:** 95% **tidak tercapai** pada data ini (plafon realistis F1 ≈ 0,91 / AUC ≈
  0,99), namun **melewati batas minimum 85%** — sesuai fallback yang disebut dosen.

> ✍️ **[UNTUK DITULIS DI TESIS]** Pada BAB 4 (dan abstrak), rumuskan output sistem sebagai
> **klasifikasi biner Phishing / Bukan Phishing** lebih dulu; baru sebutkan risk score 0–100
> sebagai turunan probabilitas untuk membantu interpretasi derajat risiko. Hindari menjadikan
> LOW/MEDIUM/HIGH sebagai output utama.

### 16.8 Reproduksi

```powershell
python ml\clean_hybrid_dataset.py   # §16.1 cleaning
python ml\eda_visualize.py          # §16.2 EDA/outlier -> eda/01..05_*.png
python ml\feature_selection.py      # §16.3 seleksi fitur -> feature_selection_report.json
python ml\tune_xgboost.py           # §16.4 tuning + imbalance -> xgb_tuning_results.json
python ml\evaluate_roc.py           # §16.5 ROC/AUC -> roc_xgb.png, roc_comparison.png
python ml\predict.py                # §16.6 smoke test skor + verdict biner + SHAP
```

### 16.9 Verifikasi Reproduksi (run ulang 11 Juli 2026)

Seluruh script §16 dijalankan ulang dari awal untuk memastikan angka yang dilaporkan
benar-benar reproducible (bukan sekadar tersalin dari run lama). Hasilnya cocok:

| Script | Angka dilaporkan | Hasil run ulang | Status |
|---|---|---|---|
| `clean_hybrid_dataset.py` | 0 duplikat · 2.127 nilai di-clip | 0 duplikat · 2.127 di-clip | ✅ identik |
| `feature_selection.py` | top_20 F1 0,8953 · top_15 F1 0,8957 | 0,8953 · 0,8957 | ✅ identik |
| `tune_xgboost.py` | tuned F1 0,9065 · CV F1 0,9295 | 0,9065 · 0,9295 | ✅ identik |
| `evaluate_roc.py` — **XGBoost** | F1 0,9102 · AUC 0,9948 | 0,9102 · 0,9948 | ✅ identik |
| `evaluate_roc.py` — GNN | F1 0,8863 · AUC 0,9880 | F1 0,8986 · AUC 0,9894 | ⚠️ beda tipis (lihat catatan §16.5) |

**Kesimpulan verifikasi:** pipeline data-centric, seleksi fitur, tuning, dan ROC/AUC
**reproducible penuh**; model produksi XGBoost identik bit-per-bit. Satu-satunya variasi
ada pada GNN (non-deterministik PyTorch), dan pada run ini GNN justru **sedikit lebih baik**
(F1 0,8986 > 0,8863) — jadi tidak ada penurunan kinerja. Semua target bimbingan tetap
terpenuhi: F1 produksi 0,9102 (**melewati minimum 85%**, di bawah target ideal 95%), AUC
0,9948, output biner Phishing/Bukan Phishing, dan skor dijelaskan via SHAP.

### 16.10 Dampak Tiap Tindakan terhadap F1 — Awal vs Sesudah (analisis jujur)

Bagian ini menjawab pertanyaan langsung: *"setelah data dibersihkan, outlier di-capping, fitur
mati dibuang, dan seleksi fitur/tuning dilakukan — apakah F1 naik?"* Jawaban jujurnya:
**F1 absolut tidak naik signifikan; model terbaik tetap model produksi awal (F1 0,9102).**
Nilai acuan awal = XGBoost 26 fitur, MulDiGraph + Etherscan, sebelum tindak lanjut bimbingan.

| Tindakan | F1 awal | F1 sesudah | Selisih | Keterangan |
|---|:---:|:---:|:---:|---|
| Buang fitur mati/konstan (`created_at_ts`) | 0,9102 | 0,9102 | **0,0000** | Varians nol → secara matematis tidak menyumbang; metrik tak berubah |
| Capping outlier (winsorize p1–p99) | 0,9102 | ~0,9065 | **−0,0037** | Sedikit turun — nilai ekstrem ternyata *sinyal* phishing, bukan noise |
| Seleksi fitur (25 → 20 fitur) | 0,8921 | 0,8953 | **+0,0032** | Naik tipis (dalam eksperimen ber-`scale_pos_weight`), tetap di bawah produksi |
| Tuning hyperparameter (RandomizedSearchCV) | 0,9050 | 0,9065 | **+0,0015** | Naik tipis; optimasi threshold tidak membantu |
| `scale_pos_weight` vs undersampling 1:5 | 0,8973 | 0,9065 | **+0,0092** | `scale_pos_weight` menang tanpa membuang data |
| **Model produksi final (yang di-deploy)** | 0,9102 | **0,9102** | **0,0000** | Model 26-fitur dipertahankan karena mengungguli hasil tuning 20-fitur (0,9065) |

**Mengapa capping outlier tidak menaikkan (malah sedikit menurunkan) F1?** Pada deteksi
phishing, **nilai ekstrem justru membawa sinyal** — wallet phishing kerap menunjukkan perilaku
tidak wajar (lonjakan saldo, ledakan jumlah transaksi dalam waktu singkat). Ketika outlier
di-*capping*, sebagian daya pembeda kelas phishing ikut terpangkas, sehingga F1 turun tipis.
Temuan ini konsisten dengan AUC yang sangat tinggi (0,9948): model sudah mendekati **plafon
kemampuan diskriminasi data**, sehingga intervensi data-centric tidak lagi mampu menaikkan F1
secara berarti.

**Makna hasil ini (untuk sidang):** tujuan pekerjaan data-centric **bukan** sekadar menaikkan
F1, melainkan **memastikan kualitas, validitas, dan reproducibility** metodologi — menjawab
tuntutan bimbingan soal duplikat, missing value, outlier, dan strategi imbalance. Hasilnya
membuktikan tiga hal secara empiris: (1) fitur dapat dipangkas dari 25 ke 20 **tanpa merugikan**
kinerja; (2) `scale_pos_weight` **lebih baik** daripada undersampling; dan (3) tidak ada
konfigurasi yang mampu menembus target 0,95 pada data ini — **plafon realistis F1 ≈ 0,91 /
AUC ≈ 0,99**. Karena capping tidak menguntungkan, **model produksi final mempertahankan data
tanpa capping** dengan 26 fitur (F1 0,9102) sebagai konfigurasi terbaik.

> ✍️ **[UNTUK DITULIS DI TESIS — BAB 4/BAB 5]** Sampaikan sebagai *honest finding*: pendekatan
> data-centric meningkatkan **kualitas metodologi**, bukan skor absolut, karena model telah
> mendekati plafon kemampuan data (AUC 0,9948). Sertakan tabel awal-vs-sesudah di atas sebagai
> bukti transparansi, dan jelaskan capping outlier sebagai keputusan yang **diuji lalu tidak
> diadopsi** pada model final karena nilai ekstrem bersifat diskriminatif untuk phishing.

---

## 17) Pendalaman Audit Data — Tabel Lengkap (Tindak Lanjut Audit 18 Catatan, Update: 13 Juli 2026)

Bagian ini menutup item yang pada audit masih berstatus **parsial** karena angkanya sudah
ada tetapi **tabel rincinya** belum ditampilkan. Semua angka dihasilkan oleh
`ml/audit_gaps.py` → `ml/artifacts/audit_report.json` (reproducible).

### 17.1 Audit Penggabungan Sumber Data (catatan #1)

Penggabungan dilakukan **by-address** (di-*lowercase*) antar tiga sumber. Tabel matched/unmatched:

| Sumber | Matched | Coverage | Keterangan |
|---|---:|---:|---|
| MulDiGraph (fitur graf) | 8.595 / 8.595 | 100% | Semua alamat punya jejak di graph (`graph_present`=1) |
| Etherscan (fitur atribut) | 12.812 / 12.815 | 99,98% | Hanya **3 alamat gagal fetch** dari set enrichment 1:10 |
| BigQuery (fitur `bq_*`) | 80 / 8.595 | 0,93% | Coverage sangat rendah → **tidak dipakai di produksi** |

Dataset final = **8.595 alamat unik** (1.165 phishing, 7.430 normal), **0 baris ganda** hasil
merge. Penggabungan tidak menciptakan duplikat karena kunci join adalah alamat unik.

### 17.2 Statistik Deskriptif Lengkap per Fitur (catatan #4)

Metode outlier: IQR (di luar Q1−1,5·IQR … Q3+1,5·IQR). Tabel lengkap min/Q1/median/mean/Q3/max/std + jumlah outlier:

| Fitur | Min | Q1 | Median | Mean | Q3 | Max | Std | #Outlier | % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| min_out_amount | 0 | 0 | 0 | 1,86 | 0,005 | 1.995 | 26,1 | 2.046 | 23,8 |
| timestamp_spread_out | 0 | 0 | 0 | 1.828.255 | 144.854 | 66.627.072 | 5.408.744 | 1.964 | 22,9 |
| total_sent | 0 | 0 | 0 | 17,45 | 0,39 | 24.986 | 375,8 | 1.837 | 21,4 |
| max_out_amount | 0 | 0 | 0 | 9,81 | 0,30 | 24.985 | 281,6 | 1.809 | 21,0 |
| avg_out_amount | 0 | 0 | 0 | 2,78 | 0,18 | 1.995 | 29,7 | 1.795 | 20,9 |
| timestamp_spread_in | 0 | 0 | 0 | 709.128 | 0 | 57.492.528 | 3.376.422 | 1.763 | 20,5 |
| min_in_amount | 0 | 0 | 0 | 3,12 | 0,061 | 6.247 | 88,8 | 1.747 | 20,3 |
| balance | 0 | 0 | 0 | 0,52 | 0,001 | 2.576 | 28,3 | 1.738 | 20,2 |
| total_received | 0 | 0 | 0 | 22,23 | 0,72 | 27.069 | 437,0 | 1.713 | 19,9 |
| std_in_amount | 0 | 0 | 0 | 1,67 | 0 | 3.906 | 44,3 | 1.704 | 19,8 |
| max_in_amount | 0 | 0 | 0 | 10,20 | 0,57 | 13.000 | 208,1 | 1.590 | 18,5 |
| sent_received_ratio | 0 | 0 | 0 | 2,69e9 | 1,0 | 4,26e12 | 5,66e10 | 1.579 | 18,4 |
| std_out_amount | 0 | 0 | 0 | 1,67 | 0 | 3.498 | 41,9 | 1.474 | 17,1 |
| avg_in_amount | 0 | 0 | 0 | 4,29 | 0,42 | 6.247 | 92,8 | 1.359 | 15,8 |
| in_degree | 0 | 0 | 0 | 7,93 | 1,0 | 11.066 | 176,7 | 1.288 | 15,0 |
| activity_score | 0 | 3 | 8 | 22,49 | 26 | 100 | 30,0 | 1.137 | 13,2 |
| tx_frequency | 0 | 0,1 | 0,27 | 0,75 | 0,87 | 3,33 | 1,0 | 1.137 | 13,2 |
| transaction_count | 0 | 3 | 8 | 51,02 | 26 | 10.000 | 309,8 | 1.132 | 13,2 |
| avg_tx_value | 0 | 0,003 | 0,22 | 6,20 | 1,34 | 4.000 | 73,9 | 1.087 | 12,6 |
| unique_receivers | 0 | 0 | 1 | 2,65 | 1,0 | 573 | 13,7 | 1.085 | 12,6 |
| unique_senders | 0 | 0 | 0 | 5,12 | 1,0 | 8.371 | 110,4 | 980 | 11,4 |
| out_degree | 0 | 0 | 1 | 5,40 | 2,0 | 2.574 | 44,1 | 903 | 10,5 |
| unique_counterparties | 0 | 3 | 6 | 12,53 | 14 | 101 | 15,5 | 897 | 10,4 |
| in_out_ratio | 0 | 0 | 0 | 3,35 | 1,0 | 11.066 | 127,8 | 660 | 7,7 |
| graph_present | 1 | 1 | 1 | 1,0 | 1 | 1 | 0 | 0 | 0,0 |

Terlihat jelas kasus yang dosen sebut ("balance mayoritas kecil, ada 1 yang ekstrem"):
mean ≫ median pada hampir semua fitur → sebaran **skewed kanan** → ditangani capping p1–p99.

### 17.3 Tabel Audit 25 Fitur Inti (catatan #5)

Coverage = % baris bernilai non-nol; importance dari `feature_selection_report.json`;
`|corr|` maks = korelasi absolut tertinggi dengan fitur lain; keputusan = hasil seleksi fitur.

| Fitur | Sumber | Coverage % | Imp. XGB | Imp. RF | \|corr\| maks | Keputusan |
|---|---|---:|---:|---:|---:|---|
| unique_senders | MulDiGraph | 45,8 | 0,396 | 0,181 | 0,93 | ✅ dipakai (top-20) |
| in_degree | MulDiGraph | 45,8 | 0,136 | 0,100 | 0,93 | ✅ dipakai (top-20) |
| timestamp_spread_in | MulDiGraph | 20,5 | 0,080 | 0,084 | 0,41 | ✅ dipakai (top-20) |
| sent_received_ratio | MulDiGraph | 34,2 | 0,053 | 0,065 | 0,35 | ✅ dipakai (top-20) |
| total_sent | MulDiGraph | 34,2 | 0,059 | 0,053 | 0,82 | ✅ dipakai (top-20) |
| max_out_amount | MulDiGraph | 34,2 | 0,072 | 0,032 | 0,97 | ✅ dipakai (top-20) |
| total_received | MulDiGraph | 44,4 | 0,011 | 0,086 | 0,89 | ✅ dipakai (top-20) |
| max_in_amount | MulDiGraph | 44,4 | 0,011 | 0,071 | 0,89 | ✅ dipakai (top-20) |
| std_in_amount | MulDiGraph | 19,8 | 0,005 | 0,069 | 0,74 | ✅ dipakai (top-20) |
| avg_out_amount | MulDiGraph | 34,2 | 0,015 | 0,045 | 0,89 | ✅ dipakai (top-20) |
| avg_in_amount | MulDiGraph | 44,4 | 0,016 | 0,042 | 0,96 | ✅ dipakai (top-20) |
| in_out_ratio | MulDiGraph | 45,8 | 0,019 | 0,033 | 0,81 | ✅ dipakai (top-20) |
| transaction_count | Etherscan | 99,3 | 0,018 | 0,015 | 0,55 | ✅ dipakai (top-20) |
| out_degree | MulDiGraph | 74,3 | 0,012 | 0,016 | 0,54 | ✅ dipakai (top-20) |
| unique_receivers | MulDiGraph | 74,3 | 0,015 | 0,013 | 0,54 | ✅ dipakai (top-20) |
| min_in_amount | MulDiGraph | 42,6 | 0,008 | 0,019 | 0,96 | ✅ dipakai (top-20) |
| unique_counterparties | Etherscan | 99,3 | 0,015 | 0,010 | 0,78 | ✅ dipakai (top-20) |
| activity_score | Etherscan | 99,3 | 0,010 | 0,013 | 1,00 | ✅ dipakai (top-20) |
| tx_frequency | Etherscan | 99,3 | 0,011 | 0,012 | 1,00 | ✅ dipakai (top-20) |
| avg_tx_value | Etherscan | 98,1 | 0,009 | 0,011 | 0,27 | ✅ dipakai (top-20) |
| min_out_amount | MulDiGraph | 26,8 | 0,005 | 0,014 | 0,89 | ⏸ cadangan (di luar top-20) |
| timestamp_spread_out | MulDiGraph | 32,8 | 0,008 | 0,006 | 0,44 | ⏸ cadangan (di luar top-20) |
| balance | Etherscan | 55,6 | 0,007 | 0,007 | 0,04 | ⏸ cadangan (di luar top-20) |
| std_out_amount | MulDiGraph | 17,1 | 0,009 | 0,003 | 0,97 | ⏸ cadangan (di luar top-20) |
| graph_present | MulDiGraph | 100,0 | 0,000 | 0,000 | 0,00 | ❌ dibuang (importance ~0) |

Catatan: `activity_score` & `tx_frequency` punya `|corr|`=1,00 (saling redundan) — salah satu
dapat dipangkas; `graph_present` konstan-1 sehingga importance 0 dan dibuang.

### 17.4 Missingness Sebelum Zero-Filling (catatan #6, #11)

**Penting (menjawab dosen):** dataset final memang 0 NaN, **tetapi** itu karena nilai kosong
dari API sudah diisi nol saat `build_hybrid_dataset.py`. Tabel berikut menampilkan kondisi
**SEBELUM** zero-fill:

| Kelompok kolom | Missing sebelum fill | % | Kebijakan | Alasan |
|---|---:|---:|---|---|
| 6 fitur Etherscan (balance, transaction_count, dst) | **3** | 0,02% | Imputasi nol | Hanya 3 alamat gagal fetch → dampak dapat diabaikan |
| 10 fitur BigQuery (`bq_*`) | **8.515** | 99,07% | Imputasi nol | Coverage sangat rendah → fitur praktis konstan-nol, **tidak dipakai produksi** |
| `created_at_ts` | 8.595 (konstan 0) | 100% | **Dibuang** | Timestamp gagal terisi → varians nol, tidak informatif |

**Kebijakan untuk paper:** (1) fitur Etherscan yang hanya 3 baris kosong → imputasi nol
(dampak minor); (2) fitur BigQuery kosong ~99% → dipertahankan hanya untuk pelaporan
eksperimen, **tidak untuk model produksi**; (3) `created_at_ts` konstan → dibuang. Tidak ada
kolom "hanya terisi 4 dari 25" pada fitur inti — seluruh 25 fitur inti terisi (nol yang ada
bersifat **struktural**, bukan missing).

### 17.5 Deteksi Salah Input Rinci (catatan #7)

Pemeriksaan menyeluruh terhadap tipe kesalahan input yang mungkin:

| Jenis cek | Hasil | Status |
|---|---|---|
| Nilai negatif (balance, count, dll) | 0 | ✅ bersih |
| Nilai tak hingga (`inf`) | 0 | ✅ bersih |
| Nilai NaN | 0 | ✅ bersih |
| Alamat duplikat | 0 | ✅ bersih |
| Timestamp tidak valid (`created_at_ts`) | 8.595 baris = 0 | ❌ dibuang (fitur mati) |
| Rasio ekstrem akibat pembagi ≈ 0 (`sent_received_ratio`) | max 4,26×10¹², **1.455 baris** > 10⁶ | ⚠️ di-capping p1–p99 |
| Rasio ekstrem (`in_out_ratio`) | max 11.066 | ⚠️ di-capping p1–p99 |

**Kesimpulan #7:** tidak ada salah input klasik (negatif/inf/NaN/duplikat). Anomali yang ada
bersifat **struktural**: (a) `created_at_ts` gagal terisi → dibuang; (b) rasio meledak karena
pembagi mendekati nol (wallet nyaris tanpa transaksi masuk) → **ditangani capping**, bukan
dibuang, agar barisnya tetap dipakai.

### 17.7 Tuning Hyperparameter GNN-GraphSAGE (catatan #13)

**Script:** `ml/tune_gnn.py` → `ml/artifacts/gnn_tuning_results.json`. Menjawab arahan agar
GNN dituning **setara** XGBoost. Grid: `hidden_dim` ∈ {64, 128, 256}, `num_layers` ∈ {2, 3},
`learning_rate` ∈ {0,005, 0,01, 0,02}, `dropout` ∈ {0,3, 0,5} = **36 konfigurasi**. Karena
GNN non-deterministik, tiap konfigurasi dilatih **2 seed (42, 7)** lalu dirata-rata (total 72
training). Skenario, split, dan edge nyata (654 edge MulDiGraph+Etherscan) identik dengan §16.5.

| Peringkat | hidden | layers | lr | dropout | F1 (mean±std) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---|---:|---:|
| **Terbaik** | 64 | 3 | 0,01 | 0,3 | **0,9093 ± 0,003** | 0,991 | 0,939 |
| 2 | 256 | 3 | 0,01 | 0,3 | 0,9078 ± 0,001 | 0,990 | 0,939 |
| 3 | 256 | 3 | 0,005 | 0,3 | 0,9077 ± 0,002 | 0,992 | 0,936 |
| 4 | 64 | 3 | 0,005 | 0,3 | 0,9063 ± 0,001 | 0,992 | 0,940 |
| … | | | | | | | |
| Baseline (default lama) | 128 | 2 | 0,01 | 0,5 | 0,8938 ± 0,002 | 0,987 | 0,916 |
| Terburuk | 256 | 3 | 0,02 | 0,5 | 0,8732 ± 0,005 | 0,983 | 0,891 |

**Kesimpulan #13:** tuning **berhasil menaikkan F1 GNN dari 0,8938 → 0,9093 (+1,55 poin)**,
sehingga GNN kini **nyaris menyamai XGBoost (0,9102)** — selisih hanya 0,0009, di dalam rentang
variasi antar-seed. Pola yang konsisten: **`dropout` 0,3 lebih baik dari 0,5**, **3 layer sedikit
unggul dari 2 layer**, dan **`lr` 0,02 selalu terburuk** (terlalu besar → tidak stabil). Ukuran
hidden 64 sudah kompetitif dengan 256 (graf sparse tidak butuh kapasitas besar). Meski begitu,
**model produksi tetap XGBoost** karena (a) F1 tetap sedikit lebih tinggi & deterministik, dan
(b) inferensi tabular jauh lebih ringan daripada membangun graf. GNN yang dituning diposisikan
sebagai **pembanding arsitektur yang kuat**, bukan model layanan.

> ✍️ **[UNTUK DITULIS DI TESIS — BAB 4]** Laporkan bahwa GNN telah dituning setara XGBoost
> (grid 36 konfigurasi × 2 seed). Konfigurasi terbaik (hidden=64, 3-layer, lr=0,01, dropout=0,3)
> mencapai F1 0,9093, praktis setara XGBoost. Kesimpulan: keunggulan XGBoost tipis dan konsisten,
> sedangkan GNN kompetitif namun dibatasi sparsity graf — bukan oleh kurang tuning.

### 17.8 Reproduksi

```powershell
python ml\audit_gaps.py         # §17.1–17.5 audit merge/deskriptif/fitur/missingness/salah-input
python ml\smote_experiment.py   # §16.4 uji SMOTE/oversampling -> smote_experiment.json
python ml\tune_gnn.py           # §17.7 tuning GNN (36 konfigurasi x 2 seed) -> gnn_tuning_results.json
python ml\evaluate_roc.py       # §16.5 ROC + PR-AUC -> roc_*.png, pr_curve_comparison.png
```
