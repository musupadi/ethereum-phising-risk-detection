# AI + Blockchain Risk Registry (Ethereum Phishing) — Hasil Implementasi

> Dokumen ini mencatat hasil nyata dari implementasi sistem, berdasarkan blueprint di `code.md`.
> Dibuat: 3 Maret 2026
> Update terakhir: **6 Agustus 2026** — seluruh angka dijalankan ulang pada dataset 1:10 penuh
> (12.815 alamat) dengan skema 25 fitur yang sudah dikoreksi. Lihat catatan revisi di bawah.
>
> ## Catatan revisi 6 Agustus 2026 (dua perubahan, keduanya sudah dieksekusi)
>
> **(1) `graph_present` bukan predictor.** Skenario `muldigraph_plus_etherscan` sebelumnya
> disebut "26 fitur" karena `graph_present` (flag audit cakupan join MulDiGraph, bukan perilaku
> wallet) ikut dihitung sebagai predictor. Kode & skema (`build_hybrid_dataset.py`,
> `train_production_model.py`, `hybrid_feature_schema.json`, `prod_feature_schema.json`) sudah
> diperbaiki: skenario ini kini benar **25 fitur** (18 graf + 7 Etherscan) dan `graph_present`
> hanya muncul di `audit_columns`. Dampaknya **nihil secara empiris** — kolom ini konstan bernilai
> 1 pada seluruh baris, sehingga varians dan importance-nya nol. Uji isolasi pada data yang sama
> memberi F1 0,8911 (25 fitur) vs 0,8966 (26 fitur), selisih yang murni berasal dari
> `colsample_bytree` yang mengacak kolom, bukan dari sinyal apa pun.
>
> **(2) Eksperimen skenario kini memakai dataset 1:10 penuh (12.815 alamat).** Seluruh angka
> §5.5, §12, §16, dan §17 pada versi sebelumnya berasal dari sampel **8.595 alamat**
> (1.165 phishing : 7.430 normal ≈ **1:6,4**), sedangkan §11 sudah menyatakan rasio yang dipilih
> adalah **1:10** (1.165 phishing : 11.650 normal = 12.815). Ini inkonsistensi internal dokumen:
> baseline EPTransNet 1:10 dan eksperimen skenario berjalan di ukuran dataset yang berbeda.
> Pada revisi ini seluruh pipeline dijalankan ulang pada **12.815 alamat**, sehingga semua bab
> kini memakai dataset, split, dan rasio yang sama.
>
> **Konsekuensi angka.** Jumlah phishing tidak berubah (1.165; test 233), yang bertambah hanya
> kelas normal — jadi *precision* turun (lebih banyak kandidat negatif yang bisa salah ditandai)
> sementara *recall* dan *accuracy* naik. F1 model produksi bergeser **0,9102 → 0,8911**.
> Penurunan ini **bukan degradasi model** melainkan efek evaluasi pada distribusi yang lebih
> timpang dan lebih jujur terhadap kondisi nyata — konsisten dengan argumen §11.4 bahwa metrik
> 1:10 memang lebih konservatif daripada 1:5.
>
> **Lima kesimpulan yang berubah** (dicatat eksplisit di tempatnya masing-masing, bukan
> disembunyikan):
>
> | § | Dulu | Sekarang |
> |---|---|---|
> | §5.5 | "XGBoost model terbaik" (unggul ±2,2 poin F1 atas RF) | XGBoost & RF **praktis seri** (89,11% vs 89,21%); XGBoost dipilih karena *recall* lebih tinggi, bukan karena F1 |
> | §16.3 | Rekomendasi seleksi fitur: `top_20` | Rekomendasi: **`corr_pruned`** (22 fitur) |
> | §16.4 | `scale_pos_weight` **menang** atas undersampling | Ketiga strategi **setara dalam derau**; undersampling malah unggul 0,32 poin. `scale_pos_weight` tetap dipakai atas alasan metodologis |
> | §16.10 | "Nilai ekstrem adalah sinyal phishing" (capping menurunkan F1) | Klaim **dicabut** — efek capping ±0,2–0,4 poin dan arahnya berbalik antar-run |
> | §17.7 | "`lr` 0,02 **selalu** terburuk" pada tuning GNN | Klaim **dicabut** — konfigurasi ber-F1 tertinggi justru memakai `lr` 0,02; hanya rata-ratanya yang lebih lemah |
>
> Selebihnya kesimpulan metodologis dokumen ini tidak berubah.

---

## Executive Summary (Ringkas)

Penelitian ini berhasil membangun sistem end-to-end untuk deteksi risiko phishing wallet Ethereum: **ML inference + penyimpanan immutable risk score ke smart contract**.

### Hasil Utama

| Komponen | Hasil | Kesimpulan |
|---|---|---|
| **Model utama eksperimen EPTransNet 1:10** | Random Forest: Accuracy 97,82% · Precision 88,31% · Recall 87,55% · **F1 87,93%** | Baseline kuat; model **terbaik & yang di-deploy** adalah XGBoost (lihat baris skenario) |
| **Baseline** | Logistic Regression: **F1 71,46%** | RF unggul +16,47 poin F1 |
| **Eksperimen skenario dataset gabungan (12.815 alamat, 4 skenario)** | MulDiGraph only → RF: **F1 87,45%**, XGBoost: **87,17%**, GNN-GraphSAGE: **77,78%**. MulDiGraph + Etherscan → RF: **89,21%**, XGBoost: **89,11%**, GNN-GraphSAGE: **85,94%** | Fitur Etherscan meningkatkan semua model; RF & XGBoost praktis seri (selisih 0,10 poin F1), GNN-GraphSAGE naik paling banyak (+8,16 poin) |
| **Blockchain write performance** | Rata-rata latency commit **13,6 ms** (localhost), gas rata-rata **35.462** | Overhead on-chain wajar untuk audit trail |
| **Integrasi data eksternal** | Pipeline Etherscan + Forta sudah tersedia (fitur dan label siap pakai) | Siap dipakai untuk eksperimen lanjutan dengan data real |

### Jawaban Singkat Research Questions

1. **RQ1 terjawab:** Random Forest konsisten mengungguli baseline Logistic Regression pada setup utama.
2. **RQ2 terjawab:** Pencatatan risk score ke smart contract feasible secara waktu dan biaya.
3. **RQ3 terjawab:** Fitur graf transaksi efektif; rasio sampling memengaruhi metrik dan trade-off generalisasi.

### Keputusan Implementasi

- **Model produksi yang di-deploy (API + Streamlit):** XGBoost pada skenario gabungan **MulDiGraph + Etherscan** (25 fitur, **F1 89,11%**, ROC-AUC 0,9938). Untuk wallet yang ada di dataset, fitur Etherscan sudah ter-precompute sehingga inferensi tidak perlu memanggil API eksternal. Artefak: `ml/artifacts/model_prod_xgb.json` (dibuat oleh `ml/train_production_model.py`).
- **Random Forest:** pada dataset 1:10 penuh, RF justru unggul **sangat tipis** atas XGBoost di skenario produksi (F1 89,21% vs 89,11% — selisih 0,10 poin, jauh di dalam rentang variasi acak). **XGBoost tetap dipilih sebagai model produksi** karena *recall*-nya lebih tinggi (94,85% vs 92,27%) — untuk deteksi phishing, phishing yang lolos (false negative) lebih berbahaya daripada satu peringatan palsu. RF tetap dilatih sebagai pembanding dan fallback inferensi. Klaim "XGBoost model terbaik" karenanya diturunkan menjadi "XGBoost dan RF setara; XGBoost dipilih atas dasar profil recall", lihat §5.5.
- **GNN-GraphSAGE:** dengan arsitektur 2-layer + edge transaksi nyata dari MulDiGraph mencapai F1 85,94% pada skenario produksi — di bawah XGBoost/RF tetapi tetap layak sebagai kontribusi ilmiah perbandingan arsitektur, dengan sparsity graf sebagai penjelas utama keterbatasannya (§14.2).

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
│       ├── model_prod_xgb.json         ✅ XGBoost MulDiGraph+Etherscan (DI-DEPLOY, 25 fitur)
│       ├── prod_feature_schema.json    ✅ 25 fitur + audit_columns: ["graph_present"]
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
> **Caption:** Gambar 4.X — Confusion Matrix XGBoost pada Skenario Terbaik (MulDiGraph + Etherscan, n=2.563)
> *(Insert di Word sebagai confusion matrix model produksi — F1-nya seri dengan Random Forest, tetapi FN-nya paling rendah; lihat §5.3 & §5.5)*

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_cm_gnn.png`
> **Caption:** Gambar 4.X — Confusion Matrix GNN-GraphSAGE V2 pada Skenario Terbaik (MulDiGraph + Etherscan, n=2.563)
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
| **XGBoost** (Sc2, n=2563) | 2288 | 42 | **12** | 221 | **5,2%** | 1,8% |
| **GNN-GraphSAGE V2** (Sc2, n=2563) | 2271 | 59 | 13 | 220 | 5,6% | 2,5% |

Seluruh baris kini memakai *test set* yang sama (n=2.563, 233 phishing) — pada versi dokumen
sebelumnya baris Sc2 masih memakai n=1.719 dari sampel 8.595, sehingga tidak sebanding dengan
dua baris baseline di atasnya.

**Interpretasi untuk konteks keamanan (phishing detection):**
- **FN (False Negative) lebih kritis** — phishing yang tidak terdeteksi = wallet korban tidak terlindungi
- **XGBoost memiliki FN terendah (12 node, 5,2%)** → paling aman dari sisi recall, hanya 12 dari 233 phishing lolos, sekaligus FP rate terendah (1,8%)
- **XGBoost paling seimbang** (FN=12, FP=42) → dasar utama pemilihannya sebagai model produksi meski F1-nya praktis seri dengan Random Forest (§5.5)
- **GNN lebih agresif mendeteksi phishing** (FP=59 lebih tinggi dari XGBoost=42) → lebih banyak false alarm tanpa imbalan FN yang lebih rendah
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

Eksperimen ini membandingkan empat kombinasi sumber fitur pada **12.815 alamat wallet** yang sama (1.165 phishing : 11.650 normal, rasio 1:10 sesuai §11), menggunakan split train/test yang identik (80/20, stratified, random seed 42 → 10.252 train / 2.563 test, 233 phishing di test), sehingga selisih performa murni mencerminkan kontribusi tiap kelompok fitur.

**Model yang dibandingkan:** Logistic Regression (baseline), Random Forest, XGBoost, GNN-GraphSAGE V2.

**GNN menggunakan edge transaksi nyata** — 708 edge diekstrak dari MulDiGraph antar alamat sampel ditambah 126 *cross-edge* dari Etherscan (total 834 edge, menjadi 440 edge unik berarah setelah deduplication) — bukan graph kNN sintetis seperti pada versi sebelumnya.

#### Skenario 1 — Hanya fitur graf dari MulDiGraph (18 fitur)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 94,58% | 71,76% | 66,52% | 69,04% |
| **Random Forest** | 97,66% | 85,31% | 89,70% | **87,45%** |
| XGBoost | 97,54% | 82,95% | 91,85% | 87,17% |
| GNN-GraphSAGE V2 | 95,47% | 70,24% | 87,12% | 77,78% |

#### Skenario 2 — Fitur graf + fitur Etherscan (25 fitur) ← **skenario produksi**

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 95,20% | 75,00% | 70,82% | 72,85% |
| **Random Forest** | **97,97%** | **86,35%** | 92,27% | **89,21%** |
| **XGBoost** (di-deploy) | 97,89% | 84,03% | **94,85%** | **89,11%** |
| GNN-GraphSAGE V2 | 97,19% | 78,85% | 94,42% | 85,94% |

#### Skenario 3 — Fitur graf + fitur BigQuery (28 fitur)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 94,42% | 70,27% | 66,95% | 68,57% |
| **Random Forest** | 97,66% | 84,74% | 90,56% | **87,55%** |
| XGBoost | 97,50% | 83,14% | 90,99% | 86,89% |
| GNN-GraphSAGE V2 | 95,32% | 69,02% | 87,98% | 77,36% |

#### Skenario 4 — Fitur graf + Etherscan + BigQuery (35 fitur)

| Model | Accuracy | Precision | Recall | F1-Score |
|---|---|---|---|---|
| Logistic Regression | 95,83% | 77,88% | 75,54% | 76,69% |
| Random Forest | 97,85% | 85,60% | 91,85% | 88,61% |
| **XGBoost** | 97,85% | 83,97% | 94,42% | **88,89%** |
| GNN-GraphSAGE V2 | 97,31% | 79,50% | 94,85% | 86,50% |

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_f1_by_scenario.png`
> **Caption:** Gambar 4.X — Perbandingan F1-Score Empat Model pada Empat Skenario Dataset
> *(Insert gambar di Word setelah keempat tabel skenario di atas — gambar ini merangkum semua tabel dalam satu visualisasi grouped bar chart)*

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_heatmap_f1.png`
> **Caption:** Gambar 4.X — Heatmap F1-Score: Perbandingan Model × Skenario Dataset
> *(Alternatif/pelengkap gambar di atas — heatmap lebih ringkas, cocok untuk versi singkat di BAB 4)*

#### Interpretasi Hasil

1. **Fitur Etherscan secara konsisten meningkatkan performa semua model** (Skenario 1 → 2): GNN naik +8,16 poin F1, LR naik +3,81, XGBoost naik +1,94, RF naik +1,76.

2. **Fitur BigQuery alone (Skenario 3) tidak menambah nilai** dibanding fitur graf saja (Skenario 1). Hal ini konsisten karena fitur BigQuery yang tersedia banyak tumpang-tindih semantik dengan fitur graf (misalnya `bq_tx_count_out` ≈ `out_degree`).

3. **Fitur gabungan Etherscan + BigQuery (Skenario 4) tidak melampaui Etherscan saja (Skenario 2)**, mengindikasikan bahwa BigQuery tidak menambah informasi independen setelah Etherscan tersedia.

4. **Random Forest dan XGBoost praktis seri sebagai model terbaik** pada Skenario 2 (F1 89,21% vs 89,11% — selisih **0,10 poin**, jauh lebih kecil daripada variasi antar-split). Pada dataset 1:6,4 sebelumnya XGBoost unggul ±2,2 poin; keunggulan itu **tidak bertahan** setelah kelas normal diperbanyak ke rasio 1:10. Yang membedakan keduanya bukan F1 melainkan **profil kesalahan**: XGBoost jauh lebih tinggi *recall* (94,85% vs 92,27%, artinya 12 vs 18 phishing lolos dari 233), RF lebih tinggi *precision* (86,35% vs 84,03%). **XGBoost tetap dipilih sebagai model produksi** karena untuk deteksi phishing false negative lebih mahal daripada false positive (§5.3) — ini keputusan berbasis *cost asymmetry*, bukan klaim keunggulan F1.

5. **GNN-GraphSAGE V2 berada jelas di bawah model tabular** pada dataset penuh (85,94% vs ~89,1%, selisih ~3,2 poin). Pada sampel 8.595 sebelumnya GNN sempat nyaris menyamai XGBoost; setelah kelas normal diperbanyak, jumlah edge tidak ikut bertambah sebanding sehingga **subgraph menjadi lebih sparse** (440 edge unik untuk 12.815 node ≈ 0,03 edge/node, turun dari ~0,06) dan sinyal struktural makin encer. Ini justru **memperkuat** argumen §14.2/§15.2: keterbatasan GNN di sini adalah sparsity graf, bukan arsitekturnya. Catatan: baseline lama GNN-GCN 4-layer kNN **tidak dijalankan ulang** pada dataset ini, sehingga klaim selisih terhadap versi tersebut dihapus dari dokumen ini.

6. **GNN mendapat manfaat terbesar dari penambahan fitur Etherscan** (+8,16 poin F1), jauh melampaui XGBoost (+1,94) dan RF (+1,76). Karena struktur graf sangat sparse, GNN paling bergantung pada kualitas fitur node — atribut wallet eksternal menutupi sebagian informasi yang tidak bisa ia peroleh lewat *message passing*.

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_best_scenario_metrics.png`
> **Caption:** Gambar 4.X — Perbandingan Accuracy, Precision, Recall, dan F1-Score pada Skenario Terbaik (MulDiGraph + Etherscan, 25 Fitur)
> *(Insert di Word pada sub-bagian pembahasan skenario terbaik)*

> 📌 **[PLACEHOLDER GAMBAR — BAB 4]**
> **File:** `ml/artifacts/thesis_etherscan_impact.png`
> **Caption:** Gambar 4.X — Dampak Penambahan Fitur Etherscan terhadap F1-Score Semua Model (Skenario 1 → Skenario 2)
> *(Insert di Word pada sub-bagian pembahasan kontribusi fitur Etherscan / jawaban RQ3)*

> **Catatan metodologi:** Semua empat skenario menggunakan baris dan split yang identik. GNN menggunakan 834 edge transaksi nyata (708 dari MulDiGraph + 126 cross-edge Etherscan) untuk 12.815 alamat sampel, menyisakan 440 edge unik di dalam graf latih — rata-rata **~0,03 edge per node**, jauh lebih sparse daripada sampel 8.595 sebelumnya (~0,06). GraphSAGE dengan self-loop dan 2-layer aggregation masih memanfaatkan sinyal struktural yang ada, tetapi pada kepadatan serendah ini keunggulannya atas model tabular tidak muncul.
>
> **Keterbatasan yang perlu disebut:** file `data/raw/etherscan_edges.csv` (126 edge, menyentuh 97 alamat) dibangun ketika populasi masih 8.595 alamat dan **belum di-crawl ulang** untuk 12.815 alamat. Jadi jumlah *cross-edge* Etherscan di sini kemungkinan **undercount** — angka sparsity di atas adalah batas bawah kepadatan yang sebenarnya tersedia. Meng-crawl ulang (`python ml/fetch_etherscan_edges.py`) membutuhkan ~12.815 alamat × 3 endpoint dan kuota API Etherscan yang besar, sehingga ditunda. Ini **tidak membatalkan** kesimpulan bahwa graf terlalu sparse untuk GNN — 708 edge dari MulDiGraph (sumber utama, sudah lengkap untuk semua 12.815 alamat) tetap dominan dan sudah sangat jarang dengan sendirinya.

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
divalidasi lewat ROC-AUC 0,9938 & PR-AUC 0,9582.

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
| **Model produksi (di-deploy)** | **XGBoost (MulDiGraph + Etherscan, 25 fitur), F1 89,11% · ROC-AUC 0,9938** — `model_prod_xgb.json`, model_version `xgb-eth-v1` |
| Metrik RF (baseline eksperimen EPTransNet 1:10) | Accuracy 97,82% · Precision 88,31% · Recall 87,55% · F1 **87,93%** |
| Hasil eksperimen skenario (12.815 alamat, 4 skenario) | RF: **F1 89,21%** · XGBoost: **F1 89,11%** (praktis seri) · GNN-GraphSAGE V2: **F1 85,94%** (semua pada skenario MulDiGraph + Etherscan) |
| Model GNN | GraphSAGE V2, 2-layer, hidden=128, edge transaksi nyata dari MulDiGraph + Etherscan (834 edges, 440 unik) |
| Contract addr | `0x5FbDB2315678afecb367f032d93F642f64180aa3` |
| Contoh tx hash | `c7c8069430427b1e8f482c49b9d26d1b0eb359f5a18f210160b8257d27de099c` |
| Mode aplikasi | **Scan by Address** (tempel wallet dataset), **Demo** (`row_id` dari 2.563 held-out), **Manual** (isi fitur) |
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
├── Load hybrid_dataset.csv  (12.815 alamat, 41 kolom)
├── Load MulDiGraph.pkl       (2,97M node / 13,55M edge — untuk ekstraksi edge GNN)
├── Build 4 scenarios dari hybrid_feature_schema.json:
│   ├── muldigraph_only                    (18 fitur)
│   ├── muldigraph_plus_etherscan          (25 fitur)
│   ├── muldigraph_plus_bigquery           (28 fitur)
│   └── muldigraph_plus_etherscan_plus_bigquery (35 fitur)
│   (graph_present = kolom audit, TIDAK dihitung sebagai predictor)
├── Per skenario, train 4 model:
│   ├── Logistic Regression  (baseline)
│   ├── Random Forest
│   ├── XGBoost
│   └── GNN-GraphSAGE V2  ← edge transaksi nyata dari MulDiGraph
└── Simpan hasil ke ml/artifacts/dataset_scenarios_comparison.csv
```

### Hasil Final Eksperimen Skenario (Dijalankan Ulang 6 Agustus 2026 — 12.815 alamat)

| Skenario | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|
| MulDiGraph only | LR | 94,58% | 71,76% | 66,52% | 69,04% |
| MulDiGraph only | RF | 97,66% | 85,31% | 89,70% | 87,45% |
| MulDiGraph only | XGBoost | 97,54% | 82,95% | 91,85% | 87,17% |
| MulDiGraph only | GNN-GraphSAGE V2 | 95,47% | 70,24% | 87,12% | 77,78% |
| MulDiGraph + Etherscan | LR | 95,20% | 75,00% | 70,82% | 72,85% |
| **MulDiGraph + Etherscan** | **RF** | **97,97%** | **86,35%** | 92,27% | **89,21%** |
| **MulDiGraph + Etherscan** | **XGBoost** (di-deploy) | 97,89% | 84,03% | **94,85%** | **89,11%** |
| MulDiGraph + Etherscan | GNN-GraphSAGE V2 | 97,19% | 78,85% | 94,42% | 85,94% |
| MulDiGraph + BigQuery | LR | 94,42% | 70,27% | 66,95% | 68,57% |
| MulDiGraph + BigQuery | RF | 97,66% | 84,74% | 90,56% | 87,55% |
| MulDiGraph + BigQuery | XGBoost | 97,50% | 83,14% | 90,99% | 86,89% |
| MulDiGraph + BigQuery | GNN-GraphSAGE V2 | 95,32% | 69,02% | 87,98% | 77,36% |
| MulDiGraph + Etherscan + BigQuery | LR | 95,83% | 77,88% | 75,54% | 76,69% |
| MulDiGraph + Etherscan + BigQuery | RF | 97,85% | 85,60% | 91,85% | 88,61% |
| MulDiGraph + Etherscan + BigQuery | XGBoost | 97,85% | 83,97% | 94,42% | 88,89% |
| MulDiGraph + Etherscan + BigQuery | GNN-GraphSAGE V2 | 97,31% | 79,50% | 94,85% | 86,50% |

### Konfigurasi Model Final

**GNN-GraphSAGE V2:**
```
Architecture  : GraphSAGE (SAGEConv), 2 layers, hidden=128, dropout=0.5
Graph edges   : 834 edge transaksi nyata (708 MulDiGraph + 126 cross-edge Etherscan),
                menjadi 440 edge unik di dalam graf latih
Training      : Adam lr=0.01, CrossEntropyLoss dengan class weights
                normal=0.550, phishing=5.500
Early stop    : patience=25, best-weight restoration
Epochs        : maks 200, berhenti di epoch 104–172 (per skenario)
Training loss : konvergen di 0.14–0.19 (CrossEntropyLoss, tergantung skenario fitur)
              Sc1 (MulDiGraph only)          : ~0.183 (stop ep 104)
              Sc2 (+ Etherscan)              : ~0.166 (stop ep 108)
              Sc3 (+ BigQuery)               : ~0.183 (stop ep 110)
              Sc4 (+ Etherscan + BigQuery)   : ~0.144 (stop ep 172)  ← loss terendah
```

**XGBoost:**
```
n_estimators  : 250, max_depth=6, learning_rate=0.05
subsample     : 0.85, colsample_bytree=0.85
scale_pos_weight: 9.99 (auto dari rasio kelas 1:10)
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
├── hybrid_feature_schema.json         ← daftar 35 fitur + 4 skenario + audit_columns
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
| **Subgraph sangat sparse** | Dari 12.815 alamat sampel, hanya 834 edge transaksi langsung ditemukan (708 MulDiGraph + 126 Etherscan), menyisakan 440 edge unik di graf latih — rata-rata **~0,03 edge/node**. GNN beroperasi hampir tanpa koneksi semantik antar node; sparsity ini **memburuk** saat dataset diperbesar ke 1:10 (dari ~0,06 edge/node pada sampel 8.595) dan tercermin langsung pada turunnya F1 GNN relatif terhadap model tabular. |

### 14.3 Construct Validity

| Ancaman | Penjelasan |
|---|---|
| **Demo mode (row_id)** | Demo Streamlit menggunakan row dari test.csv — bukan crawling real-time dari blockchain Ethereum. Hanya untuk demontstrasi antarmuka. |
| **Blockchain lokal** | Hardhat localhost ≠ jaringan publik. Latency dan gas cost nyata akan berbeda. |
| **Risk score linear** | `round(p × 100)` adalah mapping sederhana; tidak mempertimbangkan calibration probabilistik model RF |

### 14.4 Kualitas Data Fitur Eksternal (Etherscan & BigQuery)

Audit kualitas data terhadap 12.815 alamat sampel mengungkap beberapa keterbatasan pada fitur eksternal yang perlu dicatat secara jujur:

**Fitur Etherscan (coverage 99,98% — layak dipakai, dengan catatan):**

| Catatan | Detail | Mitigasi / Status |
|---|---|---|
| `created_at_ts` konstan nol | Pada run eksperimen, kolom ini tidak terisi (selalu 0) sehingga menjadi *dead feature* tanpa kontribusi ke model. Karena varians nol, fitur ini **tidak memengaruhi metrik** RF/XGBoost/LR yang dilaporkan. | Kode enrichment (`ml/data_fetchers.py`) telah diperbaiki agar mengambil timestamp transaksi pertama untuk run berikutnya. |
| `unique_counterparties > transaction_count` pada ~32% baris | Bukan korupsi acak, melainkan **beda basis hitung**: `transaction_count` dihitung dari seluruh tx normal (cap 10.000), sedangkan `unique_counterparties`, `avg_tx_value`, `tx_frequency`, dan `activity_score` dihitung hanya dari **100 transaksi terbaru** (jendela sampling) dan sempat menyertakan alamat itu sendiri. | Inkonsistensi self-count diperbaiki di kode; keterbatasan jendela 100-tx didokumentasikan di sini. |
| `transaction_count` ter-cap 10.000 | 8 alamat sangat aktif nilainya terpotong oleh limit halaman API Etherscan. | Dampak minor; dicatat sebagai keterbatasan pengukuran. |
| 3 baris (0,02%) gagal fetch | Di-imputasi nol (missing menyamar sebagai nilai nol). | Dampak dapat diabaikan; dicatat. |

**Fitur BigQuery (coverage 0,98% — keterbatasan signifikan):**

Hanya 126 dari 12.815 alamat (0,98%) yang benar-benar memiliki data BigQuery; 12.689 baris di-imputasi nol sehingga seluruh 10 fitur `bq_*` praktis konstan-nol. Karena itu Skenario 3 (graf + BigQuery) tidak menambah nilai dibanding Skenario 1 — penyebab utamanya adalah **coverage yang sangat rendah dan jendela waktu berbeda** (graf MulDiGraph 2019 vs sebagian data BigQuery yang lebih baru), bukan semata tumpang-tindih semantik. Selain itu, dari 126 baris ber-data tersebut 125 berlabel normal dan hanya 1 phishing, sehingga keberadaan data BigQuery sedikit terkorelasi dengan label (potensi bias yang, untungnya, berdampak kecil karena hampir semua baris bernilai nol). Fitur BigQuery tetap dipertahankan dalam pelaporan sebagai bagian dari eksperimen, namun **tidak direkomendasikan sebagai sumber fitur produksi** sampai coverage-nya diperbaiki.

### 14.5 Ringkasan Kontribusi vs Keterbatasan

| ✅ Kontribusi | ⚠️ Keterbatasan |
|---|---|
| Pipeline ML end-to-end dari graph pkl ke prediksi | Dataset 2019, tidak real-time |
| Bukti integrasi blockchain (immutable audit trail) | Blockchain lokal (Hardhat), bukan mainnet |
| Perbandingan 4 model × 4 skenario dengan metrik standar | Imbalance tidak sepenuhnya diatasi |
| Eksperimen empiris sampling ratio (RQ3) | Demo mode, bukan crawling on-chain |
| GNN-GraphSAGE dengan edge transaksi nyata (85,94% F1) | Subgraph sparse — hanya 834 edge (440 unik) dari 12.815 node |
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
| Buang fitur mati | Hapus `created_at_ts` (konstan 0) | **25 → 24 fitur** inti |
| Cek NaN | Verifikasi missing value | **0 NaN** → tanpa imputasi |
| Cek salah input | Cari nilai negatif tak logis | **0 negatif** |
| Baris low-activity | Tinjau baris hampir kosong | **32 baris** dipertahankan |
| Outlier | Winsorize (capping) p1–p99 | **3.202 nilai** di-cap |

Total baris tetap **12.815** (tidak ada yang dibuang — pendekatan **data-centric**: data
dirapikan, bukan dibuang). Distribusi label tetap 11.650 normal / 1.165 phishing (9,09%).
Keputusan paper: (1) tidak ada duplikat; (2) `created_at_ts` dibuang karena nol informasi;
(3) nilai 0 bersifat **struktural** (bukan missing) → dipertahankan; (4) outlier banyak
(~6–25% per fitur, IQR) → **di-capping** bukan dibuang; (5) imbalance ditangani di modeling.
**Anti-leakage:** capping/scaling di-fit pada TRAIN saja; threshold p1/p99 disimpan di
`cleaning_report.json`.

### 16.2 EDA, Outlier & Statistik Deskriptif (poin 4, 8, data-centric)

**Script:** `ml/eda_visualize.py` → `ml/artifacts/eda/01..05_*.png`.

| Figur | Isi | Menjawab |
|---|---|---|
| `01_class_distribution.png` | Distribusi kelas (imbalance 1 : 10) | poin 8 |
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
`scale_pos_weight = 10,00`, capping di-fit pada train saja. Basis fitur = 24 (25 fitur skenario
produksi dikurangi `created_at_ts` yang dibuang di §16.1); `graph_present` sudah tidak lagi ikut
karena bukan predictor.

Peringkat teratas: `unique_senders` ≫ `in_degree` > `timestamp_spread_in` >
`sent_received_ratio` > `total_sent` — **urutan lima besar identik** dengan run sebelumnya,
menunjukkan peringkat fitur stabil terhadap perubahan rasio kelas. Redundan (dibuang):
**`in_degree`**, **`tx_frequency`**.

| Skenario | Jumlah fitur | F1 | Precision | Recall |
|---|:---:|:---:|:---:|:---:|
| full | 24 | 0,8947 | 0,8467 | 0,9485 |
| **corr_pruned** | **22** | **0,8966** | 0,8500 | 0,9485 |
| top_20 | 20 | 0,8921 | 0,8488 | 0,9399 |
| top_18 | 18 | 0,8866 | 0,8391 | 0,9399 |
| top_15 | 15 | 0,8907 | 0,8429 | 0,9442 |

**Kesimpulan:** memangkas fitur **tidak menurunkan** kinerja secara berarti — seluruh varian
berada di rentang sempit 0,8866–0,8966 (selisih maksimum 1,0 poin F1). Yang terbaik kini
**`corr_pruned`** (22 fitur, 0,8966), yaitu membuang tepat dua fitur yang terbukti redundan
(`in_degree` yang berkorelasi 0,93 dengan `unique_senders`, dan `tx_frequency` yang berkorelasi
1,00 dengan `activity_score`) — sedikit **lebih baik** daripada 24 fitur penuh (0,8947). Arahan
dosen bahwa banyak fitur bisa dipangkas tetap terbukti; hanya rekomendasinya berubah dari
`top_20` menjadi **`corr_pruned`**, yang secara metodologis juga lebih mudah dipertahankan
karena kriterianya eksplisit (korelasi ≥ 0,95) alih-alih ambang peringkat yang arbitrer.

### 16.4 Hyperparameter Tuning & Penanganan Imbalance (poin 2, 8, 12)

**Script:** `ml/tune_xgboost.py` → `ml/artifacts/xgb_tuning_results.json`.
`RandomizedSearchCV` (60 iterasi × StratifiedKFold 5-fold, `scoring="f1"`) pada 20 fitur
hasil seleksi (`top_20`), capping p1–p99 di-fit pada train saja, `scale_pos_weight = 10,00`.
Catatan: skrip ini masih memakai `top_20` sebagai basis; setelah §16.3 merekomendasikan
`corr_pruned` (22 fitur), mengulang tuning di atas basis tersebut adalah pekerjaan lanjutan
yang **belum** dilakukan — selisih F1 antar-basis (0,8921 vs 0,8966) terlalu kecil untuk
mengubah kesimpulan bagian ini.

| Konfigurasi | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Baseline (parameter lama, thr 0.5) | 0,9770 | 0,8295 | 0,9399 | 0,8813 |
| Tuned (thr 0.5) | 0,9797 | 0,8494 | 0,9442 | 0,8943 |
| **Tuned (thr 0.604, optimasi F1)** | 0,9809 | 0,8594 | 0,9442 | **0,8998** |
| CV F1 terbaik (5-fold) | — | — | — | 0,8961 |

Perbandingan strategi imbalance (poin dosen: kenapa undersampling? / XGBoost harusnya bisa
tanpa undersampling):

| Strategi | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| **Undersampling 1:5** | 0,9805 | 0,8588 | 0,9399 | **0,8975** |
| `scale_pos_weight` (tanpa buang data) | 0,9797 | 0,8494 | 0,9442 | 0,8943 |
| `sample_weight` per-baris | 0,9785 | 0,8423 | 0,9399 | 0,8884 |

> ⚠️ **Perubahan kesimpulan dibanding versi 8.595 (jujur dicatat).** Pada dataset 1:6,4
> sebelumnya `scale_pos_weight` menang atas undersampling (0,9065 vs 0,8973). Pada dataset 1:10
> penuh urutannya **berbalik**: undersampling 1:5 kini sedikit unggul (0,8975 vs 0,8943).
> Ini masuk akal — makin timpang data awal, makin besar keuntungan presisi dari menyeimbangkan
> ulang kelas mayoritas. Namun selisihnya hanya **0,32 poin F1**, di dalam rentang variasi acak,
> dan arah trade-off-nya berlawanan dengan kebutuhan keamanan: undersampling menaikkan
> *precision* (0,8588 vs 0,8494) dengan mengorbankan *recall* (0,9399 vs 0,9442) **dan** membuang
> ~5.800 baris kelas normal.

**Kesimpulan:** penelitian **tetap memilih `scale_pos_weight`**, tetapi kini atas dasar yang
lebih hati-hati: bukan karena F1-nya tertinggi (ternyata tidak, pada rasio 1:10), melainkan
karena (a) selisihnya tidak signifikan, (b) *recall*-nya lebih tinggi — arah yang tepat untuk
deteksi phishing, dan (c) **tidak membuang satu baris pun**, sehingga estimasi performa lebih
stabil dan reproducible. Tuning menaikkan F1 dari 0,8813 → 0,8943; berbeda dari run sebelumnya,
kali ini **optimasi threshold membantu** (thr 0,604 → F1 0,8998, +0,55 poin). Meski begitu
**ambang 0,5 tetap dipertahankan di produksi** agar keluaran biner `p ≥ 0,5` konsisten dengan
definisi verdict di §6 dan tidak di-tuning ke *test set*.

#### Uji empiris SMOTE / oversampling (catatan #8)

**Script:** `ml/smote_experiment.py` → `ml/artifacts/smote_experiment.json`. Menjawab arahan
dosen agar SMOTE/oversampling **diuji empiris**, bukan hanya alasan konseptual. Resampling
diterapkan **hanya pada data train** (anti-leakage; test dibiarkan asli), model resampling
memakai `scale_pos_weight=1` agar tidak dobel-koreksi. Semua memakai `best_params` yang sama.

| Strategi | Precision | Recall | F1 | PR-AUC | Ukuran train |
|---|---|---|---|---|---|
| `scale_pos_weight` (acuan, tanpa resampling) | 0,8494 | 0,9442 | 0,8943 | 0,9598 | 10.252 (asli) |
| SMOTE | 0,8359 | 0,9399 | 0,8848 | 0,9495 | 18.640 (9320:9320) |
| **BorderlineSMOTE** | 0,8610 | 0,9571 | **0,9065** | 0,9543 | 18.640 |
| SMOTETomek | 0,8271 | 0,9442 | 0,8818 | 0,9484 | 17.876 |
| RandomOverSampler | 0,8571 | 0,9528 | 0,9024 | 0,9615 | 18.640 |

**Kesimpulan #8:** semua metode oversampling menghasilkan F1 dalam rentang sempit
(0,882–0,907) di sekitar `scale_pos_weight` (0,8943) — tidak ada yang unggul meyakinkan.
BorderlineSMOTE menempati posisi teratas (0,9065, +1,2 poin) tetapi dengan **PR-AUC yang justru
lebih rendah** (0,9543 vs 0,9598), artinya keunggulan F1-nya bergantung pada ambang 0,5 dan
tidak mencerminkan kemampuan *ranking* yang lebih baik. Karena tidak ada peningkatan yang
konsisten lintas metrik dan oversampling **menghasilkan data sintetis** (menggandakan train
dari 10.252 → 18.640 baris), penelitian **tetap memilih `scale_pos_weight`** yang lebih
sederhana, tidak membuang maupun mengarang data. Catatan jujur: pada run sebelumnya (1:6,4)
SMOTETomek yang unggul tipis, kini SMOTETomek justru terendah (0,8818) — ketidakkonsistenan
peringkat antar-run ini sendiri menjadi bukti bahwa selisih antar metode oversampling **berada
di dalam derau**, bukan efek nyata.

#### Mengapa XGBoost "tidak perlu takut" data tidak seimbang (penjelasan mekanisme)

Pernyataan bahwa XGBoost tidak perlu penanganan imbalance eksternal **bukan** berarti
algoritma ini kebal secara otomatis, melainkan karena XGBoost menangani ketidakseimbangan
**dari dalam fungsi loss-nya**, tanpa mengubah komposisi data. Mekanismenya:

XGBoost dilatih dengan meminimalkan *loss* (kesalahan prediksi). Secara default setiap baris
berbobot sama, sehingga pada data timpang (**11.650 normal : 1.165 phishing = 1 : 10,00**)
model cenderung "bermain aman" dengan memprediksi mayoritas (normal) — menebak normal terus
saja sudah benar ±91% waktu, tetapi recall phishing menjadi rendah. Parameter
**`scale_pos_weight`** mengoreksi ini dengan **memperbesar bobot kesalahan pada kelas
positif (phishing)** di dalam gradien loss, sehingga kesalahan mendeteksi phishing "dihukum"
jauh lebih berat daripada kesalahan pada kelas normal. Nilainya dihitung langsung dari rasio
kelas:

```text
scale_pos_weight = jumlah_negatif / jumlah_positif = 11.650 / 1.165 = 10,00
```

Angka **10,00** inilah yang dipakai di seluruh eksperimen (lihat `cleaning_report.json`,
`feature_selection_report.json`, `xgb_tuning_results.json`); pada split train saja nilainya
9,99 karena pembulatan stratifikasi.

**Perbandingan tiga pendekatan imbalance:**

| Pendekatan | Cara kerja | Konsekuensi | Analogi |
|---|---|---|---|
| Undersampling | Buang sebagian baris kelas mayoritas | **Informasi hilang**, dataset mengecil | Murid mayoritas disuruh keluar kelas agar jumlah seimbang |
| Oversampling (SMOTE) | Duplikasi/sintesis baris minoritas | Data tiruan, risiko *overfitting* | Murid minoritas difotokopi jadi banyak |
| **`scale_pos_weight`** | Bobot error minoritas dinaikkan di loss | **Tidak buang & tidak mengarang data** | Semua murid tetap, nilai murid minoritas dikali 10 |

Secara empiris pada dataset ini ketiga pendekatan berakhir **setara**: `scale_pos_weight`
(F1 0,8943) berada di antara undersampling 1:5 (0,8975) dan `sample_weight` (0,8884), dengan
rentang total hanya 0,9 poin F1. Karena tidak ada baris yang dibuang, `scale_pos_weight`
dipilih agar estimasi performa lebih **stabil dan jujur** terhadap seluruh data yang tersedia,
dan agar hasilnya dapat direproduksi tanpa bergantung pada subsampel acak.

**Trade-off yang perlu disebut saat sidang:** `scale_pos_weight` menaikkan **recall** (lebih
banyak phishing tertangkap) dengan sedikit korban **precision** (false alarm naik tipis). Untuk
konteks keamanan/phishing, arah trade-off ini **tepat** — membiarkan phishing lolos (false
negative) jauh lebih berbahaya daripada satu peringatan palsu (false positive).

> ✍️ **[UNTUK DITULIS DI TESIS]**
> **BAB 3 (Metodologi) — sub-bab Penanganan Data Tidak Seimbang:** tuliskan bahwa penelitian
> ini menangani imbalance kelas (1 : 10) menggunakan `scale_pos_weight = jumlah_negatif /
> jumlah_positif = 10,00` pada XGBoost, **bukan** undersampling/oversampling, dengan alasan
> tidak membuang informasi dan tidak menghasilkan data sintetis.
> **BAB 4 (Hasil/Pembahasan):** sertakan tabel perbandingan tiga strategi imbalance
> (undersampling 0,8975 vs `scale_pos_weight` 0,8943 vs `sample_weight` 0,8884). **Jangan**
> klaim penanganan internal XGBoost lebih unggul secara F1 — pada rasio 1:10 undersampling
> justru sedikit di atas. Rumuskan sebagai: ketiga strategi setara dalam batas derau,
> `scale_pos_weight` dipilih karena recall lebih tinggi dan tidak membuang data; tegaskan
> trade-off recall–precision yang sesuai untuk kasus keamanan.

### 16.5 ROC / AUC & Perbandingan XGBoost vs GNN (poin 13, 14)

**Script:** `ml/evaluate_roc.py` → `roc_xgb.png`, `roc_comparison.png`,
`pr_curve_comparison.png`, `roc_auc_report.json`. Skenario MulDiGraph + Etherscan, split
identik (2.563 baris uji, 233 phishing, base rate positif 0,0909).

| Model | Accuracy | Precision | Recall | F1 | **ROC-AUC** | **PR-AUC** |
|---|---|---|---|---|---|---|
| **XGBoost** | 0,9789 | 0,8403 | 0,9485 | **0,8911** | **0,9938** | **0,9582** |
| GNN-GraphSAGE | 0,9692 | 0,7873 | 0,9056 | 0,8423 | 0,9856 | 0,8762 |

Selain **kurva ROC asli (FPR–TPR)** di `roc_xgb.png`/`roc_comparison.png`, kini tersedia pula
**kurva Precision-Recall** (`pr_curve_comparison.png`) beserta **PR-AUC** (*average precision*).
PR-AUC penting untuk data **imbalance**: baseline acak hanya 0,0909 (proporsi kelas phishing),
sementara XGBoost mencapai **PR-AUC 0,9582** — lebih dari sepuluh kali baseline, menegaskan model
benar-benar membedakan phishing, bukan sekadar unggul karena kelas mayoritas. Perhatikan bahwa
base rate turun dari 0,1355 (dataset 1:6,4) ke 0,0909 (1:10), sehingga **PR-AUC 0,9582 pada
baseline 0,0909 sebenarnya lebih impresif** daripada 0,9701 pada baseline 0,1355. XGBoost unggul
pada ROC-AUC maupun PR-AUC.

ROC/AUC **valid untuk klasifikasi biner** (phishing vs bukan phishing). ROC-AUC 0,9938 (XGBoost)
dan 0,9856 (GNN) berarti **kemampuan diskriminasi sangat baik** — plafon F1 ~0,89 murni karena
trade-off precision–recall di kelas minoritas pada ambang 0,5, bukan karena model lemah.
XGBoost unggul pada F1 dan AUC → dipilih sebagai model produksi.

> **Catatan non-determinisme GNN:** angka XGBoost bersifat **deterministik** (seed tetap,
> bit-per-bit sama tiap run). Angka **GNN-GraphSAGE sedikit bervariasi antar run** karena
> inisialisasi bobot & dropout PyTorch tidak sepenuhnya deterministik pada CPU — pada
> pengujian terpisah pada dataset 1:10 ini GNN berkisar **F1 ≈ 0,842–0,859 dan AUC ≈ 0,986**
> (run §5.5 memberi F1 0,8594; run `evaluate_roc.py` di tabel atas memberi 0,8423). Variasi ini
> **tidak mengubah kesimpulan**: XGBoost tetap unggul F1 & AUC pada setiap run. Angka GNN yang
> dilaporkan di tabel adalah salah satu run representatif — karena spread antar-run mencapai
> ~1,7 poin F1, angka tunggal GNN sebaiknya **selalu dikutip bersama rentangnya** di naskah.
> GNN pada tabel ini memakai konfigurasi default; **hasil tuning GNN penuh (36 konfigurasi ×
> 2 seed) ada di §17.7** — konfigurasi terbaik menaikkan F1 GNN ke 0,8876–0,8878, praktis
> menyamai XGBoost.

### 16.6 Asal Skor Risiko & Penjelasan (Interpretability) (poin 15)

```text
25 fitur wallet ─► XGBoost ─► probabilitas p ∈ [0,1]
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
- **Model terdeploy:** XGBoost 25 fitur (**F1 0,8911, ROC-AUC 0,9938, PR-AUC 0,9582**)
  dipertahankan sebagai model produksi. Hasil tuning (F1 0,8943 pada 20 fitur, 0,8998 dengan
  threshold 0,604) unggul tipis di atasnya, tetapi selisih ≤0,9 poin berada di dalam derau dan
  threshold non-default akan memutus konsistensi dengan definisi verdict biner `p ≥ 0,5` (§6).
- **Target F1:** 95% **tidak tercapai** pada data ini (plafon realistis F1 ≈ 0,89–0,90 /
  ROC-AUC ≈ 0,99), namun **melewati batas minimum 85%** — sesuai fallback yang disebut dosen.

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

### 16.9 Verifikasi Reproduksi (run ulang 6 Agustus 2026 — dataset 1:10 penuh)

Seluruh script §16 dijalankan ulang dari awal pada dataset 12.815 alamat. Karena dataset
berubah (8.595 → 12.815), angkanya **memang berbeda** dari versi dokumen sebelumnya — yang
diverifikasi di sini adalah bahwa setiap angka di §16 benar-benar berasal dari artefak run ini,
bukan tersalin dari run lama:

| Script | Artefak | Angka yang kini dilaporkan | Berubah dari versi 8.595 |
|---|---|---|---|
| `clean_hybrid_dataset.py` | `cleaning_report.json` | 0 duplikat · 3.202 nilai di-clip · 12.815 baris | dari 2.127 clip · 8.595 baris |
| `feature_selection.py` | `feature_selection_report.json` | corr_pruned (22) F1 0,8966 · top_20 F1 0,8921 | rekomendasi berubah top_20 → corr_pruned |
| `tune_xgboost.py` | `xgb_tuning_results.json` | tuned F1 0,8943 · CV F1 0,8961 · thr-opt 0,8998 | dari 0,9065 · 0,9295 · 0,8992 |
| `smote_experiment.py` | `smote_experiment.json` | terbaik BorderlineSMOTE 0,9065 | sebelumnya SMOTETomek 0,9087 |
| `evaluate_roc.py` — **XGBoost** | `roc_auc_report.json` | F1 0,8911 · ROC-AUC 0,9938 · PR-AUC 0,9582 | dari 0,9102 · 0,9948 · 0,9701 |
| `evaluate_roc.py` — GNN | `roc_auc_report.json` | F1 0,8423 · ROC-AUC 0,9856 | non-deterministik (§16.5) |

**Kesimpulan verifikasi:** pipeline data-centric, seleksi fitur, tuning, SMOTE, dan ROC/AUC
**reproducible penuh** pada dataset baru; model produksi XGBoost deterministik (seed tetap).
Dua kesimpulan berubah dan dicatat eksplisit: **strategi imbalance** (§16.4) dan **rekomendasi
seleksi fitur** (§16.3). Semua target bimbingan tetap terpenuhi: F1 produksi 0,8911
(**melewati minimum 85%**, di bawah target ideal 95%), ROC-AUC 0,9938, output biner
Phishing/Bukan Phishing, dan skor dijelaskan via SHAP.

### 16.10 Dampak Tiap Tindakan terhadap F1 — Awal vs Sesudah (analisis jujur)

Bagian ini menjawab pertanyaan langsung: *"setelah data dibersihkan, outlier di-capping, fitur
mati dibuang, dan seleksi fitur/tuning dilakukan — apakah F1 naik?"* Jawaban jujurnya:
**tidak ada satu pun tindakan yang menaikkan F1 secara berarti; seluruh varian berkerumun di
rentang 0,881–0,900.** Nilai acuan = XGBoost 25 fitur, MulDiGraph + Etherscan, dataset 1:10.
Setiap baris di bawah diukur **terisolasi** (satu perubahan, konfigurasi & split lain identik).

| Tindakan | F1 awal | F1 sesudah | Selisih | Keterangan |
|---|:---:|:---:|:---:|---|
| Buang fitur mati/konstan (`created_at_ts`) | 0,8911 | 0,8911 | **0,0000** | Varians nol → secara matematis tidak menyumbang; metrik tak berubah |
| Buang flag audit (`graph_present`, 26 → 25) | 0,8966 | 0,8911 | **−0,0055** | Konstan-1 → juga varians nol; selisih murni dari pengacakan `colsample_bytree`, bukan sinyal |
| Capping outlier (winsorize p1–p99) | 0,8911 | 0,8929 | **+0,0018** | Praktis netral (recall identik, precision +0,3 poin) |
| Seleksi fitur (24 → 22, `corr_pruned`) | 0,8947 | 0,8966 | **+0,0019** | Naik tipis; membuang 2 fitur redundan tidak merugikan |
| Seleksi fitur (24 → 20, `top_20`) | 0,8947 | 0,8921 | **−0,0026** | Pemangkasan lebih agresif mulai sedikit merugikan |
| Tuning hyperparameter (RandomizedSearchCV) | 0,8813 | 0,8943 | **+0,0130** | Kenaikan terbesar, tapi dari basis yang sudah di-capping + 20 fitur |
| Optimasi threshold (0,5 → 0,604) | 0,8943 | 0,8998 | **+0,0055** | Membantu, tapi tidak diadopsi (memutus definisi verdict `p ≥ 0,5`, §6) |
| `scale_pos_weight` vs undersampling 1:5 | 0,8975 | 0,8943 | **−0,0032** | Kali ini undersampling sedikit unggul; tetap tidak dipakai (§16.4) |
| **Model produksi final (yang di-deploy)** | 0,8911 | **0,8911** | **0,0000** | Konfigurasi 25 fitur, tanpa capping, threshold 0,5 dipertahankan |

**Mengapa capping outlier praktis tidak berpengaruh?** Pada run 1:6,4 sebelumnya capping
tercatat sedikit **menurunkan** F1 (−0,0037) dan ditafsirkan sebagai bukti bahwa nilai ekstrem
membawa sinyal phishing. Pada dataset 1:10 arahnya **berbalik menjadi sedikit naik** (+0,0018),
dengan recall yang sama persis. Kedua selisih ini sama-sama di bawah 0,5 poin F1 dan saling
meniadakan antar-run, sehingga tafsiran yang benar adalah: **capping tidak berpengaruh material
ke salah satu arah** pada data ini. Klaim "nilai ekstrem adalah sinyal" karenanya **dicabut** —
tidak didukung bukti yang stabil. Yang tetap berlaku: ROC-AUC sangat tinggi (0,9938) menunjukkan
model sudah mendekati **plafon kemampuan diskriminasi data**, sehingga intervensi data-centric
memang tidak lagi mampu menggerakkan F1 secara berarti.

**Makna hasil ini (untuk sidang):** tujuan pekerjaan data-centric **bukan** sekadar menaikkan
F1, melainkan **memastikan kualitas, validitas, dan reproducibility** metodologi — menjawab
tuntutan bimbingan soal duplikat, missing value, outlier, dan strategi imbalance. Hasilnya
membuktikan tiga hal secara empiris: (1) fitur dapat dipangkas dari 24 ke 22 **tanpa merugikan**
kinerja; (2) ketiga strategi imbalance (undersampling, `scale_pos_weight`, `sample_weight`)
**setara dalam batas derau**, sehingga pemilihannya dapat didasarkan pada alasan metodologis
(tidak membuang data) alih-alih selisih skor; dan (3) tidak ada konfigurasi yang mampu menembus
target 0,95 pada data ini — **plafon realistis F1 ≈ 0,89–0,90 / ROC-AUC ≈ 0,99**. Karena capping
tidak memberi keuntungan yang stabil, **model produksi final mempertahankan data tanpa capping**
dengan 25 fitur (F1 0,8911) sebagai konfigurasi yang dipakai.

> ✍️ **[UNTUK DITULIS DI TESIS — BAB 4/BAB 5]** Sampaikan sebagai *honest finding*: pendekatan
> data-centric meningkatkan **kualitas metodologi**, bukan skor absolut, karena model telah
> mendekati plafon kemampuan data (ROC-AUC 0,9938). Sertakan tabel awal-vs-sesudah di atas
> sebagai bukti transparansi, dan jelaskan capping outlier sebagai keputusan yang **diuji lalu
> tidak diadopsi** karena efeknya terbukti tidak material (±0,2–0,4 poin F1, arahnya bahkan
> berbalik antar-run) — bukan karena nilai ekstrem terbukti diskriminatif.

---

## 17) Pendalaman Audit Data — Tabel Lengkap (Tindak Lanjut Audit 18 Catatan, Update: 13 Juli 2026)

Bagian ini menutup item yang pada audit masih berstatus **parsial** karena angkanya sudah
ada tetapi **tabel rincinya** belum ditampilkan. Semua angka dihasilkan oleh
`ml/audit_gaps.py` → `ml/artifacts/audit_report.json` (reproducible).

### 17.1 Audit Penggabungan Sumber Data (catatan #1)

Penggabungan dilakukan **by-address** (di-*lowercase*) antar tiga sumber. Tabel matched/unmatched:

| Sumber | Matched | Coverage | Keterangan |
|---|---:|---:|---|
| MulDiGraph (fitur graf) | 12.815 / 12.815 | 100% | Semua alamat punya jejak di graph (`graph_present`=1) |
| Etherscan (fitur atribut) | 12.812 / 12.815 | 99,98% | Hanya **3 alamat gagal fetch** |
| BigQuery (fitur `bq_*`) | 126 / 12.815 | 0,98% | Coverage sangat rendah → **tidak dipakai di produksi** |

Dataset final = **12.815 alamat unik** (1.165 phishing, 11.650 normal), **0 baris ganda** hasil
merge. Penggabungan tidak menciptakan duplikat karena kunci join adalah alamat unik.
Catatan: `graph_present` bernilai 1 pada **seluruh** baris — inilah alasan kolom ini tidak layak
menjadi predictor dan kini hanya dicatat sebagai kolom audit (`audit_columns`).

### 17.2 Statistik Deskriptif Lengkap per Fitur (catatan #4)

Metode outlier: IQR (di luar Q1−1,5·IQR … Q3+1,5·IQR). Tabel lengkap min/Q1/median/mean/Q3/max/std + jumlah outlier:

| Fitur | Min | Q1 | Median | Mean | Q3 | Max | Std | #Outlier | % |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| min_out_amount | 0 | 0 | 0 | 1,54 | 0 | 1.995 | 23,03 | 3.136 | 24,5 |
| timestamp_spread_out | 0 | 0 | 0 | 1,74e6 | 67.984 | 7,30e7 | 5,26e6 | 2.998 | 23,4 |
| total_sent | 0 | 0 | 0 | 234 | 0,143 | 2,83e6 | 24.980 | 2.860 | 22,3 |
| max_out_amount | 0 | 0 | 0 | 9,60 | 0,117 | 28.964 | 346 | 2.844 | 22,2 |
| avg_out_amount | 0 | 0 | 0 | 2,27 | 0,082 | 1.995 | 26,20 | 2.763 | 21,6 |
| total_received | 0 | 0 | 0 | 237 | 0,374 | 2,81e6 | 24.834 | 2.702 | 21,1 |
| max_in_amount | 0 | 0 | 0 | 10,58 | 0,309 | 30.000 | 316 | 2.634 | 20,6 |
| min_in_amount | 0 | 0 | 0 | 2,83 | 0,050 | 6.247 | 74,39 | 2.625 | 20,5 |
| avg_in_amount | 0 | 0 | 0 | 4,82 | 0,236 | 11.664 | 130 | 2.540 | 19,8 |
| balance | 0 | 0 | 0 | 0,397 | 0,002 | 2.576 | 23,28 | 2.530 | 19,7 |
| sent_received_ratio | 0 | 0 | 0 | 2,76e9 | 1,000 | 4,26e12 | 5,90e10 | 2.469 | 19,3 |
| timestamp_spread_in | 0 | 0 | 0 | 638.417 | 0 | 6,32e7 | 3,23e6 | 2.203 | 17,2 |
| std_in_amount | 0 | 0 | 0 | 1,94 | 0 | 6.995 | 72,22 | 2.114 | 16,5 |
| std_out_amount | 0 | 0 | 0 | 1,32 | 0 | 3.498 | 35,81 | 1.914 | 14,9 |
| avg_tx_value | 0 | 0,003 | 0,193 | 6,16 | 1,26 | 4.000 | 66,88 | 1.696 | 13,2 |
| activity_score | 0 | 3 | 8 | 22,28 | 26 | 100 | 29,94 | 1.670 | 13,0 |
| tx_frequency | 0 | 0,100 | 0,267 | 0,743 | 0,867 | 3,33 | 0,998 | 1.670 | 13,0 |
| transaction_count | 0 | 3 | 8 | 55,17 | 26 | 10.000 | 352 | 1.664 | 13,0 |
| in_degree | 0 | 0 | 0 | 5,62 | 1 | 11.066 | 145 | 1.516 | 11,8 |
| unique_receivers | 0 | 0 | 1 | 2,46 | 1 | 2.527 | 25,14 | 1.461 | 11,4 |
| unique_counterparties | 0 | 3 | 6 | 12,19 | 14 | 101 | 15,13 | 1.283 | 10,0 |
| out_degree | 0 | 0 | 1 | 5,95 | 2 | 17.954 | 163 | 1.241 | 9,7 |
| unique_senders | 0 | 0 | 0 | 3,60 | 1 | 8.371 | 90,45 | 1.042 | 8,1 |
| in_out_ratio | 0 | 0 | 0 | 2,42 | 1 | 11.066 | 105 | 798 | 6,2 |

Tabel kini memuat **24 fitur** — `graph_present` sudah tidak lagi diaudit sebagai fitur karena
berstatus kolom audit (pada versi sebelumnya ia muncul di baris terakhir dengan std 0 dan
0 outlier, yang justru menegaskan bahwa ia konstan).

Terlihat jelas kasus yang dosen sebut ("balance mayoritas kecil, ada 1 yang ekstrem"):
mean ≫ median pada hampir semua fitur → sebaran **skewed kanan** → ditangani capping p1–p99.

### 17.3 Tabel Audit 24 Fitur Inti (catatan #5)

Coverage = % baris bernilai non-nol; importance dari `feature_selection_report.json`;
`|corr|` maks = korelasi absolut tertinggi dengan fitur lain; keputusan = hasil seleksi fitur.

| Fitur | Sumber | Coverage % | Imp. XGB | Imp. RF | \|corr\| maks | Keputusan |
|---|---|---:|---:|---:|---:|---|
| unique_senders | MulDiGraph | 43,1 | 0,432 | 0,190 | 0,93 | ✅ dipakai (top-20) |
| in_degree | MulDiGraph | 43,1 | 0,127 | 0,075 | 0,93 | ✅ dipakai (top-20) |
| timestamp_spread_in | MulDiGraph | 17,2 | 0,077 | 0,083 | 0,40 | ✅ dipakai (top-20) |
| sent_received_ratio | MulDiGraph | 31,2 | 0,066 | 0,049 | 0,42 | ✅ dipakai (top-20) |
| total_sent | MulDiGraph | 31,2 | 0,056 | 0,046 | 1,00 | ✅ dipakai (top-20) |
| max_out_amount | MulDiGraph | 31,2 | 0,014 | 0,047 | 0,76 | ✅ dipakai (top-20) |
| total_received | MulDiGraph | 41,7 | 0,009 | 0,066 | 1,00 | ✅ dipakai (top-20) |
| max_in_amount | MulDiGraph | 41,7 | 0,011 | 0,059 | 0,93 | ✅ dipakai (top-20) |
| std_in_amount | MulDiGraph | 16,5 | 0,008 | 0,078 | 0,93 | ✅ dipakai (top-20) |
| avg_out_amount | MulDiGraph | 31,2 | 0,020 | 0,045 | 0,89 | ✅ dipakai (top-20) |
| avg_in_amount | MulDiGraph | 41,7 | 0,014 | 0,062 | 0,87 | ✅ dipakai (top-20) |
| in_out_ratio | MulDiGraph | 43,1 | 0,037 | 0,047 | 0,81 | ✅ dipakai (top-20) |
| transaction_count | Etherscan | 99,3 | 0,021 | 0,014 | 0,40 | ✅ dipakai (top-20) |
| out_degree | MulDiGraph | 73,2 | 0,014 | 0,015 | 0,97 | ✅ dipakai (top-20) |
| unique_receivers | MulDiGraph | 73,2 | 0,015 | 0,016 | 0,92 | ✅ dipakai (top-20) |
| min_in_amount | MulDiGraph | 40,4 | 0,008 | 0,024 | 0,58 | ✅ dipakai (top-20) |
| unique_counterparties | Etherscan | 99,3 | 0,014 | 0,012 | 0,77 | ✅ dipakai (top-20) |
| activity_score | Etherscan | 99,3 | 0,011 | 0,015 | 1,00 | ✅ dipakai (top-20) |
| tx_frequency | Etherscan | 99,3 | 0,009 | 0,014 | 1,00 | ✅ dipakai (top-20) |
| avg_tx_value | Etherscan | 98,1 | 0,009 | 0,015 | 0,25 | ✅ dipakai (top-20) |
| min_out_amount | MulDiGraph | 24,5 | 0,006 | 0,012 | 0,89 | ⏸ cadangan (di luar top-20) |
| timestamp_spread_out | MulDiGraph | 31,3 | 0,009 | 0,005 | 0,45 | ⏸ cadangan (di luar top-20) |
| balance | Etherscan | 57,0 | 0,007 | 0,006 | 0,04 | ⏸ cadangan (di luar top-20) |
| std_out_amount | MulDiGraph | 14,9 | 0,006 | 0,005 | 0,74 | ⏸ cadangan (di luar top-20) |

Catatan: `activity_score` & `tx_frequency` punya `|corr|`=1,00 (saling redundan) — salah satu
dipangkas di varian `corr_pruned` (§16.3), begitu pula `in_degree` yang berkorelasi 0,93 dengan
`unique_senders`. Peringkat importance **stabil** terhadap perubahan rasio kelas: lima fitur
teratas tetap `unique_senders` ≫ `in_degree` > `timestamp_spread_in` > `sent_received_ratio` >
`total_sent`, sama persis dengan audit pada dataset 8.595. `graph_present` tidak lagi muncul di
tabel ini karena bukan predictor (dulu tercatat importance 0,000 — konsisten dengan alasan
pengeluarannya).

### 17.4 Missingness Sebelum Zero-Filling (catatan #6, #11)

**Penting (menjawab dosen):** dataset final memang 0 NaN, **tetapi** itu karena nilai kosong
dari API sudah diisi nol saat `build_hybrid_dataset.py`. Tabel berikut menampilkan kondisi
**SEBELUM** zero-fill:

| Kelompok kolom | Missing sebelum fill | % | Kebijakan | Alasan |
|---|---:|---:|---|---|
| 6 fitur Etherscan (balance, transaction_count, dst) | **3** | 0,02% | Imputasi nol | Hanya 3 alamat gagal fetch → dampak dapat diabaikan |
| 10 fitur BigQuery (`bq_*`) | **12.689** | 99,02% | Imputasi nol | Coverage sangat rendah → fitur praktis konstan-nol, **tidak dipakai produksi** |
| `created_at_ts` | 12.815 (konstan 0) | 100% | **Dibuang** | Timestamp gagal terisi → varians nol, tidak informatif |

**Kebijakan untuk paper:** (1) fitur Etherscan yang hanya 3 baris kosong → imputasi nol
(dampak minor); (2) fitur BigQuery kosong ~99% → dipertahankan hanya untuk pelaporan
eksperimen, **tidak untuk model produksi**; (3) `created_at_ts` konstan → dibuang. Tidak ada
kolom "hanya terisi 4 dari 25" pada fitur inti — seluruh 24 fitur inti terisi (nol yang ada
bersifat **struktural**, bukan missing).

### 17.5 Deteksi Salah Input Rinci (catatan #7)

Pemeriksaan menyeluruh terhadap tipe kesalahan input yang mungkin:

| Jenis cek | Hasil | Status |
|---|---|---|
| Nilai negatif (balance, count, dll) | 0 | ✅ bersih |
| Nilai tak hingga (`inf`) | 0 | ✅ bersih |
| Nilai NaN | 0 | ✅ bersih |
| Alamat duplikat | 0 | ✅ bersih |
| Timestamp tidak valid (`created_at_ts`) | 12.815 baris = 0 | ❌ dibuang (fitur mati) |
| Rasio ekstrem akibat pembagi ≈ 0 (`sent_received_ratio`) | max 4,26×10¹², **2.270 baris** > 10⁶ | ⚠️ di-capping p1–p99 |
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
training). Skenario (25 fitur), split (2.563 baris uji), dan edge nyata (834 edge
MulDiGraph+Etherscan) identik dengan §16.5.

| Peringkat | hidden | layers | lr | dropout | F1 (mean±std) | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---|---:|---:|
| **Terbaik (F1)** | 128 | 3 | 0,02 | 0,3 | **0,8878 ± 0,0068** | 0,9868 | 0,9001 |
| 2 | 256 | 3 | 0,005 | 0,3 | 0,8876 ± 0,0026 | **0,9907** | **0,9491** |
| 3 | 128 | 3 | 0,01 | 0,3 | 0,8864 ± 0,0028 | 0,9904 | 0,9402 |
| 4 | 256 | 3 | 0,01 | 0,3 | 0,8858 ± 0,0009 | 0,9869 | 0,9245 |
| 5 | 128 | 3 | 0,01 | 0,5 | 0,8850 ± 0,0013 | 0,9897 | 0,9141 |
| … | | | | | | | |
| Baseline (default lama) | 128 | 2 | 0,01 | 0,5 | 0,8649 ± 0,0022 | 0,9890 | 0,9121 |
| Terburuk | 64 | 2 | 0,02 | 0,5 | 0,8585 ± 0,0030 | 0,9878 | 0,8891 |

Rata-rata F1 per nilai hyperparameter (36 konfigurasi, rentang total 0,8585–0,8878):

| Hyperparameter | Nilai → rata-rata F1 |
|---|---|
| `num_layers` | 2 → 0,8685 · **3 → 0,8778** |
| `dropout` | **0,3 → 0,8768** · 0,5 → 0,8695 |
| `learning_rate` | 0,005 → 0,8757 · 0,01 → 0,8752 · 0,02 → 0,8685 |
| `hidden_dim` | 64 → 0,8696 · **128 → 0,8758** · 256 → 0,8740 |

**Kesimpulan #13:** tuning **menaikkan F1 GNN dari 0,8649 → 0,8878 (+2,29 poin)** — kenaikan
yang lebih besar daripada run 8.595 sebelumnya (+1,55 poin), jadi arahan agar GNN dituning
setara XGBoost tetap terbukti berfaedah. GNN yang dituning kini **mendekati XGBoost (0,8911)**
dengan selisih **0,0033**, masih di dalam rentang variasi antar-seed GNN (±0,007).

Dua pola bertahan dari run sebelumnya: **`dropout` 0,3 lebih baik dari 0,5** dan **3 layer
unggul atas 2 layer** (seluruh 5 besar memakai 3 layer). Satu pola **tidak bertahan**: dulu
`lr` 0,02 disebut "selalu terburuk", padahal kini justru konfigurasi ber-F1 tertinggi memakai
`lr` 0,02 — meski secara rata-rata `lr` 0,02 tetap yang terlemah (0,8685). Klaim "selalu"
karenanya dicabut; yang bisa dipertahankan hanya "rata-rata `lr` 0,02 sedikit lebih lemah".

> ⚠️ **Peringatan memilih konfigurasi "terbaik".** Konfigurasi ber-F1 tertinggi
> (128/3/0,02/0,3) justru punya **PR-AUC terburuk di antara 5 besar** (0,9001 vs 0,9491) dan
> deviasi antar-seed terbesar (±0,0068). Peringkat-2 (**256/3/0,005/0,3**) lebih meyakinkan:
> F1-nya praktis sama (0,8876, selisih 0,0002) tetapi ROC-AUC **0,9907** dan PR-AUC **0,9491**
> jelas lebih tinggi, dengan deviasi lebih kecil (±0,0026). Untuk naskah, **256/3/0,005/0,3**
> lebih tepat dilaporkan sebagai konfigurasi GNN terbaik — memilih berdasarkan F1 tunggal pada
> selisih 0,0002 adalah *overfitting* terhadap satu ambang keputusan.

Meski begitu, **model produksi tetap XGBoost** karena (a) F1-nya masih sedikit lebih tinggi,
deterministik (seed tetap → bit-per-bit sama), sementara GNN bervariasi antar-run; dan
(b) inferensi tabular jauh lebih ringan daripada membangun graf untuk setiap prediksi. GNN yang
dituning diposisikan sebagai **pembanding arsitektur yang kuat**, bukan model layanan.

> ✍️ **[UNTUK DITULIS DI TESIS — BAB 4]** Laporkan bahwa GNN telah dituning setara XGBoost
> (grid 36 konfigurasi × 2 seed = 72 training). Konfigurasi terbaik (hidden=256, 3-layer,
> lr=0,005, dropout=0,3) mencapai F1 0,8876 dengan ROC-AUC 0,9907 — praktis setara XGBoost
> (0,8911). Kesimpulan: keunggulan XGBoost tipis dan konsisten, sedangkan GNN kompetitif namun
> dibatasi sparsity graf (§14.2) — **bukan** oleh kurang tuning, karena tuning penuh hanya
> memindahkan F1 sebesar 2,3 poin dan tetap tidak menembus model tabular.

### 17.8 Reproduksi

```powershell
python ml\audit_gaps.py         # §17.1–17.5 audit merge/deskriptif/fitur/missingness/salah-input
python ml\smote_experiment.py   # §16.4 uji SMOTE/oversampling -> smote_experiment.json
python ml\tune_gnn.py           # §17.7 tuning GNN (36 konfigurasi x 2 seed) -> gnn_tuning_results.json
python ml\evaluate_roc.py       # §16.5 ROC + PR-AUC -> roc_*.png, pr_curve_comparison.png
```
