# TAHAP 5 — Model dan Eksperimen 4 × 4

File: **`ml/train_dataset_scenarios.py`**

## 1. Kenapa empat model ini, bukan yang lain

Susunannya bukan acak — ada logika bertingkat. Jelaskan seperti tangga.

### Logistic Regression — baseline linier (`train_lr`, `:289-295`)
```python
scaler = StandardScaler()
lr = LogisticRegression(max_iter=1000, random_state=42, class_weight="balanced")
```
Model paling sederhana: menarik satu garis lurus pemisah di ruang fitur.
**Fungsinya sebagai pembanding dasar** — kalau model canggih tidak jauh lebih baik
dari LR, berarti masalahnya memang mudah dan kecanggihan tidak perlu.
Butuh `StandardScaler` karena LR sensitif skala (ingat `sent_received_ratio` yang bisa 10⁹).
`class_weight="balanced"` menangani ketidakseimbangan 1:6,4.

### Random Forest — baseline ensemble (`train_rf`, `:298-308`)
```python
RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=2,
                       class_weight="balanced", n_jobs=-1, random_state=42)
```
200 pohon keputusan yang dilatih pada sampel & fitur acak, lalu hasilnya divoting.
Sudah bisa menangkap hubungan **non-linier** dan **interaksi antar fitur**. Tidak butuh
scaling. Fungsinya: membuktikan bahwa kenaikan performa datang dari *non-linieritas*,
bukan dari trik XGBoost saja.

### XGBoost — model untuk data tabular (`train_xgb`, `:311-334`)
```python
XGBoostModel(n_estimators=250, max_depth=6, learning_rate=0.05,
             subsample=0.85, colsample_bytree=0.85)
xgb.fit(X_fit, y_fit, X_val, y_val, early_stopping_rounds=40)
```
Bedanya dengan Random Forest: pohon dibangun **berurutan**, tiap pohon baru memperbaiki
kesalahan gabungan pohon-pohon sebelumnya (*gradient boosting*).
Untuk data tabular berukuran ribuan baris, ini memang standar industri.

Perhatikan `train_test_split` **di dalam** `train_xgb` (`:312-318`): 20% dari data
training dipotong lagi jadi validasi untuk *early stopping*. **Data test tidak
disentuh.** Ini benar dan bisa Anda banggakan.

### GNN-GraphSAGE — model untuk data graph (`train_gnn`, `:337-377`)
```python
gnn = GraphPhiV2(model_type="sage", input_dim=len(feature_cols),
                 hidden_dim=128, num_layers=2)
```
Satu-satunya model yang **tidak** memperlakukan tiap wallet sebagai baris independen.
Ia melihat siapa tetangga wallet tersebut. Rinciannya di TAHAP 6.

**Pola argumennya:** linier → ensemble → boosting → graph. Tiap tangga menambahkan
satu kemampuan baru. Kalau performa tidak naik di suatu tangga, itu temuan, bukan kegagalan.

## 2. Arti "4 model × 4 skenario"

**Skenario = kombinasi sumber data**, bukan kombinasi model. Didefinisikan di
`ml/artifacts/hybrid_feature_schema.json` dan dipakai lewat `load_scenarios()` (`:265`).

| Skenario | Sumber | Jumlah fitur |
|---|---|---|
| `muldigraph_only` | graph saja | 19 |
| `muldigraph_plus_etherscan` | graph + Etherscan | 26 |
| `muldigraph_plus_bigquery` | graph + BigQuery | 29 |
| `muldigraph_plus_etherscan_plus_bigquery` | ketiganya | 36 |

4 × 4 = **16 kali pelatihan**, semuanya di atas **split yang sama persis**
(`:435-440`, `random_state=42`, `stratify`). Karena splitnya identik, perbedaan angka
murni berasal dari (a) model yang dipakai atau (b) fitur yang tersedia — bukan dari
kebetulan pembagian data.

Desain ini menjawab **dua pertanyaan penelitian sekaligus**:
- Baris-baris tabel → *"model mana yang terbaik?"*
- Kolom-kolom tabel → *"apakah menambah sumber data eksternal membantu?"*

## 3. Hasil sebenarnya (angka asli dari `ml/artifacts/dataset_scenarios_comparison.csv`)

F1-score, dibulatkan 4 desimal:

| Skenario (n fitur) | LogReg | RandomForest | **XGBoost** | GNN-SAGE-V2 |
|---|---|---|---|---|
| muldigraph_only (19) | 0,7850 | 0,8685 | **0,8857** | 0,8432 |
| **muldigraph + etherscan (26)** | 0,8540 | 0,8884 | **0,9102** ⭐ | 0,8960 |
| muldigraph + bigquery (29) | 0,7879 | 0,8708 | **0,8838** | 0,8333 |
| muldigraph + etherscan + bigquery (36) | 0,8546 | 0,8973 | **0,9038** | 0,8753 |

Metrik lengkap pemenang (skenario 2, XGBoost):
**Accuracy 0,9750 · Precision 0,8862 · Recall 0,9356 · F1 0,9102**
Confusion matrix: TN 1458, FP 28, FN 15, TP 218.

### Empat kesimpulan yang bisa Anda tarik (dan pertahankan)

1. **XGBoost menang di keempat skenario.** Konsisten, bukan kebetulan satu percobaan.
2. **Etherscan membantu, BigQuery tidak.** Graph→+Etherscan naik +0,0245 F1.
   Graph→+BigQuery justru turun sedikit (−0,0019). Menambah BigQuery ke skenario terbaik
   malah **menurunkan** F1 (0,9102 → 0,9038).
3. **Kenapa BigQuery tidak membantu?** Karena fitur `bq_*` sebagian besar **mengulang
   informasi yang sudah ada**: `bq_tx_count_in` ≈ `in_degree`, `bq_total_in_eth` ≈
   `total_received`. Menambah 10 kolom redundan hanya menambah dimensi dan peluang
   overfit. Ini temuan yang bagus — jangan disembunyikan, justru jadikan bahasan.
4. **Recall (0,9356) > Precision (0,8862).** Model lebih rela salah menuduh (28 FP)
   daripada melewatkan penipu (15 FN). Untuk kasus keamanan, itu trade-off yang benar.
   Sebutkan ini kalau ditanya "kenapa presisinya lebih rendah?" — itu pilihan, bukan cacat.

## 4. ⚠️ Satu kelemahan metodologi yang WAJIB Anda ketahui duluan

Di `train_gnn` (`:369-375`):
```python
gnn.fit(graph, train_mask=train_mask, test_mask=test_mask,
        epochs=200, early_stopping_patience=25)
```

Dan di dalam `GraphPhiV2.fit` (`ml/gnn_model_v2.py:339-346`), early stopping
memonitor **F1 pada test set**, lalu mengembalikan bobot terbaik menurut test F1.

**Artinya angka GNN di tabel 4×4 mengandung kebocoran (data leakage):** test set ikut
menentukan kapan pelatihan berhenti. Angka GNN di sana **optimistis**.

Kabar baiknya: **Anda sudah memperbaikinya sendiri.**
`experiments/run_graphsage_etherscan_late_fusion_v2.py` secara eksplisit memakai
validasi internal dari data training saja, dan test set hanya di-*score* sekali.
Hasil GraphSAGE yang jujur: **F1 0,8189** (bukan 0,8432/0,8960).

**Cara menyampaikannya ke dosen (posisi kuat, bukan defensif):**
> "Pada eksperimen 4×4 awal, early stopping GNN masih memantau test F1 sehingga angkanya
> optimistis. Saya menemukan itu, lalu membuat eksperimen V2 yang leakage-controlled.
> Dengan protokol yang benar, GraphSAGE graph-only mendapat F1 0,8189 — dan kesimpulan
> utama tesis tidak berubah, karena XGBoost tetap unggul dengan selisih yang justru lebih besar."

Menemukan kelemahan sendiri dan memperbaikinya adalah nilai plus akademik. Yang buruk
adalah kalau penguji yang menemukannya lebih dulu.

## 5. Memakai XGBoost ≠ menciptakan algoritma

Ini harus tegas di kepala Anda.

`ml/xgboost_model.py` adalah **kelas pembungkus** (`class XGBoostModel`) di sekitar
library `xgboost` — mengatur parameter, memanggil `.fit()`, menyimpan model, menghitung
metrik. Tidak ada satu pun rumus baru di sana.

Analogi yang bisa dipakai saat sidang:
> "Memakai XGBoost itu seperti memakai mesin diesel yang sudah ada untuk membuat kapal.
> Saya tidak menciptakan mesin diesel. Yang saya rancang adalah kapalnya: lambungnya,
> sistem kemudinya, dan bahan bakar apa yang saya masukkan."

Apa yang **benar-benar** Anda rancang:
- Definisi 18 fitur dan dua rumus rasio.
- Empat skenario sumber data dan protokol perbandingannya.
- Arsitektur late fusion 128+7 = 135 dimensi.
- Desain eksperimen ablasi.
- Seluruh sistem penyajian dan pencatatan on-chain.

## 6. Kapan sebuah penelitian boleh disebut apa

Tabel ini adalah inti persiapan TAHAP 8. Pelajari dengan serius.

| Istilah | Syarat minimal | Bukti yang harus ada | Anda memenuhinya? |
|---|---|---|---|
| **Komparasi model** | ≥2 algoritma dibandingkan pada data & protokol yang sama | Tabel perbandingan metrik | ✅ Penuh (4 model × 4 skenario) |
| **Optimasi model** | Pencarian hyperparameter yang sistematis + laporan hasilnya | Log tuning + parameter terbaik | ✅ `ml/tune_xgboost.py`, `ml/tune_gnn.py`, `ml/artifacts/*_tuning_results.json` |
| **Pengembangan model** | Ada perubahan pada **arsitektur/formulasi** model — bukan sekadar mengganti parameter — dan dibuktikan lebih baik dari model asalnya | Arsitektur baru + ablasi + pembanding | ⚠️ **Sebagian.** Late fusion adalah arsitektur rakitan Anda dan sudah diablasi — tapi **kalah** dari XGBoost biasa (0,8825 < 0,9102). Klaim "pengembangan model" jadi rapuh karena hasilnya tidak lebih baik. |
| **Pengembangan sistem** | Model tertanam dalam aplikasi yang berjalan end-to-end, ada antarmuka, API, penyimpanan hasil | Sistem yang bisa dijalankan | ✅ **Kuat.** Streamlit + FastAPI + XGBoost + web3.py + smart contract, semuanya jalan |
| **Pengembangan metode baru** | Formulasi/algoritma baru yang bisa dibuktikan secara umum, bukan hanya pada satu dataset | Turunan matematis / bukti / generalisasi lintas dataset | ❌ Tidak. Semua komponen adalah library standar |

**Kesimpulan yang jujur dan menguntungkan Anda:**
kekuatan penelitian ini ada di kolom **"komparasi + optimasi + pengembangan sistem"**,
bukan di "pengembangan model". Judul sebaiknya mencerminkan kekuatan itu.

## 7. Model produksi ≠ model eksperimen

Jangan sampai tertukar saat presentasi.

| | Eksperimen tesis | Produksi (yang dilayani API) |
|---|---|---|
| File | `ml/train_dataset_scenarios.py` | `ml/train_production_model.py` |
| Isi | 16 pelatihan, semua model | 1 model saja: XGBoost skenario 2 |
| Output | tabel CSV + confusion matrix | `model_prod_xgb.json` + `prod_feature_schema.json` + `prod_demo.csv` |
| Tujuan | menjawab pertanyaan penelitian | melayani prediksi ke pengguna |

`train_production_model.py:50-67` sengaja menyalin **split dan konfigurasi yang persis
sama** dengan eksperimen, supaya angka yang dilaporkan ke pengguna
(F1 0,9102) benar-benar angka yang sama dengan yang ada di tesis. Ini detail rapi
yang layak disebut — banyak tesis gagal di titik ini.

---

## 3 pertanyaan cek pemahaman

1. Kalau XGBoost menang di keempat skenario, apa yang bisa Anda simpulkan yang **tidak**
   bisa disimpulkan kalau ia menang hanya di satu skenario?
2. Skenario 4 (36 fitur) punya fitur paling banyak tapi F1-nya lebih rendah dari
   skenario 2 (26 fitur). Apa penjelasannya, dan istilah teknisnya apa?
3. Kenapa penelitian Anda **belum** memenuhi syarat "pengembangan model", padahal Anda
   sudah merancang arsitektur late fusion sendiri?
