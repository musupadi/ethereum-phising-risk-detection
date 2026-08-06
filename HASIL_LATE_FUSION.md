# Hasil Eksperimen: GraphSAGE–Etherscan Late Fusion

> Eksperimen tambahan (fusi graf–tabular tingkat embedding) di atas Skenario 2 (MulDiGraph + Etherscan).
> Dijalankan: 29 Juli 2026 · Skrip: `experiments/run_graphsage_etherscan_late_fusion.py`
> Status: **eksperimen baru** — tidak mengubah metrik, artefak, maupun kesimpulan eksperimen yang sudah ada.

> ⚠️ **ANGKA DI DOKUMEN INI SUDAH USANG (per 6 Agustus 2026).** Dijalankan pada populasi
> **8.595 wallet** dengan baseline **XGBoost 26 fitur, F1 91,02%**. Eksperimen utama kini memakai
> dataset **1:10 penuh (12.815 wallet)** dan skema **25 fitur** dengan baseline **F1 89,11%**
> (lihat `hasil_implementasi.md` §5.5). Jalur late-fusion belum dijalankan ulang — **jangan kutip
> angkanya ke naskah tesis** sampai `experiments/run_graphsage_etherscan_late_fusion.py`
> dijalankan ulang. Kesimpulan utamanya (late fusion tidak mengungguli XGBoost tabular) tidak
> terpengaruh perubahan ini.

---

## 1. Ringkasan Eksekutif

| Aspek | Hasil |
|---|---|
| Jenis fusi | **Embedding-level late fusion** (graph–tabular), bukan konkatenasi fitur awal dan bukan rata-rata skor prediksi |
| Dimensi fusi | 128 (embedding GraphSAGE) + 7 (Etherscan) = **135** |
| F1 late fusion | **90,27%** |
| F1 XGBoost tabular (baseline) | **91,02%** |
| F1 GraphSAGE (baseline) | **89,60%** |
| Kesimpulan | Late fusion **belum mengungguli** XGBoost tabular (−0,75 poin F1), tetapi **mengungguli** GraphSAGE murni (+0,67 poin F1) |

**Pernyataan penting:** tidak ada klaim peningkatan terhadap XGBoost 91,02%. Hasil pada data uji held-out tidak mendukung klaim tersebut.

---

## 2. Desain Eksperimen

### 2.1 Posisi fusi dalam pipeline

```
MulDiGraph (topologi + label)
        │
        ├─ 26 fitur Skenario 2 ──► GraphSAGE (2 layer, hidden 128)
        │                                │
        │                                ├─ [layer terakhir: SAGEConv → BatchNorm → ReLU]
        │                                │        │
        │                                │        └──► z_i  (embedding 128-d)  ◄── DIAMBIL DI SINI
        │                                │
        │                                └─ Linear(128, 2) → logit   ◄── TIDAK dipakai
        │
        └─ 7 fitur Etherscan ──► StandardScaler (fit di TRAIN saja) ──► e_i (7-d)

                 fused_i = [ z_i ‖ e_i ]  →  135-d  →  XGBoost  →  prediksi
```

Fusi terjadi **setelah GraphSAGE menghasilkan embedding graf dan sebelum XGBoost mengklasifikasi wallet**. Embedding diambil dari lapisan tersembunyi terakhir sebelum kepala klasifikasi — **bukan** logit dan **bukan** probabilitas prediksi.

### 2.2 Sumber data dan label

- **EPTransNet/MulDiGraph tetap menjadi sumber wallet dan label.** Semesta wallet adalah 8.595 alamat unik dataset hybrid: **1.165 phishing / 7.430 normal**.
- **Tidak ada wallet yang ditambahkan dari Etherscan dan tidak ada label yang diubah.**
- **Etherscan hanya menyumbang tujuh fitur pengayaan tabular**, di-join berdasarkan `wallet_address` yang sudah dinormalisasi ke huruf kecil: `created_at_ts`, `balance`, `transaction_count`, `activity_score`, `avg_tx_value`, `tx_frequency`, `unique_counterparties`.

### 2.3 Split data

Split stratifikasi 80:20 dengan `random_state=42` — **panggilan yang identik** dengan Skenario 2 pada `ml/train_dataset_scenarios.py`.

| | Jumlah | Phishing | Normal |
|---|---|---|---|
| Train | 6.876 | 932 | 5.944 |
| Test (held-out) | 1.719 | 233 | 1.486 |

Validitas split terkonfirmasi: kedua baseline yang dilatih ulang di skrip ini mereproduksi **persis** angka yang sudah dipublikasikan sebelumnya (XGBoost F1 91,02%; GraphSAGE F1 89,60%).

### 2.4 Konfigurasi

| Komponen | Konfigurasi |
|---|---|
| Encoder | `GraphPhiV2(model_type="sage", input_dim=26, hidden_dim=128, num_layers=2)` — kelas yang sudah ada, tidak dimodifikasi |
| Edge | 654 edge transaksi nyata (528 MulDiGraph + 126 Etherscan), 350 unik setelah deduplikasi + self-loop |
| Early stopping encoder | Patience 25, dipantau pada **slice validasi dari dalam TRAIN** (5.500 fit / 1.376 val) |
| Classifier fusi | XGBoost `n_estimators=250, max_depth=6, lr=0,05, subsample=0,85, colsample_bytree=0,85`, early stopping 40 — resep identik dengan baseline |
| Seed | numpy = torch = `random_state` = 42 |
| Perangkat | CPU |

---

## 3. Hasil pada Data Uji Held-out (n = 1.719)

| Model | Accuracy | Precision | Recall | F1-score | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost tabular (Skenario 2) | **97,50%** | **88,62%** | 93,56% | **91,02%** | **0,9948** | **0,9701** | 1.458 | 28 | 15 | 218 |
| GraphSAGE (Skenario 2) | 96,97% | 83,90% | **96,14%** | 89,60% | 0,9861 | 0,9225 | 1.443 | 43 | **9** | **224** |
| **GraphSAGE–Etherscan Late Fusion** | 97,27% | 87,20% | 93,56% | 90,27% | 0,9921 | 0,9592 | 1.454 | 32 | 15 | 218 |

### Confusion matrix late fusion

|  | Prediksi Normal | Prediksi Phishing |
|---|---|---|
| **Aktual Normal** | 1.454 (TN) | 32 (FP) |
| **Aktual Phishing** | 15 (FN) | 218 (TP) |

### Laporan dimensi

| Komponen | Dimensi |
|---|---|
| Embedding GraphSAGE (lapisan tersembunyi pra-head) | **128** |
| Fitur tabular Etherscan | **7** |
| **Vektor fusi akhir** | **135** |
| *(Referensi)* fitur tabular Skenario 2 | 26 (19 graf + 7 Etherscan) |

---

## 4. Interpretasi

### 4.1 Apakah late fusion meningkatkan F1 terhadap XGBoost 91,02%?

**Tidak.** Late fusion mencapai F1 **90,27%**, yaitu **−0,75 poin** di bawah XGBoost tabular. Selisih ini setara **4 false positive tambahan** (32 vs 28) pada jumlah true positive dan false negative yang persis sama (218 TP, 15 FN). Recall kedua model identik (93,56%); seluruh perbedaan berasal dari precision (87,20% vs 88,62%). ROC-AUC dan PR-AUC juga sedikit lebih rendah (0,9921 vs 0,9948; 0,9592 vs 0,9701).

Karena hasil pada data uji held-out tidak mendukung, **tidak ada klaim peningkatan yang dibuat**.

### 4.2 Terhadap GraphSAGE murni, fusi jelas membantu

Dibandingkan GraphSAGE Skenario 2, fusi menaikkan F1 dari 89,60% → 90,27% (**+0,67 poin**) dan precision dari 83,90% → 87,20% (**+3,30 poin**), dengan FP turun dari 43 → 32. Artinya: mengganti kepala klasifikasi linear GraphSAGE dengan XGBoost di atas embedding yang sama memang memperbaiki kualitas keputusan — meskipun ditebus dengan recall yang turun dari 96,14% → 93,56% (FN naik 9 → 15).

### 4.3 Mengapa fusi tidak melampaui XGBoost tabular

Analisis *gain* pada model fusi menjelaskan penyebabnya: dari 130 fitur yang benar-benar dipakai pohon, **fitur Etherscan mentah hanya menyumbang 1,2% total gain**; sisanya (98,8%) berasal dari dimensi embedding, dan empat dimensi saja (`sage_emb_50`, `sage_emb_0`, `sage_emb_120`, `sage_emb_108`) menguasai ~77% gain.

| Fitur teratas | Gain share |
|---|---|
| `sage_emb_50` | 27,8% |
| `sage_emb_0` | 21,5% |
| `sage_emb_120` | 18,3% |
| `sage_emb_108` | 9,2% |
| `eth_unique_counterparties` (fitur Etherscan terbaik) | 0,6% |

Ini masuk akal secara metodologis: ketujuh fitur Etherscan **sudah menjadi bagian dari 26 fitur input GraphSAGE**, sehingga informasinya sudah terserap ke dalam embedding. Blok Etherscan yang di-concat menjadi sebagian besar redundan, dan XGBoost akhirnya belajar dari representasi 128-d yang sudah terkompresi — bukan dari nilai fitur asli yang memungkinkan split bernilai-tepat. Kompresi ini bersifat *lossy*: model kehilangan sedikit ketajaman batas keputusan yang justru menjadi kekuatan XGBoost tabular.

### 4.4 Implikasi

- Untuk dataset ini, **fitur tabular langsung masih superior** dibanding representasi graf terkompresi. Sinyal fitur tabular pada 8.595 wallet sudah sangat kuat, sementara topologi transaksi antar-wallet sangat jarang (hanya 350 edge unik di antara 8.595 node) sehingga message passing memberi informasi tambahan yang terbatas.
- Arah perbaikan yang layak diuji jika eksperimen dilanjutkan: (a) melatih encoder GraphSAGE **hanya pada 19 fitur graf** agar blok Etherscan pada vektor fusi benar-benar membawa informasi baru; (b) memperkaya edge (topologi lebih padat) sebelum berharap embedding graf unggul; (c) fusi dengan XGBoost yang menerima 26 fitur tabular **penuh** ditambah embedding (155-d), bukan hanya 7 fitur Etherscan.
- Nilai ilmiah eksperimen ini tetap ada: hasilnya menunjukkan bahwa **keunggulan XGBoost pada Skenario 2 bukan artefak pemilihan classifier**, karena classifier yang sama di atas representasi graf tetap tidak melampauinya.

---

## 5. Pengendalian Kebocoran Data & Verifikasi Penjajaran

Seluruh pemeriksaan berikut ditegakkan sebagai `assert` di dalam skrip (gagal → eksekusi berhenti), bukan sekadar komentar:

| # | Pemeriksaan | Hasil |
|---|---|---|
| 1 | Himpunan wallet train dan test saling lepas (disjoint) | ✓ 6.876 / 1.719 |
| 2 | Seluruh 8.595 `wallet_address` unik dan sudah lowercase | ✓ |
| 3 | Train + test mencakup tepat semua wallet dataset (tidak ada wallet ditambah/dibuang) | ✓ |
| 4 | StandardScaler input GraphSAGE di-fit hanya pada baris TRAIN | ✓ |
| 5 | Early stopping GraphSAGE memakai slice validasi internal TRAIN (5.500 fit / 1.376 val); node test dikecualikan dari loss dan pemilihan model | ✓ |
| 6 | Embedding diambil dari output `SAGEConv → BatchNorm → ReLU` terakhir (pra-head, 128-d); logit/probabilitas tidak dipakai | ✓ |
| 7 | Urutan node graf = `[wallet train…, wallet test…]`; baris embedding sejajar berdasarkan posisi | ✓ |
| 8 | Nilai Etherscan yang kosong di-*zero-fill*, sama persis dengan praproses Skenario 2 | ✓ |
| 9 | StandardScaler Etherscan di-fit hanya pada TRAIN, lalu di-*transform* ke TEST | ✓ |
| 10 | Baris embedding, baris Etherscan, label, dan indeks split merujuk wallet yang sama pada urutan yang sama | ✓ |

**Catatan transduktif.** Encoder GraphSAGE dilatih secara transduktif di atas topologi graf penuh — perilaku yang sama dengan baseline GraphSAGE yang sudah ada, sehingga sengaja dipertahankan. Namun **label test tidak pernah dipakai**: hanya label node train yang masuk ke fungsi loss, dan early stopping digerakkan oleh slice validasi dari dalam train.

**Perbedaan yang disengaja terhadap baseline.** Baseline GraphSAGE lama (`ml/train_dataset_scenarios.py:train_gnn`) memantau **F1 test** untuk early stopping. Baris baseline pada tabel di atas direproduksi apa adanya agar tetap sebanding dengan angka tesis yang sudah dipublikasikan (dan memang menghasilkan 89,60% yang identik). Encoder late fusion **tidak** melakukan hal itu — perbedaan ini adalah pengetatan metodologis, dan konsekuensinya angka late fusion diperoleh di bawah kontrol kebocoran yang lebih ketat daripada baseline GraphSAGE pembandingnya.

**Tidak ada sampel sintetis** yang dihasilkan (tanpa SMOTE/oversampling); ketidakseimbangan kelas ditangani lewat `class_weight` pada loss GNN dan `scale_pos_weight` pada XGBoost.

---

## 6. Reproduksi

```powershell
.\.venv\Scripts\Activate.ps1
python experiments\run_graphsage_etherscan_late_fusion.py
```

Waktu eksekusi ± 1 menit di CPU, di luar ± 30 detik untuk memuat `MulDiGraph.pkl`.

## 7. Berkas yang Dihasilkan

**`outputs/`**

| Berkas | Isi |
|---|---|
| `late_fusion_metrics.json` | Seluruh metrik, dimensi, konfigurasi split, dan daftar pemeriksaan kebocoran |
| `late_fusion_comparison_table.csv` | Tabel perbandingan tiga model |
| `late_fusion_confusion_matrix.csv` | Confusion matrix late fusion |
| `late_fusion_predictions.csv` | Prediksi per wallet test (`wallet_address`, `y_true`, `y_pred`, `y_proba_phishing`) |
| `late_fusion_confusion_matrix.png` | Gambar confusion matrix |
| `late_fusion_roc_curve.png` | Kurva ROC tiga model |
| `late_fusion_pr_curve.png` | Kurva precision–recall tiga model |
| `late_fusion_model_comparison.png` | Diagram batang precision / recall / F1 |
| `late_fusion_report.md` | Ringkasan versi Inggris (dihasilkan otomatis oleh skrip) |

**`models/`**

| Berkas | Isi |
|---|---|
| `graphsage_encoder_for_fusion.pt` | Bobot encoder GraphSAGE penghasil embedding |
| `graphsage_etherscan_late_fusion_xgboost.pkl` | Classifier XGBoost pada vektor 135-d |
| `late_fusion_etherscan_scaler.pkl` | StandardScaler Etherscan (fit pada train) |

Tidak ada berkas tesis, jurnal, `ml/`, maupun `ml/artifacts/` yang dimodifikasi.
