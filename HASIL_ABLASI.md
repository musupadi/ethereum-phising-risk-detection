# Hasil Ablasi: GraphSAGE Embedding + XGBoost (No Etherscan Ablation)

> Ablasi penutup untuk rangkaian eksperimen late fusion. Melengkapi `HASIL_LATE_FUSION.md` (V1) dan `HASIL_LATE_FUSION_V2.md` (V2).
> Dijalankan: 29 Juli 2026 · Skrip: `experiments/run_embedding_only_ablation.py`
> Status: eksperimen baru bersufiks `_ablation` — artefak V1 dan V2 tidak tersentuh.

---

## 1. Ringkasan Eksekutif

| Aspek | Hasil |
|---|---|
| F1 embedding-only XGBoost | **81,84%** |
| Kontribusi penggantian kepala linear → XGBoost | **−0,06 poin** |
| Kontribusi 7 fitur Etherscan | **+6,41 poin** |
| Kesimpulan | **Cabang Etherscan menyumbang nilai tambah yang terukur.** Hampir seluruh peningkatan Late Fusion V2 berasal dari fitur Etherscan, bukan dari penggantian classifier. |
| Encoder | Tidak dilatih ulang — bobot V2 dimuat, identitasnya dibuktikan lewat `assert` |

---

## 2. Mengapa Ablasi Ini Diperlukan

Di V2, Late Fusion (F1 88,25%) mengungguli baseline GraphSAGE graph-only (F1 81,89%) sebesar **+6,35 poin**. Tetapi **dua hal berubah bersamaan**:

- **(a)** kepala klasifikasi linear GraphSAGE diganti XGBoost, dan
- **(b)** tujuh fitur Etherscan ditambahkan.

Selama keduanya bercampur, klaim "fusi Etherscan memperbaiki GraphSAGE" tidak bisa dipertanggungjawabkan. Ablasi ini membuang **(b)** dan tidak mengubah apa pun yang lain, sehingga kedua efek terpisah:

```
baris 2  GraphSAGE graph-only        : embedding → kepala linear
baris 3  Ablasi (eksperimen ini)     : embedding → XGBoost            ← mengisolasi efek (a)
baris 4  Late Fusion V2              : embedding + Etherscan → XGBoost ← menambahkan efek (b)

efek (a) = baris 3 − baris 2
efek (b) = baris 4 − baris 3
```

---

## 3. Protokol

- **Encoder GraphSAGE tidak dilatih ulang.** Bobot dimuat dari `models/graphsage_encoder_for_fusion_v2.pt` dan dijalankan dalam mode `eval()`, sehingga statistik berjalan BatchNorm dibekukan dan embedding 128-d pra-head identik dengan V2.
- Split identik (6.876 train / 1.719 test, `random_state=42`), topologi identik (264 edge unik MulDiGraph), scaler dan protokol kebocoran identik.
- **Identitas encoder dibuktikan, bukan diasumsikan.** Prediksi kepala klasifikasi encoder yang dimuat di-skor ulang dan di-`assert` sama dengan baris GraphSAGE di `outputs/late_fusion_metrics_v2.json` — toleransi 1e-9 pada setiap metrik. Kalau bobot yang dimuat keliru, skrip berhenti.
- Input XGBoost adalah embedding 128-d **saja**. Tidak ada satu pun fitur Etherscan yang masuk.
- Baris 1 (XGBoost Skenario 2) dan baris 4 (Late Fusion V2) **dibaca dari file metrik V2**, tidak dihitung ulang.
- Resep XGBoost sama persis dengan baseline: 250 estimator, depth 6, lr 0,05, subsample/colsample 0,85, early stopping 40 pada slice validasi internal train. Test di-skor sekali.

---

## 4. Hasil: Perbandingan Empat Model (test held-out, n = 1.719)

| # | Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | XGBoost tabular (Skenario 2) | 26 fitur tabular | **97,50%** | **88,62%** | **93,56%** | **91,02%** | **0,9948** | **0,9701** | 1.458 | **28** | 15 | 218 |
| 2 | GraphSAGE graph-only (leakage-controlled) | 19 fitur graf + 264 edge | 94,88% | 78,66% | 85,41% | 81,89% | 0,9714 | 0,8880 | 1.432 | 54 | 34 | 199 |
| 3 | **GraphSAGE Embedding + XGBoost (ablasi)** | embedding 128-d saja | 94,71% | 76,49% | 87,98% | **81,84%** | 0,9820 | 0,9222 | 1.423 | 63 | 28 | 205 |
| 4 | GraphSAGE–Etherscan Late Fusion V2 | embedding 128 + Etherscan 7 = 135-d | 96,68% | 84,92% | 91,85% | 88,25% | 0,9916 | 0,9529 | 1.448 | 38 | 19 | 214 |

### Tabel atribusi

| Efek | Perubahan F1 |
|---|---|
| (a) Kepala linear → XGBoost (baris 2 → 3) | **−0,06 poin** |
| (b) Penambahan 7 fitur Etherscan (baris 3 → 4) | **+6,41 poin** |
| Total Late Fusion V2 atas baseline graf (baris 2 → 4) | +6,35 poin |
| Sisa jarak model ablasi ke XGBoost Skenario 2 (baris 3 → 1) | −9,18 poin |

---

## 5. Interpretasi

### 5.1 Putusan sesuai aturan interpretasi

Aturan yang ditetapkan **sebelum** melihat hasil: "mendekati 88,25%" dioperasionalkan sebagai **selisih ≤ 1,00 poin** dari F1 Late Fusion V2. Selisih teramati: **6,41 poin** → masuk kategori *materially lower*.

Maka putusan resminya: **cabang Etherscan menyumbang nilai tambah yang terukur, di luar efek penggantian classifier.** Penggantian kepala linear dengan XGBoost, dengan sendirinya, praktis tidak menghasilkan apa-apa (−0,06 poin).

### 5.2 Koreksi terhadap dugaan di laporan V2 §5.2

Laporan V2 menduga — berdasarkan audit *gain* yang menunjukkan blok Etherscan hanya 1,8% dari total gain — bahwa mayoritas peningkatan berasal dari penggantian classifier. **Dugaan itu keliru, dan ablasi ini membuktikannya secara langsung.**

Pelajaran metodologisnya layak masuk tesis: ***gain share* bukan ukuran atribusi kausal.** Gain mengukur seberapa banyak *total pengurangan impurity* yang dikreditkan ke sebuah fitur di seluruh pohon — bukan seberapa besar dampak fitur itu terhadap metrik akhir. Ketujuh fitur Etherscan memang dipakai di sedikit split, tetapi split-split itu berada di posisi yang menentukan: FP turun dari 63 → 38 dan FN turun dari 28 → 19 ketika blok itu ditambahkan. Satu-satunya cara mengukur kontribusi modalitas secara sahih adalah **ablasi**, yaitu mencabutnya lalu melatih ulang — persis yang dilakukan di sini.

### 5.3 Temuan halus: ranking membaik, F1 tidak

Bandingkan baris 3 dengan baris 2. F1 nyaris identik (81,84% vs 81,89%), tetapi metrik berbasis peringkat naik jelas:

| Metrik | GraphSAGE (kepala linear) | Emb-only XGBoost | Δ |
|---|---|---|---|
| ROC-AUC | 0,9714 | 0,9820 | **+0,0106** |
| PR-AUC | 0,8880 | 0,9222 | **+0,0342** |
| Recall | 85,41% | 87,98% | +2,57 poin |
| Precision | 78,66% | 76,49% | −2,17 poin |

Artinya XGBoost **memang** menghasilkan pengurutan probabilitas yang lebih baik atas embedding yang sama — PR-AUC naik +0,034 bukan angka sepele pada data seimbang 1:6,4. Hanya saja, pada ambang tetap 0,5, keunggulan itu tertukar habis antara recall dan precision sehingga F1 tidak bergerak. Jadi pernyataan yang paling akurat bukanlah "penggantian classifier tidak berguna", melainkan: **penggantian classifier memperbaiki kualitas ranking tetapi tidak memperbaiki F1 pada ambang default.** Kalau ambang keputusan dioptimalkan pada data validasi, baris 3 kemungkinan besar akan melampaui baris 2 — ini bahan eksperimen lanjutan yang murah.

### 5.4 Apa artinya untuk keseluruhan rangkaian eksperimen

Setelah tiga eksperimen, gambarannya menjadi konsisten dan bisa dipertahankan di sidang:

1. **Fitur Etherscan adalah sumber sinyal yang paling menentukan.** Fitur ini menaikkan F1 di jalur tabular (Skenario 1 → 2) *dan* di jalur fusi (+6,41 poin). Kontribusinya nyata, terukur, dan konsisten di dua arsitektur berbeda.
2. **Cabang graf lemah karena topologinya jarang, bukan karena metodenya salah.** Dengan hanya 264 edge unik untuk 8.595 node, mayoritas node hanyalah pulau ber-self-loop, sehingga GraphSAGE berdegenerasi mendekati MLP. Embedding-nya, dengan classifier terbaik sekalipun, mentok di ~82% F1 — 9,18 poin di bawah fitur tabular langsung.
3. **Fusi tingkat embedding bekerja sesuai desain, tetapi tidak cukup untuk melampaui 91,02%.** Ia memulihkan sebagian besar jarak (81,84% → 88,25%) justru dengan mengembalikan informasi tabular yang sebelumnya dicabut — bukan dengan menambah informasi graf baru. Ini indikasi kuat bahwa yang menahan performa adalah kemiskinan topologi, bukan mekanisme fusinya.
4. **Konsekuensi praktisnya:** keputusan men-deploy XGBoost tabular sebagai model produksi tetap benar dan sekarang didukung tiga bukti independen, bukan sekadar satu perbandingan.

Arah lanjutan yang paling menjanjikan tetap sama: **fusi dengan 26 fitur tabular penuh + embedding (155-d)**, karena hanya konfigurasi itu yang tidak membuang informasi apa pun. Ditambah satu percobaan murah: **optimasi ambang keputusan pada validasi** (§5.3).

---

## 6. Pengendalian Kebocoran & Verifikasi

Seluruh butir ditegakkan sebagai `assert` (gagal → eksekusi berhenti):

| # | Pemeriksaan | Hasil |
|---|---|---|
| 1 | Split direproduksi identik dengan V2 (train 6.876 / test 1.719, `random_state=42`) | ✓ |
| 2 | Topologi direproduksi identik dengan V2 (264 edge unik MulDiGraph; `etherscan_edges.csv` tidak dibaca) | ✓ |
| 3 | StandardScaler fitur graf di-fit hanya pada baris TRAIN (sama seperti V2) | ✓ |
| 4 | Encoder dimuat dari checkpoint V2, **tidak dilatih ulang**, mode eval (statistik BatchNorm beku) | ✓ |
| 5 | Encoder yang dimuat mereproduksi metrik test GraphSAGE V2 **persis** (toleransi 1e-9 di 10 metrik) → identitas encoder terkonfirmasi | ✓ |
| 6 | Embedding = output `SAGEConv → BatchNorm → ReLU` terakhir (pra-head, 128-d); logit/probabilitas tidak dipakai | ✓ |
| 7 | **Tidak ada fitur Etherscan** yang masuk model ini — embedding adalah satu-satunya input | ✓ |
| 8 | XGBoost dilatih pada embedding TRAIN saja; slice validasi internal diambil dari TRAIN; test di-skor sekali | ✓ |
| 9 | Prediksi tersimpan V2 disejajarkan ulang berdasarkan `wallet_address` sebelum menggambar kurva | ✓ |

Seed: numpy = torch = `random_state` = 42. Tanpa sampel sintetis. Ketidakseimbangan kelas ditangani lewat `scale_pos_weight`.

---

## 7. Reproduksi

```powershell
.\.venv\Scripts\Activate.ps1
python experiments\run_embedding_only_ablation.py
```

± 5 detik untuk melatih XGBoost, di luar ± 1 menit memuat `MulDiGraph.pkl` (diperlukan untuk merekonstruksi topologi yang sama sebelum memuat encoder).

Prasyarat: artefak V2 harus sudah ada (`models/graphsage_encoder_for_fusion_v2.pt` dan `outputs/late_fusion_metrics_v2.json`). Skrip berhenti dengan pesan jelas kalau belum.

---

## 8. Berkas yang Dihasilkan

**`outputs/`**

| Berkas | Isi |
|---|---|
| `embedding_only_ablation_metrics.json` | Metrik empat model, tabel atribusi, putusan aturan interpretasi, audit gain, daftar pemeriksaan |
| `four_model_comparison_ablation.csv` | Tabel perbandingan empat model + kolom Input |
| `embedding_only_confusion_matrix_ablation.csv` | Confusion matrix model ablasi |
| `embedding_only_predictions_ablation.csv` | Prediksi per wallet test |
| `embedding_only_confusion_matrix_ablation.png` | Gambar confusion matrix |
| `four_model_comparison_ablation.png` | Diagram batang precision / recall / F1 empat model |
| `roc_curve_ablation.png` · `pr_curve_ablation.png` | Kurva ROC & PR (GraphSAGE vs ablasi vs fusi V2) |
| `embedding_only_ablation_report.md` | Laporan versi Inggris (dihasilkan otomatis oleh skrip) |

**`models/`**

| Berkas | Isi |
|---|---|
| `graphsage_embedding_only_xgboost_ablation.pkl` | Classifier XGBoost pada embedding 128-d |

Artefak V1 dan V2, berkas tesis, jurnal, `ml/`, serta `ml/artifacts/` tidak tersentuh. Satu-satunya berkas lama yang disunting adalah `HASIL_LATE_FUSION_V2.md` §5.2 — dokumen laporan buatan sendiri, bukan artefak eksperimen — untuk mencatat bahwa dugaan atribusinya sudah dikoreksi oleh ablasi ini.
