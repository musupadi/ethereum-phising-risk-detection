# Hasil Eksperimen V2: GraphSAGE–Etherscan Late Fusion (Leakage-Controlled, Non-Redundant)

> Versi perbaikan dari eksperimen V1 (`HASIL_LATE_FUSION.md`). V1 **tidak** ditimpa; seluruh output V2 memakai sufiks `_v2`.
> Dijalankan: 29 Juli 2026 · Skrip: `experiments/run_graphsage_etherscan_late_fusion_v2.py`
> Status: eksperimen baru — tidak mengubah metrik, artefak, maupun kesimpulan tesis yang sudah ada.

> ⚠️ **ANGKA DI DOKUMEN INI SUDAH USANG (per 6 Agustus 2026).** Dijalankan pada populasi
> **8.595 wallet** (split 6.876/1.719) dengan baseline **XGBoost 26 fitur, F1 91,02%** — termasuk
> `assert` yang mengunci reproduksi ke angka tersebut. Eksperimen utama kini memakai dataset
> **1:10 penuh (12.815 wallet, split 10.252/2.563)** dan skema **25 fitur** dengan baseline
> **F1 89,11%** (lihat `hasil_implementasi.md` §5.5). Konsekuensinya
> `experiments/run_graphsage_etherscan_late_fusion_v2.py` **akan gagal di `assert`-nya** bila
> dijalankan pada dataset baru — nilai acuan di skrip perlu diperbarui lebih dulu. **Jangan kutip
> angka di dokumen ini ke naskah tesis** sampai eksperimen dijalankan ulang.

---

## 1. Ringkasan Eksekutif

| Aspek | Hasil |
|---|---|
| F1 Late Fusion V2 | **88,25%** |
| vs XGBoost tabular Skenario 2 (91,02%) | **−2,78 poin** → tidak ada peningkatan |
| vs GraphSAGE graph-only leakage-controlled (81,89%) | **+6,35 poin** → peningkatan besar |
| Dimensi fusi | 128 (embedding, dari 19 fitur graf saja) + 7 (Etherscan) = **135** |
| Topologi | **264 edge terarah unik** dari MulDiGraph saja (528 entri mentah); edge Etherscan dikecualikan |
| Kebocoran | Tidak ada sinyal test (F1/loss/ROC-AUC/PR-AUC) dipakai untuk early stopping atau pemilihan model — untuk **semua** model |

**Pernyataan penting:** fusi terbukti memperbaiki cabang graf secara substansial, tetapi **tetap belum mengungguli XGBoost tabular**. Tidak ada klaim peningkatan terhadap 91,02%.

---

## 2. Tiga Koreksi terhadap V1

| # | Masalah V1 | Koreksi V2 |
|---|---|---|
| 1 | GraphSAGE menerima 26 fitur Skenario 2 (termasuk 7 fitur Etherscan), lalu 7 fitur yang sama di-concat lagi sebelum XGBoost → **fusi redundan** | Input GraphSAGE dibatasi ke **19 fitur graf (Skenario 1)**. Fitur Etherscan masuk ke model **tepat satu kali**, yaitu di cabang tabular. Disjoint-nya ditegakkan lewat `assert`. |
| 2 | Baseline GraphSAGE lama memantau **F1 test** untuk early stopping | **Tidak ada sinyal test** yang dipantau oleh model manapun. Early stopping digerakkan slice validasi dari dalam data train (5.500 fit / 1.376 val). Test di-skor **sekali saja**. |
| 3 | Topologi mencampur edge MulDiGraph dan edge turunan Etherscan | Topologi **hanya edge transaksi langsung EPTransNet/MulDiGraph**. `data/raw/etherscan_edges.csv` tidak pernah dibaca. |

### Klarifikasi angka edge

Angka "528 + 126" di laporan V1 adalah **entri edge mentah**, yang menghitung setiap edge dua kali (sekali dari pemindaian successor, sekali dari predecessor). V2 melaporkan keduanya secara terpisah:

| Item | Nilai |
|---|---|
| Entri edge terarah mentah (MulDiGraph) | 528 |
| **Edge terarah unik yang masuk ke GNN** | **264** |
| ├─ train–train | 150 |
| ├─ train–test | 95 |
| └─ test–test | 19 |
| Self-loop ditambahkan builder PyG | 8.595 |
| Alamat dataset yang ada di MulDiGraph | 8.595 / 8.595 |

Bandingkan V1: 654 entri mentah gabungan → 350 edge unik. Jadi V2 memangkas topologi dari 350 → 264 edge unik, konsekuensi wajar dari mengeluarkan sumber Etherscan.

---

## 3. Desain V2

```
MulDiGraph (topologi 264 edge + label)
        │
        ├─ 19 fitur graf ──► GraphSAGE (2 layer, hidden 128)
        │                          ├─ hidden pra-head ──► z_i (128-d)  ──┐
        │                          └─ Linear(128,2) ──► BASELINE GraphSAGE
        │                                                                │
        └─ 7 fitur Etherscan ──► StandardScaler (fit TRAIN) ──► e_i ────┤
                                                                         │
                              fused_i = [ z_i ‖ e_i ] = 135-d ──► XGBoost ──► Late Fusion V2
```

- Populasi 8.595 wallet (1.165 phishing / 7.430 normal); wallet dan label **hanya** dari EPTransNet.
- Split stratifikasi 80:20, `random_state=42` → 6.876 train / 1.719 test — identik Skenario 2 (ditegakkan `assert`).
- Embedding diambil dari output `SAGEConv → BatchNorm → ReLU` terakhir, sebelum `Linear(128,2)`. Bukan logit, bukan probabilitas.

**Catatan desain penting.** Baris baseline GraphSAGE dan baris fusi memakai **encoder terlatih yang sama persis**: baseline adalah prediksi kepala klasifikasi encoder itu sendiri, fusi adalah XGBoost di atas embedding pra-head-nya. Dengan begitu satu-satunya yang berubah antar dua baris adalah classifier + blok Etherscan — menjadikannya ablasi bersih atas langkah fusi.

---

## 4. Hasil pada Data Uji Held-out (n = 1.719)

| Model | Input | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC | TN | FP | FN | TP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| XGBoost tabular (Skenario 2) | 26 fitur tabular | **97,50%** | **88,62%** | **93,56%** | **91,02%** | **0,9948** | **0,9701** | 1.458 | **28** | **15** | **218** |
| GraphSAGE graph-only (leakage-controlled) | 19 fitur graf + 264 edge | 94,88% | 78,66% | 85,41% | 81,89% | 0,9714 | 0,8880 | 1.432 | 54 | 34 | 199 |
| **GraphSAGE–Etherscan Late Fusion V2** | embedding 128 ‖ Etherscan 7 = 135-d | 96,68% | 84,92% | 91,85% | 88,25% | 0,9916 | 0,9529 | 1.448 | 38 | 19 | 214 |

Baseline XGBoost Skenario 2 tereproduksi **persis** angka publikasi (97,50 / 88,62 / 93,56 / 91,02) — ditegakkan lewat `assert` dengan toleransi 5e-4, jadi kalau split pernah bergeser skrip akan gagal, bukan diam-diam melaporkan angka lain.

---

## 5. Interpretasi

### 5.1 Apakah fusi meningkatkan XGBoost?

**Tidak.** 88,25% vs 91,02% (**−2,78 poin**). Fusi menghasilkan 10 false positive lebih banyak (38 vs 28) dan 4 false negative lebih banyak (19 vs 15). ROC-AUC dan PR-AUC juga lebih rendah (0,9916 vs 0,9948; 0,9529 vs 0,9701). Hasil held-out tidak mendukung klaim peningkatan, sehingga klaim itu tidak dibuat.

Perhatikan bahwa V2 **lebih buruk** dari V1 (88,25% vs 90,27%) terhadap XGBoost. Ini konsisten dan dapat dijelaskan: V1 "curang secara tidak sengaja" — encoder-nya sudah melihat 7 fitur Etherscan, jadi embedding V1 membawa informasi tabular yang di V2 sengaja dicabut. V2 adalah angka yang jujur untuk fusi non-redundan.

### 5.2 Apakah fusi meningkatkan GraphSAGE?

**Ya, signifikan.** 88,25% vs 81,89% (**+6,35 poin**), precision +6,26 poin, recall +6,44 poin, FP turun 54 → 38, FN turun 34 → 19. Inilah temuan utama V2: begitu redundansi dihilangkan, langkah fusi benar-benar memberi nilai tambah pada cabang graf — sesuatu yang tidak bisa dibuktikan V1 karena di sana blok Etherscan sebagian besar duplikat.

**Kaveat atribusi.** Peningkatan +6,35 poin ini pada awalnya tidak bisa diatribusikan, karena dua hal berubah sekaligus: (a) kepala klasifikasi linear diganti XGBoost, dan (b) blok Etherscan ditambahkan. Audit *gain* menunjukkan blok Etherscan hanya menyumbang **1,8%** total gain (embedding 98,2%).

> **Pembaruan — kaveat ini sudah terselesaikan.** Ablasi lanjutan (`HASIL_ABLASI.md`) menjalankan XGBoost di atas embedding 128-d **tanpa** blok Etherscan dan mendapat F1 **81,84%** — praktis sama dengan baseline GraphSAGE (81,89%). Artinya penggantian classifier menyumbang **−0,06 poin** dan blok Etherscan **+6,41 poin**. Kesimpulannya terbalik dari dugaan berbasis *gain share* di atas: **hampir seluruh peningkatan V2 berasal dari fitur Etherscan, bukan dari penggantian classifier.** Angka *gain share* 1,8% di tabel berikut ternyata ukuran atribusi yang menyesatkan — ketujuh fitur Etherscan dipakai di sedikit split saja, tetapi split-split itulah yang menentukan (FP turun 63 → 38).

| Fitur teratas model fusi | Gain share |
|---|---|
| `sage_emb_89` | 39,8% |
| `sage_emb_107` | 11,1% |
| `sage_emb_127` | 10,8% |
| `sage_emb_19` | 4,5% |
| *(seluruh 7 fitur Etherscan digabung)* | **1,8%** |

Ablasi pemisah (a) dan (b) sudah dijalankan — lihat `HASIL_ABLASI.md` dan blok pembaruan di atas.

### 5.3 Mengapa baseline GraphSAGE V2 (81,89%) jauh di bawah angka publikasi (89,60%)?

Dua hal berubah sekaligus terhadap baseline lama, dan keduanya menurunkan angka:

1. **Fitur berkurang** dari 26 → 19 (fitur Etherscan dicabut dari input encoder).
2. **Early stopping tidak lagi memantau F1 test.** Baseline lama memilih epoch terbaik berdasarkan data uji — ini optimis secara sistematis.

Karena berubah bersamaan, **81,89% bukan koreksi terhadap 89,60%** dan tidak membatalkannya; keduanya sekadar mengukur dua model yang berbeda di bawah protokol yang berbeda. Yang bisa dikatakan dengan aman: angka 89,60% diperoleh dengan protokol early stopping yang lebih longgar, sehingga sebaiknya tidak dibandingkan langsung dengan angka V2 manapun. Jika ingin membandingkan apel dengan apel, jalankan ulang baseline lama (26 fitur, 350 edge) dengan protokol validasi internal V2.

### 5.4 Implikasi

- Untuk dataset ini, **fitur tabular langsung tetap superior** dibanding representasi graf. Penyebab struktural terlihat jelas di audit edge: hanya **264 edge unik untuk 8.595 node** (kerapatan ≈ 0,003 edge/node), dan hanya 19 edge yang menghubungkan sesama node test. Message passing praktis nyaris tidak punya tetangga untuk diagregasi — mayoritas node adalah pulau yang hanya punya self-loop, sehingga GraphSAGE berdegenerasi menjadi MLP pada 19 fitur graf.
- Karena itu, kesimpulan yang paling kuat dan aman untuk tesis adalah: **keterbatasan bukan pada metode fusi, melainkan pada kerapatan topologi**. Fusi tingkat embedding sudah diimplementasikan dengan benar dan terkontrol; ia tidak bisa mengungguli fitur tabular selama graf sejarang ini.
- Arah lanjutan berdasarkan bukti di atas: (a) ~~ablasi XGBoost-atas-embedding untuk memisahkan kontribusi classifier vs Etherscan~~ — **sudah dijalankan**, lihat `HASIL_ABLASI.md`; (b) memperkaya topologi (ekspansi 1-hop dari MulDiGraph penuh, bukan hanya edge antar-8.595-wallet) sebelum berharap embedding graf unggul; (c) fusi dengan XGBoost yang menerima **26 fitur tabular penuh** ditambah embedding (155-d) — ini yang paling berpeluang melampaui 91,02% karena tidak membuang informasi tabular apa pun.

---

## 6. Pengendalian Kebocoran & Verifikasi Penjajaran

Seluruh butir ditegakkan sebagai `assert` di dalam skrip (gagal → eksekusi berhenti):

| # | Pemeriksaan | Hasil |
|---|---|---|
| 1 | **Non-redundansi**: input GraphSAGE (19 fitur graf) dan blok Etherscan (7 fitur) disjoint — tidak ada fitur masuk dua kali | ✓ |
| 2 | Populasi 8.595 alamat EPTransNet unik & lowercase; tidak ada wallet ditambah/dibuang | ✓ |
| 3 | Split stratifikasi 80:20 `random_state=42` → train 6.876 / test 1.719 (identik Skenario 2) | ✓ |
| 4 | Himpunan wallet train dan test saling lepas | ✓ |
| 5 | Topologi dibangun **hanya** dari MulDiGraph (264 edge unik); `etherscan_edges.csv` tidak pernah dibaca | ✓ |
| 6 | StandardScaler fitur graf di-fit hanya pada baris TRAIN | ✓ |
| 7 | Early stopping memakai slice validasi internal TRAIN (fit 5.500 / val 1.376); **F1/loss/ROC-AUC/PR-AUC test tidak pernah dipantau** | ✓ |
| 8 | Embedding = output `SAGEConv → BatchNorm → ReLU` terakhir (pra-head, 128-d); logit/probabilitas tidak dipakai | ✓ |
| 9 | Urutan node graf = `[wallet train…, wallet test…]`; baris embedding sejajar berdasarkan posisi | ✓ |
| 10 | Nilai Etherscan di-*zero-fill* konsisten; StandardScaler Etherscan di-fit hanya pada TRAIN | ✓ |
| 11 | Baris embedding, baris Etherscan, label, dan indeks split merujuk wallet sama pada urutan sama | ✓ |
| 12 | Dimensi fusi tepat 135 | ✓ |
| 13 | Baseline XGBoost Skenario 2 tereproduksi persis angka publikasi (toleransi 5e-4) | ✓ |

**Catatan transduktif.** Encoder dilatih transduktif di atas topologi penuh (sama seperti baseline yang ada), tetapi **label test tidak pernah masuk**: hanya label node train yang berkontribusi ke loss, dan pemilihan model digerakkan slice validasi internal train. Di log, `GraphPhiV2.fit` menamai mask pemantauannya `test_mask`; V2 mengirim mask **validasi** ke sana, sehingga nilai `test_f1` di log training sebenarnya adalah F1 validasi.

**Test di-skor tepat sekali.** Tidak ada tuning, threshold, maupun pemilihan epoch yang menggunakan data uji.

**Tanpa sampel sintetis.** Ketidakseimbangan kelas ditangani lewat weighted cross-entropy (GNN) dan `scale_pos_weight` (XGBoost). Seed: numpy = torch = `random_state` = 42.

---

## 7. Reproduksi

```powershell
.\.venv\Scripts\Activate.ps1
python experiments\run_graphsage_etherscan_late_fusion_v2.py
```

± 1 menit di CPU, di luar ± 1 menit untuk memuat `MulDiGraph.pkl`.

## 8. Berkas yang Dihasilkan

**`outputs/`**

| Berkas | Isi |
|---|---|
| `late_fusion_metrics_v2.json` | Seluruh metrik, dimensi, audit edge, audit gain, daftar pemeriksaan kebocoran |
| `late_fusion_comparison_table_v2.csv` | Tabel perbandingan tiga model + kolom Input |
| `late_fusion_confusion_matrix_v2.csv` | Confusion matrix late fusion |
| `confusion_matrices_all_v2.csv` | Confusion matrix ketiga model |
| `late_fusion_predictions_v2.csv` | Prediksi per wallet test (fusi + GraphSAGE) |
| `edge_source_audit_v2.json` | Audit sumber edge (dipakai/dikecualikan, mentah vs unik, breakdown partisi) |
| `feature_source_audit_v2.csv` | Audit sumber fitur per cabang |
| `late_fusion_confusion_matrix_v2.png` · `cm_graphsage_graph_only_v2.png` · `cm_xgboost_scenario2_v2.png` | Gambar confusion matrix |
| `late_fusion_roc_curve_v2.png` · `late_fusion_pr_curve_v2.png` | Kurva ROC & PR tiga model |
| `late_fusion_model_comparison_v2.png` | Diagram batang precision / recall / F1 |
| `late_fusion_report_v2.md` | Laporan versi Inggris (dihasilkan otomatis oleh skrip) |

**`models/`**

| Berkas | Isi |
|---|---|
| `graphsage_encoder_for_fusion_v2.pt` | Bobot encoder GraphSAGE (19 fitur graf) |
| `graphsage_etherscan_late_fusion_xgboost_v2.pkl` | Classifier XGBoost pada vektor 135-d |
| `late_fusion_etherscan_scaler_v2.pkl` | StandardScaler Etherscan (fit pada train) |

Artefak V1 (tanpa sufiks `_v2`), berkas tesis, jurnal, `ml/`, dan `ml/artifacts/` tidak tersentuh.
