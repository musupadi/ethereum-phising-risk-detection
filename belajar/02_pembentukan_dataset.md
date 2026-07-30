# TAHAP 2 — Pembentukan Dataset Eksperimen

## 1. Delapan istilah yang sering tertukar

Dosen sering menguji di sini. Hafalkan bedanya, dan pasangkan dengan posisi Anda.

| Istilah | Artinya | Anda melakukannya? |
|---|---|---|
| **Membuat dataset dari nol** | Anda sendiri yang mengumpulkan data mentah **dan** memberi label. Contoh: crawl transaksi Ethereum sendiri lalu memverifikasi mana yang phishing dari laporan korban. | **TIDAK** |
| **Menggunakan dataset publik** | Memakai dataset yang sudah dilabeli orang lain apa adanya. | **YA** (EPTransNet / MulDiGraph) |
| **Sampling** | Mengambil sebagian baris dari populasi besar, karena data penuh terlalu besar / tidak seimbang. | **YA** |
| **Feature extraction** | Mengubah data mentah (graph) menjadi angka-angka yang bisa dimakan model. Informasi diringkas dari bentuk aslinya. | **YA** — 18 fitur |
| **Feature engineering** | Membuat fitur turunan dari fitur yang sudah ada, pakai pengetahuan domain. Contoh: `in_out_ratio = in_degree/(out_degree+1)`. | **YA** — rasio & std |
| **Data enrichment** | Menambah kolom baru dari **sumber data lain** untuk entitas yang sama. | **YA** — Etherscan + BigQuery |
| **Preprocessing** | Membersihkan & menyiapkan: missing value, tipe data, skala, split train/test. | **YA** |
| **EDA** | Melihat-lihat data sebelum memodelkan: distribusi, outlier, korelasi. | **YA** — `ml/eda_visualize.py`, gambar di `ml/artifacts/eda/` |

### Kalimat aman untuk dosen

> "Saya tidak membuat EPTransNet dari nol. EPTransNet adalah dataset publik berlabel.
> Yang saya bangun adalah **dataset eksperimen** turunannya: saya melakukan sampling,
> feature extraction 1-hop dari graph, feature engineering rasio, lalu pengayaan
> dengan data on-chain nyata dari Etherscan dan BigQuery. Kontribusi saya ada di
> *pembentukan dataset eksperimen*, bukan di *pembuatan dataset primer*."

## 2. Ada DUA jalur pembentukan dataset di repo ini

Ini sering membingungkan. Bedakan sejak awal.

### Jalur A — legacy / baseline (`ml/preprocess.py`)

- Sumber: **hanya** MulDiGraph.
- Sampling: **semua** node phishing + node normal sebanyak `10 × jumlah phishing`
  (`NEGATIVE_SAMPLE_RATIO = 10`, `preprocess.py:51`).
  ```python
  n_sample = min(n_phishing * NEGATIVE_SAMPLE_RATIO, len(normal_nodes))
  sampled_normal = random.sample(normal_nodes, n_sample)   # seed 42
  ```
- Output: `data/processed/train.csv` + `test.csv`, 18 fitur, split 80:20 stratified.
- Dipakai oleh `ml/train.py` (LR + RF) → `model.pkl`. **Ini bukan model yang dilayani.**

### Jalur B — dataset eksperimen utama (`ml/build_hybrid_dataset.py`)

Inilah dataset yang dipakai untuk semua angka tesis Anda.

- Tabel utamanya **bukan** graph, tapi `data/raw/addresses.csv` (daftar alamat + label).
  Lihat `build_hybrid_dataset.py:215-217`.
- Lalu di-*join* ke Etherscan, ke fitur graph, dan ke BigQuery.
- `HYBRID_NORMAL_RATIO=0` (default) → **tidak ada downsampling**, semua baris dipakai
  (`build_hybrid_dataset.py:104-107` + schema `"normal_ratio": 0`).
- Hasil: **8.595 wallet**, 1.165 phishing (13,6%) dan 7.430 normal.

> **Penting untuk dijawab kalau ditanya "sampling-nya bagaimana?"**
> Rasio 10:1 di `preprocess.py` **tidak** berlaku untuk dataset hybrid.
> Dataset hybrid memakai seluruh 8.595 alamat yang tersedia di `addresses.csv`,
> dengan rasio alami ≈ 1 : 6,4. Ketidakseimbangan ditangani di sisi model
> (`class_weight="balanced"` untuk LR & RF, bobot kelas untuk GNN), bukan di sisi data.

## 3. Alur lengkap dataset eksperimen

```
EPTransNet / MulDiGraph.pkl          data/raw/addresses.csv (alamat + label)
  (2.973.489 node, jutaan edge)               │  ← TABEL UTAMA (8.595 baris)
            │                                 │
            │                    LEFT JOIN ◄──┼──► addresses_etherscan_enriched.csv
            │                    on address   │      (7 fitur: balance, tx_count, …)
            │                                 │
  extract_node_features(G, addr) ──LEFT JOIN──┤
  untuk tiap alamat di tabel utama            │
  → 18 fitur + graph_present                  │
                                              │
                                 LEFT JOIN ◄──┼──► addresses_bigquery_enriched.csv
                                              │      (10 fitur: bq_*)
                                              ▼
                          data/processed/hybrid_dataset_full.csv
                                              │  rebalance_by_ratio(ratio=0) → hanya di-shuffle
                                              ▼
                          data/processed/hybrid_dataset.csv   (8.595 × 41)
                                              │
                        train_test_split(test_size=0.2, stratify=label, seed=42)
                                              ▼
                          hybrid_train.csv (6.876)   hybrid_test.csv (1.719)
                                              │
                                              ▼
                                    training model
```

Kode join-nya, supaya bisa Anda tunjuk saat presentasi:

```python
merged_base = base_df.merge(etherscan_df[...], on="address", how="left")   # :235
merged_df   = merged_base.merge(graph_df,     on="address", how="left")   # :267
merged_df   = merged_df.merge(bq_df,          on="address", how="left")   # :278
```

## 4. Kenapa 1-hop, bukan 2-hop atau 3-hop?

Jawaban jujur berdasarkan kode (`preprocess.py:87-105`): ekstraksi hanya menyentuh
`G.predecessors(node)` dan `G.successors(node)` — itu definisi 1-hop.

Tiga alasan yang bisa Anda pertahankan:

1. **Biaya komputasi.** MulDiGraph punya ~2,9 juta node. 2-hop dari wallet ramai bisa
   meledak jadi ratusan ribu tetangga per wallet.
2. **Interpretabilitas.** Fitur 1-hop punya arti langsung ("berapa orang mengirim ke
   wallet ini"). Fitur 2-hop sulit dijelaskan ke penyidik/pengguna.
3. **Informasi multi-hop tetap tidak hilang** — justru itu tugas GraphSAGE. GraphSAGE
   2 layer = agregasi 2-hop (`train_dataset_scenarios.py:349-356`). Jadi penelitian Anda
   membandingkan "1-hop tabular" vs "2-hop message passing" secara adil.

Poin 3 adalah jawaban terbaik Anda. Pakai itu.

## 5. Posisi EDA

`ml/eda_visualize.py` menghasilkan 6 gambar di `ml/artifacts/eda/`:

| File | Menjawab pertanyaan |
|---|---|
| `01_class_distribution.png` | Seberapa tidak seimbang datanya? |
| `02_outlier_percentage.png` | Fitur mana yang paling banyak outlier? |
| `03_boxplot_before_after.png` | Efek capping outlier |
| `04_hist_by_class.png` | Fitur mana yang memisahkan phishing vs normal? |
| `05_correlation_heatmap.png` | Fitur mana yang redundan? |
| `06_feature_importance.png` | Fitur mana yang dipakai model? |

EDA dilakukan **sebelum** memutuskan preprocessing, dan hasilnya dipakai di
`ml/clean_hybrid_dataset.py` (capping) dan `ml/feature_selection.py`.

---

## 3 pertanyaan cek pemahaman

1. Kalau dosen bertanya "Anda membuat dataset sendiri?", apa jawaban satu kalimat yang
   benar **dan** tidak merendahkan kontribusi Anda?
2. Rasio 10:1 di `preprocess.py` berlaku untuk dataset yang mana, dan **tidak** berlaku
   untuk dataset yang mana?
3. Apa beda *feature extraction* dan *data enrichment* dalam konteks penelitian Anda?
   Beri satu contoh kolom untuk masing-masing.
