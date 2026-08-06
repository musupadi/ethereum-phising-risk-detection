# Penjelasan Analisis Data — Hybrid Dataset

> Hasil pengecekan nyata pada `data/processed/hybrid_dataset.csv` untuk menjawab
> catatan bimbingan (dedup wallet, data kosong, outlier, imbalance).
> Dijalankan tanggal 2026-07-04. Semua angka di bawah ini hasil eksekusi langsung, bukan asumsi.
>
> **Update 2026-07-11:** seluruh tindakan yang direncanakan di sini **sudah dikerjakan** dan
> hasilnya sudah diverifikasi (run ulang). Tiap bagian di bawah kini ditandai statusnya, dan
> hasil lengkapnya terdokumentasi di `hasil_implementasi.md §16`.
>
> ⚠️ **Update 2026-08-06 — angka di dokumen ini sudah usang.** Pengecekan ini dilakukan pada
> `hybrid_dataset.csv` versi **8.595 baris × 26 fitur**. Dataset kini dibangun ulang menjadi
> **12.815 baris (1:10) × 25 fitur** (`graph_present` dikeluarkan dari daftar predictor menjadi
> kolom audit). Angka yang berubah antara lain: total baris, imbalance (1:6,4 → **1:10**),
> jumlah nilai di-capping (2.127 → **3.202**), dan coverage BigQuery (0,93% → **0,98%**).
> **Versi terkini dari seluruh analisis ini ada di `hasil_implementasi.md` §16 dan §17**, yang
> sudah dijalankan ulang pada dataset baru — rujuk ke sana, bukan ke angka di bawah.
> Kesimpulan kualitatifnya tidak berubah: 0 duplikat, 0 NaN, 0 nilai negatif, sebaran skewed
> kanan, nol bersifat struktural.

---

## Ringkasan cepat (untuk dibaca dosen)

| Pertanyaan bimbingan | Hasil cek | Status |
|----------------------|-----------|--------|
| Ada wallet duplikat? | **Tidak ada.** 8.595 baris, `node_id` & `address` semuanya unik | ✅ bersih |
| Ada data kosong (NaN)? | **Tidak ada NaN** di 26 fitur produksi | ✅ bersih |
| Ada salah input (nilai negatif)? | **Tidak ada** nilai negatif | ✅ bersih |
| Data balance/imbalance? | **Imbalance**: 7.430 normal vs 1.165 phishing (**13,55%** phishing) | ✅ ditangani `scale_pos_weight`=6,38 |
| Ada fitur mati/konstan? | **Ya** — `created_at_ts` = 0 di **semua** baris | ✅ sudah dibuang |
| Ada outlier? | **Ya, banyak** (~15–24% per fitur pakai IQR) | ✅ sudah di-capping (2.127 nilai) |

> **Status keseluruhan (per 2026-07-11): semua sudah dikerjakan.** Rincian tindakan &
> hasilnya ada di bagian masing-masing (✅) dan di `hasil_implementasi.md §16`.

---

## 1. Dataset secara umum

- **Ukuran:** 8.595 baris × 41 kolom (26 fitur model + 15 kolom meta/BigQuery).
- **Sumber:** seluruhnya berlabel `source_dataset = EPTransNet`.
- **Label:** biner — `0` = normal, `1` = phishing.

## 2. Deduplikasi wallet (catatan poin 1)

Dicek berdasarkan `node_id` (alamat wallet, di-lowercase) **dan** kolom `address`:

```
total baris      : 8.595
node_id unik     : 8.595   → duplikat = 0
address unik     : 8.595   → duplikat = 0
```

**Kesimpulan:** tidak ada wallet yang sama/ganda. Dataset gabungan MulDiGraph+Etherscan
**sudah ter-deduplikasi**. Untuk paper: cukup tulis "dilakukan pengecekan duplikat berdasarkan
alamat wallet, tidak ditemukan baris ganda sehingga tidak ada yang dihapus."

## 3. Data kosong / missing value (catatan poin 6, 11)

- **NaN (benar-benar kosong):** 0 pada seluruh 26 fitur → tidak perlu imputasi.
- **Yang perlu dijelaskan:** banyak nilai **0**, tapi ini **bukan** data hilang — ini
  **nol struktural**. Contoh:
  - `total_sent = 0` (5.654 baris) → wallet tidak pernah mengirim transaksi keluar.
  - `in_degree = 0` (4.661 baris) → wallet tidak pernah menerima.
  - `std_out_amount = 0` (7.121 baris) → hanya 0/1 transaksi keluar, jadi tidak ada sebaran.

  Nol ini **bermakna** (perilaku wallet), jadi **tidak dihapus / tidak diimputasi**.

- **Baris nyaris kosong:** ada **20 baris** yang ≥22 dari 26 fiturnya bernilai 0 (aktivitas
  sangat minim). Ini kandidat untuk ditinjau/dibuang saat cleaning karena informasinya tipis.

**Untuk paper:** "Tidak ditemukan missing value (NaN). Nilai nol pada fitur transaksi bersifat
struktural (wallet tanpa aktivitas tertentu) sehingga dipertahankan; sebanyak 20 baris dengan
aktivitas sangat minim ditinjau sebagai kandidat noise."

## 4. Fitur mati / tidak berguna (catatan poin 3, 5)

**Temuan penting:** `created_at_ts` bernilai **0 di seluruh 8.595 baris** → fitur konstan,
**nol informasi** untuk model. **Rekomendasi: buang** (langsung mengurangi 26 → 25 fitur tanpa
kehilangan apa pun). Ini bukti konkret bahwa seleksi fitur memang perlu.

Fitur lain dengan banyak nol (mis. `std_out_amount`, `timestamp_spread_in`) tetap dipertahankan
dulu — akan diuji relevansinya di tahap seleksi fitur (importance XGBoost/RF).

## 5. Salah input (catatan poin 7)

Dicek nilai negatif pada semua fitur numerik (balance, jumlah transaksi, dsb. tidak boleh
negatif secara logika): **tidak ditemukan nilai negatif**. Tidak ada indikasi salah input tipe ini.

## 6. Outlier (catatan poin 4, dan bagian data-centric)

Deteksi outlier metode **IQR** (di luar `Q1−1.5·IQR` s/d `Q3+1.5·IQR`). Fitur paling banyak outlier:

| Fitur | Jumlah outlier | % |
|-------|----------------|---|
| min_out_amount | 2.046 | 23,8% |
| timestamp_spread_out | 1.964 | 22,9% |
| total_sent | 1.837 | 21,4% |
| max_out_amount | 1.809 | 21,0% |
| balance | 1.738 | 20,2% |
| total_received | 1.713 | 19,9% |

Contoh sebaran ekstrem (persis kasus yang dosen sebut "balance 10 juta lalu ada 1 yang 100 juta"):

| Fitur | min | median | mean | max |
|-------|-----|--------|------|-----|
| balance | 0 | 0,000005 | 0,52 | **2.576** |
| total_received | 0 | 0 | 22,2 | **27.068** |
| transaction_count | 0 | 8 | 51 | **10.000** |
| in_degree | 0 | 0 | 7,9 | **11.066** |

Sebaran sangat **skewed** (mean ≫ median, max jauh dari mean). Karena outlier **banyak
(~20%)**, sesuai arahan dosen **tidak dibuang**, melainkan **diproses**:
- **Capping / winsorize** (mis. batasi di persentil 1–99), dan/atau
- **Transformasi log** (`log1p`) agar skala wajar.

Ini adalah inti pendekatan **data-centric**: merapikan data, bukan membuang wallet.

> ✅ **SUDAH DILAKUKAN (2026-07-11).** Winsorize/capping **persentil p1–p99** dijalankan lewat
> `ml/clean_hybrid_dataset.py` → **2.127 nilai di-cap** pada 23 fitur numerik, **tanpa membuang
> satu baris pun** (tetap 8.595 baris). Threshold p1/p99 di-*fit* pada data train saja
> (anti-leakage) dan disimpan di `ml/artifacts/cleaning_report.json`. Bukti visual sebelum vs
> sesudah capping ada di `ml/artifacts/eda/03_boxplot_before_after.png`. Dokumentasi lengkap:
> `hasil_implementasi.md §16.1–16.2`.

## 7. Imbalance kelas (catatan poin 8, 12)

```
normal (0)   : 7.430  (86,45%)
phishing (1) : 1.165  (13,55%)
```

Data **tidak seimbang**. Poin penting untuk dijawab ke dosen:
- Model produksi saat ini (`ml/train_production_model.py`) **melatih XGBoost pada data penuh
  yang imbalanced** (8.595 baris, stratified split) — jadi **sudah tidak pakai undersampling**
  di jalur produksi. Eksperimen undersampling (`sampling_experiment.json`) hanya untuk LR & RF.
- Rencana: bandingkan tiga strategi → (a) apa adanya, (b) `scale_pos_weight` (bawaan XGBoost
  untuk imbalance, tanpa buang data), (c) undersampling. Pilih yang F1-nya terbaik + alasan.
  Ini menjawab arahan "boosting harusnya bisa menangani data unbalance tanpa undersampling".

> ✅ **SUDAH DILAKUKAN (2026-07-11).** Ketiga strategi dibandingkan lewat `ml/tune_xgboost.py`.
> Hasil F1: **`scale_pos_weight` = 0,9065** (menang) > `sample_weight` = 0,9047 >
> **undersampling 1:5 = 0,8973**. Nilai `scale_pos_weight = 7.430/1.165 ≈ 6,38`. Kesimpulan:
> XGBoost menangani imbalance **tanpa membuang data** dan hasilnya lebih baik dari undersampling.
> Detail + penjelasan mekanisme & analogi: `hasil_implementasi.md §16.4`.

---

## 8. Kesimpulan & tindakan berikutnya

**Sudah bersih (tidak perlu tindakan):** duplikat, NaN, nilai negatif.

**Sudah dikerjakan (per 2026-07-11):**
1. ✅ **Buang `created_at_ts`** (fitur konstan 0) → 26 → 25 fitur. (`cleaning_report.json`)
2. ✅ **Tinjau baris nyaris kosong** — dipertahankan karena nol bersifat struktural (bukan noise acak); tidak ada baris dibuang.
3. ✅ **Capping** fitur berskala ekstrem — winsorize p1–p99, 2.127 nilai di-cap. (`§16.1`)
4. ✅ **Uji strategi imbalance** — `scale_pos_weight` (0,9065) menang atas undersampling (0,8973). (`§16.4`)
5. ✅ **Seleksi fitur & tuning** — dijalankan; F1 produksi 0,9102 (AUC 0,9948). Target ≥0,95 **tidak tercapai** (plafon realistis ~0,91), namun **lolos batas minimum 85%**. (`§16.3–16.5`)

> Detail rencana lengkap tiap tahap ada di `RENCANA_BIMBINGAN.md`; hasil eksekusi & verifikasi
> ada di `hasil_implementasi.md §16`.
