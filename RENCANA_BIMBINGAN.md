# Rencana Kerja — Tindak Lanjut Bimbingan

> Rekap catatan bimbingan (dari file `Hasil.kumpulan data itu wallet yang sama mak`) yang
> sudah dirapikan menjadi daftar tugas konkret. Dipetakan ke kondisi kode saat ini.
> Fokus utama dosen: **data centris** (data bersih, bebas noise, outlier tertangani) +
> **naikkan F1 dari 91% menuju target ≥ 95% (minimal 85%)** + **penjelasan asal skor**.

Tanggal disusun: 2026-07-04

---

## 0. Kondisi saat ini (fakta dari repo)

| Item | Nilai sekarang |
|------|----------------|
| Dataset produksi | `data/processed/hybrid_dataset.csv` — **8.595 baris**, **26 fitur** + kolom meta |
| Model dilayani | XGBoost (`ml/artifacts/model_prod_xgb.json`), skenario `muldigraph_plus_etherscan` |
| Metrik saat ini | **F1 = 0.9102**, acc 0.975, precision 0.886, recall 0.936 |
| Skema fitur | `ml/artifacts/prod_feature_schema.json` (26 fitur, urut) |
| Eksperimen sampling | `ml/artifacts/sampling_experiment.json` — baru LR & RF, rasio 1:5 & 1:10 |
| Cara skor | `ml/predict.py`: `score = round(proba*100)`; LOW <40, MEDIUM 40–69, HIGH ≥70 |

Catatan penting: dosen menyebut "25 fitur", kode sebenarnya **26 fitur**. Perlu diklarifikasi
di paper (angka pastinya 26 pada skenario produksi).

---

## 1. Rekap catatan bimbingan (18 poin, dirapikan)

1. **Deduplikasi wallet** — kalau ada wallet yang sama (duplikat) harus dihapus. Bandingkan
   antara sumber MulDiGraph vs Etherscan supaya tidak ada baris ganda.
2. **Hyperparameter tuning XGBoost** — lakukan tuning, buktikan bisa lebih baik dari 91%.
3. **Seleksi fitur** — dari 26 fitur, kurangi (mis. ke ~20) pakai metode seleksi (feature
   importance dari Gradient Boosting / Random Forest) untuk cari fitur paling berguna.
4. **Deteksi outlier saat preprocessing** — cek statistik deskriptif tiap fitur (min, max,
   mean); tandai nilai yang jauh dari rata-rata (mis. balance 100 juta padahal mayoritas 10 juta).
5. **Kaji setiap fitur** — pelajari 26 fitur, tentukan mana yang benar-benar berguna.
6. **Cek record kosong** — periksa transaksi/kolom kosong. Kalau dihapus, jelaskan alasannya;
   kalau tidak, jelaskan alasannya. Data harus bersih & bebas noise.
7. **Data salah input** — deteksi entri salah input, perlakukan sebagai outlier.
8. **Analisis balance vs imbalance** — tampilkan distribusi kelas penuh; jelaskan **kenapa
   undersampling dan bukan oversampling** (atau sebaliknya).
9. **Feature engineering** — dokumentasikan fitur turunan.
10. **Scaling** — jelaskan/normalisasi fitur.
11. **Missing value handling** — cek per kolom (dari 26 kolom, mis. hanya 4 terisi → buang;
    kalau kosong 1–2 field → imputasi mean). Tulis metodenya di paper.
12. **Boosting untuk data imbalance** — XGBoost seharusnya bisa menangani data tak seimbang
    tanpa undersampling (pakai `scale_pos_weight` / bagging-boosting). Bandingkan.
13. **Skenario XGBoost vs GNN** — buat skenario perbandingan yang fokus.
14. **Kurva ROC / AUC** — pastikan ada, dan valid untuk klasifikasi biner seperti dataset ini.
15. **Asal skor risiko** — jelaskan dari mana skor berasal (proba klasifikasi 0→1 dari 26 fitur
    penting), bagaimana proba diubah jadi risk score & level. Tulis di paper.
16. **Deploy model terbaik** — model bagus disimpan & di-deploy; input data → model deteksi
    phishing / tidak.
17. **Kategorisasi biner** — output jelas: **Phishing** atau **Bukan Phishing**.
18. **Target deployment** — F1 minimal **95%** (kalau tidak tercapai, minimal **85%**).

**Tambahan (paragraf bawah):** pastikan data RAW yang digabung bersih & bebas noise, fitur
konsisten. Pendekatan **data-centric**: tampilkan fitur mana yang nilainya ekstrem/tidak rata
(mis. distribusi balance), tentukan cara membatasinya (capping). Visualkan 8.595 baris hybrid
untuk lihat outlier — kalau cuma 1 yang aneh boleh dibuang, kalau banyak harus diproses.

---

## 2. Rencana kerja bertahap

Diurutkan sesuai alur pipeline. Tiap tahap punya **deliverable** (artefak + bagian paper).

### Tahap A — Data Quality & Cleaning (poin 1, 6, 7, 11) ✅ SELESAI
- [x] **A1. Deduplikasi wallet** → 0 duplikat (sudah unik). `ml/clean_hybrid_dataset.py`.
- [x] **A2. Audit missing value** → 0 NaN; nilai 0 bersifat struktural (dipertahankan).
- [x] **A3. Deteksi salah input** → 0 nilai negatif. Fitur mati `created_at_ts` dibuang.
- **Deliverable:** `hybrid_dataset_clean.csv`, `hybrid_dataset_capped.csv`,
  `cleaning_report.json`. Ditulis di `hasil_implementasi.md` §16.1. ✅

### Tahap B — EDA, Outlier & Statistik Deskriptif (poin 4, 8, data-centric) ✅ SELESAI
- [x] **B1/B2. Statistik & outlier** → outlier ~15–24% (IQR), di-capping p1–p99.
- [x] **B3. Distribusi kelas** → 7.430 : 1.165 (imbalance 1 : 6,4).
- **Deliverable:** `ml/artifacts/eda/01..05_*.png`. Ditulis di `hasil_implementasi.md` §16.2–16.3. ✅

### Tahap C — Feature Engineering, Scaling & Seleksi Fitur (poin 3, 5, 9, 10) ✅ SELESAI
- [x] **C1. Kaji fitur** → peringkat importance (XGBoost gain + RF).
- [x] **C2. Scaling/capping** di-fit pada train saja (anti-leakage).
- [x] **C3. Seleksi fitur** 25 → **top_20** (F1 0,8953 ≥ full_25 0,8921). `graph_present`,
      `in_degree`, `tx_frequency` redundan/lemah.
- **Deliverable:** `feature_selection_report.json`, `eda/06_feature_importance.png`. ✅

### Tahap D — Modeling, Tuning & Imbalance (poin 2, 8, 12) ✅ SELESAI
- [x] **D1. Hyperparameter tuning XGBoost** — `RandomizedSearchCV` (60×5-fold) pada top_20.
      CV F1 0.9295; F1 uji 0.9050 → **0.9065** (naik tipis). **Belum 95%, lolos min 85%.**
- [x] **D2. Imbalance handling** — `scale_pos_weight` (F1 0.9065) **> undersampling** (0.8973)
      dan sample_weight (0.9047). **XGBoost menangani imbalance tanpa buang data (poin 12 ✓).**
- **Deliverable:** `ml/tune_xgboost.py`, `ml/artifacts/xgb_tuning_results.json`. ✅

### Tahap E — Evaluasi & ROC/AUC (poin 13, 14) ✅ SELESAI
- [x] **E1. Kurva ROC + AUC** — XGBoost **AUC 0.9948**, GNN AUC 0.9880 (biner, valid ✓).
- [x] **E2. Skenario XGBoost vs GNN** — XGBoost menang F1 (0.9102 vs 0.8863) & AUC.
- **Deliverable:** `ml/evaluate_roc.py`, `roc_xgb.png`, `roc_comparison.png`,
  `roc_auc_report.json`. ✅

### Tahap F — Penjelasan Skor & Deployment (poin 15, 16, 17, 18) ✅ SELESAI
- [x] **F1. Dokumentasi asal skor** — proba×100 = score; mapping level; **kontribusi fitur SHAP**
      (`ml/predict.py:explain()`, `pred_contribs` bawaan XGBoost) di API + UI.
- [x] **F2. Output biner tegas** — `is_phishing`/`verdict` di API; banner Phishing/Bukan
      Phishing di UI. File: `api/schemas.py`, `api/main.py`, `app/streamlit_app.py`.
- [x] **F3. Deploy model terbaik** — XGBoost 26 fitur (F1 0.9102, AUC 0.9948) tetap sebagai
      model produksi karena mengungguli hasil tuning 20-fitur (0.9065). Lolos min 85%.
- **Deliverable:** ditulis di `hasil_implementasi.md` §16.4–16.7. ✅

---

## 3. Urutan eksekusi yang disarankan

```
A (cleaning) → B (EDA/outlier) → C (fitur/scaling/seleksi) → D (tuning/imbalance)
→ E (ROC/AUC & skenario) → F (skor & deploy)
```

Alasan: tiap tahap jadi input tahap berikutnya. Target akhir tiap perubahan model harus
diukur ulang F1-nya dan dicatat, supaya jelas peningkatan dari baseline **0.9102**.

---

## 4. Yang perlu dikonfirmasi ke dosen / diputuskan (setelah D–F selesai)
- **Target F1 95%:** tidak tercapai pada data ini — plafon realistis **F1 ≈ 0.91 / AUC ≈ 0.99**.
  Model **lolos batas minimum 85%**. Minta persetujuan menerima fallback 85%+ (sesuai poin 18),
  atau arahan menambah fitur baru bila 95% wajib.
- **Jumlah fitur final:** produksi tetap **26 fitur** (F1 0.9102) karena mengungguli hasil
  seleksi 20-fitur (F1 0.9065). Perlu konfirmasi apakah dosen ingin memaksa versi 20-fitur
  demi kesederhanaan meski F1 sedikit lebih rendah.
- **Strategi imbalance:** disarankan **`scale_pos_weight` (tanpa undersampling)** — terbukti
  F1 tertinggi & tak membuang data. Tinggal konfirmasi.

Semua hasil A–F terdokumentasi jadi satu di `hasil_implementasi.md` §16.
