# Lanjutkan di PC yang punya data — fix `graph_present` (26 → 25 fitur)

> ## ✅ SELESAI — dikerjakan 6 Agustus 2026, commit `ac57cb7`
>
> Seluruh langkah 1–6 di bawah sudah dijalankan. Ringkasan hasil:
>
> - **Langkah 0** tidak diperlukan: fix kode ternyata sudah ada di repo ini.
> - **Langkah 2–3** ✅ `prod_feature_schema.json` = 25 fitur + `audit_columns: ['graph_present']`;
>   `predict.py` dan API `ml_service` jalan tanpa error.
> - **Langkah 4** ✅ flag `stale_pending_retrain` hilang otomatis saat retrain.
> - **Langkah 5** ✅ `hasil_implementasi.md` di-update penuh.
> - **Langkah 6** ✅ di-commit (72 file, 24 MB data turunan; `.env` & `.pkl` tetap ter-ignore).
>   **Belum di-push** — tinggal `git push` kalau sudah oke.
>
> ### Dua temuan yang tidak diantisipasi dokumen ini
>
> 1. **`graph_present` memang tidak berpengaruh.** Konstan bernilai 1 di semua baris → varians
>    dan importance nol. Uji isolasi: F1 0,8911 (25 fitur) vs 0,8966 (26 fitur); selisihnya
>    murni dari `colsample_bytree`, bukan sinyal. Dugaan di langkah 5 benar.
> 2. **Yang benar-benar menggeser angka adalah ukuran dataset.** PC ini punya
>    **12.815 alamat (1:10)**, sedangkan §5.5/§12/§16/§17 lama ditulis dari sampel
>    **8.595 (1:6,4)**. F1 produksi bergeser 0,9102 → **0,8911** karena kelas negatif bertambah
>    (jumlah phishing tetap 1.165), bukan karena model memburuk. Ini justru **memperbaiki**
>    inkonsistensi internal: §11 sudah lama menyatakan 1:10 sebagai rasio yang dipilih.
>
> ### Yang masih tersisa (di luar cakupan fix ini)
>
> - `experiments/run_graphsage_etherscan_late_fusion_v2.py` punya `assert` yang mengunci angka
>   lama (F1 0,9102, split 6.876/1.719) → **akan gagal** kalau dijalankan sekarang. Nilai acuannya
>   perlu diperbarui sebelum jalur late-fusion di-run ulang.
> - `data/raw/etherscan_edges.csv` (126 edge) di-crawl saat populasi masih 8.595 alamat, jadi
>   cross-edge Etherscan kemungkinan undercount. Crawl ulang butuh kuota API besar; dicatat
>   sebagai keterbatasan di §5.5.
> - `laporan belajar/*.md` sengaja tidak disentuh (catatan pribadi), tapi isinya masih menyebut
>   8.595 / 26 fitur.
>
> ---
>
> <details>
> <summary>Instruksi asli (arsip)</summary>

> Dibuat: 6 Agustus 2026, dari sesi Claude Code di PC tanpa `data/` dan tanpa `git`.
> Semua perubahan KODE sudah selesai dan sudah diverifikasi (lolos `py_compile` + JSON valid),
> tapi **belum di-retrain dan belum di-commit** karena PC ini tidak punya `MulDiGraph.pkl`,
> CSV enrichment Etherscan/BigQuery, maupun `git` terpasang.

## Konteks singkat

Bug: `graph_present` (flag audit "alamat ini ketemu di MulDiGraph atau tidak") ikut
dihitung sebagai fitur ML, jadi skenario `muldigraph_plus_etherscan` yang seharusnya
**25 predictor** (18 graf + 7 Etherscan) menjadi 26. Sudah diperbaiki di kode + skema
supaya `graph_present` hanya jadi kolom audit (`audit_columns`), bukan predictor.

Skenario target setelah fix (lihat `ml/artifacts/hybrid_feature_schema.json`):

| Skenario | Predictor |
|---|---|
| `muldigraph_only` | 18 |
| `muldigraph_plus_etherscan` (**produksi**) | 25 |
| `muldigraph_plus_bigquery` | 28 |
| `muldigraph_plus_etherscan_plus_bigquery` | 35 |

## Yang HARUS dilakukan di PC yang punya data

### 0. Pindahkan perubahan ke PC itu

PC ini tidak punya `git`, jadi perubahan belum ter-commit. Cara termudah: **copy folder
repo ini (yang sudah dimodifikasi) menimpa/merge ke clone di PC satunya**, atau salin file-file
berikut satu per satu (daftar lengkap di bagian [Files yang berubah](#files-yang-berubah)).
Setelah disalin, langsung ke langkah 1.

### 1. Pastikan prasyarat ada

- `data/raw/MulDiGraph.pkl` (atau `Ethereum Phishing Transaction Network/MulDiGraph.pkl`)
- `data/raw/addresses_etherscan_enriched.csv`
- `data/raw/addresses_bigquery_enriched.csv` (opsional, untuk skenario BigQuery)
- `data/raw/addresses.csv` (daftar alamat + label)
- `.env` terisi `ETHERSCAN_API_KEY` (kalau enrichment CSV belum ada dan perlu dibuat ulang lewat
  `ml/enrich_real_addresses_etherscan.py`)

### 2. Jalankan ulang pipeline, urut

```powershell
.\.venv\Scripts\Activate.ps1
python ml\build_hybrid_dataset.py       # rebuild hybrid_dataset.csv + hybrid_feature_schema.json
python ml\train_production_model.py     # retrain model_prod_xgb.json + prod_feature_schema.json (25 fitur)
python ml\train_dataset_scenarios.py    # regenerasi tabel perbandingan 4 skenario (thesis)
```

### 3. Verifikasi hasilnya benar 25 fitur

```powershell
python -c "import json; s=json.load(open('ml/artifacts/prod_feature_schema.json')); print(len(s['features']), s.get('audit_columns'))"
# harus mencetak: 25 ['graph_present']
```

Lalu jalankan smoke test inferensi (harus tidak error — kalau schema vs booster masih
mismatch, `ml/predict.py` akan melempar `RuntimeError` yang jelas, bukan gagal diam-diam):

```powershell
python ml\predict.py
```

### 4. Bersihkan flag sementara

Setelah retrain berhasil, buka `ml/artifacts/prod_feature_schema.json` dan **hapus** dua
field ini (ditambahkan sementara supaya siapa pun tahu artefak lama masih stale):

```json
"stale_pending_retrain": true,
"retrain_note": "..."
```

(`train_production_model.py` akan menulis ulang file ini otomatis saat langkah 2 di atas
dijalankan — cek isinya, field itu seharusnya sudah tidak ada karena skrip menulis schema
baru dari nol. Kalau ternyata masih ada versi lama yang tertinggal, hapus manual.)

### 5. (Opsional tapi direkomendasikan) Update angka di `hasil_implementasi.md`

Ada catatan koreksi di bagian atas file itu (per 6 Agustus 2026) yang bilang angka F1/AUC
yang ada masih dari model 26-fitur lama. Kalau angka baru (25 fitur) beda signifikan,
update tabel-tabel hasil dan hapus catatan koreksinya. Kalau angkanya nyaris sama
(`graph_present` konstan → importance ~0 di model lama, jadi kemungkinan besar hasilnya
mirip), cukup catat itu di dokumen.

### 6. Review lalu commit

```powershell
git status
git diff --stat
```

Cek terutama:
- `.gitignore` sekarang **hanya** meng-ignore `data/raw/*.pkl` (bukan seluruh `data/raw/`
  dan `data/processed/` lagi) — supaya `hybrid_dataset.csv`, `prod_demo.csv`, dll bisa
  di-commit sebagai data turunan publik. **Pastikan tidak ada file besar/private lain**
  yang ke-stage tanpa sengaja sebelum commit (mis. `.env`, kredensial GCP — keduanya sudah
  ada aturan ignore-nya sendiri, tapi cek ulang `git status` untuk mastiin).
- `model_prod_xgb.json` dan `ml/artifacts/*.png` akan berubah setelah retrain — itu memang
  diharapkan.

```powershell
git add -A
git commit -m "Fix graph_present leaking into feature list; production scenario now 25 predictors"
git push
```

## Files yang berubah (sesi ini, di PC tanpa data)

Kode (logic fix — sudah final, tidak perlu diubah lagi):
- `ml/build_hybrid_dataset.py`
- `ml/train_dataset_scenarios.py`
- `ml/train_production_model.py`
- `ml/predict.py`
- `ml/live_features.py`
- `ml/audit_gaps.py`
- `ml/feature_selection.py`
- `ml/clean_hybrid_dataset.py`
- `ml/generate_prod_artifacts.py`
- `api/services/ml_service.py`
- `api/config.py`

Skema (ditulis manual, akan **ditimpa otomatis** saat langkah 2 di atas — ini normal):
- `ml/artifacts/hybrid_feature_schema.json`
- `ml/artifacts/prod_feature_schema.json`

Config:
- `.gitignore`

Dokumentasi:
- `README.md`
- `Projects/CLAUDE.md`
- `hasil_implementasi.md` (catatan koreksi di paling atas)

Belum disentuh (di luar cakupan fix ini, sengaja dibiarkan):
- `laporan belajar/*.md` (catatan belajar pribadi)
- `HASIL_ABLASI.md`, `HASIL_LATE_FUSION*.md` (jalur GraphSAGE embedding, bukan skenario
  18/25/28/35 ini)
- `new_system/` (live-scan model, README-nya sendiri sudah bilang ini "di luar lingkup tesis")

</details>
