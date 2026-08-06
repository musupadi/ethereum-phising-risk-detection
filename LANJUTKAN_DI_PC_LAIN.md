# Lanjutkan di PC yang punya data — fix `graph_present` (26 → 25 fitur)

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
