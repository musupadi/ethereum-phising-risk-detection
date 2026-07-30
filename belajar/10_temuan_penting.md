# TEMUAN PENTING — Ketidaksesuaian & Titik Rawan

Daftar hal yang **berbeda antara dokumen/asumsi dan kode yang benar-benar berjalan**,
plus titik yang paling mungkin diserang penguji. Baca ini terakhir, tapi jangan dilewat.

Diurutkan dari yang paling berisiko.

---

## 🔴 1. Angka GNN di tabel 4×4 mengandung kebocoran data

**Fakta kode.** `ml/train_dataset_scenarios.py:369-375` memanggil
`gnn.fit(graph, train_mask, test_mask, ...)`, dan di dalam
`ml/gnn_model_v2.py:339-346` early stopping memilih bobot terbaik berdasarkan
**F1 pada test set**.

**Akibat.** Semua angka GNN-GraphSAGE-V2 di
`ml/artifacts/dataset_scenarios_comparison.csv` (0,8432 / 0,8960 / 0,8333 / 0,8753)
adalah **estimasi optimistis**, bukan performa generalisasi murni.

**Yang benar.** Eksperimen leakage-controlled Anda sendiri (`..._late_fusion_v2.py`)
memberi GraphSAGE graph-only **F1 0,8189**.

**Tindakan sebelum sidang:**
- Tambahkan catatan kaki pada tabel 4×4 di naskah tesis.
- Siapkan kalimat: *"Saya menemukan sendiri masalah ini dan memperbaikinya di eksperimen V2;
  kesimpulan utama tidak berubah karena XGBoost tetap unggul, bahkan dengan selisih lebih besar."*
- Kalau ada waktu: ubah `train_gnn` agar memakai validasi internal dari data training,
  lalu jalankan ulang. Kalau tidak sempat, catatan kaki + penjelasan V2 sudah cukup.

**Risiko kalau dibiarkan diam-diam:** penguji yang membuka `gnn_model_v2.py` akan
menemukannya, dan validitas seluruh tabel dipertanyakan.

---

## 🟠 2. Docstring menyebut 20 fitur, kode menghitung 18

**Fakta kode.** `ml/preprocess.py:7-17` mendaftar `avg_in_timestamp` dan
`avg_out_timestamp`. Keduanya **tidak pernah dihitung** di `:117-136`.

**Yang benar:** 18 fitur graph, konsisten dengan `prod_feature_schema.json`.

**Tindakan:** perbaiki docstring (2 menit), dan pastikan naskah tesis menulis 18.
Kalau naskah menulis 20, itu kesalahan yang mudah ditemukan penguji dengan menghitung
kolom CSV.

---

## 🟠 3. `graph_present` bernilai konstan 1 — tidak berfungsi sebagai fitur

**Fakta data.** Pada `hybrid_dataset.csv`, `graph_present.mean() == 1.0` untuk seluruh
8.595 baris.

**Akibat.** Kolom ini tidak memberi informasi apa pun kepada model. Ia ikut dihitung
sebagai fitur (membuat skenario 1 punya 19 kolom, bukan 18), tapi XGBoost akan
mengabaikannya.

**Jangan klaim:** "graph_present membantu model membedakan wallet yang ada/tidak ada di graph."

**Klaim yang benar:** "Kolom ini dirancang sebagai penanda cakupan graph. Pada dataset
final nilainya seragam 1, yang justru membuktikan cakupan MulDiGraph 100% —
tidak ada fitur graph yang hasil zero-fill."

---

## 🟠 4. Sistem hanya bisa memindai 8.595 alamat dataset

**Fakta kode.** `api/services/ml_service.py:111-115` — alamat di luar `hybrid_dataset.csv`
menghasilkan `KeyError` → HTTP 404.

**Akibat.** Ini **bukan** pemindai phishing real-time. Ia sistem demonstrasi di atas
dataset tetap dari snapshot 2019.

**Tindakan:** sebut sendiri sebagai keterbatasan, dan tunjukkan jalannya:
`ml/enrich_real_addresses_etherscan.py` sudah bisa mengambil 7 fitur Etherscan untuk
alamat arbitrer; yang belum ada adalah ekstraksi fitur graph untuk alamat baru
(butuh graph, atau `fetch_etherscan_edges.py` untuk membangun tetangga dari Etherscan).

---

## 🟡 5. Tidak ada test suite sama sekali

**Fakta repo.** Tidak ada `pytest`, tidak ada folder `tests/`. Satu-satunya pengujian
adalah smoke test `python ml/predict.py` di blok `__main__`.

**Kenapa penting sekarang.** Kalau judul berubah menjadi "pengembangan sistem",
pengujian sistem menjadi kewajiban metodologis, bukan opsional.

**Tindakan minimum:** buat tabel pengujian black-box manual di bab implementasi —
kolom: skenario uji, input, output diharapkan, output aktual, status. Cukup 8–12 baris
menutup `/health`, `/analyze` (3 mode), `/risk/{wallet}`, `/records`, dan kasus error
(alamat tidak valid, alamat tidak ada di dataset, chain mati).

---

## 🟡 6. Dimensi fitur yang dilayani bergantung file di disk

**Fakta kode.** `api/config.py:47-48`:
```python
TEST_CSV   = _PROD_DEMO if _PROD_DEMO.exists() else .../test.csv
LOOKUP_CSV = _HYBRID    if _HYBRID.exists()    else TEST_CSV
```

**Akibat.** Kalau `prod_demo.csv` hilang, sistem **diam-diam** turun ke `test.csv`
(18 fitur). Tujuh fitur Etherscan akan diisi 0 oleh `predict.py:87`, prediksi jadi
salah, dan **tidak ada pesan error apa pun**.

**Tindakan:** pastikan `prod_demo.csv`, `hybrid_dataset.csv`, dan
`prod_feature_schema.json` ada dan sinkron sebelum demo. Cek cepat:
```powershell
curl http://localhost:8000/demo/rows      # harus mengembalikan 1719
```

---

## 🟡 7. Private key tertulis di dalam kode

**Fakta kode.** `api/config.py:36` memuat kunci privat akun Hardhat #0 sebagai default.

Untuk Hardhat lokal ini **aman** — kunci itu publik dan sama di semua instalasi Hardhat,
dan kodenya sendiri sudah memberi peringatan `NEVER use in production`.

**Tindakan:** kalau ditanya, jawab dengan tenang: "Itu kunci demo Hardhat yang memang
publik. Untuk produksi, penandatanganan harus memakai key management terpisah dan peran
penulis kontrak diubah menjadi himpunan oracle." Jangan gugup — ini sudah benar untuk
lingkup tesis.

⚠️ Yang **tidak** boleh: ada file `.env` dan JSON service account GCP
(`primal-library-239611-*.json`) di root repo. Pastikan keduanya masuk `.gitignore`
dan tidak pernah ikut ter-*commit*.

---

## 🟢 8. Dua jalur model yang mudah tertukar saat presentasi

| | Legacy | Produksi |
|---|---|---|
| Skrip | `ml/train.py` | `ml/train_production_model.py` |
| Model | LR + RandomForest | XGBoost |
| Fitur | 18 graph | 26 (19 graph + 7 Etherscan) |
| Artefak | `model.pkl` | `model_prod_xgb.json` |
| Dilayani API? | hanya kalau artefak produksi hilang | **ya** |

Jangan sampai menyebut "model saya Random Forest" di depan penguji. Yang dilayani
adalah **XGBoost**, dan `MODEL_VERSION` on-chain-nya `xgb-eth-v1`.

---

## 🟢 9. Klaim yang aman vs klaim yang berbahaya

| ❌ Jangan katakan | ✅ Katakan ini |
|---|---|
| "Saya mengembangkan model machine learning baru" | "Saya membangun sistem, dan memilih modelnya lewat perbandingan sistematis" |
| "Saya membuat dataset sendiri" | "Saya membentuk dataset eksperimen dari dataset publik EPTransNet melalui sampling, extraction, dan enrichment" |
| "GraphSAGE saya kurang optimal, perlu tuning" | "GraphSAGE terbatas oleh sparsitas graph: 264 edge untuk 8.595 node — dan saya buktikan dengan audit edge dan ablasi" |
| "Late fusion meningkatkan performa" | "Late fusion mencapai F1 0,8825, di bawah early fusion 0,9102; ablasi tetap membuktikan Etherscan menyumbang +0,064" |
| "Akurasi saya 97,5%" | "F1 0,9102; akurasi 97,5% menyesatkan pada data timpang 1:6,4" |
| "graph_present membantu model" | "graph_present adalah penanda cakupan; nilainya seragam 1, membuktikan cakupan graph 100%" |
| "Sistem bisa memindai wallet apa pun" | "Sistem melayani 8.595 alamat dataset; perluasan ke alamat arbitrer adalah pekerjaan lanjutan" |

---

## Checklist H-1 sebelum bimbingan

- [ ] Hafal 6 angka: **8.595 / 1.165 / 26 fitur / F1 0,9102 / 264 edge / +0,064 ablasi**
- [ ] Bisa menggambar diagram alur sistem di papan tulis tanpa catatan
- [ ] Sudah menyiapkan kalimat pembuka usulan perubahan judul (TAHAP 8 §6)
- [ ] Sudah menyiapkan pengakuan jujur soal kebocoran early stopping GNN (temuan #1)
- [ ] Sudah memperbaiki docstring 20→18 fitur (temuan #2)
- [ ] Sudah mengecek naskah tesis: tertulis 18 fitur, bukan 20
- [ ] Sudah menyiapkan draf tabel pengujian black-box (temuan #5)
- [ ] Sistem sudah dicoba jalan penuh mode B dan `/health` mengembalikan `chain_connected: true`
- [ ] `.env` dan service-account JSON dipastikan tidak ter-*commit*
