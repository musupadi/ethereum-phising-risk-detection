# Checklist Berbagi Repo ke Reviewer / Pihak Lain

> Dokumen ini untuk **kamu** (pemilik repo) sebelum membagikan kode ke reviewer IJECE,
> dosen, atau AI lain. Hapus file ini sendiri dari salinan yang dibagikan kalau tidak perlu.

## TL;DR — cara paling aman

**Bagikan lewat GitHub, jangan zip folder kerja.** Rahasia dan dataset pihak ketiga sudah
di-ignore git, jadi `git push` → link GitHub **tidak** membawa semua itu ikut.

```powershell
git status                 # pastikan tidak ada file rahasia yang ter-stage
git add -A
git commit -m "IJECE revision: multi-seed experiments, validation-based GNN early stopping"
git push
```

Kalau tetap harus zip, **jangan** zip folder kerja — buat salinan bersih dulu:

```powershell
git clone https://github.com/musupadi/ethereum-phising-risk-detection.git share-copy
# lalu hapus dokumen pribadi (Bagian B) dari share-copy, zip share-copy
```

`git clone` hanya membawa file yang ter-commit → otomatis bebas dari isi Bagian A.

---

## A. ❌ WAJIB TIDAK IKUT (rahasia / tidak boleh disebar)

| File / folder | Kenapa | Status di git |
|---|---|---|
| `.env` | Berisi **API key Etherscan** kamu | di-ignore ✅ |
| `primal-library-239611-b6941382db9f.json` | **Kunci service account Google Cloud** — orang lain bisa memakai BigQuery atas tagihanmu | di-ignore ✅ |
| `data/raw/MulDiGraph.pkl` | Dataset pihak ketiga (XBlock), sekitar 1,3 GB; arahkan reviewer ke sumber resminya, jangan disebar ulang dari repo | di-ignore ✅ |
| `Ethereum Phishing Transaction Network/` | Sama (lokasi alternatif MulDiGraph) | di-ignore ✅ |
| `.venv/` | Environment lokal Windows-mu, ratusan MB, tidak portabel | di-ignore ✅ |
| `chain/node_modules/`, `chain/artifacts/`, `chain/cache/` | Hasil `npm install` / compile, dibuat ulang otomatis | di-ignore ✅ |
| `*.docx` (naskah tesis/jurnal) | Dokumen pribadi | di-ignore ✅ |
| `.claude/`, `Projects/CLAUDE.md` | Konfigurasi asisten/editor lokal | sudah dikeluarkan dari git + di-ignore ✅ |

> ⚠️ **Kalau API key / service-account JSON pernah terkirim** (zip, email, chat),
> **rotasi** keduanya: buat API key Etherscan baru, dan di Google Cloud Console → IAM →
> Service Accounts → hapus key lama, buat baru.

## B. 🗑️ BOLEH DIHAPUS (dokumen kerja, tidak dibaca kode)

Aman dihapus — tidak ada script yang membaca file-file ini:

- `hasil_implementasi.md`, `HASIL_ABLASI.md`, `HASIL_LATE_FUSION.md`, `HASIL_LATE_FUSION_V2.md`
- `PENJELASAN_ANALISIS_DATA.md`, `RENCANA_BIMBINGAN.md`, `LANJUTKAN_DI_PC_LAIN.md`
- `Hasil.kumpulan data itu wallet yang sama mak`
- `task.md`, `task2.md`, `task3.md`, `lasttask.md`
- `laporan belajar/` (seluruh folder)
- `Projects/` (`CLAUDE.md`, `belajar.md`)
- `REVISI_IJECE.md` — **jangan dibagikan ke reviewer** (isinya draf jawaban & catatan internal); simpan untukmu sendiri
- `SHARING_CHECKLIST.md` (file ini)
- `new_system/` — sistem live-model terpisah, **bukan** bagian paper. Hapus supaya reviewer tidak bingung
- `run_daily_*.bat`, `scripts/run_daily_*.ps1` — penjadwal enrichment harian milikmu (opsional; tidak berbahaya)
- `notebooks/` — opsional; tidak dibutuhkan untuk mereproduksi tabel paper

## C. ✅ WAJIB TETAP ADA

| Path | Fungsi |
|---|---|
| `README.md` | Petunjuk instalasi & reproduksi (bagian *Reproducing the IJECE paper*) |
| `requirements.txt`, `.env.example` | Dependensi & contoh konfigurasi |
| `ml/` | Ekstraksi fitur, training, inferensi |
| `experiments/` | `revision_ijece.py`, `revision_ops.py`, late fusion, ablation |
| `data/processed/`, `data/raw/*.csv`, `data/raw/README.txt` | Populasi 12.815 wallet + hasil enrichment (supaya reproduksi tidak butuh API) |
| `ml/artifacts/` | Model produksi `model_prod_xgb.json`, skema fitur, gambar |
| `outputs/revision/` | Hasil mentah semua tabel revisi |
| `api/`, `app/`, `chain/` (tanpa `node_modules`) | Sistem FastAPI, Streamlit, RiskRegistry |
| `models/` | Model fusion/ablation yang sudah dilatih |
| `.gitignore`, `.gitattributes` | Menjaga rahasia tetap tidak ter-commit |

## D. Status uji jalan (smoke test di folder clone bersih, 27 Okt 2026)

Diuji dari `git clone` bersih + perubahan revisi, tanpa `.env`, dengan `MulDiGraph.pkl`
disalin manual (seperti yang akan dilakukan reviewer).

<!-- SMOKE_RESULTS -->

## E. Yang tetap harus disiapkan sendiri oleh penerima

1. **Python 3.13** (versi yang diuji) dan `pip install -r requirements.txt`.
2. **`MulDiGraph.pkl`** dari XBlock / Kaggle → `data/raw/`. Tanpa file ini: XGBoost/RF/LR
   tetap bisa dilatih dari `data/processed/hybrid_dataset.csv`, tetapi GraphSAGE,
   late fusion, dan `revision_ijece.py` tidak bisa jalan.
3. **API key Etherscan miliknya sendiri** di `.env` — hanya untuk scan live & re-enrichment.
   Tidak dibutuhkan untuk mereproduksi tabel paper (hasil enrichment sudah ada di CSV).
4. **Node.js 18+**, lalu `cd chain && npm install` — hanya untuk demo blockchain.
5. **Google Cloud service account miliknya sendiri** — hanya jika ingin mengulang enrichment
   BigQuery (tidak diperlukan; hasilnya sudah ada di CSV).
