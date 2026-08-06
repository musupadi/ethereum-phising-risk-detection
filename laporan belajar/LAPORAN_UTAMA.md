# LAPORAN HASIL PENELUSURAN KODE & PERSIAPAN BIMBINGAN

**Proyek:** Deteksi Risiko Phishing Wallet Ethereum berbasis Machine Learning + Smart Contract
**Tanggal laporan:** 30 Juli 2026
**Dasar:** penelusuran langsung kode dan data di repo, bukan naskah tesis

Baca file ini lebih dulu. Ia adalah ringkasan eksekutif; detailnya ada di 12 file lain
di folder yang sama.

---

## BAGIAN I — RINGKASAN EKSEKUTIF

### Apa yang sebenarnya Anda bangun

Sebuah **sistem tiga lapis yang berjalan**, bukan sebuah algoritma baru:

```
Streamlit :8502  →  FastAPI :8000  →  XGBoost (26 fitur)  →  Hardhat :8545
   antarmuka         orkestrasi          model produksi        RiskRegistry.sol
```

Ditopang jalur data:

```
MulDiGraph (EPTransNet) → 18 fitur agregat 1-hop ┐
Etherscan API v2        → 7 fitur                ├→ hybrid_dataset.csv → 4 model × 4 skenario
BigQuery public dataset → 10 fitur               ┘   (8.595 × 41)        → pilih XGBoost
```

### Enam angka yang harus Anda hafal

| Angka | Arti |
|---|---|
| **8.595** | wallet dalam dataset eksperimen final |
| **1.165 / 7.430** | phishing / normal (13,6% phishing, rasio 1:6,4) |
| **26** | fitur yang dipakai model produksi (19 graph + 7 Etherscan) |
| **F1 = 0,9102** | performa model produksi (Acc 0,975 · Prec 0,886 · Rec 0,936) |
| **264** | edge asli antara 8.595 wallet — inilah alasan GNN kalah |
| **+0,064** | kontribusi fitur Etherscan yang dibuktikan lewat ablasi |

### Kesimpulan utama laporan ini

1. **Judul harus diubah.** Dosen Anda benar. Tidak ada algoritma yang dikembangkan —
   semua model berasal dari library standar. Kekuatan penelitian ada pada
   **komparasi sistematis + pengembangan sistem end-to-end**. Judul usulan Anda
   sudah tepat (dinilai 8,5/10).
2. **Ada 4 titik rawan metodologis** yang harus Anda akui duluan sebelum ditemukan
   penguji. Semuanya punya jawaban yang aman dan bahkan menguntungkan — asalkan
   Anda yang menyebut lebih dulu.
3. **Semua angka utama tesis valid dan konsisten** dengan artefak di repo, kecuali
   angka GNN pada tabel 4×4 yang optimistis akibat kebocoran early stopping —
   dan itu sudah Anda perbaiki sendiri di eksperimen V2.

---

## BAGIAN II — TEMUAN, DIURUT DARI YANG PALING BERISIKO

### 🔴 T-8. Model gagal pada wallet di luar dataset (out-of-distribution)

> Dinomori T-8 karena ditemukan paling akhir, tetapi **diletakkan pertama karena paling
> berisiko saat sidang.**

Diuji dengan Etherscan API sungguhan pada alamat mainnet nyata di luar 8.595 wallet dataset:

| Alamat | Sebenarnya | Skor model |
|---|---|---|
| Vitalik Buterin | aman, publik | **92 HIGH → "Phishing"** |
| Uniswap V2 Router | kontrak resmi | **100 HIGH → "Phishing"** |
| ETH2 Deposit Contract | kontrak resmi | **84 HIGH → "Phishing"** |
| 8 wallet acak dari blok terbaru | tidak diketahui | **7 dari 8 HIGH** |

**Sebabnya** (dari analisis SHAP): penyumbang skor terbesar selalu `unique_senders`
dengan kontribusi +3,1 s/d +4,9. Model mempelajari aturan *"banyak pengirim unik =
phishing"* — benar di dalam dataset 2019 (kelas negatifnya wallet kecil beraktivitas
rendah), salah total di dunia nyata (bursa dan kontrak populer juga menerima dari ribuan
pengirim).

**Yang tetap sah:** F1 0,9102 tidak batal — ia sah untuk populasi terevaluasi.
**Yang harus dinyatakan:** angka itu tidak boleh ditafsirkan sebagai "91% akurat untuk
wallet Ethereum mana pun".

⚠️ **Risiko konkret saat sidang:** kalau mode scan wallet bebas dibiarkan memakai model
tesis, penguji tinggal mengetik alamat wallet-nya sendiri dan sistem akan menuduhnya
phishing.

**Keputusan akhir:** masalah ini diperbaiki di akarnya, bukan sekadar dimatikan.
Dibangun **model perluasan (Model B)** yang dilatih ulang dengan *hard negatives* —
wallet sah beraktivitas tinggi — dan fitur yang seluruhnya berasal dari satu sumber
(Etherscan). Model tesis (Model A) **tidak disentuh**: ia tetap melayani wallet di
dalam dataset, dan seluruh angka pada naskah tesis tetap berlaku.

### ✅ STATUS: SUDAH DIPERBAIKI DAN TERVERIFIKASI

| Uji | Model A | Model B |
|---|---|---|
| Vitalik Buterin | 98 HIGH ❌ | **1 LOW** ✅ |
| Uniswap V2 Router | 100 HIGH ❌ | **1 LOW** ✅ |
| ETH2 Deposit Contract | 90 HIGH ❌ | **1 LOW** ✅ |
| 8 alamat sah held-out | 3/8 benar | **8/8 benar** ✅ |
| 20 wallet acak segar | 88% false positive | **0% false positive** ✅ |
| Kontrol phishing | terdeteksi | **tetap terdeteksi** (99, 79) ✅ |

Model B: F1 0,8739 · ROC-AUC 0,9685 pada 2.700 wallet.
Regresi model tesis diperiksa: **skor 100 dan 98 tidak berubah**.

→ hasil lengkap: [14_model_live.md](14_model_live.md)

Sisi baiknya — ini adalah **evaluasi out-of-distribution yang dirancang sendiri**,
sesuatu yang jarang dilakukan tesis S2. Sajikan sebagai temuan, bukan sebagai kelemahan
yang ketahuan.

→ bukti lengkap + naskah jawaban untuk dosen: [13_uji_wallet_luar_dataset.md](13_uji_wallet_luar_dataset.md)

### 🔴 T-1. Angka GNN pada tabel 4×4 mengandung kebocoran data

`ml/train_dataset_scenarios.py:369` meneruskan `test_mask` ke `gnn.fit()`, dan
`ml/gnn_model_v2.py:339-346` memakainya untuk early stopping — bobot terbaik dipilih
berdasarkan **F1 test set**.

| | F1 GNN dilaporkan | F1 GNN jujur (V2) |
|---|---|---|
| graph-only | 0,8432 | **0,8189** |

**Sikap yang benar:** "Saya menemukan sendiri masalah ini, lalu membuat eksperimen V2
yang leakage-controlled. Kesimpulan utama tidak berubah — XGBoost tetap unggul,
bahkan dengan selisih yang lebih besar."

→ detail: `10_temuan_penting.md` §1, `05_model_dan_eksperimen.md` §4

### 🔴 T-2. Dataset hanya memakai 8.595 dari 12.815 alamat yang tersedia

`addresses.csv` berisi 12.815 alamat. Enrichment Etherscan selesai penuh
(12.815 baris) pada **3 Juni**, tetapi `hybrid_dataset.csv` dibangun **23 April** —
saat enrichment baru mencapai 8.595 alamat, dan **tidak pernah dibangun ulang**.

Sudah diverifikasi: 8.595 baris = seluruh 1.165 phishing + tepat 7.430 normal pertama
menurut urutan abjad. Pola khas proses berurutan yang berhenti di tengah.

**Merusak validitas?** Tidak — alamat Ethereum acak, jadi "7.430 pertama menurut abjad"
tetap sampel acak. Yang berubah hanya rasio kelas: 1:6,4, bukan 1:10 seperti rancangan.

⚠️ **Jangan menjalankan `build_hybrid_dataset.py` sebelum sidang.** Ia akan menimpa
dataset dan membuat seluruh angka di naskah tesis tidak cocok lagi.

→ detail: `11_asal_data_etherscan.md` §6-D

### 🟠 T-3. Dua fitur tidak berfungsi sebagaimana diklaim

| Fitur | Masalah | Bukti |
|---|---|---|
| `graph_present` | konstan **1** untuk 100% baris → tidak memberi informasi | `hybrid_dataset.csv` |
| `created_at_ts` | konstan **0** untuk 100% baris → fitur mati | `created_at` kosong di seluruh 12.815 baris CSV enrichment |

Artinya klaim "7 fitur Etherscan" secara efektif adalah **6 fitur informatif + 1 mati**,
dan skenario 1 yang tercatat 19 kolom sebenarnya 18 kolom informatif + 1 konstan.

Model tetap valid — XGBoost mengabaikan kolom konstan. Yang harus diperbaiki adalah
**klaimnya**, bukan modelnya.

→ detail: `11_asal_data_etherscan.md` §6-A, `04_penggabungan_data.md` §6

### 🟠 T-4. Docstring menyebut 20 fitur, kode menghitung 18

`ml/preprocess.py:7-17` mendaftar `avg_in_timestamp` dan `avg_out_timestamp` yang
**tidak pernah dihitung** di `:117-136`. Yang benar 18.

**Tindakan:** perbaiki docstring (2 menit) dan pastikan naskah tesis menulis 18.
Penguji bisa membuktikan sendiri hanya dengan menghitung kolom CSV.

### 🟡 T-5. Dua fitur Etherscan redundan sempurna

`tx_frequency = activity_score / 30` **persis** (korelasi 1,0000), karena
`len(txs) / max(1, days)` dengan `days` selalu 30.

Nama `tx_frequency` juga menyesatkan: 100 transaksi yang diambil bukan dari jendela
30 hari, melainkan 100 transaksi terakhir kapan pun terjadinya.

### 🟡 T-6. Angka yang tampak mustahil di data

Ada **2.774 baris** dengan `unique_counterparties` > `transaction_count`.
Penjelasannya ada dan sudah didokumentasikan di `data_fetchers.py:179-183`: kedua kolom
dihitung dari jendela sampel berbeda (100 tx terakhir vs hingga 10.000 tx).
`activity_score` terpotong di 100, `transaction_count` terpotong di 10.000.

**Siapkan jawabannya** — angka aneh seperti ini mudah tertangkap penguji teliti.

### 🟡 T-7. Belum ada pengujian sistem sama sekali

Repo tidak punya `tests/` maupun pytest. Selama judul masih "pengembangan model",
ini bisa dimaklumi. Begitu judul menjadi **"pengembangan sistem"**, pengujian menjadi
kewajiban metodologis.

**Tindakan minimum:** tabel pengujian black-box manual, 8–12 baris, menutup `/health`,
`/analyze` (3 mode), `/risk/{wallet}`, `/records`, dan kasus error.

---

## BAGIAN III — HASIL EKSPERIMEN (ANGKA ASLI DARI ARTEFAK)

### Komparasi 4 model × 4 skenario — F1-score

Sumber: `ml/artifacts/dataset_scenarios_comparison.csv`

| Skenario (n fitur) | LogReg | RandomForest | **XGBoost** | GNN-SAGE-V2 |
|---|---|---|---|---|
| muldigraph_only (19) | 0,7850 | 0,8685 | **0,8857** | 0,8432 ⚠ |
| **muldigraph + etherscan (26)** | 0,8540 | 0,8884 | **0,9102** ⭐ | 0,8960 ⚠ |
| muldigraph + bigquery (29) | 0,7879 | 0,8708 | **0,8838** | 0,8333 ⚠ |
| muldigraph + etherscan + bigquery (36) | 0,8546 | 0,8973 | **0,9038** | 0,8753 ⚠ |

⚠ = angka optimistis, lihat temuan T-1.

**Tiga kesimpulan:**
1. XGBoost menang di **keempat** skenario — konsisten, bukan kebetulan.
2. Etherscan menaikkan F1 **+0,0245**; BigQuery justru **−0,0019** karena fiturnya
   redundan dengan fitur graph.
3. Menambah BigQuery ke skenario terbaik malah **menurunkan** F1 (0,9102 → 0,9038).

### Eksperimen fusi & ablasi

Sumber: `outputs/four_model_comparison_ablation.csv` (semua leakage-controlled)

| Konfigurasi | Input | F1 | ROC-AUC |
|---|---|---|---|
| **XGBoost tabular (Skenario 2)** | 26 fitur tabular | **0,9102** | 0,9948 |
| GraphSAGE–Etherscan Late Fusion V2 | 128-d embedding ‖ 7 Etherscan = 135-d | 0,8825 | 0,9916 |
| GraphSAGE graph-only | 19 fitur graph + 264 edge | 0,8189 | 0,9714 |
| Embedding-only + XGBoost (ablasi) | 128-d embedding saja | 0,8184 | 0,9820 |

**Pembuktian ablasi:** 0,8825 − 0,8184 = **+0,064** → kontribusi fitur Etherscan
terbukti nyata, bukan asumsi.

**Kenapa GNN kalah** (`outputs/edge_source_audit_v2.json`): dari 8.595 wallet hanya ada
**264 edge berarah unik** — rata-rata 0,03 edge per node, sementara self-loop-nya 8.595.
Untuk >99% node, GraphSAGE tidak punya tetangga untuk diagregasi, sehingga efektif
berperilaku seperti MLP biasa.

Bukti pendukung: GraphSAGE graph-only (0,8189) dan embedding-only + XGBoost (0,8184)
hampir identik → batasannya bukan pada classifier, melainkan pada kualitas informasi graph.

---

## BAGIAN IV — REKOMENDASI JUDUL

### Vonis atas empat opsi

| Opsi | Vonis | Alasan singkat |
|---|---|---|
| A. Pengembangan **model** ML | ❌ Tidak dipertahankan | Tidak ada algoritma yang dikembangkan; menyembunyikan kontribusi sistem |
| B. **Komparasi** model ML | ⚠️ Aman tapi merugikan | Jujur, tapi terlalu rendah untuk S2 dan membuang kontribusi sistem |
| C. **Pengembangan sistem** deteksi berbasis ML | ✅ **Paling tepat** | Sesuai isi repo; komparasi tetap terpakai sebagai tahap pemilihan model |
| D. Pengembangan **metode hybrid** | ❌ Sangat berisiko | Late fusion Anda **kalah** dari baseline (0,8825 < 0,9102) |

### Judul usulan Anda: **8,5/10 — pakai ini**

> *"Pengembangan Sistem Deteksi Wallet Phishing Berbasis Machine Learning dengan
> Pencatatan Risk Score pada Smart Contract Ethereum"*

Kelebihan terpentingnya: **setiap bab tesis Anda mendapat tempat.**
Komparasi → bab pemilihan model. Late fusion → bab eksperimen lanjutan.
API/UI/kontrak → bab implementasi. Tidak ada pekerjaan yang terbuang.

Catatan kecil: 17 kata (di batas atas), dan "Risk Score" bercampur bahasa Inggris —
cek pedoman kampus, kalau perlu ganti menjadi "Skor Risiko".

**Alternatif terbaik** kalau dosen ingin komparasi terlihat eksplisit:
> *"Pengembangan Sistem Deteksi Wallet Phishing pada Jaringan Ethereum:
> Komparasi Model Machine Learning dan Integrasi Smart Contract"*

→ analisis lengkap + naskah pembuka bimbingan: `08_masalah_judul_tesis.md`

---

## BAGIAN V — RENCANA TINDAKAN

### Sebelum bimbingan (wajib, ±2 jam)

- [ ] Baca `10_temuan_penting.md` dan `09_latihan_presentasi.md` sampai tuntas
- [ ] Hafal 6 angka kunci di Bagian I
- [ ] Perbaiki docstring `ml/preprocess.py` (20 → 18 fitur) — 2 menit
- [ ] Cek naskah tesis: pastikan tertulis 18 fitur, bukan 20
- [ ] Latih naskah pembuka usulan perubahan judul (`08_masalah_judul_tesis.md` §6)
- [ ] Siapkan pengakuan jujur untuk T-1 (kebocoran GNN) dan T-2 (8.595 vs 12.815)
- [ ] Hafal naskah jawaban T-8 (Vitalik dapat skor 92 pada Model A) — [13_uji_wallet_luar_dataset.md](13_uji_wallet_luar_dataset.md) §8
- [ ] Baca [15_panduan_demo.md](15_panduan_demo.md) — urutan klik, wallet yang aman didemokan, rencana cadangan
- [ ] Pahami pembagian peran Model A vs Model B ([14_model_live.md](14_model_live.md))
- [ ] Coba jalankan sistem mode B sampai `/health` mengembalikan `chain_connected: true`

### Sesudah bimbingan (kalau judul disetujui berubah)

- [ ] Tulis metodologi pengembangan sistem eksplisit (SDLC / Prototyping)
- [ ] Susun analisis kebutuhan fungsional & non-fungsional dari endpoint yang ada
- [ ] Formalkan diagram: arsitektur, use case, sequence
- [ ] **Buat tabel pengujian black-box** (lubang terbesar saat ini — T-7)
- [ ] Tambahkan catatan kaki keterbatasan: T-1, T-2, T-3, T-5, **T-8**
- [ ] Tulis T-8 sebagai sub-bab tersendiri di bab keterbatasan + saran penelitian lanjutan
      (*hard negative sampling* — rumusannya sudah siap pakai di dokumen 13 §5)

### JANGAN dilakukan menjelang sidang

- ❌ Menjalankan `ml/build_hybrid_dataset.py` → menimpa dataset, semua angka berubah
- ❌ Menjalankan ulang enrichment Etherscan → hitungan jam, memicu rebuild berantai
- ❌ Melatih ulang model tanpa alasan → angka tidak akan cocok dengan naskah

### Kalau ada waktu longgar (perbaikan opsional, berdampak besar)

- Perbaiki `train_gnn` agar early stopping memakai validasi internal dari data training,
  lalu jalankan ulang tabel 4×4. Ini menutup temuan T-1 secara tuntas.
- Jalankan ulang seluruh pipeline dengan 12.815 alamat dan `created_at_ts` yang terisi.
  Hasilnya lebih kuat, tapi **semua angka tesis harus ditulis ulang**.

---

## BAGIAN VI — DAFTAR ISI LENGKAP

| File | Isi |
|---|---|
| [00_INDEX.md](00_INDEX.md) | Indeks + tabel angka kunci |
| [01_gambaran_besar_sistem.md](01_gambaran_besar_sistem.md) | Analogi, node/edge, graph→tabel, diagram alur |
| [02_pembentukan_dataset.md](02_pembentukan_dataset.md) | 8 istilah dataset, dua jalur pembentukan |
| [03_feature_extraction_18_fitur.md](03_feature_extraction_18_fitur.md) | Bedah kode + tabel 18 fitur dengan contoh angka |
| [04_penggabungan_data.md](04_penggabungan_data.md) | LEFT JOIN, zero-filling, `graph_present`, 7 pengaman |
| [05_model_dan_eksperimen.md](05_model_dan_eksperimen.md) | 4×4, kapan disebut komparasi/optimasi/pengembangan |
| [06_graphsage_dan_late_fusion.md](06_graphsage_dan_late_fusion.md) | GraphSAGE dari nol, 264 edge, ablasi |
| [07_sistem_end_to_end.md](07_sistem_end_to_end.md) | 9 langkah Streamlit→FastAPI→model→web3→kontrak |
| [08_masalah_judul_tesis.md](08_masalah_judul_tesis.md) | Analisis A/B/C/D + naskah pembuka bimbingan |
| [09_latihan_presentasi.md](09_latihan_presentasi.md) | 15 pertanyaan + jawaban + 7 pertanyaan tak terduga |
| [10_temuan_penting.md](10_temuan_penting.md) | Daftar ketidaksesuaian + checklist H-1 |
| [11_asal_data_etherscan.md](11_asal_data_etherscan.md) | Asal data Etherscan + 4 temuan enrichment |
| [12_dapp_lokal.md](12_dapp_lokal.md) | Lapisan DApp pembaca on-chain (pendukung judul baru) |
| [13_uji_wallet_luar_dataset.md](13_uji_wallet_luar_dataset.md) | **Uji OOD: kenapa scan wallet bebas belum diaktifkan** |

---

## CATATAN METODOLOGI LAPORAN INI

Seluruh isi laporan disusun dari **pembacaan langsung** kode dan data:

- Kode: `ml/preprocess.py`, `build_hybrid_dataset.py`, `train_dataset_scenarios.py`,
  `train_production_model.py`, `gnn_model_v2.py`, `predict.py`, `data_fetchers.py`,
  `enrich_real_addresses_etherscan.py`, `extract_addresses_from_graph.py`,
  `experiments/*.py`, `api/*.py`, `chain/contracts/RiskRegistry.sol`
- Artefak: `ml/artifacts/*.json`, `outputs/*.csv`
- Data: pemeriksaan langsung `hybrid_dataset.csv`, `addresses*.csv` dengan pandas

Setiap angka dalam laporan ini dapat ditelusuri ke file dan baris kodenya.
Tidak ada angka yang disalin dari naskah tesis tanpa verifikasi ulang ke sumbernya.
