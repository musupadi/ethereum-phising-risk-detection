# TAHAP 8 — Masalah Judul Tesis

## 1. Duduk perkaranya

Judul sekarang:
> *"Pengembangan Model Machine Learning untuk Deteksi Wallet Phishing pada Jaringan Ethereum"*

Keberatan dosen: **"pengembangan model"** menjanjikan bahwa Anda mengembangkan *model* —
arsitektur, formulasi, atau algoritma. Yang penguji temukan di isi tesis dan kode adalah
perbandingan algoritma, perbandingan skenario data, dan sebuah sistem aplikasi.

**Dosen Anda benar.** Ini bukan serangan, ini penyelamatan sebelum sidang.

## 2. Bukti dari kode: apa yang benar-benar Anda kerjakan

| Aktivitas | Bobot dalam repo | Bukti |
|---|---|---|
| Membangun dataset eksperimen | Besar | `build_hybrid_dataset.py`, 3 skrip enrichment, `clean_hybrid_dataset.py`, `audit_gaps.py` |
| Feature engineering | Sedang | 18 fitur di `preprocess.py`, 2 rumus rasio |
| **Komparasi model & skenario** | **Besar** | `train_dataset_scenarios.py`, 16 pelatihan |
| Optimasi hyperparameter | Sedang | `tune_xgboost.py`, `tune_gnn.py`, `smote_experiment.py`, `feature_selection.py` |
| Eksperimen arsitektur fusi | Sedang | 3 skrip di `experiments/`, lengkap dengan V2 leakage-controlled + ablasi |
| **Membangun sistem end-to-end** | **Besar** | `api/`, `app/`, `chain/`, `predict.py`, `train_production_model.py` |
| Menciptakan algoritma baru | **Nol** | tidak ada |

Dua kotak "Besar" — komparasi dan sistem — itulah tulang punggung tesis Anda.

## 3. Analisis empat opsi

### A. Pengembangan Model Machine Learning

**Kelebihan**
- Judul sekarang, tidak perlu mengubah apa pun secara administratif.
- Terdengar "ML banget" — nyaman di prodi berbasis komputasi.

**Risiko** ⚠️ **Tinggi**
- Penguji akan bertanya: *"Model apa yang Anda kembangkan? Tunjukkan bedanya dengan
  XGBoost standar."* Jawaban jujurnya: tidak ada.
- Anda memakai `XGBClassifier` dan `SAGEConv` apa adanya.
- Kalaupun Anda menunjuk late fusion sebagai "model yang dikembangkan", hasilnya
  **lebih buruk** (F1 0,8825) dari baseline (0,9102). Klaim pengembangan yang tidak
  meningkatkan performa akan langsung diserang.
- Judul ini juga **menyembunyikan** kontribusi terbesar Anda: sistem + blockchain.
  Dua hal itu tidak muncul sama sekali di judul.

**Vonis: tidak dipertahankan.**

### B. Komparasi Model Machine Learning

**Kelebihan**
- 100% jujur terhadap isi. 4 model × 4 skenario adalah komparasi yang solid,
  dengan split identik dan protokol konsisten.
- Sangat mudah dibela — semua angkanya ada.

**Risiko**
- Terlalu rendah untuk tesis S2. Penguji bisa bilang *"ini setara skripsi S1"*.
- **Membuang** kontribusi sistem dan blockchain — padahal itu yang membedakan
  penelitian Anda dari ratusan paper komparasi phishing detection yang sudah ada.
- Kebaruannya lemah: perbandingan LR/RF/XGB/GNN untuk deteksi phishing Ethereum
  sudah banyak dipublikasikan.

**Vonis: aman tapi merugikan Anda.**

### C. Pengembangan Sistem Deteksi Berbasis Machine Learning

**Kelebihan** ✅
- **Paling sesuai dengan isi repo.** Anda memang membangun sistem tiga lapis yang
  berjalan: UI → API → model → smart contract.
- Komparasi model tetap masuk — sebagai **tahap pemilihan model** di dalam metodologi
  pengembangan sistem. Tidak ada pekerjaan yang terbuang.
- Kebaruan jadi jelas: bukan "algoritma baru", tapi **integrasi ML + on-chain
  auditability**, yang memang jarang.
- Aman terhadap pertanyaan *"apa yang Anda kembangkan?"* → jawabannya sistemnya,
  dan sistemnya bisa didemokan hidup-hidup di depan penguji.

**Risiko**
- Penguji bisa bertanya: *"Kalau ini pengembangan sistem, mana pengujian sistemnya?"*
  Anda perlu menyiapkan: pengujian fungsional endpoint, uji integrasi UI–API–chain,
  dan idealnya sedikit evaluasi kegunaan. Saat ini repo **tidak punya test suite**.
  Ini pekerjaan rumah yang harus Anda akui.
- Bobot ML bisa dianggap berkurang. Bantahannya: metodologi tetap memuat 16 eksperimen
  pelatihan, tuning, ablasi, dan analisis kebocoran data.

**Vonis: paling tepat.**

### D. Pengembangan Metode Hybrid Graph–Tabular

**Kelebihan**
- Terdengar paling "berkontribusi ilmiah".
- Anda memang punya bahannya: late fusion 128+7, ablasi, dan V2 yang leakage-controlled.

**Risiko** ⚠️ **Sangat tinggi**
- Sebuah metode baru harus **terbukti lebih baik**. Milik Anda kalah:
  0,8825 vs 0,9102. Penguji akan langsung menunjuk itu.
- Perlu validasi lintas dataset untuk klaim "metode". Anda hanya punya satu dataset.
- Diagnosis Anda sendiri (264 edge, graph terlalu sparse) menjelaskan **kenapa** metode
  hybrid tidak bisa unggul di dataset ini. Anda akan berargumen melawan temuan Anda sendiri.
- Menyembunyikan kontribusi sistem, sama seperti opsi A.

**Vonis: jangan.** Late fusion tetap dilaporkan — sebagai **eksperimen pendukung dan
temuan negatif yang informatif**, bukan sebagai judul.

## 4. Penilaian judul usulan Anda

> *"Pengembangan Sistem Deteksi Wallet Phishing Berbasis Machine Learning dengan
> Pencatatan Risk Score pada Smart Contract Ethereum"*

**Nilai: 8,5/10. Sudah sesuai. Pakai ini.**

Yang sudah benar:

| Unsur | Ada? | Terbukti di |
|---|---|---|
| "Pengembangan Sistem" | ✅ | `api/`, `app/`, `chain/` — jalan sungguhan |
| "Deteksi Wallet Phishing" | ✅ | label `isp`, F1 0,9102 |
| "Berbasis Machine Learning" | ✅ | XGBoost + 3 pembanding |
| "Pencatatan Risk Score" | ✅ | `setRisk()` menyimpan skor 0–100 |
| "Smart Contract Ethereum" | ✅ | `RiskRegistry.sol` |

Tiga catatan kecil:

1. **Panjang** (17 kata). Masih dalam batas wajar untuk tesis Indonesia, tapi di titik
   maksimum. Kalau kampus Anda membatasi ~15 kata, potong "Wallet".
2. **Kata "Risk Score" bercampur bahasa Inggris.** Kalau pedoman kampus mewajibkan
   judul berbahasa Indonesia penuh, ganti "Risk Score" → "Skor Risiko". Cek pedoman.
3. **Sumber data tidak muncul.** Kekuatan Anda (multi-sumber: graph + Etherscan +
   BigQuery) tidak terlihat di judul. Ini pilihan — judul tidak wajib memuat semuanya,
   dan bisa ditonjolkan di rumusan masalah.

**Yang paling penting: judul ini membuat setiap bab tesis Anda punya tempat.**
Komparasi model → bab pemilihan model. Late fusion → bab eksperimen lanjutan.
API/UI/kontrak → bab implementasi sistem. Tidak ada bab yang jadi yatim.

## 5. Alternatif judul (arah penelitian tidak berubah)

Semua opsi di bawah tetap berporos pada "pengembangan sistem".

**Opsi 1 — versi ringkas (14 kata)**
> Pengembangan Sistem Deteksi Phishing Wallet Ethereum Berbasis Machine Learning
> dengan Pencatatan Skor Risiko On-Chain

**Opsi 2 — menonjolkan multi-sumber data** *(paling menonjolkan kontribusi data Anda)*
> Pengembangan Sistem Deteksi Wallet Phishing Ethereum Berbasis Machine Learning
> Multi-Sumber Data dengan Pencatatan Skor Risiko pada Smart Contract

**Opsi 3 — memberi ruang untuk bab komparasi**
> Pengembangan Sistem Deteksi Wallet Phishing pada Jaringan Ethereum:
> Komparasi Model Machine Learning dan Integrasi Smart Contract

**Opsi 4 — bahasa Indonesia penuh**
> Pengembangan Sistem Deteksi Dompet Phishing Ethereum Berbasis Pembelajaran Mesin
> dengan Pencatatan Skor Risiko pada Kontrak Pintar

**Rekomendasi:** pakai **judul usulan Anda sendiri** (yang sudah dinilai 8,5), atau
**Opsi 3** kalau dosen ingin komparasi tetap terlihat eksplisit di judul.
Opsi 3 punya keunggulan diplomatis: ia mengakui komparasi tanpa menjadikannya inti,
sehingga tidak ada penguji yang bisa bilang "ini cuma komparasi" atau
"komparasinya tidak relevan dengan judul".

## 6. Naskah pembuka bimbingan (hafalkan)

> "Pak/Bu, saya sudah menelusuri ulang isi penelitian dan kode saya. Saya setuju bahwa
> istilah *pengembangan model* tidak tepat — saya tidak menciptakan atau memodifikasi
> algoritma. Saya memakai XGBoost dan GraphSAGE dari library standar.
>
> Yang sebenarnya saya bangun ada tiga: **pertama**, dataset eksperimen dari EPTransNet
> yang diperkaya data on-chain nyata dari Etherscan dan BigQuery; **kedua**, protokol
> perbandingan 4 model × 4 skenario di atas split identik, plus eksperimen fusi dan
> ablasi; **ketiga**, sistem end-to-end dari antarmuka pengguna sampai pencatatan skor
> risiko di smart contract.
>
> Kontribusi terbesar saya ada di nomor tiga, dan itu tidak terlihat sama sekali pada
> judul lama. Karena itu saya usulkan judul menjadi *Pengembangan Sistem Deteksi Wallet
> Phishing Berbasis Machine Learning dengan Pencatatan Risk Score pada Smart Contract
> Ethereum*. Dengan judul ini, seluruh pekerjaan komparasi tetap terpakai — posisinya
> menjadi tahap pemilihan model di dalam metodologi pengembangan sistem."

## 7. Yang harus Anda siapkan kalau judul jadi "pengembangan sistem"

Judul baru membawa tuntutan baru. Siapkan sebelum ditagih:

| Tuntutan | Status sekarang | Tindakan |
|---|---|---|
| Metodologi pengembangan sistem eksplisit (Waterfall / Prototyping / SDLC) | Belum ditulis | Pilih satu, petakan bab ke fasenya |
| Analisis kebutuhan fungsional & non-fungsional | Belum ada | Tulis dari endpoint yang sudah ada |
| Diagram sistem (arsitektur, use case, sequence, ERD/skema data) | Sebagian di README | Formalkan; diagram di TAHAP 1 & 7 bisa jadi bahan |
| Pengujian sistem (black-box tiap endpoint) | **Tidak ada test suite** | Buat tabel uji manual: input → output diharapkan → hasil |
| Uji integrasi UI–API–chain | Manual saja | Dokumentasikan skenario + tangkapan layar |
| Evaluasi model | ✅ Lengkap | Sudah kuat |

Poin pengujian sistem adalah lubang terbesar Anda saat ini. Repo tidak punya test suite
sama sekali. Untuk tesis, tabel pengujian black-box manual yang rapi sudah cukup —
tapi harus ada.

---

## 3 pertanyaan cek pemahaman

1. Kenapa opsi D ("pengembangan metode hybrid") justru berbahaya, padahal Anda sudah
   punya arsitektur late fusion dan eksperimen ablasi?
2. Dengan judul baru, di mana posisi pekerjaan komparasi 4×4 dalam struktur tesis?
3. Kewajiban baru apa yang muncul begitu judul memakai kata "pengembangan sistem",
   dan bagian mana yang belum Anda punya?
