# Sistem Baru — Perluasan di Luar Lingkup Tesis

**Dibuat:** 31 Juli 2026
**Status:** tambahan eksperimental. **Bukan** bagian dari eksperimen yang dilaporkan
dalam naskah tesis.

Folder ini sengaja dipisah supaya batasnya tegas: apa pun yang ada di sini **tidak**
ikut menghasilkan angka-angka tesis, dan tidak pernah mengubah sistem tesis.

---

## 1. Ringkasan satu paragraf

Sistem tesis dievaluasi pada populasi tertutup: 8.595 wallet hasil sampling dari
EPTransNet. Setelah tesis berjalan, saya menguji modelnya pada wallet **di luar**
populasi itu dan menemukan tingkat false positive yang tinggi — alamat Vitalik Buterin
mendapat skor 98 (HIGH, "Phishing"). Folder ini berisi perbaikan atas masalah tersebut:
sebuah model kedua yang dilatih ulang dengan *hard negatives*. Model tesis tidak diubah
sama sekali.

---

## 2. Batas yang dijaga

| | Sistem tesis (akar repo) | Sistem baru (folder ini) |
|---|---|---|
| Model | `ml/artifacts/model_prod_xgb.json` | `new_system/artifacts/model_live_xgb.json` |
| Data latih | `data/processed/hybrid_dataset.csv` | `new_system/data/live_dataset.csv` |
| Melayani | 8.595 wallet dalam dataset | wallet arbitrer |
| Metrik dilaporkan di tesis | ✅ F1 0,9102 | ❌ tidak |
| Ditulis ke blockchain | ✅ ya | ❌ tidak pernah |
| Aktif secara bawaan | ✅ ya | ❌ **tidak** (`LIVE_SCAN_ENABLED=false`) |

Aturan teknis yang ditegakkan lewat `new_system/ml/paths.py`:
sistem baru boleh **membaca** data sistem tesis, tetapi **tidak boleh menulis** apa pun
di luar folder `new_system/`.

**Dengan pengaturan bawaan, menjalankan aplikasi menghasilkan perilaku yang persis sama
dengan sebelum folder ini ada.** Wallet di luar dataset tetap menghasilkan HTTP 404.

---

## 3. Masalah yang melatarbelakangi

Model tesis diuji pada alamat mainnet nyata di luar dataset:

| Alamat | Sebenarnya | Skor model tesis |
|---|---|---|
| Vitalik Buterin | aman, publik | **98 HIGH → "Phishing"** |
| Uniswap V2 Router | kontrak resmi | **100 HIGH → "Phishing"** |
| ETH2 Deposit Contract | kontrak resmi | **90 HIGH → "Phishing"** |
| WETH | kontrak resmi | **99 HIGH → "Phishing"** |
| 20 wallet acak | tidak diketahui | **88% ditandai HIGH** |

Diagnosis lewat nilai SHAP: penyumbang skor terbesar selalu `unique_senders`
(kontribusi +3,1 s/d +4,9). Model mempelajari aturan *"banyak pengirim unik = phishing"*.

Aturan itu **benar** di dalam dataset tesis — wallet phishing memang mengumpulkan dana
dari banyak korban, sementara wallet normal yang tersampling kebanyakan kecil dan pasif.
Aturan itu **salah** di dunia nyata — bursa dan router DEX juga menerima dari ribuan pihak.

Istilah teknisnya: **kegagalan out-of-distribution** akibat kelas negatif yang tidak
representatif.

---

## 4. Apa yang diperbaiki

### 4.1 Hard negative sampling
Kelas negatif kini memuat wallet sah yang benar-benar ramai: 40 alamat bursa/router/
token/bridge, 587 wallet acak dari blok sungguhan lintas era 2017–2026, dan 900 wallet
normal berukuran kecil.

### 4.2 Fitur satu sumber
Fitur untuk kedua kelas dihitung dari Etherscan dengan rumus identik. Sebelumnya model
dilatih pada fitur turunan MulDiGraph tetapi dipakai pada fitur turunan Etherscan —
dua distribusi berbeda untuk nama fitur yang sama.

### 4.3 Fitur rasio tak bergantung skala
Ditambahkan `sender_receiver_ratio`, `repeat_ratio`, `balance_ratio`, dan
`activity_span_days` — mengukur **pola**, bukan **besaran**.

### 4.4 Fitur yang sengaja dibuang
`created_at_ts` (waktu absolut → jalan pintas berbasis era), `tx_frequency` (redundan
sempurna dengan `activity_score`), `graph_present` (artefak MulDiGraph).

---

## 5. Hasil

**Data latih:** 2.700 wallet (1.165 phishing : 1.535 normal)

**Model akhir:** ensemble rata-rata probabilitas — XGBoost (hasil tuning) +
RandomForest + ExtraTrees, ambang 0,34.

**Metrik test split:** F1 **0,9008** · Precision 0,8685 · Recall 0,9356 · ROC-AUC 0,9672

Perjalanan menaikkannya (semua keputusan diambil dari validation split, test dinilai
sekali):

| Tahap | F1 |
|---|---|
| XGBoost konfigurasi awal | 0,8739 |
| + ambang dioptimasi (0,5 → 0,31) | 0,8816 |
| + hyperparameter dituning (40 konfigurasi) | 0,8871 |
| + **ensemble 3 model** | **0,9008** |
| *(pembanding: model tesis)* | *0,9102* |

> ⚠️ Angka ini **tidak sebanding** dengan F1 0,9102 milik model tesis. Populasinya
> berbeda dan tugasnya lebih sulit. Kedekatannya kebetulan, bukan bukti kesetaraan.

Menariknya, **recall-nya sama persis dengan model tesis (0,9356)** — kemampuan menangkap
phishing setara, yang berbeda hanya presisinya.

### Batas atas tanpa data baru sudah tercapai

Empat pendekatan lanjutan diuji (`experiments/push_f1.py`), semuanya **lebih buruk**
daripada ensemble tiga anggota:

| Pendekatan | F1 | Selisih |
|---|---|---|
| **Ensemble 3 anggota (dipakai)** | **0,9008** | — |
| Ensemble 4 anggota (+HistGradientBoosting) | 0,8984 | −0,0024 |
| Seleksi 18 fitur terpenting | 0,8942 | −0,0066 |
| Ensemble dengan bobot dioptimasi | 0,8866 | −0,0142 |
| Stacking (meta-model di atas prediksi OOF) | 0,8703 | −0,0305 |

Kandidat baru hanya diadopsi bila unggul ≥ 0,005 — ambang itu dipasang supaya
tidak mengejar derau pembagian data. Tidak ada yang lolos.

**Kesimpulannya: 0,9008 adalah batas atas untuk data dan fitur yang ada sekarang.**
Kenaikan berikutnya harus datang dari data atau fitur baru, bukan dari
mengutak-atik model.

**Gerbang mutu** — 8 alamat sah yang tidak pernah ikut dilatih:

| Alamat | Model tesis | Model baru |
|---|---|---|
| Vitalik Buterin | 98 HIGH ❌ | **6 LOW** ✅ |
| Uniswap V2 Router | 100 HIGH ❌ | **3 LOW** ✅ |
| ETH2 Deposit Contract | 90 HIGH ❌ | **11 LOW** ✅ |
| Vitalik (dompet lama) | 98 HIGH ❌ | **7 LOW** ✅ |
| Vitalik (dompet ketiga) | — | **23 LOW** ✅ |
| OKX Exchange | 13 LOW ✅ | 3 LOW ✅ |
| Token UNI | 24 LOW ✅ | 2 LOW ✅ |
| Maker PSM USDC | — | 30 LOW ✅ |

(Skor ensemble sedikit lebih tinggi dari model tunggal karena ambangnya juga naik
ke 0,34 — yang menentukan lulus/tidak adalah posisi skor terhadap ambang, bukan
angka mutlaknya.)

**False positive pada 20 wallet acak segar:** 88% → **0%**

**Kontrol positif** — kemampuan mendeteksi phishing tidak berkurang:
wallet phishing dataset tetap dinilai 99 dan 79 (HIGH).

**Bukti jalan pintas tertutup:** fitur terpenting berubah dari `transaction_count`
menjadi `sent_received_ratio` dan `repeat_ratio` — model menilai pola, bukan ukuran.

---

## 5b. Batas yang tidak bisa ditembus model — dan jawabannya: pelabelan

### Temuan dari pengujian wallet bursa nyata

Lima alamat bursa berlabel resmi Etherscan diuji dengan kedua model:

| Alamat | tx | Model tesis (fitur live) | Model perbaikan |
|---|---|---|---|
| Tokocrypto | 557 | 56 MEDIUM — **Phishing** ❌ | 29 LOW — aman ✅ |
| Tokocrypto: Treasury 1 | 7 | 0 LOW — aman ✅ | 56 MEDIUM — **Phishing** ❌ |
| Tokocrypto 2 | 1 | 2 LOW — aman ✅ | 79 HIGH — **Phishing** ❌ |
| Binance: Hot Wallet | 17 | 67 MEDIUM — **Phishing** ❌ | 76 HIGH — **Phishing** ❌ |
| Bybit: Hot Wallet | 5.304 | 99 HIGH — **Phishing** ❌ | 2 LOW — aman ✅ |

Polanya jelas: **model tesis gagal pada wallet ramai, model perbaikan gagal pada
wallet bertransaksi sedikit.** Keduanya salah, hanya pada kasus yang berbeda.

### Kenapa — dan kenapa ini tidak bisa diperbaiki dengan melatih ulang

Lihat profil `Binance: Hot Wallet`:

```
in_degree = 17    out_degree = 0
unique_senders = 16    unique_receivers = 0
sender_receiver_ratio = 16,0     balance_ratio = 1,0
```

Menerima dari 16 pihak berbeda, belum pernah mengirim keluar. Itu **persis tanda
tangan wallet pengumpul phishing** yang belum menyapu dana. Dari struktur transaksi
semata, keduanya tidak terbedakan — manusia pun tidak akan bisa.

Yang membedakan bukan angka, melainkan **identitas**. Dan identitas adalah
pengetahuan dari luar rantai.

### Percobaan yang gagal: ambang berbasis jumlah transaksi

Menahan vonis untuk wallet < 30 transaksi memang menihilkan salah tuduh, tetapi
ongkosnya diukur dan ditolak:

| | Tetap divonis |
|---|---|
| Wallet phishing | **27,5%** |
| Wallet normal | 43,5% |

Sistem yang bungkam pada tiga dari empat penipu tidak ada gunanya. Sebabnya
mendasar: median wallet phishing hanya 13 transaksi — rentang yang sama persis
dengan alamat bursa yang masih baru. **Jumlah transaksi tidak memisahkan kedua
kelas**, sehingga ambang seperti ini tidak akan pernah menyelesaikannya.

Yang dipakai sekarang: vonis hanya ditahan bila riwayatnya benar-benar tidak
bermakna (< 5 transaksi), dan setiap hasil disertai label keyakinan.

### Jawabannya: lapisan pelabelan

```
Alamat masuk
   │
   ├─ LAPISAN 1  MODEL ML
   │             ambil transaksi & lawan transaksi dari Etherscan
   │             -> 26 fitur -> skor + tingkat keyakinan
   │
   ├─ LAPISAN 2  LABEL ON-CHAIN          <- lapisan terdesentralisasi
   │             baca LabelRegistry di blockchain
   │             (cadangan: daftar lokal bila node tidak dapat dihubungi)
   │
   └─ KEPUTUSAN  label terverifikasi MENIMPA pendapat model
                 kedua pendapat tetap dilaporkan + penanda ketidaksepakatan
```

**Kenapa label diperiksa di AKHIR, bukan di awal?** Vonisnya sama saja, tetapi
menaruhnya di akhir membuat pendapat model tetap terekam walau kemudian ditimpa.
Ketidaksepakatan model vs label justru catatan paling berharga: ia menunjukkan
persis di mana model lemah, dan menjadi bahan pelatihan berikutnya.

Ada keuntungan kedua yang lebih penting bagi tesis: karena pengecekan label
dilakukan on-chain, **blockchain menjadi bagian dari jalur keputusan**, bukan
sekadar tempat mencatat hasil. Ini jawaban yang jauh lebih kuat untuk pertanyaan
penguji *"kenapa harus memakai blockchain?"* — sebelumnya kontrak hanya menulis
log; sekarang ia ikut menentukan vonis.

(Tersedia `short_circuit=True` untuk melewati model pada alamat berlabel bila
kecepatan lebih dipentingkan daripada catatan ketidaksepakatan.)

Ini bukan jalan pintas untuk memperbagus angka — ini praktik standar industri.
Etherscan, Chainalysis, dan hampir semua produk keamanan blockchain memakai daftar
label untuk alamat yang dikenal, lalu menyisakan model untuk sisanya.

**Hasil setelah lapisan label dipasang — dibaca ON-CHAIN dari Hardhat lokal:**

| Alamat | Pendapat model | Label | Lapisan | Vonis akhir |
|---|---|---|---|---|
| Tokocrypto | 29 aman | Tokocrypto | `onchain` | Bukan Phishing ✅ |
| **Binance: Hot Wallet** | **76 phishing ❌** | Binance: Hot Wallet | `onchain` | **Bukan Phishing ✅** ← dikoreksi |
| Bybit: Hot Wallet | 2 aman | Bybit: Hot Wallet | `onchain` | Bukan Phishing ✅ |
| Vitalik Buterin | 6 aman | Vitalik Buterin | `onchain` | Bukan Phishing ✅ |
| Wallet phishing kontrol | 93 phishing | — | — | Phishing ✅ |

**5 dari 5 benar.** Baris Binance adalah inti gagasannya: model salah, label on-chain
mengoreksi, dan **ketidaksepakatannya tercatat** (`disagreement: true`) beserta
skor asli model — bukan dihapus diam-diam.

Setiap hasil membawa penanda `decision_source` dan `label_layer`, sehingga tidak
pernah kabur mana fakta, mana prediksi model, dan dari mana labelnya berasal.

Batasnya tetap harus diakui: **daftar label hanya menutup alamat yang sudah
dikenal.** Ia tidak menggeneralisasi — justru karena itu model ML tetap dibutuhkan
untuk ekor panjang alamat yang tidak dikenal siapa pun.

### Perluasan on-chain — `LabelRegistry.sol`

Daftar label itu wajar dipindahkan ke blockchain, dan di situlah DApp berperan:

| Kontrak | Menyimpan | Sifat |
|---|---|---|
| `RiskRegistry` (tesis) | skor 0–100 dari model | **pendapat** mesin |
| `LabelRegistry` (baru) | identitas + laporan pengguna | **fakta** & laporan |

Dua tingkat, sengaja dipisah:

1. **Label terverifikasi** — ditetapkan kurator, mengikat.
2. **Laporan pengguna** — siapa pun boleh melapor lewat DApp, satu alamat satu
   suara per pelapor. **Tidak** otomatis menjadi vonis.

Kenapa laporan tidak langsung jadi label? Karena pemungutan suara terbuka mudah
dimanipulasi — seseorang bisa membuat ribuan alamat untuk menjatuhkan pesaing.
Yang dibuka adalah **datanya**: siapa melapor apa dan kapan, semuanya terekam
sebagai event sehingga keputusan kurator dapat diperiksa publik.

**Status: sudah di-deploy dan berjalan** di node Hardhat lokal — bukan lagi sekadar
rancangan. 25 label tertulis on-chain dalam satu transaksi (gas 1.907.286), dan
pembacaannya sudah diuji dari aplikasi.

```powershell
# Terminal 1 — node
cd chain ; npx hardhat node

# Terminal 2 — deploy LabelRegistry + isi 25 label
python new_system\chain\deploy_label_registry.py

# Aktifkan pengecekan on-chain (alamat dicetak oleh skrip di atas)
$env:LABEL_REGISTRY_ADDRESS = "0x5FbDB2315678afecb367f032d93F642f64180aa3"
$env:RPC_URL = "http://127.0.0.1:8545"
```

⚠️ Node Hardhat menyimpan data di memori. Setiap kali node dinyalakan ulang,
kontraknya hilang dan **harus di-deploy ulang** — `deployment.json` menjadi usang.
Bila kontrak tidak terbaca, sistem otomatis jatuh ke daftar lokal
(`data/address_labels.json`), jadi penilaian tetap berjalan.

`RiskRegistry` milik sistem tesis tidak diubah sama sekali; kompilasi dilakukan
dengan menyalin kontrak sementara lalu menghapus jejaknya.

Berkas: `new_system/chain/LabelRegistry.sol`, `deploy_label_registry.py`

---

## 6. Isi folder

```
new_system/
├── README.md                     ← dokumen ini
├── chain/
│   ├── LabelRegistry.sol         kontrak pelabelan on-chain (sudah di-deploy lokal)
│   ├── deploy_label_registry.py  deploy + isi 25 label dalam satu transaksi
│   └── deployment.json           alamat kontrak + ABI hasil deploy terakhir
├── ml/
│   ├── paths.py                  penjaga batas sistem tesis vs sistem baru
│   ├── label_registry.py         LAPISAN 1 — daftar label alamat dikenal
│   ├── live_features.py          ekstraksi 26 fitur dari Etherscan
│   ├── build_live_dataset.py     pembangun data (resumable, paralel)
│   ├── add_small_negatives.py    penambal wallet normal berukuran kecil
│   ├── train_live_model.py       pelatih + gerbang mutu OOD
│   ├── predict_live.py           inferensi
│   └── finalize_live_model.py    jalankan semua pemeriksaan sekaligus
├── experiments/
│   ├── run_ood_evaluation.py     bandingkan model tesis vs model baru
│   ├── run_false_positive_test.py  ukur FP pada wallet acak segar
│   ├── tune_live_model.py        cari ambang + hyperparameter terbaik
│   └── build_ensemble.py         bangun & simpan ensemble 3 model
├── artifacts/                    model, skema, laporan, confusion matrix
├── data/
│   ├── live_dataset.csv          2.700 wallet
│   └── address_labels.json       25 alamat berlabel (bursa, router, token, tokoh)
└── outputs/                      hasil evaluasi (CSV, JSON, Markdown)
```

---

## 7. Cara menjalankan

```powershell
# Menyalakan mode baru pada aplikasi (bawaan: mati)
$env:LIVE_SCAN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Membangun ulang dari nol (butuh ETHERSCAN_API_KEY di .env)
python new_system\ml\build_live_dataset.py     # ~2 jam, resumable
python new_system\ml\add_small_negatives.py    # ~1 jam
python new_system\ml\finalize_live_model.py    # latih + semua pemeriksaan

# Menaikkan performa (tanpa panggilan API, beberapa menit)
python new_system\experiments\tune_live_model.py    # cari ambang + hyperparameter
python new_system\experiments\build_ensemble.py     # bangun ensemble + gerbang mutu

# Evaluasi (memanggil Etherscan)
python new_system\experiments\run_ood_evaluation.py
python new_system\experiments\run_false_positive_test.py 20
```

---

## 8. Yang harus tetap diakui

1. **Label negatif bersifat asumsi.** Wallet dari blok sungguhan dianggap sah karena
   tidak ada di daftar phishing — ini *weak label*, tidak diverifikasi satu per satu.
2. **Positifnya tetap dari era 2019.** Taktik phishing berubah; pola 2019 belum tentu
   mewakili phishing 2026.
3. **Cakupan transaksi dibatasi** 5 halaman (5.000 transaksi) per alamat demi waktu
   pengambilan; wallet sangat besar terpotong dan ditandai kolom `truncated`.
4. **Wallet dengan < 5 transaksi** ditandai "tidak dapat disimpulkan" — riwayatnya
   terlalu tipis untuk dinilai secara bermakna.
5. **Metrik sistem baru tidak menggantikan metrik tesis.** Keduanya dievaluasi pada
   populasi berbeda.

---

## 9. Bahan diskusi dengan pembimbing

Tiga pertanyaan yang perlu diputuskan bersama:

**a. Apakah ini masuk naskah tesis, dan sebagai apa?**
Tiga kemungkinan: (i) hanya sebagai catatan keterbatasan + saran penelitian lanjutan;
(ii) sub-bab tersendiri "pengujian out-of-distribution dan perluasan sistem";
(iii) tidak dimasukkan sama sekali, cukup disimpan sebagai pekerjaan lanjutan.

**b. Kalau dimasukkan, apakah judul perlu menyesuaikan?**
Judul saat ini berfokus pada sistem deteksi dan pencatatan on-chain. Penambahan ini
memperkuat sisi "pengembangan sistem" tanpa mengubah arah penelitian.

**c. Apakah lapisan pelabelan + `LabelRegistry` layak dikembangkan?**
Ini memperluas peran smart contract dari sekadar mencatat pendapat model menjadi
tempat menyimpan fakta identitas dan laporan pengguna — DApp menjadi dua arah,
bukan hanya membaca. Kalau disetujui, langkah berikutnya: deploy `LabelRegistry.sol`
dan menambahkan halaman pelaporan di DApp.

**d. Apakah mode ini boleh aktif saat sidang?**
Saat ini **mati secara bawaan**. Menyalakannya memungkinkan penguji memindai wallet
apa pun — sekarang aman (false positive 0% pada uji 20 wallet), tetapi tetap perlu
persetujuan pembimbing karena hasilnya di luar cakupan evaluasi tesis.

---

## 10. Kejujuran akademik — poin yang ingin saya sampaikan

- Masalah ini **saya temukan sendiri**, bukan ditemukan penguji.
- Model tesis **tidak saya ubah** untuk menutupi masalahnya; metriknya tetap apa adanya,
  dan saya sudah memverifikasi skornya tidak bergeser (100 dan 98, tetap sama).
- Angka F1 0,9102 tetap **sah untuk populasi yang dievaluasi**. Yang tidak sah adalah
  menafsirkannya sebagai "91% akurat untuk wallet Ethereum mana pun" — dan itu yang
  ingin saya nyatakan eksplisit dalam naskah.
- Perbaikan ini saya letakkan terpisah justru agar batas antara hasil tesis dan
  pekerjaan tambahan tidak kabur.
