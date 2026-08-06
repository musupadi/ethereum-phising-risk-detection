# Model Perluasan (Model B) — Wallet di Luar Dataset

**Status:** ✅ selesai, dilatih, dan diverifikasi. Semua gerbang mutu lulus.

📁 **Kodenya tinggal di folder terpisah [`new_system/`](../new_system/README.md)**, bukan
menyatu dengan sistem tesis — supaya batas antara hasil tesis dan pekerjaan tambahan
tidak kabur. Mode ini **mati secara bawaan** (`LIVE_SCAN_ENABLED=false`).

---

## 1. Masalah yang diperbaiki

Model tesis (Model A) memberi skor HIGH kepada Vitalik Buterin (92), Uniswap V2 Router
(100), dan ETH2 Deposit Contract (84) — semuanya alamat sah. Dari 8 wallet acak,
7 dinilai HIGH.

Dua akar penyebabnya:

| # | Akar masalah | Penjelasan |
|---|---|---|
| 1 | **Kelas negatif tidak representatif** | Wallet normal pada data latih diambil acak dari MulDiGraph, mayoritas wallet kecil beraktivitas rendah. Model tidak pernah melihat wallet sah yang ramai, sehingga menyimpulkan "banyak pengirim unik = phishing". |
| 2 | **Ketidakcocokan sumber fitur** | Model dilatih pada fitur turunan MulDiGraph (snapshot subgraph 2019) tetapi dipakai pada fitur turunan Etherscan (riwayat penuh sampai hari ini). Dua distribusi berbeda untuk nama fitur yang sama. |

---

## 2. Keputusan desain: dua model berdampingan

**Model A tidak diubah sama sekali.** Ini keputusan sadar: mengubahnya berarti seluruh
angka tesis (tabel 4×4, late fusion, ablasi) harus dihitung ulang.

| | Model A — tesis | Model B — perluasan |
|---|---|---|
| Artefak | `ml/artifacts/model_prod_xgb.json` | `new_system/artifacts/model_live_xgb.json` |
| Melayani | 8.595 wallet di dalam dataset | wallet arbitrer |
| Sumber fitur | MulDiGraph + Etherscan (precomputed) | Etherscan, diambil saat itu juga |
| Kelas negatif | wallet acak MulDiGraph | + *hard negatives* |
| Ditulis on-chain | ✅ ya | ❌ tidak |
| Angka di naskah tesis | F1 0,9102 — tetap | dilaporkan terpisah |

---

## 3. Perbaikan yang diterapkan

### 3.1 Hard negative sampling

Kelas negatif kini terdiri dari tiga kelompok:

| Kelompok | Jumlah | Alasan |
|---|---|---|
| Alamat sah terkenal (bursa, router DEX, token, bridge, kontrak staking) | 40 | Justru kelompok inilah yang salah dinilai Model A |
| Wallet acak dari blok sungguhan, tersebar 2017–sekarang | ~2.200 | Mewakili pengguna Ethereum biasa lintas era |
| Alamat sah **held-out** (Vitalik, Uniswap Router, ETH2 Deposit, dll) | 8 | Tidak pernah dilatih — dipakai sebagai gerbang mutu |

Pemisahan held-out itu penting: kalau Vitalik ikut dilatih lalu diuji pada Vitalik,
evaluasinya sirkular dan tidak membuktikan apa pun.

### 3.2 Fitur satu sumber

Seluruh fitur — untuk kedua kelas — dihitung dari Etherscan memakai rumus yang sama
persis dengan `ml/preprocess.py`. Ketidakcocokan sumber hilang.

Kesetaraan rumusnya sudah diverifikasi: untuk wallet phishing yang tidak aktif sejak
2019, ke-18 fiturnya **identik sampai 4 desimal** antara versi MulDiGraph dan versi
Etherscan (lihat `13_uji_wallet_luar_dataset.md` §2).

### 3.3 Fitur rasio yang tak bergantung skala

Ditambahkan empat fitur baru yang mengukur **pola**, bukan **besaran** — inilah kunci
agar wallet ramai yang sah tidak otomatis dicurigai:

| Fitur | Rumus | Pola yang ditangkap |
|---|---|---|
| `sender_receiver_ratio` | `unique_senders / (unique_receivers + 1)` | Phishing: banyak korban masuk, sedikit penampung keluar |
| `repeat_ratio` | `total_tx / (counterparties + 1)` | Bursa & router: berinteraksi berulang dengan pihak yang sama |
| `balance_ratio` | `balance / (total_received + ε)` | Phishing: saldo dikuras habis, mendekati nol |
| `activity_span_days` | `max(spread_in, spread_out) / 86400` | Phishing: umur pendek |

Contoh nyata pada wallet phishing `0x60afc576…`: `sender_receiver_ratio` = 16,7
(50 pengirim → 2 penerima) dan `balance_ratio` = 0 (dikuras habis).

### 3.4 Deteksi kontrak yang benar

Ditambahkan `is_contract` lewat `eth_getCode`. Saat pengujian ditemukan bahwa alamat
Vitalik mengembalikan `0xef0100…` — itu **delegasi EIP-7702**, yang menandakan EOA
dengan akun terdelegasi, bukan kontrak. Tanpa penanganan khusus, ia salah
diklasifikasikan sebagai kontrak. Sudah diperbaiki di `live_features.py:_fetch_is_contract`.

### 3.5 Fitur yang SENGAJA tidak dipakai

| Fitur | Alasan dibuang |
|---|---|
| `created_at_ts` | Waktu absolut. Karena seluruh wallet phishing berasal dari era 2019, model bisa mengambil jalan pintas "wallet tua = phishing" — memindahkan bias, bukan menghapusnya. |
| `tx_frequency` | Persis `activity_score / 30`; redundan sempurna. |
| `graph_present` | Artefak MulDiGraph; tidak bermakna untuk wallet arbitrer. |

Penghindaran jalan pintas berbasis era juga menjadi alasan hard negatives disampel
**lintas era 2017–sekarang**, bukan hanya dari blok terbaru.

---

## 4. Cara menjalankan ulang

```powershell
python new_system\ml\build_live_dataset.py      # ambil data (resumable, ~2 jam)
python new_system\ml\train_live_model.py        # latih + gerbang mutu OOD
python new_system\experiments\run_ood_evaluation.py   # bandingkan Model A vs Model B
```

---

## 5. Integrasi ke aplikasi

```
POST /analyze { wallet }
   ├── ada di hybrid_dataset.csv ?  ── ya ──► Model A ──► skor ──► ditulis on-chain
   └── tidak ada                            ► Model B ──► skor ──► TIDAK ditulis on-chain
                                                          + out_of_distribution: true
                                                          + warning ditampilkan di UI
```

Pengaman yang dipasang:

| Pengaman | Letak |
|---|---|
| Skor mode live tidak pernah ditulis ke blockchain | `api/main.py` |
| Respons ditandai `out_of_distribution` + `warning` | `api/schemas.py`, `ml_service.py` |
| Banner peringatan tampil **sebelum** skor | `app/streamlit_app.py` |
| **Mati secara bawaan** (`LIVE_SCAN_ENABLED=false`) | `api/config.py` |
| Sistem tetap jalan bila model live belum ada | `ml_service._predict_live` |
| Wallet dengan < 5 transaksi ditandai "tidak dapat disimpulkan" | `new_system/ml/predict_live.py` |
| Kode terpisah di `new_system/`, tidak menimpa apa pun milik tesis | `new_system/ml/paths.py` |

### Bug yang ditemukan saat uji ujung-ke-ujung

Uji lewat server HTTP sungguhan (bukan pemanggilan fungsi langsung) menemukan bahwa
**uvicorn tidak membaca `.env`**, sehingga mode live gagal dengan
`ML error: ETHERSCAN_API_KEY tidak tersedia`. Diperbaiki dengan memuat `.env` di
`api/config.py` saat impor. Ini contoh kenapa uji lewat jalur nyata penting — uji
tingkat fungsi tidak akan menangkapnya.

Verifikasi akhir lewat `POST /analyze`:

| Wallet | model_used | Skor | out_of_distribution | tx_hash |
|---|---|---|---|---|
| `0x60afc576…` (dataset) | `dataset` | 100 HIGH | false | ditulis on-chain |
| `0xd8dA6BF2…` (Vitalik) | `live` | **1 LOW** | true | `null` — sengaja |

---

## 6. Hasil pelatihan

### 6.1 Komposisi data latih

| Sumber | Label | Jumlah |
|---|---|---|
| `phishing_eptransnet` — wallet phishing EPTransNet | 1 | 1.165 |
| `block_sample` — wallet acak dari blok lintas era | 0 | 587 |
| `block_sample_small` — wallet normal berukuran kecil | 0 | 900 |
| `curated` — bursa, router DEX, token, bridge | 0 | 40 |
| `curated_heldout` — alamat sah uji (tidak dilatih) | 0 | 8 |
| **Total** | | **2.700** (1.165 : 1.535) |

Split: 1.722 latih / 431 validasi / 539 uji.

### 6.2 Metrik pada test split

| Metrik | Nilai |
|---|---|
| F1 | **0,8739** |
| Precision | 0,8855 |
| Recall | 0,8627 |
| Accuracy | 0,8924 |
| ROC-AUC | 0,9685 |
| PR-AUC | 0,9604 |
| Confusion | TN 280 · FP 26 · FN 32 · TP 201 |

> ⚠️ **Jangan bandingkan 0,8739 ini dengan F1 0,9102 milik model tesis.** Keduanya
> dievaluasi pada populasi yang berbeda. Model B menghadapi tugas yang jauh lebih sulit:
> membedakan phishing dari wallet normal yang **beragam ukurannya**, bukan dari wallet
> kecil pasif belaka.

### 6.3 Gerbang mutu out-of-distribution — LULUS 8/8

Delapan alamat ini tidak pernah ikut dilatih:

| Alamat | Model A (tesis) | Model B (perluasan) |
|---|---|---|
| Vitalik Buterin | 98 HIGH ❌ | **1 LOW** ✅ |
| Uniswap V2 Router | 100 HIGH ❌ | **1 LOW** ✅ |
| ETH2 Deposit Contract | 90 HIGH ❌ | **1 LOW** ✅ |
| Vitalik (dompet lama) | 98 HIGH ❌ | **1 LOW** ✅ |
| WETH | 99 HIGH ❌ | **1 LOW** ✅ |
| OKX Exchange | 13 LOW ✅ | 0 LOW ✅ |
| Token UNI | 24 LOW ✅ | 2 LOW ✅ |
| Token USDC | 18 LOW ✅ | 1 LOW ✅ |

### 6.4 Kontrol positif — deteksi phishing tetap tajam

| Wallet | Model A | Model B |
|---|---|---|
| Phishing dataset #1 | 100 HIGH ✅ | **99 HIGH** ✅ |
| Phishing dataset #2 | 100 HIGH ✅ | **79 HIGH** ✅ |

Perbaikan ini **tidak** mengorbankan kemampuan mendeteksi phishing.

### 6.5 Uji false positive pada 20 wallet acak segar

Alamat diambil dari blok acak dengan seed berbeda, dan alamat yang pernah dilatih
maupun dievaluasi disingkirkan lebih dulu.

| | Model A | Model B |
|---|---|---|
| Ditandai phishing | 7 dari 8 (**88%**) | **0 dari 20 (0%)** |

Artefak: `outputs/false_positive_test.csv`, `outputs/ood_evaluation_report.md`

### 6.6 Ringkasan perbandingan langsung

Pada 10 alamat uji (8 sah + 2 kontrol phishing):

| | Model A | Model B |
|---|---|---|
| Benar | 5/10 (50%) | **10/10 (100%)** |
| False positive pada alamat sah | **5** | **0** |

### 6.7 Bukti jalan pintas sudah tertutup

Peringkat kepentingan fitur (gain) berubah total setelah wallet normal kecil ditambahkan:

| Peringkat | Model sementara (sebelum ditambal) | Model final |
|---|---|---|
| 1 | `transaction_count` (13) | **`sent_received_ratio`** (15) |
| 2 | `repeat_ratio` (12) | `repeat_ratio` (9) |
| 3 | `avg_out_amount` (6) | `unique_senders` (8) |
| … | | `min_out_amount` (7), `avg_out_amount` (7) |
| — | | `transaction_count` turun ke peringkat 7 (5) |

Artinya model kini memutuskan berdasarkan **pola perilaku** — perbandingan dana masuk
terhadap dana keluar, dan seberapa berulang interaksinya dengan pihak yang sama —
bukan lagi berdasarkan **seberapa besar** wallet itu.

Bukti komposisi yang mendukung: proporsi wallet kecil (≤60 transaksi) kini
**83% pada positif vs 68% pada negatif** — sebelum ditambal 83% vs 24%.

### 6.8 Uji regresi

Model tesis diperiksa lewat jalur produksi yang sesungguhnya (`ml_service`):

| Wallet | Diharapkan | Didapat | Model dipakai |
|---|---|---|---|
| `0x60afc576…` | 100 | **100** | dataset ✅ |
| `0x903bb9cd…` | 98 | **98** | dataset ✅ |

Model tesis tidak berubah sedikit pun.

---

## 7. Batasan yang harus tetap diakui

Perbaikan ini **tidak** membuat sistem siap produksi. Yang tetap berlaku:

1. **Label negatif bersifat asumsi.** Wallet dari blok sungguhan dianggap sah karena
   tidak ada di daftar phishing — ini *weak label*, tidak diverifikasi satu per satu.
   Ada kemungkinan sebagian kecil justru wallet phishing yang belum ketahuan.
2. **Positifnya tetap dari era 2019.** Taktik phishing berubah; pola 2019 belum tentu
   mewakili phishing 2026.
3. **Cakupan transaksi dibatasi** 5 halaman (5.000 transaksi) per alamat demi waktu
   pengambilan. Wallet sangat besar terpotong, dan ini ditandai lewat kolom `truncated`.
4. **Metrik Model B tidak menggantikan metrik tesis.** Keduanya dievaluasi pada populasi
   berbeda dan tidak boleh dibandingkan langsung.

---

## 8. Cara menyampaikannya ke dosen

> "Saya menemukan bahwa model tesis saya, meskipun mencapai F1 91% pada populasi yang
> dievaluasi, memberi false positive tinggi ketika diuji pada wallet di luar populasi
> itu — Vitalik Buterin mendapat skor 92. Analisis SHAP menunjukkan penyebabnya:
> model mempelajari bahwa banyak pengirim unik berarti phishing, karena kelas negatif
> pada data latih didominasi wallet beraktivitas rendah.
>
> Saya tidak mengubah model tesis, karena metriknya sah untuk populasi yang dievaluasi.
> Sebagai gantinya saya membangun model perluasan terpisah dengan dua perbaikan:
> *hard negative sampling* yang menyertakan wallet sah beraktivitas tinggi, dan fitur
> yang seluruhnya berasal dari satu sumber sehingga tidak ada ketidakcocokan distribusi.
> Hasil model perluasan itu selalu ditandai sebagai estimasi dan tidak dicatat ke
> blockchain."

Yang ditunjukkan jawaban ini: Anda menguji sistem sendiri di luar test set, memahami
sebabnya sampai level fitur, memperbaikinya, dan tetap jujur soal batasnya.
