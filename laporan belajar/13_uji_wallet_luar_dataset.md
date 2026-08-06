# Uji Wallet di Luar Dataset (Out-of-Distribution)

**Latar belakang:** muncul dari pertanyaan Anda sendiri — *"kalau wallet-nya tidak ada di
data training, berarti tidak terdeteksi dong? Harusnya ada layer 1 yang ambil dari
Etherscan?"*

**Status:** sudah diuji dengan Etherscan API sungguhan pada alamat mainnet nyata.
Belum diintegrasikan ke aplikasi — dan laporan ini menjelaskan kenapa sebaiknya belum.

⭐ **Ini bukan sekadar catatan teknis. Ini temuan penelitian yang layak masuk bab
keterbatasan dan bab saran penelitian lanjutan.**

---

## 1. Idenya: layer 1 fallback

```
Pengguna masukkan wallet
        │
        ├── ada di hybrid_dataset.csv?  ──ya──►  pakai fitur precomputed  (yang ada sekarang)
        │
        └── tidak ada  ──►  LAYER 1: ambil live dari Etherscan
                             ├─ txlist  →  hitung 18 fitur graph (rumus sama)
                             └─ fetch_all → 7 fitur Etherscan
                             └─►  prediksi
```

Secara rekayasa, ini masuk akal dan bahannya sudah ada di repo.

---

## 2. Uji Bagian A — apakah 18 fitur bisa dihitung ulang dari Etherscan?

Metode: ambil wallet yang **sudah ada** di dataset, hitung 18 fitur dua kali —
dari MulDiGraph (nilai di `hybrid_dataset.csv`) dan dari `txlist` Etherscan
memakai rumus identik dengan `ml/preprocess.py:extract_node_features` — lalu bandingkan.

### Kasus A1 — `0x60afc576…4fc74b1` (phishing, tidak aktif sejak 2019)

**Seluruh 18 fitur identik sampai 4 desimal.** Contoh:

| Fitur | MulDiGraph | Etherscan live |
|---|---|---|
| `in_degree` | 68 | 68 |
| `unique_senders` | 50 | 50 |
| `total_received` | 52,9489 | 52,9489 |
| `sent_received_ratio` | 0,9997 | 0,9997 |
| `timestamp_spread_in` | 13.585.497 | 13.585.497 |

Skor: **100 vs 100**.

➡️ Membuktikan rumusnya benar dan sumbernya setara — untuk wallet yang aktivitasnya
sudah berhenti sebelum snapshot 2019 diambil.

### Kasus A2 — `0x903bb9cd…ccf06e5` (phishing, masih aktif setelah 2019)

| Fitur | MulDiGraph | Etherscan live | rasio |
|---|---|---|---|
| `in_degree` | 5.097 | 4.831 | 0,95× |
| `out_degree` | 135 | 358 | 2,65× |
| `unique_senders` | 1.626 | 1.685 | 1,04× |
| `sent_received_ratio` | 0,4705 | 1,0000 | 2,13× |
| `timestamp_spread_in` | 23.434.691 | 29.283.449 | 1,25× |

Skor: **98 vs 95** — keduanya HIGH / Phishing.

➡️ Ada selisih, dua sebabnya: (a) MulDiGraph hanya subgraph tersampling, tidak memuat
seluruh transaksi wallet; (b) ada transaksi baru setelah 2019. Tapi **vonisnya tetap sama**.

### Temuan teknis: batas paging Etherscan

Etherscan v2 mengembalikan **maksimal 1.000 record per halaman**, berapa pun nilai
parameter `offset`. Terverifikasi:

| Permintaan | Hasil |
|---|---|
| `offset=10000, page=1` | 1.000 record |
| `offset=1000, page=1` | 1.000 record |
| `offset=1000, page=2` | 1.000 record |
| `offset=1000, page=6` | 233 record (habis) |

⚠️ Ini berdampak pada kode yang sudah ada: `ml/data_fetchers.py:111` memakai
`offset=10000` dan `get_transaction_count()` mengembalikan `len(result)`. Artinya
**`transaction_count` di dataset Anda tertutup di 1.000, bukan 10.000** — walaupun nilai
maksimum di CSV tercatat 10.000, kemungkinan besar berasal dari proses enrichment lama.
Perlu dicek ulang kalau enrichment dijalankan lagi.

---

## 3. Uji Bagian B — wallet nyata di luar dataset 🔴

Semua fitur diambil live (18 graph dari `txlist` + 7 dari `fetch_all`), `graph_present=0`.

### B1 — alamat terkenal yang jelas aman

| Alamat | Identitas | Skor model | Vonis |
|---|---|---|---|
| `0xd8dA6BF2…aA96045` | Vitalik Buterin | **92** | 🔴 HIGH — "Phishing" |
| `0x7a250d56…9F2488D` | Uniswap V2 Router | **100** | 🔴 HIGH — "Phishing" |
| `0x00000000…d7705Fa` | ETH2 Deposit Contract | **84** | 🔴 HIGH — "Phishing" |

### B2 — 8 wallet acak dari blok terbaru

| Alamat | in_deg | out_deg | uniq_senders | Skor | Level |
|---|---|---|---|---|---|
| `0x43e21e22…` | 225 | 1.331 | 25 | 82 | HIGH |
| `0x2713eabb…` | 132 | 131 | 3 | 70 | HIGH |
| `0x7c5e8f00…` | 70 | 29 | 12 | 99 | HIGH |
| `0xcef64000…` | 3 | 3 | 1 | **18** | LOW ✅ |
| `0xa63712f1…` | 507 | 1.150 | 53 | 92 | HIGH |
| `0xb992ad53…` | 61 | 2.127 | 34 | 98 | HIGH |
| `0xd97beffd…` | 192 | 1.280 | 66 | 93 | HIGH |
| `0xf05850cf…` | 117 | 12 | 9 | 99 | HIGH |

**7 dari 8 dinyatakan HIGH.** Satu-satunya yang LOW adalah wallet dengan hanya
3 transaksi masuk dan 3 keluar.

---

## 4. Diagnosis: kenapa gagal

Penjelasan SHAP dari model konsisten di semua kasus — penyumbang terbesar selalu
`unique_senders`, dengan kontribusi **+3,1 hingga +4,9**:

```
Vitalik            unique_senders = 6189   kontribusi +3,156
Uniswap Router     unique_senders = 2316   kontribusi +3,700
ETH2 Deposit       unique_senders = 1098   kontribusi +4,872
```

Model mempelajari aturan: **"banyak pengirim unik ⇒ phishing"**.

**Kenapa aturan itu muncul?** Kembali ke cara sampling (lihat `02_pembentukan_dataset.md`):

- Kelas positif = 1.165 wallet phishing → memang menerima dari banyak korban.
- Kelas negatif = wallet normal **acak** dari MulDiGraph → mayoritas wallet kecil,
  aktivitas rendah, banyak yang cuma punya 1–5 transaksi.

Model tidak pernah melihat contoh **wallet sah yang sangat aktif** — bursa, router DEX,
kontrak populer, tokoh publik. Jadi ia menyimpulkan aktivitas tinggi = phishing.

Di dalam populasi 8.595 wallet snapshot 2019, aturan itu memisahkan dengan sangat baik
(F1 0,9102). Di luar populasi itu, aturannya **salah**.

Istilah teknisnya: **out-of-distribution failure** akibat **kelas negatif yang tidak
representatif** (kurangnya *hard negatives*).

---

## 5. Arti pentingnya untuk tesis

**Yang tetap sah:** F1 0,9102 tidak batal. Ia sah untuk populasi yang dievaluasi —
wallet dari EPTransNet snapshot 2019. Itu memang yang dilaporkan.

**Yang harus dinyatakan:** metrik itu **tidak** boleh ditafsirkan sebagai
"91% akurat untuk wallet Ethereum mana pun". Batas keberlakuannya harus eksplisit.

**Rumusan untuk bab keterbatasan:**
> "Model dievaluasi pada populasi tertutup, yaitu 8.595 wallet hasil sampling dari
> EPTransNet. Pengujian tambahan terhadap alamat di luar populasi tersebut menunjukkan
> tingkat false positive yang tinggi pada wallet sah beraktivitas tinggi, karena kelas
> negatif pada data latih didominasi wallet beraktivitas rendah. Dengan demikian sistem
> ini sahih sebagai purwarupa untuk populasi terevaluasi, dan belum layak dipakai untuk
> penilaian wallet arbitrer tanpa pelatihan ulang memakai *hard negatives*."

⭐ Menyajikan ini sebagai **evaluasi out-of-distribution yang Anda rancang sendiri**
jauh lebih kuat daripada menunggu penguji menemukannya. Banyak tesis S2 tidak pernah
menguji modelnya di luar test set — Anda melakukannya.

---

## 6. Tiga pilihan tindakan

### Pilihan 1 — Dokumentasikan saja, jangan bangun layer 1 (paling aman)
- **Kerja:** menulis, ~1 jam.
- **Risiko:** nol. Semua angka tesis tetap.
- **Untung:** dapat bab keterbatasan yang kuat + saran penelitian lanjutan yang konkret.
- ⚠️ **Kalau layer 1 dibangun tanpa perbaikan model, penguji tinggal mengetik alamat
  wallet-nya sendiri saat sidang dan sistem akan menuduhnya phishing.** Ini risiko nyata.

### Pilihan 2 — Bangun layer 1 sebagai "Mode Eksplorasi" berlabel jelas
- **Kerja:** ~3–4 jam (modul fitur live + fallback di `ml_service` + peringatan di UI).
- **Syarat mutlak:** hasilnya diberi label tegas *"estimasi di luar distribusi data latih —
  bukan penilaian yang tervalidasi"*, dan skor tidak ditulis ke blockchain.
- **Untung:** menunjukkan kelengkapan rekayasa sistem.
- **Risiko:** tetap ada kemungkinan salah paham saat demo.

### Pilihan 3 — Perbaiki akar masalahnya: latih ulang dengan *hard negatives* (paling benar)
- **Kerja:** 2–4 hari.
- **Caranya:** tambahkan kelas negatif berupa wallet sah beraktivitas tinggi — sampel dari
  blok terbaru, daftar bursa/kontrak populer, alamat berlabel di Etherscan.
- **Untung:** model jadi benar-benar bisa dipakai, dan ini kontribusi ilmiah nyata.
- **Risiko:** ⚠️ **seluruh angka tesis berubah** — 4×4, late fusion, ablasi, model produksi
  harus dilatih ulang semua. Jangan dikerjakan menjelang bimbingan.

**Rekomendasi:** **Pilihan 1 sekarang**, tulis Pilihan 3 sebagai saran penelitian lanjutan.
Pertimbangkan Pilihan 2 hanya kalau dosen secara khusus meminta demo wallet bebas.

---

## 7. Kalau nanti layer 1 tetap dibangun — rancangan pengamannya

| Pengaman | Cara | Alasan |
|---|---|---|
| Deteksi kontrak vs EOA | `eth_getCode` — kalau ada bytecode, itu kontrak | Uniswap Router & ETH2 Deposit bukan wallet pengguna; model tidak dilatih untuk itu |
| Daftar putih | Bursa, router, kontrak populer | Mencegah false positive paling memalukan |
| Ambang berbeda untuk mode live | Naikkan ambang phishing dari 0,5 | Mengurangi FP di distribusi yang berbeda |
| Label distribusi | Tandai hasil sebagai `out_of_distribution: true` | Kejujuran ke pengguna |
| Larang tulis on-chain | Skor mode live tidak dicatat ke RiskRegistry | Catatan permanen tidak boleh berisi tuduhan yang tidak tervalidasi |
| Paging penuh | `offset=1000`, ulang sampai < 1000 | Kalau tidak, fiturnya terpotong (terbukti di §2) |

---

## 8. Cara menjawab kalau dosen bertanya soal ini

> "Sistem saat ini melayani 8.595 wallet dalam populasi terevaluasi. Saya sudah menguji
> kemungkinan perluasan ke wallet arbitrer dengan mengambil fitur langsung dari Etherscan,
> dan secara teknis 18 fitur graph itu bisa direproduksi persis — saya verifikasi ada
> wallet yang nilainya identik sampai empat desimal.
>
> Namun pengujian out-of-distribution menunjukkan model memberi false positive tinggi
> pada wallet sah beraktivitas tinggi: Vitalik Buterin mendapat skor 92, Uniswap Router
> 100. Analisis SHAP menunjukkan penyebabnya `unique_senders` — model mempelajari bahwa
> banyak pengirim unik berarti phishing, karena kelas negatif pada data latih didominasi
> wallet beraktivitas rendah.
>
> Karena itu saya tidak mengaktifkan mode tersebut, dan saya laporkan sebagai
> keterbatasan sekaligus arah penelitian lanjutan: pelatihan ulang dengan *hard negative
> sampling* yang menyertakan wallet sah beraktivitas tinggi."

Jawaban itu memperlihatkan Anda menguji sistem sendiri, memahami sebabnya sampai ke level
fitur, dan tahu cara memperbaikinya. Itu justru yang dicari penguji.

---

## Lampiran — cara mengulang pengujian ini

Skrip PoC ada di folder scratchpad sesi ini (di luar repo, tidak ikut ter-commit):
`poc_live.py` (bandingkan MulDiGraph vs Etherscan), `poc_live2.py` (alamat terkenal),
`poc_live3.py` (wallet acak dari blok terbaru).

Kalau ingin dipermanenkan ke repo, tempat yang wajar: `experiments/run_ood_evaluation.py`,
supaya hasilnya bisa dikutip di naskah tesis sebagai eksperimen resmi.
