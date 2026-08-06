# Dari Mana Data Etherscan Itu Datang?

Materi tambahan di luar TAHAP 1–9, menjawab: *"7 fitur Etherscan itu asalnya dari mana,
dan bagaimana cara mengambilnya?"*

---

## 1. Salah paham yang harus diluruskan dulu

**Etherscan bukan dataset.** Tidak ada file "dataset Etherscan" yang Anda unduh.

Etherscan adalah **block explorer** — situs web yang mengindeks seluruh isi blockchain
Ethereum dan menyediakan **API**. Anda **bertanya satu alamat pada satu waktu**, dan
Etherscan menjawab dengan data alamat itu.

Jadi jawaban yang benar untuk dosen:

> "Data Etherscan tidak saya unduh sebagai dataset. Saya mengambilnya sendiri lewat
> Etherscan API v2, satu alamat per permintaan, untuk daftar alamat yang saya susun
> dari EPTransNet. Hasilnya saya simpan sebagai `data/raw/addresses_etherscan_enriched.csv`."

Inilah yang membuat dataset Anda **tidak ada duanya** — kombinasi kolomnya Anda yang
membuat, bukan hasil unduhan.

---

## 2. Rantai lengkapnya — 3 langkah

```
LANGKAH 1                     LANGKAH 2                        LANGKAH 3
MulDiGraph.pkl        ──►     Etherscan API v2         ──►     hybrid_dataset.csv
(punya alamat + label)        (punya data on-chain terkini)     (gabungan keduanya)

ml/extract_addresses          ml/enrich_real_addresses          ml/build_hybrid
  _from_graph.py                _etherscan.py                     _dataset.py
       ↓                              ↓                                ↓
 addresses.csv               addresses_etherscan             hybrid_dataset.csv
 (12.815 alamat)               _enriched.csv                  (8.595 × 41)
```

Kuncinya: **daftar alamatnya berasal dari MulDiGraph, bukan dari Etherscan.**
Etherscan hanya diminta melengkapi informasi untuk alamat-alamat itu.

---

## 3. LANGKAH 1 — Menyusun daftar alamat (`ml/extract_addresses_from_graph.py`)

Skrip ini membuka MulDiGraph dan menyaring node-nya:

```python
for node, attrs in graph.nodes(data=True):
    if not ADDRESS_RE.match(str(node)):     # harus 0x + 40 hex   (:51)
        continue
    label = int(attrs.get("isp", 0) or 0)   # label phishing dari graph  (:54)
    row = {"address": node_str, "label": label,
           "class_name": "phishing" if label == 1 else "normal",
           "source_dataset": "EPTransNet"}
```

Lalu sampling:
```python
ratio = int(os.getenv("NORMAL_SAMPLE_RATIO", "10"))     # :70
max_normals = len(phishing_df) * ratio                  # 1.165 × 10 = 11.650
random.seed(RANDOM_SEED)                                # 42, reproducible
```

**Hasilnya — hafalkan angka ini:**

| File | Isi |
|---|---|
| `addresses_phishing.csv` | **1.165** alamat phishing (semua diambil, tidak di-sampling) |
| `addresses_normal.csv` | **11.650** alamat normal (sampel acak, rasio 10:1, seed 42) |
| `addresses.csv` | **12.815** baris — gabungan keduanya |

Perhatikan: **semua** alamat phishing dipakai, yang di-sampling hanya yang normal.
Itu keputusan yang benar — kelas minoritas jangan dikurangi.

---

## 4. LANGKAH 2 — Bertanya ke Etherscan (`ml/enrich_real_addresses_etherscan.py`)

### Endpoint yang dipakai

```python
ETHERSCAN_API_BASE = "https://api.etherscan.io/v2/api"   # data_fetchers.py:19
```

Butuh **API key gratis** dari etherscan.io, disimpan di `.env`:
```
ETHERSCAN_API_KEY=xxxxx
ETHERSCAN_REQUEST_DELAY=0.4
ETHERSCAN_PER_RUN_LIMIT=0
```

### Untuk SATU alamat, ada 4 panggilan API

Lihat `EtherscanFetcher.fetch_all()` (`data_fetchers.py:337-347`):

| # | Panggilan | `module` / `action` | Menghasilkan |
|---|---|---|---|
| 1 | `get_account_info` | `account` / `balance` | `balance` (Wei ÷ 1e18 → ETH) |
| 2 | `get_transaction_count` | `account` / `txlist`, `offset=10000` | `transaction_count` |
| 3 | `get_created_at_ts` | `account` / `txlist`, `offset=1`, `sort=asc` | `created_at` (timestamp tx pertama) |
| 4 | `get_wallet_activity` | `account` / `txlist`, `offset=100`, `sort=desc` | `activity_score`, `avg_tx_value`, `tx_frequency`, `unique_counterparties` |

**4 panggilan × 12.815 alamat = ±51.000 permintaan HTTP.**
Dengan jeda 0,4 detik → sekitar **5,7 jam** berjalan non-stop. Itulah kenapa skripnya
dibuat *resumable* dan ada `run_daily_etherscan.bat`.

### Kenapa skripnya resumable

```python
if CHECKPOINT_CSV.exists():                                  # :41
    done_df = pd.read_csv(CHECKPOINT_CSV)
pending = source_df[~source_df["address"].isin(done_addresses)]   # :51
...
if idx % 25 == 0:                                            # :77
    merged.to_csv(CHECKPOINT_CSV, index=False)               # simpan tiap 25 alamat
```

Kalau internet putus atau kuota API habis, jalankan lagi besok — ia melanjutkan dari
alamat yang belum selesai, tidak mengulang dari nol. Ini poin rekayasa yang layak
disebut saat presentasi.

---

## 5. Tujuh fitur Etherscan: asal & rumusnya

| # | Fitur | Rumus persisnya | Sumber |
|---|---|---|---|
| 1 | `balance` | `float(result) / 1e18` — saldo Wei diubah ke ETH | `data_fetchers.py:83` |
| 2 | `transaction_count` | `len(result)` dari txlist dengan `offset=10000` | `:125` |
| 3 | `activity_score` | `len(txs)` dari 100 transaksi terakhir | `:198` |
| 4 | `avg_tx_value` | `sum(values) / len(values)` dari 100 tx terakhir | `:199` |
| 5 | `tx_frequency` | `len(txs) / 30` | `:200` |
| 6 | `unique_counterparties` | jumlah alamat lawan transaksi unik (diri sendiri dikecualikan) | `:186-195, 201` |
| 7 | `created_at_ts` | timestamp transaksi paling awal → detik epoch | `:332` + `build_hybrid_dataset.py:134-138` |

### Kenapa fitur-fitur ini masuk akal untuk deteksi phishing

- `balance` **mendekati nol padahal `transaction_count` tinggi** → wallet yang uangnya
  selalu langsung disapu keluar. Ini pola penampung phishing.
- `unique_counterparties` **tinggi** → berinteraksi dengan banyak pihak berbeda,
  konsisten dengan "banyak korban".
- `avg_tx_value` **kecil tapi sering** → pola pengumpulan dana receh dari banyak korban.
- `created_at_ts` **baru** → wallet muda yang langsung sangat aktif itu mencurigakan.

### Nilai nyata di dataset

```
balance                  nunique= 4170   mean=0,5233   nol=44,4%
transaction_count        nunique=  475   mean=51,02    nol= 0,7%
activity_score           nunique=  101   mean=22,49    nol= 0,7%
avg_tx_value             nunique= 7696   mean=6,1959   nol= 1,9%
tx_frequency             nunique=  101   mean=0,7498   nol= 0,7%
unique_counterparties    nunique=  102   mean=12,53    nol= 0,7%
created_at_ts            nunique=    1   mean=0,0000   nol=100,0%   ← MATI
```

---

## 🔴 6. EMPAT TEMUAN PENTING

### Temuan A — `created_at_ts` adalah fitur mati (konstan 0)

**Fakta:** di `addresses_etherscan_enriched.csv`, kolom `created_at` bernilai kosong
untuk **seluruh 12.815 baris**. Akibatnya `created_at_ts` di `hybrid_dataset.csv`
bernilai **0 untuk 100% baris**.

**Sebabnya** tertulis di docstring kode Anda sendiri (`data_fetchers.py:306-308`):
> *"Previously this value was hard-coded to None (stored as 0), making `created_at_ts`
> a dead constant feature — fixed here so future enrichment runs populate a real value."*

Jadi: bug-nya sudah diperbaiki di kode, **tapi enrichment belum dijalankan ulang**,
sehingga CSV-nya masih memuat nilai lama yang kosong.

**Akibat untuk tesis:** klaim "7 fitur Etherscan" secara teknis adalah
**6 fitur informatif + 1 fitur konstan**. Model tetap valid (XGBoost mengabaikan kolom
konstan), tapi jangan sampai penguji yang menemukannya duluan.

**Pilihan tindakan:**
- *Cepat:* akui di bab keterbatasan — "`created_at_ts` tidak terisi pada proses
  enrichment yang dipakai; efektifnya 6 fitur Etherscan yang informatif."
- *Benar:* jalankan ulang enrichment (kode sudah diperbaiki), bangun ulang dataset,
  latih ulang model. Butuh berjam-jam dan semua angka tesis berubah — **jangan lakukan
  kalau bimbingan sudah dekat.**

### Temuan B — `activity_score` dan `tx_frequency` itu fitur yang sama

**Fakta:** `tx_frequency = activity_score / 30` **persis**, karena
`len(txs) / max(1, days)` dengan `days=30` selalu.

Diverifikasi: `np.allclose(activity_score/30, tx_frequency) → True`, korelasi = 1,0000.

**Artinya** satu dari dua kolom itu tidak menambah informasi apa pun — hanya salinan
berskala. Untuk model berbasis pohon ini tidak berbahaya, tapi kalau ditanya
*"apakah ada fitur redundan?"*, jawab jujur: ya, pasangan ini.

Tambahan: nama `tx_frequency` juga **menyesatkan**. Ia bukan "transaksi per hari",
karena 100 transaksi yang diambil bukan berasal dari jendela 30 hari — melainkan
100 transaksi terakhir kapan pun terjadinya. Pembagi 30 itu konstanta tetap.

### Temuan C — `activity_score` terpotong di 100, `transaction_count` di 10.000

**Fakta:** `activity_score` maksimum = **100**, karena `offset=100`.
`transaction_count` maksimum = **10.000**, karena `offset=10000`.

Jadi wallet dengan 50.000 transaksi tetap tercatat 10.000. Ini *censoring* — nilai
terpotong di batas atas.

Efek sampingnya sudah didokumentasikan di kode (`data_fetchers.py:179-183`):
> *"`unique_counterparties` can legitimately exceed `transaction_count` for very
> active wallets."*

Terbukti di data: ada **2.774 baris** dengan `unique_counterparties > transaction_count`.
Terlihat mustahil, tapi ada penjelasannya — kedua kolom dihitung dari jendela sampel
yang berbeda. **Siapkan jawaban ini**, karena angka aneh seperti ini gampang tertangkap
mata penguji yang teliti.

### Temuan D 🔴 — Dataset Anda hanya memakai 8.595 dari 12.815 alamat

Ini yang paling penting.

**Fakta:**

| File | Baris | Tanggal |
|---|---|---|
| `addresses.csv` | **12.815** | 14 April |
| `addresses_etherscan_enriched.csv` | **12.815** (selesai penuh) | **3 Juni** |
| `hybrid_dataset.csv` | **8.595** | **23 April** |

`hybrid_dataset.csv` dibangun **23 April** — saat itu enrichment Etherscan baru selesai
sekitar 8.595 alamat. Enrichment baru rampung 3 Juni, **tapi dataset tidak pernah
dibangun ulang.**

**Buktinya sudah saya verifikasi:** 8.595 baris itu terdiri dari **seluruh 1.165 alamat
phishing + tepat 7.430 alamat normal pertama menurut urutan abjad** — persis pola
skrip enrichment yang memproses berurutan lalu berhenti di tengah jalan.

**Akibatnya:**
- Rasio kelas menjadi 1 : 6,4, padahal desain aslinya 1 : 10.
- 4.220 alamat yang datanya sudah terkumpul lengkap **tidak terpakai**.

**Apakah ini merusak validitas?** Tidak fatal, karena alamat Ethereum pada dasarnya
acak, jadi "7.430 pertama menurut abjad" tetap merupakan sampel acak — bukan sampel
bias. Tapi Anda harus bisa menjelaskannya kalau ditanya
*"kenapa 8.595, padahal `addresses.csv` isinya 12.815?"*

**Pilihan tindakan:**

| Pilihan | Konsekuensi |
|---|---|
| **Biarkan, jelaskan di bab keterbatasan** (aman, disarankan kalau bimbingan dekat) | Semua angka tesis tetap berlaku. Cukup tulis: "dataset final memakai 8.595 alamat yang enrichment-nya sudah lengkap pada saat pembangunan dataset." |
| **Bangun ulang** (`python ml\build_hybrid_dataset.py`) | Dataset jadi 12.815 baris. **Semua angka tesis berubah** — harus melatih ulang seluruh eksperimen 4×4, late fusion, ablasi, dan model produksi. Hitungan hari, bukan jam. |

⚠️ **Jangan menjalankan `build_hybrid_dataset.py` menjelang bimbingan.** Ia akan
menimpa `hybrid_dataset.csv`, dan seluruh angka di naskah tesis Anda langsung
tidak cocok lagi dengan datanya.

---

## 7. Ringkasan untuk dihafal

> "Daftar 12.815 alamat saya susun dari MulDiGraph — seluruh 1.165 alamat phishing
> ditambah 11.650 alamat normal hasil sampling acak rasio 10:1 dengan seed 42.
> Untuk tiap alamat, saya memanggil Etherscan API v2 sebanyak empat kali —
> endpoint `balance` dan tiga varian `txlist` — sehingga total sekitar 51.000
> permintaan, dijalankan bertahap dengan mekanisme checkpoint karena ada batas
> rate limit. Hasilnya tujuh fitur: saldo, jumlah transaksi, skor aktivitas,
> rata-rata nilai transaksi, frekuensi transaksi, jumlah rekanan unik, dan waktu
> transaksi pertama. Ketujuhnya kemudian di-LEFT JOIN ke dataset berdasarkan
> alamat yang sudah dinormalisasi huruf kecil."

Lalu tambahkan sendiri, sebelum ditanya:

> "Ada dua keterbatasan yang saya catat: `created_at_ts` tidak terisi pada proses
> enrichment yang dipakai sehingga nilainya konstan nol, dan `activity_score`
> dengan `tx_frequency` bersifat redundan karena yang kedua adalah yang pertama
> dibagi konstanta 30."

Menyebut keterbatasan sendiri = Anda menguasai data Anda. Itu justru menaikkan nilai.

---

## 3 pertanyaan cek pemahaman

1. Kalau Etherscan bukan dataset, dari mana daftar 12.815 alamat itu berasal, dan
   dari mana labelnya?
2. Berapa panggilan API yang dibutuhkan untuk satu alamat, dan kenapa skripnya harus
   dibuat resumable?
3. Kenapa ada 2.774 baris dengan `unique_counterparties` lebih besar dari
   `transaction_count`, padahal secara logika itu mustahil?
