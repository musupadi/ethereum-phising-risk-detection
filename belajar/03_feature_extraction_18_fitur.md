# TAHAP 3 — Feature Extraction: Bedah 18 Fitur

Semua ada di satu fungsi: **`ml/preprocess.py:77` → `extract_node_features(G, node)`**.
Fungsi yang sama juga dipakai ulang oleh `ml/build_hybrid_dataset.py:34` (di-*import*),
jadi tidak ada dua versi rumus yang berbeda. Ini bagus, sebutkan saat presentasi.

## 1. Bedah kode baris demi baris

### Bagian A — kumpulkan transaksi masuk (`preprocess.py:83-93`)

```python
in_amounts, in_timestamps = [], []
unique_senders = set()

for pred in G.predecessors(node):          # 1. semua wallet yang pernah kirim ke `node`
    unique_senders.add(pred)               # 2. simpan sebagai himpunan (otomatis unik)
    for edge_data in G[pred][node].values():   # 3. SEMUA transaksi dari pred → node
        in_amounts.append(edge_data.get("amount", 0) or 0)
        ts = edge_data.get("timestamp")
        if ts is not None:
            in_timestamps.append(float(ts))
```

Yang perlu dipahami:

- **Baris 1** — `predecessors` = tetangga masuk. Ini yang membuat ekstraksinya "1-hop".
- **Baris 2** — `set` bukan `list`, jadi kalau A mengirim 5 kali, A tetap dihitung 1 pengirim.
- **Baris 3** — inilah kenapa graph-nya *Multi*DiGraph. `G[pred][node]` bukan satu edge,
  melainkan *dictionary* berisi banyak edge paralel. `.values()` mengambil semuanya.
  **Akibatnya `len(in_amounts)` = jumlah transaksi, bukan jumlah tetangga.**
- `or 0` menangani kasus `amount` bernilai `None`.

### Bagian B — kumpulkan transaksi keluar (`preprocess.py:95-105`)

Sama persis, tapi pakai `G.successors(node)` dan `G[node][succ]`.

### Bagian C — statistik aman (`preprocess.py:63-74`)

```python
def _safe_stats(values):
    if not values:
        return dict(total=0.0, mean=0.0, mx=0.0, mn=0.0, std=0.0)
    arr = np.asarray(values, dtype=float)
    return dict(total=..., mean=..., mx=..., mn=...,
                std=float(arr.std()) if len(arr) > 1 else 0.0)
```

Dua keputusan desain yang harus Anda bisa bela:

1. **List kosong → semua 0.** Wallet yang tidak pernah menerima apa pun mendapat
   `total_received=0`, `min_in_amount=0`, dst. Ini pilihan sadar: 0 di sini berarti
   "tidak ada aktivitas", bukan "data hilang".
2. **Satu transaksi → std = 0.** Karena standar deviasi dari satu angka tidak bermakna.

### Bagian D — rentang waktu (`preprocess.py:113-115`)

```python
ts_spread_in  = (max(in_timestamps)  - min(in_timestamps))  if len(in_timestamps)  > 1 else 0.0
ts_spread_out = (max(out_timestamps) - min(out_timestamps)) if len(out_timestamps) > 1 else 0.0
```

Satuannya **detik** (unix epoch). Nilai 86400 = wallet aktif selama 1 hari.

### Bagian E — kembalikan dict 18 kunci (`preprocess.py:117-136`)

Ini yang menjadi 18 kolom tabel.

## 2. Tabel lengkap 18 fitur

Contoh angka memakai wallet imajiner **X** dengan riwayat:

| Dari/Ke | Arah | Amount (ETH) | Timestamp |
|---|---|---|---|
| A → X | masuk | 2,0 | 1.000.000 |
| A → X | masuk | 4,0 | 1.086.400 |
| B → X | masuk | 6,0 | 1.100.000 |
| X → C | keluar | 11,0 | 1.100.500 |

| # | Nama fitur | Dari data apa | Cara hitung | Nilai untuk X | Kenapa berguna melacak phishing |
|---|---|---|---|---|---|
| 1 | `in_degree` | edge masuk | banyaknya transaksi masuk (`len(in_amounts)`) | **3** | Wallet phishing menerima kiriman dari banyak korban |
| 2 | `out_degree` | edge keluar | banyaknya transaksi keluar | **1** | Penipu biasanya menyapu keluar dalam sedikit transaksi besar |
| 3 | `unique_senders` | edge masuk | ukuran `set(predecessors)` | **2** (A, B) | Banyak pengirim **berbeda** = pola korban; beda dengan bot yang kirim berulang |
| 4 | `unique_receivers` | edge keluar | ukuran `set(successors)` | **1** (C) | Penipu menyapu ke 1–2 wallet penampungan |
| 5 | `total_received` | amount masuk | `sum(in_amounts)` | **12,0** | Nilai total kerugian korban |
| 6 | `total_sent` | amount keluar | `sum(out_amounts)` | **11,0** | Berapa yang sudah dilarikan |
| 7 | `avg_in_amount` | amount masuk | `mean(in_amounts)` | **4,0** | Nominal khas korban tertipu |
| 8 | `avg_out_amount` | amount keluar | `mean(out_amounts)` | **11,0** | Keluar >> masuk per transaksi = pola sapu |
| 9 | `max_in_amount` | amount masuk | `max` | **6,0** | Korban terbesar |
| 10 | `max_out_amount` | amount keluar | `max` | **11,0** | Transaksi penyapuan terbesar |
| 11 | `min_in_amount` | amount masuk | `min` | **2,0** | Nilai uji-coba kecil dari korban |
| 12 | `min_out_amount` | amount keluar | `min` | **11,0** | — |
| 13 | `std_in_amount` | amount masuk | `std` (0 jika ≤1 data) | **1,633** | Nominal masuk yang seragam = kemungkinan bot/airdrop, bukan korban acak |
| 14 | `std_out_amount` | amount keluar | `std` | **0,0** (hanya 1 tx) | idem, sisi keluar |
| 15 | `in_out_ratio` | derajat | `n_in / (n_out + 1)` | **1,5** | >1 berarti lebih banyak menerima daripada mengirim — profil penampung |
| 16 | `sent_received_ratio` | amount | `total_sent / (total_received + 1e-9)` | **0,9167** | Mendekati 1 = uang masuk langsung dikuras keluar; ciri klasik phishing |
| 17 | `timestamp_spread_in` | timestamp masuk | `max - min` (0 jika ≤1) | **100.000** detik | Wallet phishing hidup singkat lalu ditinggalkan |
| 18 | `timestamp_spread_out` | timestamp keluar | `max - min` | **0,0** | idem |

Perhatikan dua rumus yang **bukan** statistik polos — ini bagian *feature engineering*
Anda, dan layak disebut sebagai kontribusi kecil:

- `in_out_ratio` memakai `+1` di penyebut supaya tidak pernah membagi nol.
- `sent_received_ratio` memakai `+1e-9`, sehingga wallet yang tidak pernah menerima tapi
  banyak mengirim menghasilkan angka **sangat besar** — sinyal ekstrem yang bisa
  ditangkap XGBoost, tapi **berbahaya untuk Logistic Regression** (skala meledak).
  Ini salah satu alasan LR kalah telak (F1 0,785 vs XGBoost 0,886 di skenario 1).

## 3. Bedakan dengan tegas: kelompok kolom

Total kolom di `hybrid_dataset.csv` = **41**. Rinciannya:

### (a) 18 fitur hasil ekstraksi MulDiGraph
`in_degree, out_degree, unique_senders, unique_receivers, total_received, total_sent,
avg_in_amount, avg_out_amount, max_in_amount, max_out_amount, min_in_amount,
min_out_amount, std_in_amount, std_out_amount, in_out_ratio, sent_received_ratio,
timestamp_spread_in, timestamp_spread_out`

### (b) 1 kolom indikator: `graph_present`
Dibuat di `build_hybrid_dataset.py:273`:
```python
merged_df["graph_present"] = merged_df[graph_feature_cols].sum(axis=1).gt(0).astype(int)
```
Artinya: 1 kalau wallet ini benar-benar ditemukan di MulDiGraph (paling tidak satu
fiturnya bukan nol), 0 kalau fiturnya hasil zero-fill.

> ⚠️ **Fakta yang harus Anda tahu sebelum ditanya:** di `hybrid_dataset.csv` saat ini,
> `graph_present` bernilai **1 untuk 100% baris**. Jadi kolom ini **konstan** dan tidak
> memberi informasi apa pun kepada model. Ia tidak merusak hasil (XGBoost akan
> mengabaikannya), tapi jangan mengklaim kolom ini "membantu model membedakan".
> Klaim yang benar: *"kolom ini adalah pengaman desain; pada dataset final ternyata
> semua 8.595 alamat ada di MulDiGraph, sehingga nilainya seragam 1."*
> Dalam skema, ia ikut dihitung sehingga skenario 1 punya **19** kolom, bukan 18.

### (c) 7 fitur Etherscan
`created_at_ts, balance, transaction_count, activity_score, avg_tx_value, tx_frequency,
unique_counterparties`
(`build_hybrid_dataset.py:67-74` + `created_at_ts` dari `:134-138`)

`created_at_ts` = tanggal pengambilan data diubah jadi detik epoch. Sisanya atribut
wallet nyata dari API Etherscan, **bukan** dari graph 2019.

### (d) 10 fitur BigQuery
`bq_tx_count_out, bq_tx_count_in, bq_total_out_eth, bq_total_in_eth, bq_unique_to,
bq_unique_from, bq_active_days_out, bq_active_days_in, bq_first_seen_ts, bq_last_seen_ts`
(`build_hybrid_dataset.py:76-87`)

### (e) Label
`label` (1 = phishing, 0 = normal) dan `class_name` (teksnya).

### (f) Identitas & metadata
`address`, `node_id` (salinan `address`, `build_hybrid_dataset.py:279`), `source_dataset`.

Hitungan: 18 + 1 + 7 + 10 + 2 + 3 = **41** ✔

## 4. Kenapa label dan alamat wallet TIDAK boleh jadi fitur input

### Label (`label`)
Kalau label ikut masuk sebagai kolom input, model tinggal menyalinnya → akurasi 100%
di training dan tidak berguna sama sekali di dunia nyata. Ini namanya **target leakage**.
Model tidak "belajar" apa pun.

### Alamat wallet (`node_id` / `address`)
Tiga alasan:

1. **Tipe data.** Ia string heksadesimal 42 karakter. XGBoost hanya menerima angka.
2. **Tidak ada pola.** Alamat Ethereum adalah hash dari public key — acak. Digit "0xab…"
   tidak lebih berisiko dari "0xcd…". Tidak ada informasi di dalamnya.
3. **Memorisasi (overfitting ekstrem).** Kalau dipaksakan jadi angka (misal encoding),
   model akan menghafal *"alamat nomor 731 itu phishing"*, bukan mempelajari *perilaku*.
   Begitu ada wallet baru yang belum pernah dilihat, model buta total.

Perlindungan di kode ada dua lapis:

```python
# ml/preprocess.py:200
feature_cols = [c for c in df.columns if c not in ("label", "node_id")]

# api/services/ml_service.py:52
_META_COLS = {"label", "node_id", "address", "class_name", "source_dataset",
              "error", "_split", "_addr"}
```

Kalau Anda menambah kolom non-fitur baru, **wajib** ditambahkan ke `_META_COLS`,
kalau tidak ia akan bocor ke model saat inferensi.

## 5. Satu ketidakcocokan yang harus Anda tahu

Docstring di `ml/preprocess.py:7-17` menyebut **20** fitur, termasuk
`avg_in_timestamp` dan `avg_out_timestamp`. Cek kodenya
(`preprocess.py:117-136`): **dua fitur itu tidak pernah dihitung**. Yang benar
adalah **18 fitur**, dan itu yang konsisten dengan
`ml/artifacts/prod_feature_schema.json`.

Kalau di naskah tesis Anda tertulis 20, perbaiki naskahnya — jangan sebaliknya.
Kalau tertulis 18, Anda aman, tapi docstring-nya sebaiknya dirapikan supaya penguji
yang membuka kode tidak bingung.

---

## 3 pertanyaan cek pemahaman

1. Wallet Y menerima 10 transaksi, semuanya dari 1 alamat yang sama. Berapa `in_degree`
   dan `unique_senders`? Fitur mana yang membuat kasus ini bisa dibedakan dari
   "10 transaksi dari 10 korban berbeda"?
2. Kenapa `sent_received_ratio` bisa bernilai sangat besar (misal 10⁹), dan model apa
   yang paling terganggu oleh nilai seperti itu?
3. Kalau besok Anda menambah kolom `first_seen_label` bertipe teks ke
   `hybrid_dataset.csv`, satu file mana yang **wajib** Anda ubah supaya API tidak error?
