# TAHAP 4 — Penggabungan Tiga Sumber Data

File yang bertanggung jawab: **`ml/build_hybrid_dataset.py`**.

## 1. Tabel utamanya SIAPA?

Jawaban singkat: **`data/raw/addresses.csv`** — daftar alamat + label.
Bukan graph, bukan Etherscan.

```python
# build_hybrid_dataset.py:215-217
if LABEL_SOURCE_CSV.exists():
    base_df = prepare_label_df(pd.read_csv(LABEL_SOURCE_CSV))
```

Ini keputusan desain yang penting dan **harus Anda sebut duluan** kalau ditanya,
karena konsekuensinya: **jumlah baris dataset akhir tidak akan pernah lebih dari
jumlah baris `addresses.csv`.** Semua join berikutnya `how="left"`, jadi hanya
menambah kolom, tidak pernah menambah/mengurangi baris.

## 2. Urutan join yang sebenarnya

```python
merged_base = base_df.merge(etherscan_df[...], on="address", how="left")   # :235
merged_df   = merged_base.merge(graph_df,      on="address", how="left")   # :267
merged_df   = merged_df.merge(bq_df,           on="address", how="left")   # :278
```

Ketiganya **LEFT JOIN**, kunci `address` yang sudah dinormalisasi:

```python
out["address"] = out["address"].astype(str).str.lower().str.strip()   # :125, :149, :167
```

Normalisasi ini wajib karena alamat Ethereum sering ditulis dengan campuran huruf
besar-kecil (EIP-55 checksum). Tanpa `.lower()`, `0xAbC…` dan `0xabc…` dianggap dua
wallet berbeda dan join-nya gagal diam-diam.

Tiap sumber juga di-`drop_duplicates(subset=["address"])` (`:143`, `:158`, `:176`)
**sebelum** join. Ini pengaman anti-*row explosion*: kalau tabel kanan punya alamat ganda,
LEFT JOIN akan menggandakan baris tabel kiri.

## 3. Contoh konkret 3 tabel kecil

**Tabel utama — `addresses.csv`**

| address | label |
|---|---|
| 0xaaa… | 1 |
| 0xbbb… | 0 |
| 0xccc… | 0 |

**Tabel Etherscan** (0xccc tidak pernah berhasil diambil)

| address | balance | transaction_count |
|---|---|---|
| 0xaaa… | 0,00 | 412 |
| 0xbbb… | 3,50 | 12 |

**Tabel graph** (0xbbb tidak ada di MulDiGraph)

| address | in_degree | total_received |
|---|---|---|
| 0xaaa… | 87 | 120,4 |
| 0xccc… | 2 | 1,1 |

**Hasil LEFT JOIN, sebelum pengisian nol:**

| address | label | balance | transaction_count | in_degree | total_received |
|---|---|---|---|---|---|
| 0xaaa… | 1 | 0,00 | 412 | 87 | 120,4 |
| 0xbbb… | 0 | 3,50 | 12 | NaN | NaN |
| 0xccc… | 0 | NaN | NaN | 2 | 1,1 |

**Setelah pengisian nol + `graph_present`:**

| address | label | balance | transaction_count | in_degree | total_received | graph_present |
|---|---|---|---|---|---|---|
| 0xaaa… | 1 | 0,00 | 412 | 87 | 120,4 | 1 |
| 0xbbb… | 0 | 3,50 | 12 | 0 | 0 | **0** |
| 0xccc… | 0 | **0** | **0** | 2 | 1,1 | 1 |

Tetap **3 baris**, label tidak berubah. Itu tujuannya.

## 4. Apa yang terjadi kalau wallet tidak ditemukan?

### Tidak ada di Etherscan
```python
# build_hybrid_dataset.py:239-242
for col in ["created_at_ts"] + NUMERIC_ONCHAIN_COLS:
    merged_base[col] = pd.to_numeric(merged_base[col], errors="coerce").fillna(0)
```
→ 7 kolom Etherscan jadi 0.

### Tidak ada di MulDiGraph
```python
# build_hybrid_dataset.py:268-270
for col in graph_feature_cols:
    merged_df[col] = pd.to_numeric(merged_df[col], errors="coerce").fillna(0)
```
→ 18 kolom graph jadi 0, **dan `graph_present` jadi 0** sebagai penanda.

Komentar di kode (`:265-266`) jujur menyebut alasannya:
> *"Keep all enriched rows and backfill missing graph features with zeros. This allows
> larger training sets when some addresses are absent in MulDiGraph."*

### Tidak ada di BigQuery
```python
out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)   # :171
```
→ 10 kolom `bq_*` jadi 0.

## 5. Kenapa boleh diisi nol — dan kapan berbahaya

### Kenapa boleh (argumen yang bisa dipertahankan)

Untuk fitur **hitungan dan jumlah** (`in_degree`, `total_received`, `bq_tx_count_out`,
`transaction_count`), nilai 0 punya **makna semantik yang benar**:
"tidak ada transaksi tercatat di sumber ini". Ini bukan menebak-nebak;
ini menyatakan ketiadaan.

Ditambah lagi, XGBoost (model produksi Anda) bekerja dengan *split* berbasis ambang
(`fitur ≤ t`), jadi ia bisa mempelajari sendiri bahwa "kelompok nilai 0" adalah
kelompok tersendiri.

### Kapan berbahaya (sebutkan sendiri sebelum dosen menyebutkannya)

| Risiko | Penjelasan | Fitur yang terdampak |
|---|---|---|
| **0 ambigu** | 0 bisa berarti "benar-benar nol" atau "data tidak tersedia". Model tidak bisa membedakan. | `balance` (wallet kosong vs API gagal) |
| **0 tidak masuk akal secara fisik** | Timestamp 0 = 1 Januari 1970. Bukan "tidak ada", tapi "tahun 1970". | `created_at_ts`, `bq_first_seen_ts`, `bq_last_seen_ts` |
| **Distribusi terdistorsi** | Mean & std jadi bias ke bawah; korelasi ikut melenceng. | semua fitur numerik |
| **Sinyal palsu** | Kalau ketiadaan data kebetulan lebih sering pada satu kelas, model belajar "0 = phishing" — padahal itu artefak proses pengambilan data, bukan perilaku. | fatal kalau terjadi |

**Yang harus Anda katakan:** risiko paling serius adalah yang terakhir. Cara mengeceknya:
bandingkan proporsi baris ber-nilai-nol antar kelas. Kalau proporsinya mirip, zero-filling
aman. Kalau timpang, itu kebocoran proses. `ml/audit_gaps.py` dan
`ml/artifacts/audit_report.json` adalah tempat pemeriksaan seperti ini.

**Kabar baiknya untuk model produksi Anda:** `graph_present` = 1 untuk seluruh 8.595
baris, artinya **tidak ada satu pun fitur graph yang hasil zero-fill**. Jadi risiko
zero-filling di sisi graph = nol untuk dataset final ini. Risiko tersisa hanya di sisi
Etherscan/BigQuery.

## 6. Fungsi kolom `graph_present`

```python
merged_df["graph_present"] = merged_df[graph_feature_cols].sum(axis=1).gt(0).astype(int)
```

Tiga fungsinya:

1. **Penanda sumber data (provenance).** Membedakan "nol beneran" dari "nol karena tidak ditemukan".
2. **Bahan audit.** `graph_coverage = merged_df["graph_present"].mean() * 100` (`:274`)
   dicatat di log — ukuran seberapa banyak dataset benar-benar ditopang graph.
3. **Fitur kondisional untuk model.** Secara teori XGBoost bisa belajar
   *"kalau graph_present=0, abaikan fitur graph"*.

⚠️ Fungsi ke-3 **tidak aktif pada dataset ini** karena nilainya konstan 1.
Jangan mengklaim yang tidak terjadi. Rumusan aman:
> "Kolom `graph_present` dirancang sebagai penanda cakupan graph. Pada dataset final
> nilainya seragam 1 karena seluruh 8.595 alamat berhasil ditemukan di MulDiGraph,
> sehingga ia berfungsi sebagai bukti audit cakupan 100%, bukan sebagai fitur pembeda."

## 7. Cara mencegah jumlah wallet & label berubah tanpa sengaja

Kode Anda sudah punya beberapa pengaman. Hafalkan, ini pertanyaan favorit penguji metodologi.

| Pengaman | Kode | Mencegah apa |
|---|---|---|
| Semua join `how="left"` | `:235, :267, :278` | Baris hilang karena inner join |
| `drop_duplicates(subset=["address"])` di tiap sumber | `:143, :158, :176` | Baris menggandakan diri |
| Normalisasi `.lower().str.strip()` | `:125, :149, :167` | Join gagal diam-diam karena beda kapital |
| Label **hanya** dari tabel utama | `:217` | Label ikut tertimpa sumber lain |
| Cek silang label graph vs label dataset | `:281-284` | Label graph & label CSV bertentangan |
| `stratify=final_df["label"]` saat split | `:308` | Proporsi kelas berubah di train/test |
| `random_state=42` di mana-mana | `:57` | Hasil tidak reproducible |

Yang paling elegan adalah pengaman ke-5:

```python
mismatch_count = int((merged_df.loc[mask, "graph_label"] != merged_df.loc[mask, "label"]).sum())
if mismatch_count > 0:
    log.warning("Found %d graph/enriched label mismatches", mismatch_count)
merged_df = merged_df.drop(columns=["graph_label"], errors="ignore")
```

Kode mengambil label dari graph (`isp`), **membandingkannya** dengan label dari
`addresses.csv`, melaporkan kalau berbeda, lalu **membuang** label graph agar hanya ada
satu sumber kebenaran. Ini praktik yang benar — sebutkan.

> Catatan: kode hanya *memperingatkan*, tidak menggagalkan proses. Kalau di log Anda
> pernah muncul angka mismatch > 0, Anda harus siap menjelaskan berapa dan kenapa.

## 8. Yang BELUM ada (kalau ditanya "apa keterbatasannya")

- Tidak ada `assert len(merged_df) == len(base_df)` — pengaman jumlah baris masih
  mengandalkan sifat left join, belum diverifikasi eksplisit.
- Tidak ada penanda terpisah `etherscan_present` / `bq_present` seperti `graph_present`.
  Jadi untuk fitur Etherscan/BigQuery, "0 karena kosong" dan "0 karena gagal ambil"
  memang tidak bisa dibedakan. **Ini keterbatasan nyata — akui saja, jangan dibela.**

---

## 3 pertanyaan cek pemahaman

1. Kalau `addresses.csv` punya 8.595 baris dan Etherscan hanya punya 6.000, berapa baris
   dataset akhir? Kenapa?
2. Kenapa `.str.lower()` pada kolom `address` bukan sekadar kerapian, tapi penentu
   benar-salahnya hasil join?
3. Untuk fitur mana pengisian nol paling tidak bisa dibenarkan, dan apa alternatif yang
   lebih jujur untuk fitur tersebut?
