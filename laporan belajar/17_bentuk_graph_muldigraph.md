# BENTUK & ISI MulDiGraph — deskripsi data untuk Bab 3

Semua angka di halaman ini dihitung ulang langsung dari `data/raw/MulDiGraph.pkl`
memakai `scripts/profile_muldigraph.py`. Keluarannya ada di
`ml/artifacts/graph_profile/` (`graph_stats.json`, 2 PNG, 1 CSV contoh).
Tidak ada angka yang disalin dari paper atau dari draf tesis.

Halaman ini menutup lubang yang selama ini ada: file `02` dan `03` menjelaskan
*apa yang dilakukan pada* graph, tapi belum pernah menjelaskan *graph-nya sendiri
berbentuk apa*.

---

## 1. Network itu punya bentuk — dua komponen saja

MulDiGraph adalah objek `networkx.MultiDiGraph`. Isinya hanya dua hal:

| Komponen | Mewakili | Atribut | Jumlah |
|---|---|---|---|
| **Node** | 1 alamat wallet Ethereum | `isp` (1 = phishing, 0 = normal) | **2.973.489** |
| **Edge** | 1 transaksi ETH, berarah | `amount` (ETH), `timestamp` (unix) | **13.551.303** |

Arti tiga suku kata namanya:

- **Mul**ti → boleh ada **banyak edge** antara pasangan wallet yang sama, karena
  dua wallet bisa bertransaksi berkali-kali. Terbukti: 13.551.303 edge hanya
  menempati **5.355.155 pasangan unik** → rata-rata **2,53 transaksi per pasangan**.
- **Di**rected → edge punya arah: ada pengirim dan penerima. Inilah yang membuat
  `in_degree` dan `out_degree` bisa dibedakan.
- **Graph** → strukturnya simpul-dan-panah, bukan baris-dan-kolom.

### Kalau di-*flatten*, bentuk mentahnya adalah edge list

Ini 10 transaksi asli dari graph (`ml/artifacts/graph_profile/sample_edge_list.csv`),
semuanya keluar dari satu wallet phishing:

| from_address | to_address | amount_eth | tanggal |
|---|---|---|---|
| 0xb0054877…e8f01 | 0x147a26af…e8be9 | 0,01 | 2017-10-31 |
| 0xb0054877…e8f01 | 0x86fa0498…cfdb0 | 0,0 | 2017-11-01 |
| 0xb0054877…e8f01 | 0x1f495329…e70478 | 0,005 | 2017-10-29 |
| 0xb0054877…e8f01 | 0x8d12a197…cc6819 | 0,0 | 2017-10-29 |
| 0xb0054877…e8f01 | **0x00bcb8ad…9122d9** | 9,27964624 | 2017-10-29 |
| 0xb0054877…e8f01 | 0x8d12a197…cc6819 | 0,0 | 2017-10-29 |
| 0xb0054877…e8f01 | **0x00bcb8ad…9122d9** | 2,8 | 2017-11-01 |
| 0xb0054877…e8f01 | **0x00bcb8ad…9122d9** | 30,0 | 2017-11-01 |
| 0xb0054877…e8f01 | 0xb3bd49e2…323502 | 0,0 | 2017-10-31 |
| 0xb0054877…e8f01 | **0x00bcb8ad…9122d9** | 22,2 | 2017-10-29 |

Perhatikan `0x00bcb8ad…9122d9` muncul **4 kali** — itulah wujud nyata "Multi":
empat edge paralel antara pasangan wallet yang sama, dengan nominal berbeda
(9,28 + 2,8 + 30,0 + 22,2 ETH). Kalau ditanya dosen "kenapa MultiDiGraph, bukan
DiGraph biasa?", tunjuk baris-baris ini.

---

## 2. Statistik graph — tabel untuk Bab 3

| Aspek | Nilai |
|---|---|
| Tipe objek | `networkx.MultiDiGraph` |
| Jumlah node (wallet) | 2.973.489 |
| Jumlah edge (transaksi) | 13.551.303 |
| Pasangan wallet unik | 5.355.155 |
| Rata-rata edge paralel per pasangan | 2,53 |
| Node berlabel phishing (`isp=1`) | **1.165** (0,039 %) |
| Node berlabel normal (`isp=0`) | 2.972.324 |
| Kepadatan graph | 1,53 × 10⁻⁶ (sangat renggang) |
| Degree rata-rata (in+out) | 9,12 |
| Degree median | 1 |
| Degree maksimum | 247.547 |
| Node tanpa transaksi masuk | 1.854.465 (62,4 %) |
| Node tanpa transaksi keluar | 860.396 (28,9 %) |
| Total nilai ditransaksikan | 128.262.607,66 ETH |
| Nilai transaksi rata-rata / median / maks | 9,465 / **0,0** / 800.000 ETH |
| Rentang waktu | **2015-08-07 s.d. 2019-01-19** |
| Edge tanpa timestamp | 0 |

### Cara membaca tabel ini saat presentasi

> "Graph-nya sangat renggang dan sangat *skewed*. Median degree-nya 1, artinya
> separuh lebih wallet hanya pernah tersentuh satu transaksi. Tapi ada wallet
> dengan degree 247.547 — itu bursa atau kontrak besar. Distribusi seperti ini
> khas jaringan blockchain, dan itulah alasan saya melakukan penskalaan fitur
> sebelum pelatihan."

---

## 3. Gambar 1 — Ego-network 1-hop (`ego_network_1hop.png`)

Inilah gambar yang menjawab "18 fitur itu dari mana?".

Dua wallet asli dari graph, digambar bersama tetangga langsungnya:

| | Wallet | Jumlah tetangga 1-hop |
|---|---|---|
| Phishing (`isp=1`) | `0xf11f53674c211d42449aae57cabaf955a6fed8f1` | 10 |
| Normal (`isp=0`) | `0x35d54d17940cfe8677dbd1748d866422303610eb` | 7 |

Yang harus ditunjukkan dari gambar itu:

1. **Lingkaran besar = wallet yang dinilai.** Lingkaran kecil = tetangga langsung.
   Persis inilah wilayah yang dibaca `extract_node_features()` lewat
   `G.predecessors(node)` dan `G.successors(node)` (`ml/preprocess.py:87-105`).
   Tidak lebih jauh dari itu → itulah definisi **1-hop**.
2. **Arah panah menentukan fitur mana yang terisi.** Panah masuk → menyumbang
   `in_degree`, `total_received`, `avg_in_amount`. Panah keluar → `out_degree`,
   `total_sent`, dan seterusnya.
3. **Pola visual yang berbeda.** Wallet phishing di gambar didominasi panah
   **masuk** (banyak korban mengirim ke satu penampung, sedikit keluar). Wallet
   normal jauh lebih seimbang — masuk dan keluar berimbang. Itu intuisi di balik
   fitur `in_out_ratio` dan `sent_received_ratio`.

Kalimat untuk sidang:

> "Dari graph 2,9 juta node, model saya tidak melihat semuanya sekaligus. Untuk
> tiap wallet, model hanya membaca lingkungan 1-hop seperti di gambar ini, lalu
> meringkasnya jadi 18 angka. GraphSAGE-lah yang kemudian memperluas jangkauan
> jadi 2-hop lewat *message passing*."

---

## 4. Gambar 2 — Sebaran degree (`degree_distribution.png`)

Dinormalisasi **per kelas** (proporsi, bukan jumlah). Ini penting: kalau memakai
jumlah absolut, batang normal (2,97 juta wallet) akan menelan batang phishing
(1.165 wallet) dan bentuk sebarannya tidak terlihat sama sekali.

| Kelas | Degree rata-rata | Degree median |
|---|---|---|
| Phishing | **51,73** | **13** |
| Normal | 9,10 | **1** |

Wallet phishing rata-rata **5,7× lebih aktif** dan median-nya **13× lebih tinggi**.

---

## 5. Tiga temuan yang wajib masuk tesis (jangan disembunyikan)

### Temuan A — 67,5 % transaksi bernilai 0 ETH

**9.152.968 dari 13.551.303 edge punya `amount = 0`.** Itu sebabnya median nilai
transaksi = 0,0 ETH padahal rata-ratanya 9,47 ETH.

Artinya: fitur berbasis nilai (`avg_in_amount`, `min_in_amount`, `total_received`,
dst.) jauh lebih jarang terisi daripada yang terkesan dari daftar 18 fitur.
Edge bernilai 0 umumnya adalah pemanggilan smart contract atau transfer token
ERC-20 — nilai ETH-nya nol, tapi ada perpindahan aset yang tidak terekam di graph ini.

**Implikasi jujur yang harus ditulis:** MulDiGraph hanya merekam aliran **ETH
native**, bukan token. Padahal banyak penipuan phishing justru menguras token
ERC-20. Ini keterbatasan sumber data, bukan kesalahan model.

### Temuan B — data berhenti di Januari 2019

Rentang graph: **2015-08-07 s.d. 2019-01-19**. Tidak ada transaksi setelah itu.

Konsekuensi yang harus diakui di bab keterbatasan: model dilatih pada pola
phishing era 2017–2018. Modus phishing 2024–2026 (drainer, permit signature,
approval phishing) **tidak terwakili**. Ini juga penjelasan paling kuat kenapa
skor untuk wallet modern di luar dataset harus dibaca hati-hati — sambungkan ke
[[13_uji_wallet_luar_dataset]] dan [[14_model_live]].

### Temuan C — ketimpangan label 0,039 % vs 13,6 %

Di graph asli, phishing hanya **0,039 %** (1.165 dari 2,97 juta). Di dataset
eksperimen Anda, phishing jadi **13,6 %** (1.165 dari 8.595).

Pembilangnya **sama persis: 1.165**. Jadi Anda memakai **seluruh** wallet
phishing yang ada, lalu mengambil sampel wallet normal — bukan membuang phishing.
Itu keputusan yang benar dan mudah dipertahankan.

Tapi konsekuensinya harus disebut: **metrik F1 0,91 berlaku pada populasi 13,6 %
phishing, bukan pada populasi nyata 0,039 %.** Di dunia nyata, dengan basis
kejadian seribu kali lebih rendah, presisi akan jatuh drastis (*base rate
fallacy*). Kalau dosen menanyakan ini dan Anda sudah menyiapkan jawabannya, nilai
Anda naik. Kalau ditanya dan Anda kaget, nilai Anda turun.

> Kalimat aman: "F1 0,91 saya laporkan pada distribusi dataset eksperimen
> (13,6 % phishing). Saya sadar populasi sebenarnya hanya 0,039 %, sehingga
> presisi di lapangan akan lebih rendah. Karena itu keluaran sistem saya
> posisikan sebagai **skor risiko untuk triase**, bukan vonis phishing."

### Kaitan dengan kelemahan model yang sudah Anda temukan

Temuan degree di bagian 4 (median 13 vs 1) adalah **bukti kuantitatif langsung**
untuk dugaan di [[model-weakness-poc]]: kelas phishing memang jauh lebih aktif
daripada kelas normal di dataset ini, sehingga model **bisa** mencapai F1 tinggi
hanya dengan belajar aturan "aktivitas tinggi → phishing". Sebutkan sendiri
sebelum penguji yang menyebutkannya.

---

## 6. Apa yang masuk ke tesis, dan di mana

| Bahan | File | Letak di tesis |
|---|---|---|
| Tabel statistik graph (bagian 2) | `graph_stats.json` | Bab 3, sub-bab Sumber Data — **wajib** |
| Contoh edge list (bagian 1) | `sample_edge_list.csv` | Bab 3, tepat sebelum penjelasan 18 fitur |
| Gambar ego-network 1-hop | `ego_network_1hop.png` | Bab 3, sebagai jembatan graph → fitur — **paling berdampak** |
| Gambar sebaran degree | `degree_distribution.png` | Bab 3 (deskripsi data) atau Bab 4 (EDA) |
| Temuan A, B, C | halaman ini | Bab 3 (deskripsi) + bab Keterbatasan |

**Jangan** menggambar seluruh graph 2,9 juta node. Hasilnya "bola benang" hitam
yang tidak terbaca dan justru memberi kesan Anda tidak memahami datanya.

Urutan penyajian yang disarankan di Bab 3:

```
3.x Sumber Data
    3.x.1 Dataset MulDiGraph        → tabel statistik + penjelasan node/edge
    3.x.2 Struktur data mentah      → potongan edge list
    3.x.3 Lingkup ekstraksi 1-hop   → gambar ego-network
    3.x.4 Keterbatasan sumber data  → temuan A & B
3.y Ekstraksi Fitur                 → tabel 18 fitur (sudah ada)
```

---

## 7. Cara menghasilkan ulang

```powershell
python scripts/profile_muldigraph.py
```

Perlu ± 2 menit (memuat pickle 1,2 GB) dan RAM lapang. Keluaran ditimpa di
`ml/artifacts/graph_profile/`. Array degree disimpan sebagai `.npy` supaya gambar
bisa dibuat ulang tanpa memuat pickle lagi.

Terkait: [[02_pembentukan_dataset]] · [[03_feature_extraction_18_fitur]] ·
[[10_temuan_penting]]
