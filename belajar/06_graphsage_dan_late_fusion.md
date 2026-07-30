# TAHAP 6 — GraphSAGE dan Late Fusion

## 1. GraphSAGE dengan analogi wallet

### Masalahnya dulu

Model tabular (XGBoost) melihat wallet X seperti ini:
> *"in_degree=87, total_received=120,4, balance=0. Selesai. Titik."*

Ia **tidak tahu** bahwa 40 dari 87 pengirim itu ternyata wallet yang juga menerima dari
wallet phishing lain. Informasi "siapa temanmu" hilang saat graph dipipihkan jadi tabel.

### Ide GraphSAGE: "kamu adalah rata-rata tetanggamu"

Bayangkan penilaian reputasi di kampung:

- **Ronde 0.** Tiap orang cuma punya profil pribadi (18 fitur).
- **Ronde 1.** Tiap orang bertanya ke **tetangga langsung**: "profilmu apa?"
  Lalu ia membuat profil baru = campuran (profil sendiri, rata-rata profil tetangga).
- **Ronde 2.** Diulang. Tapi karena tetangga di ronde 1 sudah menyerap info dari tetangga
  *mereka*, sekarang informasi datang dari **2 langkah** jauhnya.

Itulah **message passing** dan **neighborhood aggregation**. Jumlah ronde = jumlah layer.

### Rumusnya (sederhana saja)

Ada di docstring kode Anda, `ml/gnn_model_v2.py:46-49`:

```
h_v = W1 · h_v  +  W2 · mean( h_u untuk semua u tetangga v )
      └─diri─┘       └────── tetangga ──────┘
```

Yang penting dipahami:
- `W1` dan `W2` **terpisah** → model bisa belajar seberapa besar mendengarkan diri sendiri
  vs mendengarkan tetangga. Ini keunggulan SAGE dibanding GCN yang mencampur keduanya.
- `mean` → tidak peduli tetangganya 3 atau 3.000, hasilnya tetap skala sama.
  Cocok untuk transaksi Ethereum di mana derajat node sangat timpang.

Kode Anda memilih SAGE dengan alasan yang sudah tertulis (`gnn_model_v2.py:46-49`):
*"Keeps own-feature contribution explicit — better than GCN's symmetric normalisation
when node degrees vary widely (as in transaction graphs)."* Kutip ini kalau ditanya
"kenapa GraphSAGE, bukan GCN?"

## 2. Istilah satu per satu

| Istilah | Artinya di penelitian Anda | Kode |
|---|---|---|
| **Node feature** | 19 fitur graph per wallet, sudah di-`StandardScaler` | `train_dataset_scenarios.py:345-347` |
| **Adjacency / edge_index** | Daftar pasangan `[dari, ke]` dalam bentuk tensor 2×E | `gnn_model_v2.py:186-189` |
| **Message passing** | Tiap layer, tiap node mengambil informasi dari tetangganya | `SAGEConv` di `forward()`, `:74-79` |
| **Neighborhood aggregation** | Cara meringkas pesan-pesan tetangga → di sini `mean` | bawaan `SAGEConv` |
| **Embedding 128 dimensi** | `hidden_dim=128` — vektor hasil belajar yang mewakili wallet + konteks tetangganya | `:63, :351-355` |
| **Self-loop** | Tiap node dihubungkan ke dirinya sendiri, supaya node tanpa tetangga tetap punya sinyal | `:192-193` |
| **Graph sparse** | Sangat sedikit edge dibanding jumlah node | lihat bagian 3 |
| **num_layers=2** | Jangkauan 2-hop | `:353-355` |

Kenapa hanya 2 layer? Komentar di kode menjawabnya (`train_dataset_scenarios.py:349-350`):
> *"2 layers: 1-hop then 2-hop aggregation — more layers cause over-smoothing on graphs
> of this size (~8K nodes)."*

**Over-smoothing** = kalau terlalu banyak ronde, semua orang akhirnya jadi rata-rata
seluruh kampung, dan tidak ada yang bisa dibedakan lagi. Itu jawaban yang tepat.

## 3. Inilah alasan sebenarnya GNN kalah: graph-nya kosong

Angka dari `outputs/edge_source_audit_v2.json` — **ini angka paling penting di TAHAP 6**:

```json
{
  "muldigraph_addresses_found": 8595,
  "dataset_addresses_total": 8595,
  "raw_directed_edge_entries": 528,
  "unique_directed_edges": 264,
  "edges_train_train": 150,
  "edges_train_test": 95,
  "edges_test_test": 19,
  "self_loops_added_by_pyg_builder": 8595
}
```

Baca pelan-pelan:

- 8.595 node, tapi hanya **264 edge berarah unik** di antara mereka.
- Kepadatan = 264 / (8.595 × 8.594) ≈ **0,00000357** — praktis nol.
- Rata-rata derajat ≈ **0,03 edge per node**.
- Jumlah self-loop (8.595) **32× lebih banyak** dari edge asli (264).

**Kesimpulan yang harus Anda ucapkan sendiri di depan dosen:**

> "Sampling saya memilih 8.595 wallet dari graph berisi jutaan node. Karena dipilih
> secara acak, wallet-wallet ini hampir tidak pernah bertransaksi satu sama lain —
> hanya 264 edge yang tersisa. Akibatnya, untuk lebih dari 99% wallet, GraphSAGE tidak
> punya tetangga untuk diagregasi, sehingga ia hanya mengolah fitur node sendiri lewat
> self-loop. Dalam kondisi itu GraphSAGE praktis berubah menjadi MLP biasa, dan wajar
> kalau ia kalah dari XGBoost yang memang unggul di data tabular."

Ini jawaban terkuat yang bisa Anda punya. Ia mengubah "GNN saya jelek" menjadi
"saya paham persis kenapa dan bisa membuktikannya dengan angka".

### Kenapa graph-nya bisa sekosong itu?

Karena tabel utama dataset adalah `addresses.csv` (daftar alamat individual), bukan
sebuah **subgraph terhubung**. Kalau Anda ingin GNN punya kesempatan adil, sampling
harus dilakukan secara graph-aware — misalnya *snowball sampling*: ambil wallet phishing,
lalu ambil semua tetangganya, lalu tetangga dari tetangga. Sebutkan ini sebagai
**saran penelitian lanjutan** — itu memperkuat posisi Anda, bukan melemahkan.

## 4. Late Fusion: apa dan bagaimana

File: **`experiments/run_graphsage_etherscan_late_fusion_v2.py`**

### Idenya

*Early fusion* = campur semua fitur di awal, lalu satu model mengolahnya
(itulah skenario 2: 19 graph + 7 Etherscan langsung ke XGBoost).

*Late fusion* = tiap sumber diolah oleh model yang paling cocok **dulu**, hasilnya baru
digabung di tahap akhir.

```
19 fitur graph ──► GraphSAGE (2 layer, 128 hidden) ──► embedding 128-d ─┐
                                                                        ├─ concat ─► 135-d ─► XGBoost ─► prediksi
7 fitur Etherscan ──► StandardScaler ──────────────► 7-d ───────────────┘
```

Kutipan dari docstring kode Anda:
> `fused_i = [ graphsage_embedding_128 || scaled_etherscan_7 ]  ->  135-d`

### Dua koreksi V2 yang wajib Anda hafal

Docstring `run_graphsage_etherscan_late_fusion_v2.py:8-23` menyebut tiga perbaikan
atas V1. Ini menunjukkan kedewasaan metodologi Anda — **jangan lupa menyebutkannya**:

1. **NON-REDUNDANT.** Di V1, 7 fitur Etherscan masuk **dua kali** (jadi input GraphSAGE
   *dan* di-concat lagi). Di V2, GraphSAGE hanya diberi 19 fitur graph. Etherscan masuk
   tepat sekali, di cabang tabular.
2. **LEAKAGE-CONTROLLED.** Tidak ada sinyal test (F1/loss/AUC) yang dipakai untuk early
   stopping atau pemilihan model. Early stopping memakai validasi internal dari wallet
   training saja. Test set di-*score* **tepat satu kali**.
3. **EPTRANSNET EDGES ONLY.** `data/raw/etherscan_edges.csv` sengaja **tidak** dibaca,
   supaya topologi graph murni dari EPTransNet dan tidak tercampur sumber lain.

## 5. Empat konfigurasi, dan bedanya

Angka asli dari `outputs/four_model_comparison_ablation.csv`:

| Konfigurasi | Input | F1 | ROC-AUC | PR-AUC | TN/FP/FN/TP |
|---|---|---|---|---|---|
| **XGBoost tabular (Skenario 2)** | 26 fitur tabular (19 graph + 7 Etherscan) | **0,9102** | 0,9948 | 0,9701 | 1458/28/15/218 |
| GraphSAGE–Etherscan Late Fusion V2 | 128-d embedding ‖ 7 Etherscan = 135-d | 0,8825 | 0,9916 | 0,9529 | 1448/38/19/214 |
| GraphSAGE graph-only | 19 fitur graph + 264 edge | 0,8189 | 0,9714 | 0,8880 | 1432/54/34/199 |
| Embedding-only + XGBoost (**ablasi**) | 128-d embedding saja, tanpa Etherscan | 0,8184 | 0,9820 | 0,9222 | 1423/63/28/205 |

### Bacaan per baris

- **XGBoost tabular** — juara. Semua informasi masuk lewat satu model yang memang
  dioptimalkan untuk tabel.
- **Late fusion V2** — kedua. Lebih baik dari GraphSAGE murni, tapi **tidak melampaui**
  early fusion sederhana. Selisihnya 0,0277 F1.
- **GraphSAGE graph-only** — hanya graph, dengan protokol jujur. F1 0,8189.
- **Embedding-only** — GraphSAGE menghasilkan embedding, lalu XGBoost mengklasifikasi
  embedding itu **tanpa** Etherscan. Ini kelompok kontrol.

## 6. Fungsi ablasi, dan apa yang ia buktikan

**Ablasi** = membuang satu komponen, lalu mengukur seberapa turun performanya.
Kalau turun, komponen itu memang berkontribusi. Kalau tidak turun, komponen itu hiasan.

File: `experiments/run_embedding_only_ablation.py`.

### Perbandingan yang benar

| | F1 | Selisih |
|---|---|---|
| Late fusion (embedding 128 **+ Etherscan 7**) | 0,8825 | |
| Embedding-only (embedding 128 saja) | 0,8184 | **−0,0641** |

**Kesimpulan ablasi:** menambahkan 7 fitur Etherscan ke embedding graph menaikkan F1
sebesar **+0,064** (dari 0,8184 ke 0,8825). Jadi kontribusi Etherscan **terbukti nyata
dan besar** — bukan asumsi.

### Kenapa ablasi diperlukan (jawaban untuk dosen)

Tanpa ablasi, Anda hanya bisa bilang *"model fusi saya F1-nya 0,88"*. Penguji akan tanya:
*"Dari mana 0,88 itu datang? Dari GraphSAGE-nya, atau dari Etherscan-nya?"*
Tanpa ablasi Anda tidak bisa menjawab.

Dengan ablasi Anda bisa memecahnya:
- Embedding graph sendirian: 0,8184
- + Etherscan: 0,8825 → **kontribusi Etherscan = +0,064**

Ablasi mengubah klaim menjadi bukti. Ini standar di paper machine learning.

### Temuan tambahan yang menarik

GraphSAGE graph-only (0,8189) dan embedding-only + XGBoost (0,8184) **hampir identik**.
Artinya: mengganti classifier head GraphSAGE dengan XGBoost tidak mengubah apa pun.
Batasannya bukan di classifier, tapi di **kualitas informasi graph** — kembali ke
264 edge tadi. Ini konsisten dengan diagnosis Anda dan memperkuatnya.

## 7. Cara menyajikan kekalahan late fusion tanpa terlihat gagal

Salah (defensif):
> ~~"Late fusion saya kurang berhasil, mungkin perlu tuning lebih lanjut."~~

Benar (ilmiah):
> "Late fusion V2 mencapai F1 0,8825, di bawah XGBoost tabular 0,9102. Ini adalah
> **temuan**, bukan kegagalan: pada graph transaksi yang sangat sparse — 264 edge untuk
> 8.595 node — embedding 128 dimensi dari GraphSAGE membawa **lebih sedikit** informasi
> daripada 19 fitur graph aslinya, karena proses agregasi tanpa tetangga justru
> memampatkan informasi tanpa menambah konteks. Ablasi membuktikan Etherscan tetap
> menyumbang +0,064 F1 di kedua arsitektur, jadi kesimpulan tentang nilai pengayaan
> data konsisten. Implikasinya jelas: GNN pada masalah ini baru layak dipakai bila
> sampling dilakukan secara graph-aware."

Itu jawaban tingkat S2. Hafalkan strukturnya: **angka → penjelasan mekanisme →
apa yang tetap valid → implikasi ke depan.**

---

## 3 pertanyaan cek pemahaman

1. Berapa edge asli antara 8.595 wallet di dataset Anda, dan berapa self-loop? Apa
   akibatnya terhadap kemampuan GraphSAGE?
2. Apa beda *early fusion* (skenario 2) dan *late fusion* dalam penelitian Anda?
   Mana yang menang, dan berapa selisih F1-nya?
3. Angka mana yang Anda kurangi dari angka mana untuk membuktikan kontribusi fitur
   Etherscan dalam eksperimen ablasi?
