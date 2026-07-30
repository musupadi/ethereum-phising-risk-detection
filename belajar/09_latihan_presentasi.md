# TAHAP 9 — Latihan Presentasi ke Dosen

**Cara memakai file ini:** tutup bagian jawaban, baca pertanyaannya, jawab dengan suara
keras, baru buka jawabannya. Jangan menghafal kalimat — hafalkan **angka** dan
**struktur argumen**.

---

## 1. Pendekatan apa yang digunakan dalam membangun dataset?

**Jawaban:**
Pendekatan **turunan berbasis pengayaan** (*derived + enriched dataset*). Tiga langkah:

1. **Sampling** — 8.595 alamat berlabel diambil dari EPTransNet, dengan `addresses.csv`
   sebagai tabel utama.
2. **Feature extraction** — untuk tiap alamat, 18 fitur agregat dihitung dari
   1-hop neighborhood di MulDiGraph (`ml/preprocess.py:77`).
3. **Data enrichment** — 7 fitur Etherscan dan 10 fitur BigQuery digabung lewat
   LEFT JOIN pada alamat yang sudah dinormalisasi huruf kecil.

Hasilnya `hybrid_dataset.csv`: 8.595 baris × 41 kolom, 1.165 phishing (13,6%).

❌ **Jangan bilang:** "saya pakai dataset Kaggle." Sebutkan namanya: EPTransNet/MulDiGraph.

---

## 2. Apakah Anda membuat dataset sendiri?

**Jawaban:**
"Tidak untuk dataset primernya, ya untuk dataset eksperimennya."

EPTransNet adalah dataset publik — graph transaksi Ethereum berlabel phishing yang
sudah ada. Saya tidak mengumpulkan transaksinya dan tidak membuat labelnya.

Yang saya bangun adalah **dataset eksperimen turunan**: saya menentukan populasi
sampel, merancang 18 fitur agregat 1-hop, dan mengambil sendiri data on-chain nyata
dari Etherscan dan BigQuery untuk 8.595 alamat tersebut — proses pengambilannya
resumable dan dibatasi rate limit, ada di `ml/enrich_real_addresses_*.py`.
Jadi kombinasi kolom pada dataset akhir ini tidak ada di dataset publik mana pun.

⭐ **Kunci:** akui batasnya di awal, lalu tunjukkan kontribusi nyatanya. Jangan defensif.

---

## 3. Bagaimana proses feature extraction dilakukan dalam program?

**Jawaban:**
Di fungsi `extract_node_features(G, node)`, `ml/preprocess.py:77`. Untuk tiap wallet:

1. `G.predecessors(node)` → semua wallet yang pernah mengirim ke dia.
2. Karena MulDiGraph mengizinkan edge paralel, tiap pasangan di-*loop* dengan
   `G[pred][node].values()` agar **semua** transaksi terhitung, bukan cuma satu.
3. `amount` dan `timestamp` dikumpulkan ke dalam list.
4. `G.successors(node)` → sisi keluar, cara yang sama.
5. `_safe_stats()` menghitung total, mean, max, min, std — dengan penanganan list kosong
   (semua nol) dan std=0 bila hanya satu transaksi.
6. Dikembalikan sebagai dict 18 kunci → `pd.DataFrame` → tabel.

Fungsi yang sama di-*import* oleh `build_hybrid_dataset.py`, jadi tidak ada risiko dua
rumus berbeda antara jalur eksperimen dan jalur produksi.

---

## 4. Mengapa menggunakan 1-hop neighborhood?

**Jawaban:** Tiga alasan.

1. **Skalabilitas.** MulDiGraph punya jutaan node. Ekspansi 2-hop dari wallet berderajat
   tinggi bisa mencakup ratusan ribu node per wallet.
2. **Interpretabilitas.** Fitur 1-hop punya arti langsung: "87 orang mengirim ke wallet ini".
   Agregat 2-hop sulit dijelaskan ke penyidik atau pengguna.
3. **Yang terpenting:** informasi multi-hop tidak saya buang — saya pindahkan ke model
   yang tepat. GraphSAGE 2 layer memang melakukan agregasi 2-hop. Jadi desain penelitian
   saya membandingkan secara adil: fitur 1-hop tabular vs message passing 2-hop.

⭐ Alasan ke-3 mengubah pertanyaan menjebak menjadi kelebihan desain. Utamakan itu.

---

## 5. Mengapa fiturnya 18?

**Jawaban:**
18 muncul dari **struktur sistematis**, bukan angka pilihan acak:

- 2 sisi arah (masuk / keluar) × 5 statistik nominal (total, mean, max, min, std) = **10**
- 2 hitungan derajat (`in_degree`, `out_degree`) = **2**
- 2 hitungan pihak unik (`unique_senders`, `unique_receivers`) = **2**
- 2 rentang waktu (`timestamp_spread_in/out`) = **2**
- 2 rasio hasil rekayasa (`in_out_ratio`, `sent_received_ratio`) = **2**

Total **18**. Setiap fitur punya pasangan arah, sehingga model bisa membedakan perilaku
menerima dan mengirim — yang justru inti pola phishing: banyak masuk dari banyak pihak,
lalu disapu keluar ke sedikit pihak.

Di skema skenario, ditambah kolom indikator `graph_present`, sehingga skenario 1
tercatat 19 kolom.

⚠️ **Waspada jebakan:** docstring `preprocess.py` menyebut 20 fitur (termasuk
`avg_in_timestamp`/`avg_out_timestamp`) yang **tidak pernah dihitung**. Kalau penguji
membuka kode dan menghitung sendiri, ia akan menemukan 18. Perbaiki docstring dan
pastikan naskah tesis Anda menulis 18.

---

## 6. Mengapa ditambah Etherscan dan BigQuery?

**Jawaban:**
MulDiGraph adalah **snapshot statis tahun 2019**. Ia hanya tahu perilaku transaksi
di dalam graph itu. Etherscan memberi keadaan wallet **saat ini** (saldo, total transaksi,
jumlah rekanan) dan BigQuery memberi agregat transaksi skala penuh dari seluruh
riwayat blockchain.

Hipotesisnya: menggabungkan perilaku historis dan keadaan terkini akan meningkatkan
deteksi.

**Hasil pengujian hipotesis — dan di sini letak temuannya:**

| Skenario | F1 XGBoost | Selisih dari baseline |
|---|---|---|
| Graph saja | 0,8857 | — |
| Graph + Etherscan | **0,9102** | **+0,0245** ✅ |
| Graph + BigQuery | 0,8838 | −0,0019 ❌ |
| Ketiganya | 0,9038 | +0,0181 |

**Etherscan terbukti membantu, BigQuery tidak.** Penyebabnya: fitur `bq_*` sebagian
besar redundan dengan fitur graph — `bq_tx_count_in` mengukur hal yang mirip dengan
`in_degree`, `bq_total_in_eth` mirip `total_received`. Menambah 10 kolom redundan hanya
menambah dimensi dan risiko overfit.

⭐ Ini jawaban terbaik karena Anda melaporkan **hasil negatif** dengan penjelasan
mekanismenya. Itu tanda peneliti, bukan tukang lapor angka bagus.

---

## 7. Bagaimana ketiga sumber data digabungkan?

**Jawaban:**
Tiga kali **LEFT JOIN** dengan kunci `address` yang dinormalisasi ke huruf kecil,
di `ml/build_hybrid_dataset.py`:

```python
merged_base = base_df.merge(etherscan_df, on="address", how="left")   # :235
merged_df   = merged_base.merge(graph_df,  on="address", how="left")  # :267
merged_df   = merged_df.merge(bq_df,       on="address", how="left")  # :278
```

Tabel utamanya `addresses.csv` (daftar alamat + label), sehingga jumlah baris dan label
tidak pernah berubah — LEFT JOIN hanya menambah kolom.

Wallet yang tidak ditemukan di suatu sumber mendapat 0 untuk kolom sumber tersebut,
dan kolom `graph_present` mencatat apakah fitur graph asli atau hasil pengisian nol.
Setiap tabel juga di-`drop_duplicates` pada `address` sebelum join, untuk mencegah
penggandaan baris.

Pada dataset final, `graph_present` bernilai 1 untuk seluruh 8.595 baris — artinya
semua alamat berhasil ditemukan di MulDiGraph, tidak ada fitur graph yang hasil zero-fill.

---

## 8. Mengapa disebut pengembangan?

**Jawaban (versi judul baru):**
Yang saya kembangkan adalah **sistemnya**, bukan algoritmanya.

Saya membangun pipeline lengkap: pembentukan dataset multi-sumber → pemilihan model
lewat eksperimen 4 model × 4 skenario → penyajian model sebagai layanan API →
antarmuka pengguna → pencatatan skor risiko ke smart contract dengan jejak audit
berbasis event log.

Pengembangan di sini bermakna rekayasa: merancang, membangun, dan mengintegrasikan
komponen menjadi sistem yang berjalan — bukan menciptakan algoritma baru.

⚠️ **Kalau judul Anda masih "pengembangan model", pertanyaan ini adalah jebakan
mematikan dan tidak ada jawaban yang aman.** Ubah judulnya (lihat TAHAP 8).

---

## 9. Model apa yang sebenarnya dikembangkan?

**Jawaban jujur:**
Tidak ada algoritma baru. Yang saya latih dan pilih adalah **XGBoost pada skenario
MulDiGraph + Etherscan dengan 26 fitur**, F1 0,9102 — model inilah yang dilayani sistem,
tersimpan sebagai `ml/artifacts/model_prod_xgb.json`.

Yang saya rancang sendiri: himpunan fiturnya, skenario sumber datanya, dan satu
arsitektur fusi (embedding GraphSAGE 128 dimensi digabung dengan 7 fitur Etherscan
menjadi 135 dimensi, lalu diklasifikasi XGBoost). Arsitektur fusi itu saya uji dan
saya ablasi — hasilnya F1 0,8825, di bawah XGBoost tabular, dan saya laporkan apa adanya.

⭐ Menjawab jujur di sini **menyelamatkan** Anda. Kalau Anda mengklaim mengembangkan
model, pertanyaan berikutnya pasti "tunjukkan bedanya dengan XGBoost standar" —
dan tidak ada yang bisa ditunjukkan.

---

## 10. Apa kebaruan penelitian ini?

**Jawaban — empat poin, urut dari yang terkuat:**

1. **Integrasi ML dengan pencatatan on-chain yang dapat diaudit.** Skor risiko, waktu
   penilaian, dan versi model dicatat di smart contract, dan riwayatnya bisa
   direkonstruksi dari event log. Sebagian besar penelitian deteksi phishing berhenti
   di tabel metrik.
2. **Dataset eksperimen multi-sumber** yang memadukan graph EPTransNet 2019 dengan
   data on-chain nyata terkini dari Etherscan dan BigQuery untuk 8.595 alamat yang sama.
3. **Temuan empiris tentang nilai pengayaan data:** Etherscan menaikkan F1 +0,0245,
   BigQuery justru tidak, disertai penjelasan redundansi fitur.
4. **Bukti empiris keterbatasan GNN pada graph transaksi hasil sampling acak:** hanya
   264 edge asli untuk 8.595 node, dan ablasi menunjukkan embedding graph sendirian
   (F1 0,8184) tidak mengungguli fitur agregat tabular.

❌ **Jangan bilang** "kebaruannya karena saya pakai GraphSAGE". GraphSAGE terbit 2017.

---

## 11. Mengapa XGBoost menjadi model terbaik?

**Jawaban — tiga alasan teknis:**

1. **Bentuk datanya tabular, dan XGBoost memang unggul di sana.** Setelah agregasi,
   tiap wallet adalah baris berisi 26 angka. Boosting membangun pohon secara berurutan,
   tiap pohon memperbaiki kesalahan pohon sebelumnya.
2. **Tahan terhadap skala ekstrem dan outlier.** `sent_received_ratio` bisa mencapai
   orde 10⁹ karena penyebutnya 1e-9. Pohon keputusan hanya membandingkan ambang, jadi
   tidak terganggu — sementara Logistic Regression rusak berat oleh nilai seperti itu
   (F1-nya hanya 0,7850 di skenario 1).
3. **Menangkap interaksi antar fitur secara otomatis.** Pola phishing bukan satu fitur
   tunggal, melainkan kombinasi: `unique_senders` tinggi **dan** `unique_receivers`
   rendah **dan** `sent_received_ratio` ≈ 1. Pohon bertingkat menyatakan kombinasi
   seperti itu secara alami; model linier tidak bisa.

Buktinya bukan satu percobaan: XGBoost menang di **keempat** skenario.

---

## 12. Mengapa GraphSAGE tidak mengalahkan XGBoost?

**Ini pertanyaan terpenting di seluruh sidang Anda. Jawaban ada di satu angka.**

**Jawaban:**
Karena graph-nya praktis kosong. Audit di `outputs/edge_source_audit_v2.json`
menunjukkan: dari 8.595 wallet, hanya ada **264 edge berarah unik** di antara mereka.
Rata-rata 0,03 edge per node. Jumlah self-loop (8.595) 32 kali lebih banyak dari edge asli.

Penyebabnya: sampling saya memilih alamat secara individual dari graph berisi jutaan
node, sehingga wallet-wallet terpilih hampir tidak pernah bertransaksi satu sama lain.

Akibatnya untuk lebih dari 99% node, GraphSAGE tidak punya tetangga untuk diagregasi —
ia hanya mengolah fitur node sendiri melalui self-loop, sehingga secara efektif berubah
menjadi MLP biasa. Dalam kondisi itu, wajar ia kalah dari XGBoost.

Bukti pendukungnya: GraphSAGE graph-only F1 0,8189 dan embedding-only + XGBoost
F1 0,8184 — hampir identik. Artinya batasannya bukan pada classifier, melainkan pada
kualitas informasi graph itu sendiri.

**Implikasinya untuk penelitian lanjutan:** GNN baru layak diuji pada masalah ini bila
sampling dilakukan secara graph-aware, misalnya snowball sampling yang mempertahankan
konektivitas subgraph.

⭐ Struktur jawaban ini: **angka → mekanisme → bukti pendukung → implikasi.** Kuasai.

---

## 13. Apa fungsi late fusion?

**Jawaban:**
Late fusion menguji hipotesis bahwa tiap jenis data sebaiknya diolah oleh model yang
paling cocok terlebih dahulu, baru digabungkan di tahap akhir.

Arsitekturnya: 19 fitur graph masuk ke GraphSAGE menghasilkan embedding 128 dimensi;
7 fitur Etherscan diskalakan terpisah; keduanya digabung menjadi vektor 135 dimensi
yang diklasifikasi XGBoost.

Ini kontras dengan *early fusion* pada skenario 2, di mana 26 fitur langsung masuk
satu model.

**Hasilnya: F1 0,8825, di bawah early fusion 0,9102.** Ini temuan, bukan kegagalan —
pada graph yang sangat sparse, kompresi menjadi embedding 128 dimensi justru
menghilangkan informasi dibanding 19 fitur aslinya, karena agregasi tanpa tetangga
tidak menambah konteks apa pun.

Versi V2 eksperimen ini saya rancang leakage-controlled: early stopping hanya memakai
validasi internal dari data training, test set di-*score* tepat satu kali, dan
7 fitur Etherscan masuk model persis sekali (di V1 sempat masuk dua kali).

---

## 14. Apa kontribusi smart contract?

**Jawaban — empat fungsi:**

1. **Auditability.** Setiap penilaian tercatat permanen beserta `timestamp` dan
   `modelVersion`. Bisa dibuktikan bahwa wallet X dinilai 91 pada waktu tertentu
   oleh model versi `xgb-eth-v1` — dan catatan itu tidak bisa diubah belakangan.
2. **Akuntabilitas versi model.** Kalau model diperbarui dan hasilnya berubah,
   jejak keputusan lama tetap ada. Ini penting untuk sistem yang menuduh sebuah
   wallet sebagai penipu.
3. **Interoperabilitas.** Kontrak lain (DEX, wallet, bursa) bisa memanggil `getRisk()`
   langsung on-chain sebelum memproses transaksi, tanpa perlu mempercayai API saya.
4. **Transparansi tanpa membuka data.** Yang dipublikasikan hanya skor, waktu, dan
   versi model — bukan 26 fitur atau profil finansial wallet.

**Batasan yang saya sadari:** penulisan dibatasi `onlyOwner`, jadi masih terpusat pada
satu penandatangan. Dalam sistem produksi, peran ini harus menjadi himpunan oracle
yang disetujui — dan itu sudah dicatat sebagai keterbatasan di dokumentasi kontrak.

⭐ Menyebut batasannya sendiri jauh lebih kuat daripada menunggu ditemukan penguji.

---

## 15. Apa perbedaan model dan sistem?

**Jawaban:**

| | Model | Sistem |
|---|---|---|
| **Wujud** | File `model_prod_xgb.json` — sekumpulan pohon keputusan | Beberapa proses yang berjalan dan saling berbicara |
| **Fungsi** | 26 angka masuk → 1 probabilitas keluar | Menerima permintaan manusia → menghasilkan keputusan tercatat |
| **Ukuran keberhasilan** | F1, precision, recall, ROC-AUC | Berfungsi tidaknya, keandalan, kemudahan pakai, waktu respons |
| **Dalam tesis saya** | XGBoost 26 fitur, F1 0,9102 | Streamlit + FastAPI + model + web3.py + RiskRegistry |

**Analogi:** model itu **mesinnya**, sistem itu **mobilnya**. Mesin yang bagus tidak ada
gunanya kalau tidak ada roda, kemudi, dan dasbor.

Model saya adalah **satu komponen** di dalam sistem saya. Itu sebabnya judul yang tepat
adalah "pengembangan sistem" — karena yang saya hasilkan adalah keseluruhan mobilnya,
sementara mesinnya saya pilih lewat perbandingan yang sistematis.

---

# Pertanyaan yang mungkin muncul tapi tidak Anda duga

| Pertanyaan | Jawaban singkat |
|---|---|
| "Kenapa presisinya cuma 0,886?" | Trade-off yang disengaja. Recall 0,936 lebih tinggi — model lebih rela salah tuduh (28 FP) daripada melewatkan penipu (15 FN). Untuk keamanan, itu arah yang benar. |
| "Datanya tidak seimbang, bagaimana Anda menanganinya?" | 1.165 vs 7.430 (1:6,4). LR & RF pakai `class_weight="balanced"`, GNN pakai bobot kelas dari distribusi training. Ada juga eksperimen SMOTE di `ml/smote_experiment.py`. Metrik utama F1, bukan accuracy — accuracy 0,975 menyesatkan pada data timpang. |
| "Bisa mendeteksi wallet baru?" | Belum. Sistem hanya melayani 8.595 alamat dalam dataset; alamat di luar itu mengembalikan 404. Infrastrukturnya sudah ada (`enrich_real_addresses_etherscan.py`), tinggal disambungkan ke jalur inferensi. Ini keterbatasan yang saya catat. |
| "Sudah diuji sistemnya?" | Repo belum punya test suite otomatis. Pengujian dilakukan manual per endpoint. Ini pekerjaan yang saya akui belum lengkap. **Siapkan tabel uji black-box sebelum sidang.** |
| "Kenapa ambangnya 0,5 dan 40/70?" | 0,5 adalah ambang keputusan biner standar. 40/70 hanya pemetaan komunikasi ke LOW/MEDIUM/HIGH untuk pengguna, bukan ambang statistik. Keduanya berbeda peran dan bisa disetel terpisah. |
| "Bagaimana pengguna tahu kenapa skornya tinggi?" | API mengembalikan 5 fitur paling berpengaruh memakai nilai SHAP bawaan XGBoost (`pred_contribs=True`, `ml/predict.py:117`). Jadi sistem bisa menjelaskan asal skornya. |
| "Datanya 2019, masih relevan?" | Pola perilakunya masih relevan (menerima dari banyak korban, menyapu ke sedikit penampung), dan itu justru alasan saya memperkaya dengan Etherscan yang mencerminkan keadaan terkini. Tapi validasi pada data phishing terbaru adalah pekerjaan lanjutan yang wajar. |

---

# Kunci jawaban — pertanyaan cek pemahaman TAHAP 1–8

**TAHAP 1**
1. `in_degree = 4` (jumlah transaksi), `unique_senders = 2` (A dan B). Berbeda karena
   `unique_senders` memakai `set`, sehingga A yang mengirim 3 kali tetap dihitung satu.
2. Karena alamat adalah hash acak tanpa pola, bertipe string, dan kalau dipaksakan akan
   membuat model menghafal alamat spesifik alih-alih mempelajari perilaku.
3. Hapus smart contract → masih ML. Hapus FastAPI + Streamlit → tinggal skrip
   eksperimen, tidak lagi bisa disebut sistem.

**TAHAP 2**
1. "Dataset primernya publik (EPTransNet), tetapi dataset eksperimennya saya bentuk
   sendiri melalui sampling, feature extraction 1-hop, dan pengayaan Etherscan+BigQuery."
2. Berlaku untuk jalur legacy `preprocess.py` → `train.csv`/`test.csv`. **Tidak** berlaku
   untuk `hybrid_dataset.csv`, yang memakai seluruh 8.595 baris (`HYBRID_NORMAL_RATIO=0`).
3. Extraction: mengubah bentuk data yang sudah ada (graph → `in_degree`).
   Enrichment: menambah kolom dari sumber lain (`balance` dari Etherscan).

**TAHAP 3**
1. `in_degree = 10`, `unique_senders = 1`. Justru `unique_senders` (dan turunannya)
   yang membedakan kedua kasus itu — `in_degree` saja identik.
2. Karena penyebutnya `total_received + 1e-9`; kalau wallet tidak pernah menerima,
   hasilnya meledak. Logistic Regression paling terganggu (butuh skala wajar);
   model berbasis pohon tidak.
3. `api/services/ml_service.py` — tambahkan nama kolomnya ke `_META_COLS`.

**TAHAP 4**
1. Tetap 8.595 baris, karena semua join `how="left"` dengan `addresses.csv` sebagai
   tabel kiri. 2.595 baris hanya akan punya kolom Etherscan bernilai 0.
2. Karena alamat Ethereum sering ditulis dengan checksum EIP-55 (campur besar-kecil).
   Tanpa `.lower()`, `0xAbC…` dan `0xabc…` dianggap berbeda dan join gagal diam-diam —
   tidak error, tapi hasilnya salah.
3. Fitur timestamp (`created_at_ts`, `bq_first_seen_ts`, `bq_last_seen_ts`), karena 0
   berarti tahun 1970, bukan "tidak ada". Alternatif yang lebih jujur: `NaN` dibiarkan
   (XGBoost menanganinya secara native) atau kolom penanda `*_present` terpisah.

**TAHAP 5**
1. Bahwa keunggulannya konsisten dan tidak bergantung pada himpunan fitur tertentu —
   jadi bukan kebetulan satu percobaan.
2. Fitur BigQuery redundan dengan fitur graph; menambah dimensi tanpa informasi baru
   menaikkan risiko overfit. Istilahnya: *curse of dimensionality* / fitur redundan.
3. Karena "pengembangan model" menuntut bukti bahwa arsitektur baru **lebih baik**.
   Late fusion Anda F1 0,8825, kalah dari baseline 0,9102.

**TAHAP 6**
1. 264 edge berarah unik vs 8.595 self-loop. Akibatnya >99% node tidak punya tetangga
   untuk diagregasi, sehingga GraphSAGE efektif berperilaku seperti MLP.
2. Early fusion = 26 fitur langsung ke satu model (F1 0,9102). Late fusion = graph diolah
   GraphSAGE dulu jadi 128-d, lalu digabung 7 Etherscan jadi 135-d (F1 0,8825).
   Early fusion menang, selisih 0,0277.
3. 0,8825 (late fusion) − 0,8184 (embedding-only) = **+0,064**, itulah kontribusi
   fitur Etherscan.

**TAHAP 7**
1. `set_risk` melempar exception, ditangkap di `main.py:122-124`, `tx_hash` menjadi
   `null`. Hasil ML tetap dikembalikan lengkap — API tidak error.
2. Karena `mapping` di Solidity tidak bisa di-*iterate*; tidak ada cara menanyakan
   "siapa saja isinya". Karena itu kontrak memancarkan event `RiskUpdated`, dan aplikasi
   merekonstruksi daftarnya dari log.
3. Untuk menjaga integritas data: yang tercatat permanen harus alamat dataset yang
   sebenarnya, bukan teks bebas yang bisa salah ketik oleh pengguna.

**TAHAP 8**
1. Karena klaim "metode baru" menuntut bukti superioritas dan generalisasi lintas
   dataset. Late fusion Anda kalah dari baseline dan hanya diuji pada satu dataset.
2. Sebagai **tahap pemilihan model** di dalam metodologi pengembangan sistem — menjawab
   "model mana yang layak dilayani sistem".
3. Metodologi SDLC eksplisit, analisis kebutuhan, diagram sistem, dan **pengujian
   sistem** — yang terakhir ini belum ada sama sekali di repo (tidak ada test suite).
