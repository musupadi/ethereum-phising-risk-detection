# TAHAP 1 — Gambaran Besar Sistem

## 1. Analogi dulu, jangan rumus

Bayangkan Anda petugas bank yang ingin tahu apakah sebuah rekening itu rekening penipu.

- Anda **tidak** melihat wajah pemilik rekening.
- Anda hanya punya **buku mutasi**: siapa mengirim ke siapa, berapa, kapan.
- Dari buku mutasi itu Anda membuat **ringkasan per rekening**: berapa kali menerima,
  berapa kali mengirim, rata-rata nominal, dari berapa orang berbeda, rentang waktunya.
- Ringkasan itu Anda masukkan ke **kalkulator penilai risiko** yang sudah "belajar" dari
  ribuan rekening yang sudah diketahui penipu / bukan penipu.
- Kalkulator mengeluarkan angka 0–100.
- Angka itu Anda **stempel dan catat di buku besar yang tidak bisa dihapus** (blockchain),
  supaya siapa pun bisa memverifikasi kapan penilaian dibuat dan versi model apa yang dipakai.

Itu persis sistem Anda. Sisanya hanya detail teknis.

## 2. Data awalnya berbentuk apa

File `data/raw/MulDiGraph.pkl` — sebuah **objek graph NetworkX (MultiDiGraph)** yang di-*pickle*.
Bukan CSV, bukan tabel. Dibaca di `ml/preprocess.py:160-161`:

```python
with open(graph_path, "rb") as f:
    G = pickle.load(f)
```

Isinya: kumpulan **node** dan **edge**, masing-masing punya atribut.

## 3. Arti node dan edge

| Istilah | Di penelitian Anda | Bukti di kode |
|---|---|---|
| **Node** | Satu wallet address Ethereum (`0x...`) | `G.nodes(data=True)` dipakai di `preprocess.py:169` |
| Atribut node | `isp` = 1 kalau phishing, 0 kalau normal | `data.get("isp", 0)` — `preprocess.py:170` |
| **Edge** | Satu transaksi dari wallet A ke wallet B | `G[pred][node].values()` — `preprocess.py:89` |
| Atribut edge | `amount` (jumlah ETH), `timestamp` (waktu) | `preprocess.py:90-93` |

**Kenapa "Multi"DiGraph?** Karena A bisa mengirim ke B lebih dari sekali. Setiap transaksi
adalah edge tersendiri (edge paralel). Karena itu kode meng-*loop* `G[pred][node].values()` —
mengambil **semua** transaksi antara pasangan yang sama, bukan cuma satu.

**Di** = *directed* (berarah). Kirim ≠ terima. Ini penting: fitur Anda memisahkan
`in_*` (masuk) dan `out_*` (keluar).

## 4. Bagaimana graph berubah jadi tabel

Ini bagian yang paling sering ditanya dosen. Jawabannya: **agregasi 1-hop**.

Untuk satu wallet X:
1. Ambil semua tetangga langsung yang **mengirim ke X** (`G.predecessors(X)`).
2. Ambil semua tetangga langsung yang **menerima dari X** (`G.successors(X)`).
3. Kumpulkan nominal & timestamp dari edge-edge tersebut ke dalam list.
4. Hitung ringkasan statistik: total, rata-rata, max, min, std.
5. Hasilnya **satu baris tabel** berisi 18 angka.

```
Graph (tak beraturan)  ──agregasi──▶  Tabel (baris = wallet, kolom = fitur)
```

Setelah semua wallet diproses, `pd.DataFrame(records)` (`preprocess.py:196`) mengubahnya
menjadi tabel biasa yang bisa dimakan Logistic Regression / Random Forest / XGBoost.

> **Ini kompromi penting.** Begitu graph jadi tabel, model kehilangan informasi
> *"siapa tetangga saya dan tetangga saya siapa lagi"*. Yang tersisa hanya ringkasan angka.
> Justru karena itulah Anda juga menguji GraphSAGE, yang tidak melakukan pemipihan ini.

## 5. Apa yang masuk ke model

Untuk model yang benar-benar dilayani (production): **26 angka per wallet**.

- 18 fitur agregat graph
- 1 kolom indikator `graph_present`
- 7 fitur Etherscan

Daftar persisnya ada di `ml/artifacts/prod_feature_schema.json`.

Yang **tidak** masuk: `node_id` (alamat wallet), `label`, `class_name`, `source_dataset`.
Dijaga oleh `_META_COLS` di `api/services/ml_service.py:52-53`.

## 6. Apa yang dihasilkan model

`ml/predict.py:145-167`:

```
probabilitas phishing (0..1)
  → risk_score = round(proba * 100)          # 0..100
  → risk_level: LOW <40, MEDIUM 40-69, HIGH ≥70
  → is_phishing: proba ≥ 0.5
  → top_contributors: 5 fitur paling berpengaruh (SHAP bawaan XGBoost)
```

Perhatikan: **ambang keputusan (0,5) dan ambang level risiko (40/70) itu dua hal berbeda.**
Yang menentukan "Phishing / Bukan Phishing" adalah 0,5. Level LOW/MEDIUM/HIGH hanya
label komunikasi ke pengguna.

## 7. Bagaimana hasil dikirim ke smart contract

`api/main.py:117-124` → `chain_service.set_risk(wallet, risk_score, MODEL_VERSION)`
→ web3.py memanggil fungsi Solidity:

```solidity
function setRisk(address wallet, uint16 score, string calldata modelVersion) external onlyOwner
```
(`chain/contracts/RiskRegistry.sol:57`)

Yang disimpan on-chain hanya **3 hal**: skor, timestamp, versi model
(`RiskRegistry.sol:20-24`). Fitur dan probabilitas **tidak** disimpan on-chain — itu
keputusan desain yang benar (murah, dan tidak membocorkan data).

Kegagalan tulis chain **tidak mematikan API** (`main.py:122-124`, `tx_hash` jadi `null`).

## 8. Mana yang machine learning, mana yang sistem, mana yang komparasi

Ini inti masalah judul Anda. Petakan dengan jujur:

| Bagian | Nama kegiatannya | File |
|---|---|---|
| Ekstraksi 18 fitur dari graph | **Feature engineering** (bagian dari ML) | `ml/preprocess.py` |
| Pengayaan Etherscan & BigQuery | **Data enrichment** (rekayasa data) | `ml/enrich_real_addresses_*.py` |
| Latih 4 model × 4 skenario | **Komparasi model + komparasi skenario data** | `ml/train_dataset_scenarios.py` |
| Tuning hyperparameter | **Optimasi model** | `ml/tune_xgboost.py`, `ml/tune_gnn.py` |
| Late fusion GraphSAGE+Etherscan | **Eksperimen arsitektur hybrid** (paling dekat ke "pengembangan model") | `experiments/run_graphsage_etherscan_late_fusion_v2.py` |
| Ablasi | **Validasi eksperimen** | `experiments/run_embedding_only_ablation.py` |
| Simpan model & skema, layani inferensi | **Rekayasa perangkat lunak / MLOps ringan** | `ml/train_production_model.py`, `ml/predict.py` |
| FastAPI + Streamlit | **Pengembangan sistem** | `api/`, `app/` |
| Smart contract + web3.py | **Pengembangan sistem (integrasi blockchain)** | `chain/`, `api/services/chain_service.py` |

**Kesimpulan jujur:** tidak ada satu baris pun di repo ini yang menciptakan algoritma
baru. Anda memakai `LogisticRegression`, `RandomForestClassifier`, `XGBoost`, dan
`SAGEConv` dari PyTorch Geometric — semuanya library standar. Yang Anda bangun sendiri
adalah **pipeline, fitur, skenario data, eksperimen fusi, dan sistem end-to-end**.
Itu bukan kelemahan — itu justru argumen untuk mengubah judul jadi "pengembangan sistem"
(lihat TAHAP 8).

## Diagram satu layar

```
data/raw/MulDiGraph.pkl  (graph: node=wallet, edge=transaksi)
        │
        │  ml/preprocess.py :: extract_node_features()   ← agregasi 1-hop
        ▼
   18 fitur per wallet (tabel)
        │
        │  ml/build_hybrid_dataset.py  ← LEFT JOIN by address
        ├──◄ addresses_etherscan_enriched.csv   (7 fitur)
        └──◄ addresses_bigquery_enriched.csv    (10 fitur)
        ▼
data/processed/hybrid_dataset.csv   (8.595 baris × 41 kolom)
        │
        ├─► ml/train_dataset_scenarios.py  → 4 model × 4 skenario  → tabel tesis
        ├─► experiments/…late_fusion_v2.py → GraphSAGE 128-d + 7 Etherscan → XGBoost
        └─► ml/train_production_model.py   → model_prod_xgb.json (26 fitur, F1 0,9102)
                                                   │
        Streamlit :8502 ──HTTP──► FastAPI :8000 ───┤
                                                   │  ml/predict.py
                                                   ▼
                                     probability → risk_score 0-100 → level
                                                   │
                                    web3.py ───────┤ CHAIN_ENABLED=true
                                                   ▼
                                    RiskRegistry.sol @ Hardhat :8545
                                    (setRisk → event RiskUpdated)
```

---

## 3 pertanyaan cek pemahaman

1. Kalau wallet X menerima 3 transaksi dari wallet A dan 1 dari wallet B, berapa nilai
   `in_degree` dan berapa `unique_senders`? Kenapa berbeda?
2. Kenapa alamat wallet (`node_id`) tidak boleh ikut menjadi fitur input model?
3. Sebutkan satu bagian sistem yang kalau dihapus, penelitian Anda **masih** bisa disebut
   "machine learning", dan satu bagian yang kalau dihapus penelitian Anda **tidak lagi**
   bisa disebut "pengembangan sistem".

*(Kunci jawaban ada di akhir [09_latihan_presentasi.md](09_latihan_presentasi.md).)*
