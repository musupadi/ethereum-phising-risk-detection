# Cara Menjalankan Semuanya — Dari Nol

Panduan lengkap untuk mencoba sendiri. Semua perintah untuk **Windows PowerShell**.

---

## Peta dulu: ada dua sistem, jangan tertukar

| | Sistem tesis | Sistem baru |
|---|---|---|
| Letak | akar repo (`api/`, `app/`, `ml/`, `chain/`) | `new_system/` |
| Model | `ml/artifacts/model_prod_xgb.json` — XGBoost 26 fitur, F1 91,02% | ensemble 3 model, F1 90,08% |
| Cara jalan | lewat **aplikasi** (Streamlit + FastAPI) | lewat **skrip** di terminal |
| Tersambung ke aplikasi? | ✅ ya | ❌ **belum** — sengaja, agar batasnya jelas |

⚠️ **Yang paling sering bikin bingung:** kalau kamu menyalakan `LIVE_SCAN_ENABLED=true`
di aplikasi, yang dipakai adalah **model tesis** dengan fitur diambil live — *bukan*
model perbaikan di `new_system/`. Mode itu memang sengaja dibuat untuk
**memperlihatkan kelemahannya** ke dosen.

---

## Persiapan (sekali saja)

```powershell
cd C:\Kuliah\ethereum-phising-risk-detection
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Pastikan `.env` berisi:
```
ETHERSCAN_API_KEY=<kunci kamu>
```
Kunci ini hanya dibutuhkan untuk mode scan live dan skrip `new_system/`.
Aplikasi dasar jalan tanpa itu.

---

## SKENARIO 1 — Aplikasi dasar, tanpa blockchain (paling cepat, 2 terminal)

Cocok untuk mendemokan bagian machine learning saja.

```powershell
# Terminal 1 — API
.\.venv\Scripts\Activate.ps1
$env:CHAIN_ENABLED = "false"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 2 — Antarmuka
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py --server.port 8502
```

Buka **http://localhost:8502**

**Coba ini:** tempel `0x60afc5761212ed59608b0559b4ccbaad24fc74b1` → harus keluar
skor **100 HIGH — Phishing**, lengkap dengan tabel "Mengapa skor ini?".

Wallet di luar dataset akan ditolak dengan pesan 404. Itu perilaku bawaan dan benar.

---

## SKENARIO 2 — Aplikasi + blockchain (RiskRegistry), 4 terminal

Skor risiko tercatat permanen ke smart contract.

```powershell
# Terminal 1 — node blockchain (biarkan terbuka)
cd chain
npx hardhat node

# Terminal 2 — deploy RiskRegistry (ulangi tiap node dinyalakan ulang)
cd chain
npx hardhat run scripts/deploy.ts --network localhost

# Terminal 3 — API dengan chain aktif
.\.venv\Scripts\Activate.ps1
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 4 — Antarmuka
.\.venv\Scripts\Activate.ps1
streamlit run app/streamlit_app.py --server.port 8502
```

Cek dulu sebelum demo:
```powershell
curl http://localhost:8000/health      # chain_connected harus true
```

Scan wallet → hasilnya muncul dengan `tx_hash`, artinya sudah tercatat on-chain.

---

## SKENARIO 3 — Tambah halaman DApp (baca langsung dari blockchain)

Lanjutan Skenario 2, tambah satu terminal:

```powershell
# Terminal 5
.\chain\dapp\run_dapp.ps1
```

Buka **http://localhost:8503/dapp/** → klik **Hubungkan** → tempel wallet →
**Baca on-chain**.

**Demo paling meyakinkan:** matikan API di Terminal 3 (Ctrl+C), lalu baca lagi di
halaman DApp. Tetap jalan — karena halaman itu bicara langsung ke blockchain,
tanpa melewati backend.

---

## SKENARIO 4 — Scan wallet mana pun (bukti kelemahan model)

Lanjutan mana pun di atas. Bedanya cuma satu variabel:

```powershell
# Di terminal API, sebelum menjalankan uvicorn:
$env:LIVE_SCAN_ENABLED = "true"
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Sekarang wallet di luar dataset tidak ditolak lagi — fiturnya diambil dari Etherscan
saat itu juga, lalu dinilai **model tesis**.

**Coba alamat ini** dan perhatikan hasilnya:

| Alamat | Siapa | Hasil yang muncul |
|---|---|---|
| `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` | Vitalik Buterin | **98 HIGH — Phishing** ❌ |
| `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D` | Uniswap V2 Router | **100 HIGH — Phishing** ❌ |
| `0xf89d7b9c864f589bbf53a82105107622b35eaa40` | Bybit Hot Wallet | **99 HIGH — Phishing** ❌ |

Inilah **bahan diskusi dengan dosen**: model yang F1-nya 91% pada populasi
penelitiannya ternyata menuduh Vitalik Buterin sebagai penipu ketika diberi wallet
di luar populasi itu.

Satu scan memakan 5–20 detik karena memanggil Etherscan beberapa kali. Wajar.

---

## SKENARIO 5 — Sistem baru: model perbaikan + label on-chain

Ini **belum tersambung ke Streamlit**, jadi dijalankan lewat terminal.

### 5a. Uji model perbaikan saja

```powershell
.\.venv\Scripts\Activate.ps1
python new_system\ml\predict_live.py 0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045
```
Hasilnya **skor 6 — LOW**, bukan 98. Itu perbedaan sebelum dan sesudah perbaikan.

### 5b. Nyalakan lapisan label on-chain

Butuh node Hardhat hidup (Terminal 1 dari Skenario 2).

```powershell
# Deploy LabelRegistry + isi 25 label sekaligus
python new_system\chain\deploy_label_registry.py
```

Skrip mencetak alamat kontraknya. Salin ke variabel lingkungan:

```powershell
$env:LABEL_REGISTRY_ADDRESS = "<alamat yang dicetak tadi>"
$env:RPC_URL = "http://127.0.0.1:8545"
```

Lalu uji:
```powershell
python new_system\ml\predict_live.py 0x631fc1ea2270e98fbd9d92658ece0f5a269aa161
```

Yang akan terlihat: model menilai **76 — phishing**, tetapi label on-chain
mengoreksinya menjadi **Bukan Phishing**, dan ketidaksepakatan itu tercatat.

⚠️ Node Hardhat menyimpan data di memori. Setiap kali node dinyalakan ulang,
**LabelRegistry hilang dan harus di-deploy ulang.** Bila kontrak tidak terbaca,
sistem otomatis memakai daftar lokal `new_system/data/address_labels.json`.

### 5c. Menjalankan ulang evaluasi

```powershell
python new_system\ml\finalize_live_model.py              # latih + semua pemeriksaan
python new_system\experiments\run_ood_evaluation.py      # bandingkan kedua model
python new_system\experiments\run_false_positive_test.py 20
```

---

## Ringkasan: mode mana pakai model apa

| Yang kamu jalankan | Wallet dalam dataset | Wallet luar dataset |
|---|---|---|
| Aplikasi bawaan | model tesis ✅ | ditolak 404 |
| Aplikasi + `LIVE_SCAN_ENABLED=true` | model tesis ✅ | **model tesis** (memperlihatkan kelemahan) |
| `new_system/ml/predict_live.py` | — | **model perbaikan** + label |

---

## Kalau ada masalah

| Gejala | Sebab | Tindakan |
|---|---|---|
| `chain_connected: false` | node mati / kontrak belum di-deploy | Jalankan `npx hardhat node`, lalu `deploy.ts` |
| DApp bilang "belum pernah dinilai" | node di-restart, data hilang | Deploy ulang + scan ulang satu wallet |
| `ETHERSCAN_API_KEY tidak tersedia` | `.env` belum diisi | Isi `ETHERSCAN_API_KEY` di `.env` |
| Scan lama sekali (>20 detik) | wallet di luar dataset, memanggil Etherscan | Wajar, tunggu saja |
| Label on-chain tidak terbaca | node restart / alamat kontrak usang | Deploy ulang, perbarui `LABEL_REGISTRY_ADDRESS` |
| `address already in use` | proses lama masih jalan | Tutup terminal lama, atau ganti nomor port |

**Kalau semuanya kacau menjelang demo:** jalankan Skenario 1 saja. Bagian machine
learning — inti tesis — tetap tampil utuh tanpa blockchain.

---

## Urutan yang saya sarankan untuk mencoba pertama kali

1. **Skenario 1** — pastikan bagian ML jalan (5 menit)
2. **Skenario 2** — tambah blockchain, lihat `tx_hash` muncul
3. **Skenario 3** — buka DApp, matikan API, buktikan tetap jalan
4. **Skenario 4** — masukkan alamat Vitalik, lihat sistem salah menuduh
5. **Skenario 5a** — jalankan model perbaikan pada alamat yang sama, lihat 98 → 6

Lima langkah itu adalah alur cerita utuh untuk dibawa ke bimbingan: sistem jalan,
tercatat on-chain, dapat diaudit, **ditemukan kelemahannya**, lalu **diperbaiki**.
