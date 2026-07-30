# TAHAP 7 — Sistem End-to-End

Ini bagian terkuat penelitian Anda. Kuasai alurnya sampai bisa digambar di papan tulis
tanpa melihat catatan.

## 1. Tiga proses terpisah

```
┌───────────────────┐   HTTP    ┌───────────────────┐   JSON-RPC   ┌──────────────────┐
│ Streamlit  :8502  │ ────────► │ FastAPI    :8000  │ ───────────► │ Hardhat   :8545  │
│ (antarmuka)       │ ◄──────── │ (otak + model ML) │ ◄─────────── │ (blockchain)     │
└───────────────────┘           └───────────────────┘              └──────────────────┘
                                          │
                                          ▼
                              ml/artifacts/model_prod_xgb.json
```

Ketiganya jalan sebagai **proses terpisah** — itu sebabnya butuh 2–4 terminal.
Blockchain **opsional**: matikan dengan `CHAIN_ENABLED=false` dan sistem tetap jalan
penuh kecuali pencatatan on-chain (`api/config.py:57`).

## 2. Alur satu permintaan, langkah demi langkah

### Langkah 1 — Pengguna memasukkan wallet di Streamlit
`app/streamlit_app.py` menyediakan **mode scan-by-address**: pengguna memilih contoh
wallet dari daftar atau mengetik alamat `0x...`. Streamlit **tidak** memuat model —
ia hanya mengirim HTTP POST ke API.

### Langkah 2 — Request masuk ke FastAPI
```
POST /analyze
{ "wallet": "0x4f66f7e2de0a1f9848ac61278f61a69c2bd276be" }
```
Ditangani `api/main.py:81-137`. Ada tiga jalur input (`main.py:93-99`):

| Input | Mode | Sumber fitur |
|---|---|---|
| `row_id` | Demo | baris ke-N dari `prod_demo.csv` (data held-out, ada label sebenarnya) |
| `features` | Manual | dipakai apa adanya; kunci yang tidak diisi → 0 |
| hanya `wallet` | Scan by address | dicari di `hybrid_dataset.csv` berdasarkan `node_id` |

### Langkah 3 — Pengambilan & pembersihan fitur
`api/services/ml_service.py:96-123`:
```python
key = str(wallet).strip().lower()
if not re.fullmatch(r"0x[0-9a-f]{40}", key):   # validasi format
    raise KeyError("Alamat tidak valid — harus 42 karakter …")
if key not in df.index:
    raise KeyError("Wallet tidak ada di dataset MulDiGraph (snapshot 2019) …")
row = df.loc[key]
features = _row_features(row)   # buang kolom meta/string
```

> ⚠️ **Batasan yang harus Anda sebut sendiri:** sistem ini **hanya** bisa memindai
> wallet yang sudah ada di dataset (8.595 alamat, snapshot 2019). Alamat baru akan
> mendapat HTTP 404. Ia bukan pemindai real-time. Kalau ditanya, jawab jujur dan
> tunjukkan bahwa infrastrukturnya sudah siap: tinggal menyambungkan
> `ml/enrich_real_addresses_etherscan.py` ke jalur inferensi.

### Langkah 4 — Preprocessing di titik inferensi
`ml/predict.py:86-87` — ini penting dan sering terlewat:
```python
feature_cols = _schema["features"]      # urutan 26 fitur dari prod_feature_schema.json
row = np.array([[float(features.get(col, 0.0)) for col in feature_cols]])
```
Dua hal terjadi di sini:
1. **Urutan fitur dipaksa mengikuti skema.** Kalau urutan berbeda dari saat training,
   prediksi jadi omong kosong tanpa error. Ini pengaman yang benar.
2. **Fitur yang hilang diisi 0.** Sama seperti perlakuan saat membangun dataset — konsisten.

### Langkah 5 — Model menghasilkan probabilitas
```python
dmatrix = xgb.DMatrix(row, feature_names=feature_cols)
return float(_model.predict(dmatrix)[0])       # ml/predict.py:92-93
```
Model di-*load* sekali saja (singleton `_model`, `predict.py:34-41`), jadi request kedua
dan seterusnya cepat.

### Langkah 6 — Probabilitas → risk score → status
```python
score      = round(prob * 100)                             # predict.py:156
is_phishing = prob >= 0.5                                  # DECISION_THRESHOLD, :114
risk_level = "LOW" if score<40 else "MEDIUM" if score<70 else "HIGH"   # :103-110
```
Ditambah **penjelasan** — 5 fitur paling berpengaruh via SHAP bawaan XGBoost
(`predict.py:117-142`, `pred_contribs=True`). Ini yang membuat sistem Anda tidak
sekadar kotak hitam. Sebutkan saat presentasi.

### Langkah 7 — Penentuan alamat kanonik
`api/main.py:112-115`:
```python
node_id = str(ml_result.get("node_id", "") or "")
wallet = node_id if re.fullmatch(r"0x[0-9a-fA-F]{40}", node_id) else req.wallet
```
Yang dicatat on-chain adalah **alamat dataset yang sebenarnya**, bukan teks bebas yang
diketik pengguna. Detail integritas data yang bagus.

### Langkah 8 — Penulisan ke smart contract
`api/main.py:118-124` → `chain_service.set_risk()` (`chain_service.py:63-90`):
```python
nonce = _w3.eth.get_transaction_count(_account.address)
tx = _contract.functions.setRisk(wallet_addr, score, model_version).build_transaction({...})
signed = _w3.eth.account.sign_transaction(tx, private_key=SIGNER_PRIVATE_KEY)
tx_hash = _w3.eth.send_raw_transaction(signed.raw_transaction)
receipt = _w3.eth.wait_for_transaction_receipt(tx_hash)
```
Kalau gagal → ditangkap, dicatat sebagai warning, `tx_hash = None`, API **tetap
mengembalikan hasil ML**. Desain yang benar: kegagalan lapisan opsional tidak boleh
menjatuhkan lapisan inti.

### Langkah 9 — Respons ke pengguna
`api/main.py:126-137`, sesuai `api/schemas.py`:
```json
{
  "wallet": "0x...", "probability": 0.9123, "risk_score": 91,
  "risk_level": "HIGH", "is_phishing": true, "verdict": "Phishing",
  "top_contributors": [...], "tx_hash": "0x...", "true_label": 1,
  "dataset_split": "dataset"
}
```

## 3. Fungsi tiap komponen (versi hafalan)

| Komponen | Perannya dalam satu kalimat | Tanpanya apa yang hilang |
|---|---|---|
| **Streamlit** | Antarmuka web untuk manusia; tidak punya logika ML sama sekali | Sistem tetap jalan lewat `curl` / Swagger |
| **FastAPI** | Otak orkestrasi: menerima request, mengambil fitur, memanggil model, menulis ke chain | Semuanya berhenti |
| **Model ML (XGBoost)** | Mengubah 26 angka jadi satu probabilitas | Tidak ada prediksi |
| **web3.py** | Penerjemah Python ⇄ Ethereum: menyusun, menandatangani, mengirim transaksi | Skor tidak bisa masuk blockchain |
| **Hardhat** | Node Ethereum lokal untuk pengembangan (blockchain "mainan" di laptop) | Tidak ada tempat kontrak dijalankan |
| **RiskRegistry.sol** | Buku besar on-chain: `mapping(address => Record)` | Hasil tidak permanen & tidak bisa diaudit publik |

## 4. Isi data on-chain

Dari `chain/contracts/RiskRegistry.sol:20-24` — **hanya tiga field**:

| Field | Tipe | Isi | Kenapa disimpan |
|---|---|---|---|
| `riskScore` | `uint16` | 0–100 | Hasil penilaian |
| `timestamp` | `uint64` | `block.timestamp` | Bukti **kapan** dinilai — tidak bisa dipalsukan |
| `modelVersion` | `string` | `"xgb-eth-v1"` | Bukti **model mana** yang memutuskan → auditability |

Yang **tidak** disimpan: probabilitas mentah, 26 fitur, identitas pengguna. Ini benar —
menyimpan fitur on-chain berarti mempublikasikan profil finansial wallet selamanya.

Kontrol akses: `onlyOwner` (`:39-42`) — hanya deployer yang boleh menulis. Docstring
kontrak sendiri mengakui ini penyederhanaan demo:
> *"in production this role can be an approved set of oracle addresses."*

Kutip ini kalau ditanya "bukankah ini terpusat?" — Anda sudah menyadarinya, dan itu
tercatat di kode, bukan alasan dadakan.

## 5. Membaca kembali: dua cara

| Endpoint | Cara kerja | Kode |
|---|---|---|
| `GET /risk/{wallet}` | Panggil `getRisk()`, baca langsung dari storage kontrak | `chain_service.py:93-107` |
| `GET /records` | Baca **event log** `RiskUpdated` dari block 0, ambil catatan terbaru per wallet | `chain_service.py:110-146` |

Kenapa dua cara? Karena Solidity `mapping` **tidak bisa di-*iterate***. Anda tidak bisa
bertanya "siapa saja yang pernah dicatat?" ke sebuah mapping. Solusinya: kontrak
memancarkan `event RiskUpdated` setiap penulisan (`RiskRegistry.sol:65`), dan aplikasi
merekonstruksi tabel audit dari log tersebut. **Ini pola standar Ethereum dan poin
teknis yang bagus untuk disebut saat sidang.**

## 6. Konfigurasi yang harus Anda tahu

Dari `api/config.py`:

| Variabel | Default | Fungsi |
|---|---|---|
| `CHAIN_ENABLED` | `true` | Saklar utama lapisan blockchain |
| `RPC_URL` | `http://127.0.0.1:8545` | Alamat node Hardhat |
| `SIGNER_PRIVATE_KEY` | akun Hardhat #0 | Penandatangan transaksi — **demo saja** |
| `MODEL_VERSION` | `xgb-eth-v1` | Tercatat on-chain |
| `TEST_CSV` | `prod_demo.csv` (fallback `test.csv`) | Sumber mode demo |
| `LOOKUP_CSV` | `hybrid_dataset.csv` (fallback `TEST_CSV`) | Sumber mode scan |

⚠️ **Risiko yang harus Anda sadari:** private key tertulis langsung di
`config.py:36`. Untuk Hardhat lokal itu tidak masalah (kunci itu publik dan diketahui
semua orang). Tapi **jangan pernah** mengatakan sistem ini siap produksi tanpa
menyebut bahwa manajemen kunci harus diganti. Kalau penguji menemukannya duluan,
posisi Anda lemah; kalau Anda menyebutnya duluan, itu tanda kematangan.

Perhatikan juga `config.py:47-48`: dimensi fitur yang dilayani **bergantung pada file
CSV mana yang ada di disk**. Kalau `prod_demo.csv` hilang, sistem diam-diam turun ke
`test.csv` yang hanya punya 18 fitur, dan 8 fitur Etherscan akan diisi 0 tanpa
peringatan. Jaga agar `prod_demo.csv`, `hybrid_dataset.csv`, dan
`prod_feature_schema.json` selalu sinkron.

## 7. Perintah menjalankan (untuk demo di depan dosen)

```powershell
.\.venv\Scripts\Activate.ps1

# Terminal 1 — blockchain
cd chain ; npx hardhat node

# Terminal 2 — deploy kontrak (sekali saja per sesi node)
cd chain ; npx hardhat run scripts/deploy.ts --network localhost

# Terminal 3 — API
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 4 — UI
streamlit run app/streamlit_app.py --server.port 8502
```

Cek kesehatan sebelum presentasi:
```powershell
curl http://localhost:8000/health      # chain_connected harus true
```

Tanpa blockchain (lebih aman kalau waktu mepet): `$env:CHAIN_ENABLED = "false"`,
lalu cukup terminal 3 dan 4.

---

## 3 pertanyaan cek pemahaman

1. Kalau node Hardhat mati di tengah demo, apa yang terjadi pada respons `/analyze`?
   Field mana yang berubah?
2. Kenapa `/records` harus membaca event log, tidak bisa langsung membaca `mapping`
   di kontrak?
3. Kenapa alamat wallet yang dicatat on-chain diambil dari `node_id`, bukan dari
   teks yang diketik pengguna?
