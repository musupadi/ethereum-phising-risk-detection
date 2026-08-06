# Lapisan DApp Lokal — Pendukung Judul Baru

Dibuat menyusul arahan dosen untuk memasukkan **RiskRegistry / DApp** ke dalam judul.
Status: **berjalan dan sudah diuji di jaringan lokal (Hardhat, chainId 31337).**

---

## 1. Kenapa ini perlu dibuat

Judul yang diminta dosen menyebut **DApp**. Sebelum ini, sistem Anda **belum** memenuhi
definisi DApp sama sekali — semua akses ke blockchain harus melewati FastAPI.

| Syarat DApp | Sebelum | Sesudah |
|---|---|---|
| Klien berbicara langsung ke blockchain | ❌ selalu lewat FastAPI | ✅ browser → node, tanpa perantara |
| Bisa dibaca tanpa backend | ❌ matikan API, semua mati | ✅ matikan API, pembacaan tetap jalan |
| Dukungan wallet pengguna (MetaMask) | ❌ tidak ada | ✅ ada (opsional) |
| Jaringan publik | ❌ localhost | ⏳ belum — masih lokal, sesuai keputusan |
| Logika inti on-chain | ❌ model ML off-chain | ❌ **tidak akan pernah** — lihat §6 |

Sekarang klaim "DApp" punya dasar yang bisa ditunjukkan, walau masih terbatas pada
lapisan registry.

---

## 2. FastAPI TIDAK dihapus — ini dua jalur berdampingan

Ini pertanyaan yang wajar muncul: *"loh, FastAPI kan sudah ada?"*

Betul, dan FastAPI **tetap dipakai**. Yang bertambah adalah jalur kedua:

```
JALUR TULIS (butuh model ML — wajib lewat backend)
  Streamlit → FastAPI → XGBoost → web3.py → setRisk() → RiskRegistry
                                                              │
JALUR BACA (tidak butuh model — langsung ke chain)            │
  Browser (halaman DApp) ──── eth_call / eth_getLogs ─────────┘
```

**Kenapa jalur tulis tidak bisa jadi DApp?** Karena menulis skor butuh menjalankan model
XGBoost 26 fitur. Model itu mustahil dijalankan di dalam smart contract — biaya gas-nya
tidak masuk akal, dan Solidity tidak punya floating point. Jadi inferensi ML **wajib**
off-chain. Ini bukan kelemahan desain Anda; ini keterbatasan fundamental blockchain,
dan berlaku untuk semua sistem ML + blockchain.

**Kenapa jalur baca bisa jadi DApp?** Karena membaca skor tidak butuh model sama sekali —
datanya sudah tersimpan di kontrak. Siapa pun yang punya alamat kontrak bisa memverifikasi
sendiri tanpa mempercayai server Anda.

⭐ **Inilah argumen terkuat Anda:** sistem ini memisahkan *kepercayaan* dengan benar.
Perhitungan boleh terpusat karena hasilnya bisa diverifikasi siapa saja secara terdesentralisasi.

---

## 3. Cara menjalankan (4 terminal)

```powershell
# Terminal 1 — blockchain lokal
cd chain ; npx hardhat node

# Terminal 2 — deploy kontrak (sekali tiap node baru dinyalakan)
cd chain ; npx hardhat run scripts/deploy.ts --network localhost

# Terminal 3 — API + UI (jalur tulis, untuk mengisi data)
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
streamlit run app/streamlit_app.py --server.port 8502

# Terminal 4 — halaman DApp (jalur baca)
.\chain\dapp\run_dapp.ps1
```

Lalu buka: **http://localhost:8503/dapp/**

⚠️ Harus lewat `http://`, jangan buka file HTML-nya langsung (`file://`) — kalau langsung,
alamat kontrak tidak terisi otomatis dan sebagian browser memblokir permintaan jaringannya.

---

## 4. Isi halamannya

| Bagian | Fungsi | Panggilan RPC |
|---|---|---|
| **Koneksi** | Sambung ke node, verifikasi ada kontrak di alamat itu, tampilkan chainId, block, owner | `eth_chainId`, `eth_blockNumber`, `eth_getCode`, `eth_call` → `owner()` |
| **Cek skor wallet** | Baca skor, level, versi model, waktu penilaian | `eth_call` → `getRisk(address)` |
| **Riwayat penilaian** | Tabel semua wallet yang pernah dinilai | `eth_getLogs` → event `RiskUpdated` |

Dua cara menyambung:
- **Hubungkan** — langsung ke `http://127.0.0.1:8545`, tanpa MetaMask. Untuk demo cepat.
- **MetaMask** — lewat `window.ethereum`. Ini yang membuktikan sifat DApp-nya paling nyata.
  (Perlu menambahkan jaringan Hardhat di MetaMask: RPC `http://127.0.0.1:8545`, chainId `31337`.)

---

## 5. Detail teknis yang layak dibanggakan saat sidang

### Tanpa satu pun library eksternal

Halaman ini **tidak memakai ethers.js, web3.js, atau CDN apa pun**. Encoding dan decoding
ABI dilakukan sendiri di JavaScript. Konsekuensinya: halaman ini jalan sepenuhnya offline,
tidak akan rusak karena CDN mati saat sidang.

Selector fungsi dihitung dari keccak256 tanda tangan fungsi:

| Tanda tangan | Selector / topic |
|---|---|
| `getRisk(address)` | `0xcca7ee41` |
| `owner()` | `0x8da5cb5b` |
| `RiskUpdated(address,uint16,uint64,string)` | `0x714c3a9c073adc76bbe4b255f69bb4feba383e720ad06bdd9f16d7d1cce79fd4` |

Decoding hasilnya (`decodeString`) menangani ABI dinamis: 3 slot kepala 32-byte, slot ke-3
berisi offset menuju panjang string dan datanya.

### Masalah CORS yang ditemukan dan dipecahkan

Saat pengujian, node Hardhat ternyata menjawab preflight dengan:
```
Access-Control-Allow-Methods: OPTIONS, GET
```
`POST` tidak diizinkan — padahal JSON-RPC memakai POST. Kalau memakai
`Content-Type: application/json`, browser mengirim preflight lebih dulu, dan permintaannya
**akan diblokir**.

Solusinya: kirim dengan `Content-Type: text/plain;charset=UTF-8`. Ini membuat permintaannya
tergolong *simple request*, sehingga browser tidak melakukan preflight sama sekali.
Hardhat tetap mem-parse body-nya sebagai JSON. Sudah diverifikasi terhadap node lokal.

Kalau penguji bertanya *"kenapa content-type-nya text/plain, bukan application/json?"* —
ini jawabannya, dan itu pertanyaan yang menunjukkan Anda menguasai detailnya.

---

## 6. Batasan yang HARUS Anda sebut sendiri

Jangan mengklaim lebih dari yang ada. Tiga batasan ini akan ditanyakan:

1. **Masih jaringan lokal.** Hardhat `chainId 31337`, bukan testnet publik. Kontraknya
   belum bisa dilihat orang lain. → lihat §7 untuk langkah berikutnya.
2. **Penulisan tetap terpusat.** `setRisk()` dibatasi `onlyOwner` dan ditandatangani satu
   private key di server. Yang terdesentralisasi baru **pembacaannya**.
3. **Model ML tetap off-chain, dan memang harus.** Yang disebut DApp adalah lapisan
   registry-nya, bukan seluruh sistem.

**Rumusan yang aman untuk tesis:**
> "Sistem ini terdiri atas dua lapis: lapisan inferensi terpusat yang menjalankan model
> XGBoost, dan lapisan registry terdesentralisasi berupa smart contract RiskRegistry yang
> dapat dibaca langsung oleh klien tanpa melalui backend. Inferensi ML tidak
> didesentralisasi karena keterbatasan komputasi on-chain; yang didesentralisasi adalah
> verifikasi hasilnya."

Kalimat itu jujur, teknis, dan justru terdengar lebih matang daripada mengklaim
"sistem saya DApp penuh".

---

## 7. Langkah berikutnya kalau mau naik ke testnet publik

Belum dikerjakan sesuai keputusan Anda (lokal dulu). Kalau nanti diperlukan:

1. Tambahkan network Sepolia di `chain/hardhat.config.ts`:
   ```ts
   sepolia: { url: process.env.SEPOLIA_RPC_URL, accounts: [process.env.PRIVATE_KEY] }
   ```
2. Ambil ETH testnet gratis dari faucet Sepolia.
3. `npx hardhat run scripts/deploy.ts --network sepolia`
4. Di halaman DApp, ganti RPC endpoint ke URL Sepolia — **halamannya tidak perlu diubah
   sama sekali**, karena sudah dirancang menerima RPC dan alamat kontrak dari input.
5. Verifikasi kontrak di Etherscan Sepolia → jadi bukti publik yang bisa dicantumkan
   di naskah tesis sebagai tautan.

Estimasi: setengah hari. Efeknya besar — kontrak Anda bisa dibuka siapa pun di
`sepolia.etherscan.io`, dan itu bukti paling kuat untuk kata "DApp".

---

## 8. Bukti pengujian (sudah dijalankan, bukan asumsi)

Node lokal dijalankan, kontrak di-deploy ke `0x5FbDB2315678afecb367f032d93F642f64180aa3`,
lalu tiga wallet dinilai lewat jalur ML asli dan ditulis on-chain:

| Wallet | Label asli | Skor | Level |
|---|---|---|---|
| `0x60afc576…4fc74b1` | 1 (phishing) | **100** | HIGH |
| `0x1006f68a…e52f3ea` | 1 (phishing) | 34 | LOW |
| `0x1fbce20f…7994f75b` | 0 (normal) | 0 | LOW |

Pembacaan mentah lewat JSON-RPC — meniru persis logika JavaScript di halaman — berhasil:
```
getRisk     -> score=100  ts=1785429156  model='xgb-eth-v1'
eth_getLogs -> 3 event RiskUpdated (block 2, 3, 4)
```

Selector, decoding string dinamis, dan rekonstruksi event log semuanya terverifikasi benar.

> 📌 Catatan jujur untuk bahan diskusi: wallet kedua berlabel phishing tetapi diberi skor
> 34 (LOW) — ini **false negative**. Konsisten dengan recall 0,9356, artinya sekitar 6%
> wallet phishing memang lolos. Jangan pakai wallet ini untuk demo; pakai yang pertama.
> Tapi kalau ditanya, jawab terbuka — model dengan recall 93,6% memang punya FN, dan
> Anda tahu persis berapa banyak.
