# Panduan Demo — Urutan Klik & Naskah Bicara

Dibuat untuk dipakai langsung saat menghadap dosen. Baca sekali sebelum mulai,
lalu ikuti urutannya.

---

## A. Persiapan 15 menit sebelum mulai

### 1. Nyalakan sistemnya (4 terminal)

```powershell
.\.venv\Scripts\Activate.ps1

# Terminal 1 — blockchain
cd chain ; npx hardhat node

# Terminal 2 — deploy kontrak (WAJIB diulang tiap node dinyalakan ulang)
cd chain ; npx hardhat run scripts/deploy.ts --network localhost

# Terminal 3 — API
$env:CHAIN_ENABLED = "true"
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000

# Terminal 4 — UI
streamlit run app/streamlit_app.py --server.port 8502

# Terminal 5 (opsional, kalau mau menunjukkan sisi DApp)
.\chain\dapp\run_dapp.ps1
```

### 2. Cek semuanya hidup

```powershell
curl http://localhost:8000/health        # chain_connected harus true
```

### 3. Panaskan dulu (PENTING)

Scan satu wallet lewat Streamlit **sebelum** dosen datang, supaya:
- model sudah ter-load (scan pertama selalu lebih lambat),
- sudah ada catatan on-chain untuk ditunjukkan di bagian riwayat.

Pakai: `0x60afc5761212ed59608b0559b4ccbaad24fc74b1` → harusnya skor **100 HIGH**.

---

## B. Wallet yang aman untuk didemokan

### ✅ Pakai ini

| Wallet | Hasil | Untuk menunjukkan |
|---|---|---|
| `0x60afc5761212ed59608b0559b4ccbaad24fc74b1` | 100 HIGH, label asli phishing | Deteksi benar + penjelasan SHAP |
| `0x4dad23689bef624af5d1757a56135901c915fcfd` | ≈100 HIGH | Ada di dropdown contoh |
| `0x2a7cf914a7825bb067703042eff254d54963234c` | ≈25 LOW, label asli normal | Sisi negatif juga benar |

### ⚠️ Jangan dipakai untuk demo

| Wallet | Kenapa |
|---|---|
| `0x1006f68abfe6ebd96d7a4f74f811713a6e52f3ea` | Label phishing tapi skor 34 (LOW) — false negative |

### 🆕 Wallet di luar dataset — sekarang AMAN didemokan

Sejak Model B dibangun, penguji boleh mengetik alamat apa pun. Sistem beralih ke model
perluasan, menampilkan peringatan kuning "mode estimasi", dan **tidak** mencatat skornya
ke blockchain.

Alamat yang bagus untuk memamerkan ini:

| Alamat | Hasil | Kesan yang ditimbulkan |
|---|---|---|
| `0xd8dA6BF26964aF9D7eEd9e03E53415D37aA96045` | **1 LOW** — Bukan Phishing | Vitalik Buterin dinilai benar |
| `0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D` | **1 LOW** — Bukan Phishing | Uniswap Router dinilai benar |

> **Naskahnya:** "Ini alamat Vitalik Buterin. Perhatikan peringatan kuningnya — sistem
> menyadari alamat ini di luar dataset penelitian, jadi ia memakai model perluasan dan
> menandai hasilnya sebagai estimasi. Model tesis saya dulu menilai alamat ini 98 HIGH;
> setelah saya perbaiki dengan *hard negative sampling*, skornya menjadi 1."

Ini justru **poin terkuat** yang bisa kamu tunjukkan: kamu menemukan kelemahan sistemmu
sendiri, memahami sebabnya sampai level fitur, lalu memperbaikinya.

---

## C. Urutan demo (8 menit)

### Menit 1–2 — Gambaran besar
Buka diagram di `laporan belajar/01_gambaran_besar_sistem.md`, katakan:

> "Sistem ini tiga lapis. Streamlit sebagai antarmuka, FastAPI sebagai otak yang
> menjalankan model XGBoost, dan smart contract RiskRegistry sebagai buku besar
> yang mencatat skor beserta waktu dan versi modelnya. Datanya berasal dari
> EPTransNet — graph transaksi Ethereum berlabel phishing — yang saya perkaya
> dengan data on-chain nyata dari Etherscan dan BigQuery."

### Menit 3–4 — Scan wallet dataset
Tempel `0x60afc576…`, klik Scan. Tunjukkan:
- Vonis **Phishing**, skor **100**, level **HIGH**
- Tabel **"Mengapa skor ini?"** — kontribusi SHAP
- Baris label asli dataset: **cocok ✅**

> "Modelnya bukan kotak hitam. Untuk tiap penilaian, sistem menunjukkan lima fitur
> yang paling menentukan skornya memakai nilai SHAP."

### Menit 5 — Bukti on-chain
Tunjukkan bagian "Tersimpan di smart contract": tx hash, skor, waktu, versi model.

> "Yang tercatat permanen hanya tiga hal: skor, waktu penilaian, dan versi model.
> Fitur dan data pribadi wallet tidak ikut disimpan. Ini memberi jejak audit —
> bisa dibuktikan bahwa wallet ini dinilai 100 pada waktu tertentu oleh model
> versi xgb-eth-v1, dan catatan itu tidak bisa diubah belakangan."

### Menit 6 — Sisi DApp (kalau ditanya soal judul)
Buka `http://localhost:8503/dapp/` → Hubungkan → tempel wallet → Baca on-chain.
Lalu **matikan FastAPI** (Ctrl+C di terminal 3) dan ulangi pembacaannya.

> "Halaman ini membaca langsung dari smart contract tanpa melewati backend saya.
> Buktinya, API-nya saya matikan dan pembacaan tetap berjalan. Inilah lapisan
> yang benar-benar terdesentralisasi."

### Menit 7–8 — Hasil penelitian
Buka tabel 4×4 di `laporan belajar/05_model_dan_eksperimen.md`.

> "Saya membandingkan empat model pada empat skenario sumber data, semuanya di atas
> pembagian data yang identik. XGBoost menang di keempat skenario, terbaik pada
> skenario MulDiGraph + Etherscan dengan F1 91,02%. Temuan menariknya, penambahan
> BigQuery justru tidak membantu karena fiturnya redundan dengan fitur graph."

---

## D. Kalau ditanya hal sulit — jawaban 30 detik

| Pertanyaan | Jawab |
|---|---|
| "Kalau saya masukkan wallet saya sendiri?" | "Silakan. Sistem akan beralih ke model perluasan dan menandainya sebagai estimasi. Awalnya model saya salah menuduh alamat sah — Vitalik Buterin dapat skor 98. Saya telusuri lewat SHAP, ternyata model belajar 'banyak pengirim unik = phishing' karena kelas negatifnya didominasi wallet pasif. Saya latih ulang dengan hard negatives, dan sekarang Vitalik dapat skor 1, dengan false positive 0 dari 20 wallet acak." |
| "Kok skor model perluasannya cuma F1 0,87, lebih rendah dari 0,91?" | "Karena tugasnya berbeda dan lebih sulit. Model tesis membedakan phishing dari wallet kecil pasif; model perluasan harus membedakannya dari wallet normal beragam ukuran, termasuk bursa dan router. Keduanya dievaluasi pada populasi berbeda, jadi angkanya tidak sebanding." |
| "Kenapa GraphSAGE kalah?" | "Karena graph-nya terlalu jarang. Dari 8.595 wallet hasil sampling, hanya ada 264 edge di antara mereka. Untuk lebih dari 99% node, GraphSAGE tidak punya tetangga untuk diagregasi sehingga efektif berperilaku seperti MLP biasa." |
| "Apa kebaruannya?" | "Integrasi machine learning dengan pencatatan on-chain yang dapat diaudit, dataset eksperimen multi-sumber, dan temuan empiris bahwa pengayaan Etherscan menaikkan F1 sementara BigQuery tidak." |
| "Model apa yang Anda kembangkan?" | "Saya tidak menciptakan algoritma baru. Saya memakai XGBoost dan GraphSAGE dari library standar. Yang saya kembangkan adalah sistemnya, dan modelnya saya pilih lewat perbandingan yang sistematis." |
| "Kenapa akurasinya 97% tapi presisi 88%?" | "Karena datanya timpang 1:6,4, jadi akurasi menyesatkan. Metrik utama saya F1. Presisi lebih rendah dari recall karena itu memang trade-off yang saya pilih — lebih baik salah menuduh 28 wallet daripada melewatkan penipu." |

Jawaban lengkap 15 pertanyaan ada di `09_latihan_presentasi.md`.

---

## E. Kalau sistem bermasalah saat demo

| Gejala | Penyebab paling mungkin | Tindakan cepat |
|---|---|---|
| `tx_hash: null` terus | Node Hardhat mati / belum deploy | Deploy ulang: `npx hardhat run scripts/deploy.ts --network localhost` |
| Halaman DApp bilang "belum pernah dinilai" | Node di-restart → data hilang | Deploy ulang + scan ulang 1 wallet |
| Scan lama sekali (>20 detik) | Wallet di luar dataset → mode live memanggil Etherscan | Wajar. Jelaskan sambil menunggu |
| Streamlit error koneksi | FastAPI belum jalan | Nyalakan terminal 3 |
| Semua kacau | — | `$env:CHAIN_ENABLED = "false"`, jalankan API + Streamlit saja. Inferensi ML tetap jalan penuh |

**Rencana cadangan:** kalau blockchain bermasalah dan waktu mepet, demokan Mode A saja
(tanpa chain). Bagian machine learning — yang merupakan inti tesis — tetap tampil utuh.

---

## F. Checklist terakhir sebelum masuk ruangan

- [ ] Keempat terminal jalan, `/health` menunjukkan `chain_connected: true`
- [ ] Sudah scan 1 wallet sebagai pemanasan
- [ ] Tab browser sudah terbuka: Streamlit, `:8503/dapp/`, dan `laporan belajar/`
- [ ] Hafal 6 angka: **8.595 / 1.165 / 26 fitur / F1 0,9102 / 264 edge / +0,064**
- [ ] Siap mengakui 3 hal duluan: kebocoran early stopping GNN, dataset 8.595 dari 12.815,
      dan keterbatasan out-of-distribution
- [ ] Laptop dicolok ke listrik 🔌
