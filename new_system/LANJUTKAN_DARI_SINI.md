# Lanjutkan Dari Sini — Eksperimen Fitur v2

**Dihentikan:** 31 Juli 2026, 19:20 — kehabisan waktu sebelum bimbingan.
**Status:** menggantung di tengah, tetapi aman dan bisa dilanjutkan kapan saja.

---

## Berhenti di mana

Pengambilan ulang dataset dengan fitur generasi kedua berhenti di
**2.600 dari 2.700 alamat** (96%), nol kegagalan.

Checkpoint: `new_system/data/live_dataset_v2.partial.csv`

---

## Cara melanjutkan (sisa ~15 menit)

```powershell
cd C:\Kuliah\ethereum-phising-risk-detection
.\.venv\Scripts\Activate.ps1

# 1. Selesaikan 100 alamat terakhir — otomatis lanjut dari checkpoint
python new_system\ml\rebuild_dataset_v2.py

# 2. Bandingkan v1 vs v2 pada split identik + gerbang mutu OOD
python new_system\experiments\compare_v1_v2.py
```

Skrip pertama melewati alamat yang sudah diambil, jadi aman dijalankan berkali-kali.

---

## Hasil sementara (pada 2.400 baris)

| | F1 | Precision | Recall | ROC-AUC |
|---|---|---|---|---|
| v1 — 26 fitur | 0,8774 | 0,8793 | 0,8755 | 0,9514 |
| **v2 — 37 fitur** | **0,9036** | 0,9017 | 0,9056 | 0,9647 |
| selisih | **+0,0262** | | | |

Fitur baru penyumbang terbesar (gain):

| Fitur | Gain |
|---|---|
| `sweep_gap_median_s` | 8,2 |
| `sweep_ratio_1h` | 7,2 |
| `sweep_gap_min_s` | 5,5 |
| `token_out_ratio` | 4,3 |

**Gerbang mutu OOD: 7 dari 8 lulus.** Yang gagal: **Maker PSM USDC** (skor 49,
dituduh phishing).

---

## ⚠️ Yang harus diwaspadai saat melanjutkan

Fitur sapu bekerja **berlawanan arah dengan dugaan awal**:

| | Phishing | Normal |
|---|---|---|
| `sweep_gap_median_s` | 168.144 dtk (~2 hari) | **612 dtk (~10 menit)** |
| `sweep_ratio_1h` | 0,00 | **0,67** |

Yang menyapu cepat justru wallet **normal**, bukan phishing.

**Kenapa ini berisiko:** wallet "normal" pada dataset ini kebanyakan bot dan relayer
yang memang meneruskan dana dalam hitungan detik. Wallet manusia biasa tidak begitu —
manusia menyimpan dananya berhari-hari, sama seperti wallet phishing 2019.

Artinya model berpeluang mempelajari jalan pintas baru: **"lambat memindahkan dana
= phishing"**, yang akan salah menuduh wallet manusia biasa. Ini jebakan yang sama
persis dengan `transaction_count` sebelumnya, hanya berganti baju.

Bukti awalnya sudah muncul: Vitalik Buterin punya `sweep_gap_median_s` = 223.140 detik
— mirip median phishing (168.144), bukan median normal (612).

**Karena itu jangan mengadopsi v2 hanya karena F1-nya naik.** Syarat yang sudah
dipasang di `compare_v1_v2.py`: naik ≥ 0,005 **DAN** lulus gerbang OOD penuh.
Saat ini syarat kedua belum terpenuhi.

---

## Kalau gerbang tetap 7/8 setelah data lengkap

Tiga pilihan, dari yang paling jujur:

1. **Tolak v2.** Laporkan sebagai temuan: fitur sapu menaikkan F1 tetapi
   memperkenalkan bias baru terhadap wallet yang menahan dana. Ini temuan yang
   sah dan menarik secara ilmiah.
2. **Adopsi v2 + tambahkan Maker PSM ke daftar label.** Jujur, tetapi artinya
   lubangnya ditutup dengan daftar, bukan dengan model. Harus dinyatakan terbuka.
3. **Perbaiki akar masalahnya:** tambahkan wallet manusia biasa yang menahan dana
   (bukan bot) ke kelas negatif, lalu latih ulang. Paling benar, paling lama.

Pilihan 3 adalah yang sebenarnya menyelesaikan masalah — sama seperti penambahan
900 wallet kecil dulu menyelesaikan jebakan `transaction_count`.

---

## Yang TIDAK terpengaruh sama sekali

- Model tesis (F1 0,9102) — tidak tersentuh
- Model perluasan yang sekarang dipakai (ensemble, F1 0,9008) — masih aktif
- Lapisan label + `LabelRegistry.sol` — berfungsi penuh
- Semua artefak dan laporan lain

Eksperimen v2 ini murni tambahan yang menggantung; tidak ada yang rusak karena
dihentikan di tengah.

---

## Catatan jujur untuk diri sendiri nanti

Klaim awal saya — "wallet phishing menyapu dana dalam hitungan menit, selisihnya
450×" — **salah**, karena disimpulkan dari satu wallet contoh yang kebetulan paling
cepat menyapu. Setelah dilihat pada 1.165 wallet phishing, mediannya justru ~2 hari.

Pelajarannya: jangan menyimpulkan pola dari satu contoh. Periksa dulu sebarannya
pada seluruh populasi sebelum membangun fitur di atasnya.
