# Materi Belajar Tesis — Deteksi Wallet Phishing Ethereum

Ditulis untuk persiapan bimbingan. Semua penjelasan dikaitkan langsung ke file kode
yang benar-benar ada di repo ini (bukan karangan / bukan salinan tesis).

| Tahap | File | Isi |
|---|---|---|
| 1 | [01_gambaran_besar_sistem.md](01_gambaran_besar_sistem.md) | Alur sistem dari data mentah sampai smart contract, pakai analogi |
| 2 | [02_pembentukan_dataset.md](02_pembentukan_dataset.md) | Beda dataset dari nol / publik / sampling / extraction / enrichment / preprocessing / EDA |
| 3 | [03_feature_extraction_18_fitur.md](03_feature_extraction_18_fitur.md) | Bedah `ml/preprocess.py` baris demi baris + arti tiap fitur |
| 4 | [04_penggabungan_data.md](04_penggabungan_data.md) | Left join MulDiGraph + Etherscan + BigQuery, zero-filling, `graph_present` |
| 5 | [05_model_dan_eksperimen.md](05_model_dan_eksperimen.md) | 4 model × 4 skenario, hasil angka asli, istilah komparasi vs pengembangan |
| 6 | [06_graphsage_dan_late_fusion.md](06_graphsage_dan_late_fusion.md) | GraphSAGE dari nol, late fusion 128+7=135, ablasi |
| 7 | [07_sistem_end_to_end.md](07_sistem_end_to_end.md) | Streamlit → FastAPI → model → web3.py → RiskRegistry |
| 8 | [08_masalah_judul_tesis.md](08_masalah_judul_tesis.md) | Analisis A/B/C/D + penilaian judul baru + alternatif |
| 9 | [09_latihan_presentasi.md](09_latihan_presentasi.md) | 15 pertanyaan dosen + kunci jawaban + jebakan |
| — | [10_temuan_penting.md](10_temuan_penting.md) | **Daftar ketidaksesuaian tesis vs kode. Baca ini sebelum bimbingan.** |
| — | [11_asal_data_etherscan.md](11_asal_data_etherscan.md) | Asal data Etherscan: API, 4 panggilan/alamat, 4 temuan enrichment |
| — | [12_dapp_lokal.md](12_dapp_lokal.md) | Lapisan DApp pembaca on-chain — pendukung judul baru |
| — | [13_uji_wallet_luar_dataset.md](13_uji_wallet_luar_dataset.md) | **Uji OOD: Vitalik dapat skor 92. Kenapa scan wallet bebas tidak diaktifkan.** |
| — | [14_model_live.md](14_model_live.md) | Model perluasan untuk wallet di luar dataset (hasil pelatihan ulang) |
| 🎯 | [15_panduan_demo.md](15_panduan_demo.md) | **Panduan demo: urutan klik, naskah bicara, rencana cadangan** |
| — | [17_bentuk_graph_muldigraph.md](17_bentuk_graph_muldigraph.md) | **Bentuk & isi MulDiGraph: statistik asli, ego-network 1-hop, bahan Bab 3** |
| ⭐ | [LAPORAN_UTAMA.md](LAPORAN_UTAMA.md) | **Ringkasan eksekutif — baca ini lebih dulu** |

## Angka kunci yang harus hafal

| Hal | Angka | Sumber |
|---|---|---|
| Wallet dalam dataset eksperimen | 8.595 | `ml/artifacts/hybrid_feature_schema.json` → `rows_final` |
| Phishing / Normal | 1.165 / 7.430 (13,6% phishing) | `data/processed/hybrid_dataset.csv` |
| Fitur graph | 18 (+1 kolom `graph_present` = 19) | `ml/preprocess.py:117-136` |
| Fitur Etherscan | 7 | `ml/build_hybrid_dataset.py:67-74` + `created_at_ts` |
| Fitur BigQuery | 10 | `ml/build_hybrid_dataset.py:76-87` |
| Model produksi | XGBoost, 26 fitur (19 graph + 7 Etherscan) | `ml/artifacts/prod_feature_schema.json` |
| F1 model produksi | 0,9102 (Acc 0,9750; Prec 0,8862; Rec 0,9356) | `ml/artifacts/dataset_scenarios_comparison.csv` |
| Late fusion V2 F1 | 0,8825 | `outputs/late_fusion_comparison_table_v2.csv` |
| GraphSAGE graph-only F1 | 0,8189 | idem |
| Embedding-only (ablasi) F1 | 0,8184 | `outputs/four_model_comparison_ablation.csv` |
| Edge asli antar 8.595 wallet | 264 edge berarah unik | `outputs/edge_source_audit_v2.json` |
| MulDiGraph: node / edge | 2.973.489 / 13.551.303 | `ml/artifacts/graph_profile/graph_stats.json` |
| MulDiGraph: phishing | 1.165 (0,039% dari graph) | idem |
| Degree median phishing vs normal | 13 vs 1 | idem |
| Edge bernilai 0 ETH | 9.152.968 (67,5%) | idem |
| Rentang waktu data | 2015-08-07 s.d. 2019-01-19 | idem |
