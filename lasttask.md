Tolong perbaiki repository berikut agar sesuai dengan kebutuhan reproduksibilitas jurnal/tesis:

Repository:
https://github.com/musupadi/ethereum-phising-risk-detection

Konteks penelitian:
- Data EPTransNet berbentuk MultiDiGraph ditransformasikan menjadi 18 fitur tabular per wallet.
- Terdapat 7 atribut tambahan yang diperoleh melalui Etherscan API.
- Skenario produksi seharusnya menggunakan total 25 predictor: 18 graph-derived features + 7 Etherscan-derived wallet attributes.
- Saat ini kode memasukkan `graph_present` sebagai predictor sehingga menjadi 26 fitur.
- `graph_present` hanya boleh menjadi kolom audit kualitas join, bukan fitur Machine Learning.
- Jumlah populasi eksperimen adalah 8.595 wallet dan split menggunakan random_state=42.
- Repository harus menyediakan code sekaligus data turunan publik agar penelitian dapat direproduksi.
- Jangan mengunggah ulang file mentah EPTransNet/MulDiGraph.pkl karena berasal dari pihak ketiga dan ukurannya besar.

Kerjakan langsung pada repository dan jangan hanya memberikan saran.

Tugas utama:

1. PERBAIKI JUMLAH FITUR MENJADI 25

Periksa:
- `ml/build_hybrid_dataset.py`
- `ml/train_dataset_scenarios.py`
- `ml/train_production_model.py`
- `ml/artifacts/hybrid_feature_schema.json`
- `ml/artifacts/prod_feature_schema.json`
- file lain yang menggunakan daftar fitur.

Pastikan:
- 18 fitur graph:
  - in_degree
  - out_degree
  - unique_senders
  - unique_receivers
  - total_received
  - total_sent
  - avg_in_amount
  - avg_out_amount
  - max_in_amount
  - max_out_amount
  - min_in_amount
  - min_out_amount
  - std_in_amount
  - std_out_amount
  - in_out_ratio
  - sent_received_ratio
  - timestamp_spread_in
  - timestamp_spread_out

- 7 fitur hasil pengambilan Etherscan:
  - created_at_ts
  - balance
  - transaction_count
  - activity_score
  - avg_tx_value
  - tx_frequency
  - unique_counterparties

- `graph_present` tetap dibuat untuk audit coverage, tetapi dikeluarkan dari seluruh feature list dan input model.

Skenario harus menjadi:
- MultiDiGraph only: 18 predictor
- MultiDiGraph + Etherscan: 25 predictor
- MultiDiGraph + BigQuery: 28 predictor
- MultiDiGraph + Etherscan + BigQuery: 35 predictor

Tambahkan pada schema:
```json
"audit_columns": ["graph_present"]