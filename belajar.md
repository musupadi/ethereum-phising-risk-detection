Saya sedang mempersiapkan bimbingan tesis dan perlu benar-benar memahami sistem yang saya bangun, bukan hanya menghafal isi dokumen.

Judul tesis saya saat ini:

“Pengembangan Model Machine Learning untuk Deteksi Wallet Phishing pada Jaringan Ethereum”

Namun, dosen saya mempertanyakan arti “pengembangan model” karena isi penelitian terlihat lebih banyak membandingkan algoritma dan skenario data. Kemungkinan arah yang lebih tepat adalah “pengembangan sistem”, karena saya membangun pipeline end-to-end dari dataset sampai pencatatan risk score ke smart contract.

Pelajari file tesis dan source code yang saya lampirkan. Ajari saya secara bertahap menggunakan bahasa Indonesia sederhana, seperti menjelaskan kepada programmer yang memahami coding tetapi belum terlalu kuat dalam konsep machine learning, graph, dan penelitian akademik.

Sistem penelitian saya secara umum mencakup:

1. Dataset utama EPTransNet dalam format MulDiGraph.
2. Node merepresentasikan wallet Ethereum.
3. Edge merepresentasikan transaksi antarwallet.
4. Feature extraction dari subgraph lokal atau 1-hop neighborhood.
5. Perhitungan 18 fitur agregat dari incoming dan outgoing transaction.
6. Pengayaan data dengan 7 fitur Etherscan dan 10 fitur BigQuery.
7. Penggabungan data berdasarkan wallet address.
8. Preprocessing, sampling, missing value, outlier, dan pembagian train-test.
9. Empat model:
   - Logistic Regression
   - Random Forest
   - XGBoost
   - GNN-GraphSAGE
10. Beberapa skenario kombinasi fitur.
11. Eksperimen GraphSAGE–Etherscan late fusion dan ablasi.
12. Pemilihan model terbaik.
13. FastAPI dan Streamlit sebagai sistem aplikasi.
14. Smart contract RiskRegistry untuk mencatat risk score secara on-chain.

Ajari saya dengan urutan berikut.

TAHAP 1 — GAMBARAN BESAR SISTEM

Jelaskan terlebih dahulu sistem saya dari awal sampai akhir menggunakan analogi sederhana.

Jelaskan:

- Data awalnya berbentuk apa.
- Apa arti node dan edge.
- Bagaimana data graph berubah menjadi data tabular.
- Apa yang masuk ke model.
- Apa yang dihasilkan model.
- Bagaimana hasil model dikirim ke smart contract.
- Bagian mana yang disebut machine learning.
- Bagian mana yang disebut pengembangan sistem.
- Bagian mana yang hanya komparasi algoritma.

Jangan langsung masuk ke rumus.

TAHAP 2 — PEMBENTUKAN DATASET EKSPERIMEN

Jelaskan perbedaan antara:

- membuat dataset dari nol;
- menggunakan dataset publik;
- sampling dataset;
- feature extraction;
- feature engineering;
- data enrichment;
- preprocessing;
- EDA.

Jelaskan posisi penelitian saya:

Saya tidak membuat EPTransNet dari nol, tetapi membentuk dataset eksperimen dari EPTransNet melalui sampling, feature extraction, dan pengayaan data.

Berikan contoh alur seperti:

EPTransNet MulDiGraph
→ pemilihan wallet
→ pengambilan incoming dan outgoing edge
→ perhitungan 18 fitur
→ DataFrame
→ join dengan Etherscan dan BigQuery
→ preprocessing
→ training model.

TAHAP 3 — FEATURE EXTRACTION 18 FITUR

Cari di kode saya bagian yang membaca MulDiGraph dan menghitung 18 fitur.

Jelaskan kode tersebut baris demi baris.

Untuk setiap fitur, jelaskan:

- nama fitur;
- diambil dari data apa;
- cara menghitungnya;
- contoh angka sederhana;
- alasan fitur tersebut dapat membantu mendeteksi wallet phishing.

Bedakan dengan jelas:

- 18 fitur hasil ekstraksi MulDiGraph;
- 7 fitur Etherscan;
- 10 fitur BigQuery;
- kolom indikator seperti graph_present;
- label phishing;
- wallet address.

Jelaskan mengapa label dan wallet address tidak dihitung sebagai fitur input.

TAHAP 4 — PENGGABUNGAN DATA

Jelaskan bagaimana data MulDiGraph, Etherscan, dan BigQuery digabungkan.

Gunakan contoh tiga tabel kecil dengan wallet address yang sama dan berbeda.

Jelaskan:

- apakah prosesnya menggunakan left join, inner join, atau pendekatan lain;
- dataset mana yang menjadi tabel utama;
- apa yang terjadi ketika wallet tidak ditemukan di Etherscan atau BigQuery;
- mengapa missing value dapat diisi nol;
- risiko dari zero-filling;
- fungsi kolom graph_present;
- cara mencegah jumlah wallet dan label berubah secara tidak sengaja.

TAHAP 5 — MODEL DAN EKSPERIMEN

Jelaskan mengapa penelitian memakai:

- Logistic Regression sebagai baseline sederhana;
- Random Forest sebagai baseline ensemble;
- XGBoost untuk data tabular;
- GraphSAGE untuk data graph.

Jelaskan arti eksperimen 4 model × 4 skenario.

Tegaskan bahwa menggunakan XGBoost atau GraphSAGE tidak otomatis berarti saya menciptakan algoritma baru.

Jelaskan kapan penelitian boleh disebut:

- komparasi model;
- optimasi model;
- pengembangan model;
- pengembangan sistem;
- pengembangan metode baru.

TAHAP 6 — GRAPHSAGE DAN LATE FUSION

Ajari saya GraphSAGE dari dasar menggunakan analogi wallet dan tetangganya.

Kemudian jelaskan:

- node feature;
- adjacency atau edge;
- message passing;
- neighborhood aggregation;
- embedding 128 dimensi;
- graph yang sparse;
- mengapa graph sparse membatasi GNN.

Setelah itu jelaskan late fusion saya:

GraphSAGE
→ menghasilkan embedding 128 dimensi
→ digabung dengan 7 fitur Etherscan
→ menjadi 135 dimensi
→ digunakan oleh XGBoost untuk klasifikasi.

Jelaskan perbedaan:

- graph-only GraphSAGE;
- embedding-only + XGBoost;
- GraphSAGE–Etherscan late fusion;
- XGBoost tabular biasa.

Jelaskan juga fungsi eksperimen ablasi dan mengapa ablasi diperlukan untuk membuktikan kontribusi fitur Etherscan.

TAHAP 7 — SISTEM END-TO-END

Jelaskan aliran aplikasi saya:

Pengguna memasukkan wallet di Streamlit
→ request dikirim ke FastAPI
→ preprocessing fitur
→ model menghasilkan probabilitas phishing
→ probabilitas dikonversi menjadi risk score
→ risk score diklasifikasikan
→ data dicatat ke RiskRegistry
→ transaction hash dan data risiko dikembalikan ke pengguna.

Jelaskan fungsi masing-masing komponen:

- Streamlit;
- FastAPI;
- model machine learning;
- web3.py;
- Hardhat;
- smart contract RiskRegistry;
- wallet address;
- probability;
- risk score;
- risk status;
- timestamp;
- transaction hash.

TAHAP 8 — MASALAH JUDUL TESIS

Analisis berdasarkan isi tesis dan kode saya:

Apakah penelitian ini lebih tepat disebut:

A. Pengembangan model Machine Learning
B. Komparasi model Machine Learning
C. Pengembangan sistem deteksi berbasis Machine Learning
D. Pengembangan metode hybrid graph–tabular

Jelaskan kelebihan dan risikonya satu per satu.

Fokuskan analisis pada kenyataan bahwa penelitian saya membandingkan model, memilih model terbaik, membangun API dan antarmuka, serta mengintegrasikan hasil prediksi dengan smart contract.

Nilai apakah judul berikut sudah sesuai:

“Pengembangan Sistem Deteksi Wallet Phishing Berbasis Machine Learning dengan Pencatatan Risk Score pada Smart Contract Ethereum”

Berikan alternatif judul lain, tetapi jangan mengubah arah penelitian secara drastis.

TAHAP 9 — PERSIAPAN PRESENTASI KE DOSEN

Latih saya menjawab pertanyaan berikut:

1. Pendekatan apa yang digunakan dalam membangun dataset?
2. Apakah Anda membuat dataset sendiri?
3. Bagaimana proses feature extraction dilakukan dalam program?
4. Mengapa menggunakan 1-hop neighborhood?
5. Mengapa fiturnya 18?
6. Mengapa ditambah Etherscan dan BigQuery?
7. Bagaimana ketiga sumber data digabungkan?
8. Mengapa disebut pengembangan?
9. Model apa yang sebenarnya dikembangkan?
10. Apa kebaruan penelitian ini?
11. Mengapa XGBoost menjadi model terbaik?
12. Mengapa GraphSAGE tidak mengalahkan XGBoost?
13. Apa fungsi late fusion?
14. Apa kontribusi smart contract?
15. Apa perbedaan model dan sistem?

Untuk setiap pertanyaan:

- minta saya menjawab terlebih dahulu;
- nilai jawaban saya;
- tunjukkan bagian yang salah atau kurang tepat;
- berikan versi jawaban yang lebih baik;
- lanjutkan ke pertanyaan berikutnya.

Jangan memberikan semua jawaban sekaligus saat sesi latihan.

ATURAN MENGAJAR

- Gunakan bahasa Indonesia sederhana.
- Jangan hanya menyalin tesis.
- Kaitkan penjelasan dengan kode yang benar-benar tersedia.
- Jangan mengarang proses yang tidak ada di kode.
- Bila isi tesis berbeda dengan implementasi kode, beri tahu saya secara tegas.
- Tunjukkan nama file, fungsi, class, atau bagian kode yang relevan.
- Bedakan fakta dari file dengan interpretasi Anda.
- Berikan contoh dengan maksimal 3–5 wallet agar mudah dipahami.
- Setelah setiap tahap, berikan 3 pertanyaan singkat untuk mengecek pemahaman saya.
- Jangan lanjut ke tahap berikutnya sebelum saya menjawab “lanjut”.
- Mulai dari TAHAP 1 saja.