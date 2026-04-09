# BAB 2: METODOLOGI

## 2.1 Gambaran Umum Pipeline

LPSE-X dibangun sebagai pipeline offline untuk mendeteksi risiko anomali pengadaan dari data OCDS. Alur kerja utama terdiri dari lima tahap: akuisisi dan perapihan data, pemisahan temporal untuk mencegah kebocoran data, rekayasa fitur split-aware, pelabelan heuristik berbasis red flag, serta pemodelan dan explainability berbasis XGBoost + SHAP.

Pipeline ini mengikuti constraint kompetisi Track C: seluruh inferensi berjalan lokal, pemisahan train/test dilakukan sebelum feature engineering, dan seluruh output penjelasan dapat dijalankan tanpa ketergantungan cloud API.

## 2.2 Sumber dan Kualitas Data

Sumber data kerja proyek ini disimpan dalam artefak kanonik `data/processed/ocds_flat.parquet`, dengan ringkasan kualitas pada `data/processed/quality_report.md`. Dataset yang tersedia berisi 5.000 baris dengan 24 kolom utama, rentang waktu 2014-01-02 hingga 2023-12-30, serta 5.000 OCID unik. Cakupan untuk kolom inti berada di atas 92%, sehingga cukup kuat untuk eksperimen hackathon.

Beberapa temuan penting dari quality report:

- `tender_numberOfTenderers` tersedia pada 96,9% baris, sehingga fitur terkait kompetisi tender dapat diaktifkan.
- `award_value_amount` tersedia pada 96,9% baris.
- `contract_value_amount` tersedia pada 92,1% baris.
- Tidak ada field inti dengan missing value di atas 30%.

## 2.3 Strategi Split Data dan Anti-Leakage

Sesuai hard rule kompetisi, pemisahan train/test dilakukan pada level **raw split** sebelum feature engineering. Implementasi berada di `src/split.py` dan menghasilkan:

- `train_data/raw.parquet`
- `test_data/raw.parquet`
- `data/processed/split_metadata.json`

Hasil split final:

- Train: 4.003 baris (2014-01-02 s.d. 2021-12-27)
- Test: 997 baris (2021-12-28 s.d. 2023-12-30)

Di dalam train split, data dipecah lagi menjadi tiga dev split temporal:

- `train_fit`
- `val_hpo`
- `val_calibration`

Dengan aturan ini, `test_data/` tidak pernah dipakai untuk HPO, kalibrasi, maupun threshold tuning. Semua fitur temporal pada Tier 2 dibangun dengan expanding-window berbasis histori masa lalu saja.

## 2.4 Rekayasa Fitur

Sistem menggunakan **30 feature families** yang dibagi menjadi dua kelompok:

### Tier 1: Fitur langsung dari field pengadaan

Contoh fitur Tier 1:

- log nilai tender
- log nilai award
- rasio deviasi harga
- durasi tender
- jumlah peserta tender
- indikator single bidder
- panjang judul dan deskripsi
- encoding metode pengadaan
- indikator Q4 dan Desember
- rasio kontrak terhadap award

### Tier 2: Fitur historis dan agregat temporal

Contoh fitur Tier 2:

- rata-rata historis nilai buyer
- deviasi z-score nilai buyer
- jumlah kemenangan historis supplier
- frekuensi buyer-supplier berulang
- jumlah tender historis buyer
- jumlah unique buyer per supplier
- laju pertumbuhan nilai buyer
- kapasitas supplier terhadap histori award

Semua fitur diserialisasi ke:

- `train_data/features.parquet`
- `test_data/features.parquet`
- `data/processed/feature_manifest.json`

## 2.5 Pelabelan Heuristik Risiko

Karena tidak tersedia label fraud terverifikasi pada skala kompetisi, proyek ini menggunakan weak-labeling berbasis red flag untuk mengklasifikasikan risiko menjadi tiga kelas:

- 0 = Rendah
- 1 = Sedang
- 2 = Tinggi

Indikator utama yang dipakai meliputi:

- peserta tunggal
- jendela tender pendek
- deviasi harga terhadap nilai referensi
- supplier menang berulang pada buyer yang sama
- jumlah bidder rendah

Distribusi label pada train split (`train_data/labels.parquet`):

- Low: 954
- Medium: 2.658
- High: 391

Pelabelan ini bersifat **indikator risiko**, bukan pembuktian fraud. Keterbatasan ini dijelaskan eksplisit pada protokol clean-label dan pada pembahasan hasil.

## 2.6 Pemodelan

Model inti yang dipakai adalah **XGBoost multi-class** dengan objective `multi:softprob`. XGBoost dipilih karena:

1. kuat untuk data tabular,
2. efisien di CPU,
3. kompatibel dengan SHAP,
4. dapat diekspor ke format yang mendukung inferensi offline.

Hyperparameter terbaik yang tersimpan di `models/best_params.json` meliputi:

- `max_depth = 3`
- `learning_rate = 0.1865`
- `subsample = 0.8693`
- `colsample_bytree = 0.6505`
- `min_child_weight = 8`
- `gamma = 0.0259`
- `reg_alpha = 0.0003047`
- `reg_lambda = 0.0000580`
- `n_rounds = 449`

## 2.7 Kalibrasi dan Clean Labels

Kalibrasi probabilitas dilakukan dengan temperature scaling menggunakan subset `val_calibration` yang telah melalui clean-label review. Protokol review disimpan pada `data/processed/clean_labels_protocol.md`, sementara hasilnya tersedia di `data/processed/clean_labels_100.csv`.

Konfigurasi kalibrasi akhir (`models/calibration.json`):

- enabled: true
- temperature: 9.999901
- n_calibration_samples: 95
- n_high_confidence: 94

## 2.8 Explainable AI

Komponen explainability berada di `src/explain.py` dan `src/narrative.py`.

Pipeline penjelasan terdiri dari:

1. prediksi probabilitas multi-kelas,
2. ekstraksi faktor SHAP teratas,
3. narasi Bahasa Indonesia yang dapat dibaca auditor,
4. saran counterfactual berbasis SHAP.

Output `explain_single(...)` menjaga kontrak yang konsisten untuk kebutuhan proposal, notebook, dan jalur inferensi.

## 2.9 Artefak dan Reproduksibilitas

Artefak utama yang digunakan oleh metodologi ini adalah:

- `train_data/*.parquet`
- `test_data/*.parquet`
- `models/metrics.json`
- `models/calibration.json`
- `models/imputation_values.json`
- `proposal/figures/*.png`
- `training.ipynb`
- `inference.ipynb`

Dengan struktur tersebut, seluruh pipeline dapat dijalankan ulang pada lingkungan CPU lokal dengan dependency yang dipin pada `requirements.txt`.
