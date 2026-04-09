# BAB 2: METODOLOGI

## 2.1 Gambaran Umum Pipeline

LPSE-X dibangun sebagai pipeline offline untuk mendeteksi risiko anomali pengadaan dari data OCDS. Alur kerja utama terdiri dari lima tahap: akuisisi dan perapihan data, pemisahan temporal untuk mencegah kebocoran data, rekayasa fitur split-aware, pelabelan heuristik berbasis red flag, serta pemodelan dan explainability berbasis XGBoost + SHAP.

Pipeline ini mengikuti constraint kompetisi Track C: seluruh inferensi berjalan lokal, pemisahan train/test dilakukan sebelum feature engineering, dan seluruh output penjelasan dapat dijalankan tanpa ketergantungan cloud API.

## 2.2 Sumber dan Kualitas Data

Sumber data kerja proyek ini berasal dari publikasi resmi Indonesia pada `https://data.open-contracting.org/en/publication/101`, dengan metadata lokal tersimpan di `data/processed/source_manifest.json`. Untuk menjaga repo tetap runnable selama Phase 2, benchmark saat ini memakai **slice data riil tahun 2021-2023** yang kemudian diflatten menjadi `data/processed/ocds_flat.parquet`.

Setelah pembersihan tanggal tidak valid, benchmark ini berisi:

- 465.184 baris usable
- 618 buyer unik
- 60.976 supplier unik
- train split: 372.150 baris
- test split: 93.034 baris

Ringkasan kualitas berada di `data/processed/quality_report.md`, sedangkan provenance ringkas berada di `data/processed/data_provenance.json`.

Temuan penting dari quality report dan inspeksi lapangan:

- `award_value_amount` tersedia luas dan dapat dipakai untuk evaluasi nilai award
- `tender.value.amount` sering kosong pada sumber asli, sehingga pipeline memakai fallback `tender.minValue.amount`
- `tender.description` sering kosong pada sumber asli, sehingga pipeline memakai fallback judul tender untuk menjaga sinyal teks minimum
- `tender_numberOfTenderers` dan `contracts` sangat jarang tersedia, sehingga sebagian fitur kompetisi/kontrak tetap lemah atau kosong
- `tender_procurementMethod` kosong pada benchmark ini, sehingga flag direct procurement tidak memberikan sinyal pada slice saat ini

Dengan demikian, benchmark riil multi-tahun ini jauh lebih kredibel daripada benchmark sintetis sebelumnya dan juga lebih stabil daripada benchmark riil satu tahun.

## 2.3 Strategi Split Data dan Anti-Leakage

Sesuai hard rule kompetisi, pemisahan train/test dilakukan pada level **raw split** sebelum feature engineering. Implementasi berada di `src/split.py` dan menghasilkan:

- `train_data/raw.parquet`
- `test_data/raw.parquet`
- `data/processed/split_metadata.json`

Hasil split final pada benchmark riil saat ini:

- Train: 372.150 baris (2015-07-09 s.d. 2023-03-10 07:27:51)
- Test: 93.034 baris (2023-03-10 07:38:45 s.d. 2023-12-20 23:00:00)

Di dalam train split, data dipecah lagi menjadi tiga dev split temporal:

- `train_fit`
- `val_hpo`
- `val_calibration`

Dengan aturan ini, `test_data/` tidak pernah dipakai untuk HPO, kalibrasi, maupun threshold tuning. Semua fitur temporal pada Tier 2 dibangun dengan expanding-window berbasis histori masa lalu saja.

## 2.4 Rekayasa Fitur

Sistem menggunakan **30 feature families** yang dibagi menjadi dua kelompok:

### Tier 1: Fitur langsung dari field pengadaan

Contoh fitur Tier 1:

- log nilai tender (dengan fallback dari `minValue.amount`)
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
- sinyal nilai tinggi dan timing akhir tahun

Distribusi label pada train split (`train_data/labels.parquet`) setelah migrasi ke benchmark riil multi-tahun:

- Low: 154.848
- Medium: 213.640
- High: 3.662

Pelabelan ini tetap bersifat **indikator risiko**, bukan pembuktian fraud. Pada benchmark riil multi-tahun, distribusi kelas menjadi lebih stabil daripada benchmark satu tahun, tetapi label tetap tidak sama dengan fraud ground truth.

## 2.6 Pemodelan

Model inti yang dipakai adalah **XGBoost multi-class** dengan objective `multi:softprob`. XGBoost dipilih karena:

1. kuat untuk data tabular,
2. efisien di CPU,
3. kompatibel dengan SHAP,
4. dapat diekspor ke format yang mendukung inferensi offline.

Pada benchmark riil 2021-2023, pipeline retraining memakai parameter terbaik yang tersimpan di `models/best_params.json`.

## 2.7 Kalibrasi dan Clean Labels

Kalibrasi probabilitas dilakukan dengan temperature scaling menggunakan subset `val_calibration` yang telah melalui clean-label review. Protokol review disimpan pada `data/processed/clean_labels_protocol.md`, sementara hasilnya tersedia di `data/processed/clean_labels_100.csv`.

Konfigurasi kalibrasi akhir (`models/calibration.json`) pada benchmark riil multi-tahun mengikuti artefak yang tersimpan di repo dan tetap dipakai sebagai pelunak probabilitas untuk inferensi.

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
- `models/benchmark_comparison.json`
- `proposal/figures/*.png`
- `training.ipynb`
- `inference.ipynb`

Dengan struktur tersebut, seluruh pipeline dapat dijalankan ulang pada lingkungan CPU lokal dengan dependency yang dipin pada `requirements.txt`.

## 2.10 Audit Circularity dan Robustness

Untuk mengukur seberapa besar performa model didorong oleh fitur yang sangat dekat dengan aturan pelabelan, dilakukan audit robustness pada tiga kelompok fitur yang diringkas pada `models/robustness.json` dan `proposal/figures/robustness_ablation.png`:

- **baseline_all_features (30 fitur)** → Macro-F1 0,9432
- **proxy_core_removed (21 fitur)** → Macro-F1 0,3854
- **proxy_broad_removed (18 fitur)** → Macro-F1 0,3781

Hasil ini menunjukkan bahwa ketergantungan pada fitur proksi langsung tetap kuat, bahkan setelah benchmark diperluas menjadi multi-tahun. Jadi, migrasi ke data riil multi-tahun memperbaiki kredibilitas eksternal, tetapi tidak menghapus circularity risk.

## 2.11 Perbandingan Benchmark Sintetis vs Riil

Artefak `models/benchmark_comparison.json` membandingkan benchmark sintetis sebelumnya dengan benchmark riil multi-tahun saat ini.

Ringkasan utama:

- Macro-F1 sintetis: 0,9950
- Macro-F1 riil 2021-2023: 0,9349
- Delta: -0,0601

Kesimpulan metodologisnya jelas: benchmark sintetis tetap terlalu optimistis, tetapi benchmark riil multi-tahun menunjukkan bahwa pipeline mentransfer lebih baik daripada benchmark riil satu tahun yang sebelumnya dipakai. Ini memperkuat validitas Phase 2 tanpa kembali ke klaim yang berlebihan.
