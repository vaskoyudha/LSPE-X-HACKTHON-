# LPSE-X Hackathon Proposal Final

## Identitas Dokumen

- Proyek: **LPSE-X**
- Kompetisi: **Find IT! 2026 Track C — The Explainable Oracle**
- Fokus: Deteksi risiko anomali pengadaan pemerintah berbasis XGBoost + SHAP

---

## Daftar Isi

1. Bab 1 — Pendahuluan
2. Bab 2 — Metodologi
3. Bab 3 — Kepatuhan dan Implementasi
4. Bab 4 — Hasil dan Pembahasan

---

# Bab 1 — Pendahuluan

# BAB 1: PENDAHULUAN

## 1.1 Latar Belakang

Pengadaan barang dan jasa pemerintah merupakan salah satu sektor yang paling rentan terhadap praktik korupsi di Indonesia. Menurut data Indonesia Corruption Watch (ICW), sektor pengadaan secara konsisten menjadi sektor dengan kasus korupsi tertinggi setiap tahunnya. Nilai kerugian negara akibat korupsi pengadaan mencapai triliunan rupiah, yang berdampak langsung pada kualitas layanan publik dan pembangunan infrastruktur.

Sistem Layanan Pengadaan Secara Elektronik (LPSE) yang dikelola oleh Lembaga Kebijakan Pengadaan Barang/Jasa Pemerintah (LKPP) telah menghasilkan volume data pengadaan yang sangat besar. Data ini dipublikasikan dalam format Open Contracting Data Standard (OCDS), yang mencakup informasi tender, penawaran, kontrak, dan pelaksanaan. Namun, volume data yang besar justru menyulitkan pengawasan manual oleh auditor dan aparat penegak hukum.

Pendekatan berbasis kecerdasan buatan (AI) menawarkan solusi untuk menganalisis pola-pola anomali dalam data pengadaan secara sistematis. Dengan memanfaatkan teknik machine learning, khususnya model berbasis gradient boosting, dimungkinkan untuk mengidentifikasi indikator risiko (red flag) yang sulit dideteksi secara manual. Lebih penting lagi, pendekatan Explainable AI (XAI) memungkinkan setiap prediksi risiko disertai penjelasan yang dapat dipahami oleh auditor non-teknis.

## 1.2 Rumusan Masalah

Berdasarkan latar belakang di atas, rumusan masalah dalam penelitian ini adalah:

1. Bagaimana membangun pipeline pemrosesan data pengadaan dari format OCDS menjadi fitur-fitur yang merepresentasikan pola risiko korupsi?
2. Bagaimana merancang sistem pelabelan heuristik yang transparan berdasarkan indikator risiko yang diakui secara internasional?
3. Bagaimana melatih model klasifikasi risiko yang akurat dengan tetap menjaga kepatuhan terhadap prinsip anti-kebocoran data (anti-leakage)?
4. Bagaimana menghasilkan penjelasan prediksi yang interpretatif dan actionable dalam Bahasa Indonesia untuk mendukung pengambilan keputusan auditor?

## 1.3 Tujuan

Penelitian ini bertujuan untuk:

1. Mengembangkan pipeline end-to-end untuk deteksi risiko anomali pengadaan pemerintah Indonesia menggunakan data OCDS.
2. Mengimplementasikan sistem pelabelan heuristik berbasis indikator Potential Fraud Analysis (PFA) dari ICW.
3. Melatih model XGBoost dengan hyperparameter optimization (HPO) yang ketat, termasuk validasi temporal dan pemisahan data yang memenuhi standar kompetisi.
4. Menyediakan penjelasan berbasis SHAP (SHapley Additive exPlanations) dengan narasi otomatis dalam Bahasa Indonesia, termasuk analisis kontrafaktual.
5. Mengemas seluruh pipeline dalam format yang dapat direproduksi secara offline pada lingkungan CPU.

## 1.4 Batasan Masalah

1. Data yang digunakan terbatas pada data pengadaan publik Indonesia yang tersedia dalam format OCDS melalui platform Opentender.net.
2. Label risiko bersifat heuristik berdasarkan indikator red flag, bukan hasil investigasi atau putusan hukum yang terkonfirmasi.
3. Model dioptimalkan untuk inferensi pada CPU tanpa ketergantungan pada GPU atau layanan cloud.
4. Sistem berjalan sepenuhnya offline tanpa memerlukan koneksi internet saat inferensi.
5. Penjelasan prediksi dihasilkan secara algoritmik, bukan oleh model bahasa generatif.

## 1.5 Manfaat

1. **Bagi auditor dan pengawas**: Menyediakan alat skrining awal yang mampu memprioritaskan proses pengadaan berisiko tinggi untuk investigasi lebih lanjut.
2. **Bagi transparansi publik**: Mendukung akuntabilitas pengadaan dengan menyediakan penjelasan yang dapat dipahami masyarakat umum.
3. **Bagi pengembangan ilmu pengetahuan**: Mendemonstrasikan penerapan XAI pada domain pengadaan publik dengan pendekatan yang reproducible dan terdokumentasi.

## 1.6 Sistematika Penulisan

- **Bab 1 — Pendahuluan**: Latar belakang, rumusan masalah, tujuan, batasan, dan manfaat penelitian.
- **Bab 2 — Metodologi**: Profil data, strategi pelabelan, rekayasa fitur, arsitektur model, dan desain evaluasi.
- **Bab 3 — Kepatuhan dan Implementasi**: Pemenuhan kriteria kompetisi, validasi anti-kebocoran, dan bukti implementasi.
- **Bab 4 — Hasil dan Pembahasan**: Metrik evaluasi, analisis SHAP, contoh narasi, dan diskusi keterbatasan.


---

# Bab 2 — Metodologi

# BAB 2: METODOLOGI

## 2.1 Gambaran Umum Pipeline

LPSE-X dibangun sebagai pipeline offline untuk mendeteksi risiko anomali pengadaan dari data OCDS. Alur kerja utama terdiri dari lima tahap: akuisisi dan perapihan data, pemisahan temporal untuk mencegah kebocoran data, rekayasa fitur split-aware, pelabelan heuristik berbasis red flag, serta pemodelan dan explainability berbasis XGBoost + SHAP.

Pipeline ini mengikuti constraint kompetisi Track C: seluruh inferensi berjalan lokal, pemisahan train/test dilakukan sebelum feature engineering, dan seluruh output penjelasan dapat dijalankan tanpa ketergantungan cloud API.

## 2.2 Sumber dan Kualitas Data

Sumber data kerja proyek ini disimpan dalam artefak kanonik `data/processed/ocds_flat.parquet`, dengan ringkasan kualitas pada `data/processed/quality_report.md` dan provenance ringkas pada `data/processed/data_provenance.json`. Dataset yang tersedia berisi 5.000 baris dengan 24 kolom utama, rentang waktu 2014-01-02 hingga 2023-12-30, serta 5.000 OCID unik. Provenance audit menunjukkan bahwa snapshot kerja saat ini bersifat **synthetic structured dataset** (`synthetic_ocid_ratio = 1.0`) dengan 50 buyer dan 200 supplier. Karena itu, dataset ini cocok untuk membuktikan pipeline Phase 2, tetapi belum cukup untuk mengklaim validitas operasional pada data LPSE riil.

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


## 2.10 Audit Circularity dan Robustness

Untuk mengukur seberapa besar performa model didorong oleh fitur yang sangat dekat dengan aturan pelabelan, dilakukan audit robustness pada tiga kelompok fitur yang diringkas pada `models/robustness.json` dan `proposal/figures/robustness_ablation.png`:

- **baseline_all_features (30 fitur)** → Macro-F1 0,9970
- **proxy_only (9 fitur proksi langsung)** → Macro-F1 0,9990
- **proxy_reduced (21 fitur non-proksi)** → Macro-F1 0,3911

Hasil ini menunjukkan bahwa sebagian besar kekuatan model saat ini berasal dari fitur yang sangat dekat dengan red-flag heuristic rules. Dengan kata lain, model Phase 2 sangat efektif sebagai **risk-rule recovery engine**, namun belum bisa diposisikan sebagai bukti kuat generalisasi terhadap fraud outcome yang independen dari aturan labeling.


---

# Bab 3 — Kepatuhan dan Implementasi

# BAB 3: KEPATUHAN DAN IMPLEMENTASI

## 3.1 Kepatuhan terhadap Constraint Kompetisi

Bab ini menyajikan pemetaan langsung antara implementasi LPSE-X dengan constraint wajib pada Track C.

### C-C1 — Explainability wajib

Constraint ini dipenuhi melalui penggunaan SHAP pada `src/explain.py`. Sistem menghasilkan faktor utama yang memengaruhi prediksi, baik untuk analisis satu baris maupun ringkasan global. Artefak bukti yang relevan adalah `proposal/figures/shap_summary.png` dan fungsi `explain_single(...)`.

### C-C2 — Output penjelasan yang dapat dibaca manusia

Constraint ini dipenuhi melalui dua lapisan output:

1. daftar faktor SHAP dengan arah pengaruh (`factors`),
2. narasi Bahasa Indonesia pada `src/narrative.py`.

Dengan demikian, auditor non-teknis dapat melihat bukan hanya skor risiko, tetapi juga alasan utama di balik prediksi.

### C-C3 — Anti-black-box

Model yang digunakan adalah XGBoost, bukan model opaque tanpa kontrol explainability. Selain itu, seluruh jalur inferensi utama dapat diaudit melalui:

- fitur input yang eksplisit,
- file parameter terbaik,
- metrik final,
- penjelasan SHAP,
- narasi terstruktur.

### C-C4 — Validasi data leakage

Constraint ini merupakan titik paling kritis dari desain sistem. Implementasi menggunakan kebijakan split-aware:

- `src/split.py` melakukan pemisahan raw train/test sebelum feature engineering,
- `test_data/` tidak dipakai untuk HPO,
- `test_data/` tidak dipakai untuk temperature scaling,
- fitur Tier 2 hanya memakai histori masa lalu.

Validasi ini diperkuat oleh test suite leakage guard dan hasil temporal split yang tidak overlap.

### C-C5 — Offline total

Seluruh komponen inti dapat berjalan lokal:

- pelatihan XGBoost,
- explainability SHAP,
- narasi Bahasa Indonesia,
- model `.ubj`,
- model `.onnx`,
- notebook training dan inference.

Tidak ada cloud inference API pada jalur training maupun inferensi.

## 3.2 Arsitektur Implementasi

Arsitektur kode dibagi ke beberapa modul yang memiliki kontrak jelas:

- `src/data.py` — akuisisi, flattening, cleaning, quality report
- `src/split.py` — split temporal eksternal dan internal
- `src/features.py` — 30 feature families split-aware
- `src/labels.py` — red-flag heuristic labeling + calibration helpers
- `src/model.py` — training, evaluation, calibration, ONNX/export helpers
- `src/explain.py` — SHAP, explain_single, counterfactual path
- `src/narrative.py` — render narasi Bahasa Indonesia

Struktur ini membantu pemisahan tanggung jawab serta memudahkan verifikasi oleh panel software engineering dan architect.

## 3.3 Status Gate Implementasi

### Gate 0 — Foundation

Sudah terpenuhi:

- scaffold proyek tersedia,
- marker pytest tersedia,
- import package berhasil,
- `pytest -m p0` dan `pytest -q` berjalan pada environment proyek.

### Gate 1 — Data freeze dan compliance split

Sudah terpenuhi secara implementasi internal:

- raw split train/test tersedia,
- metadata split tersedia,
- feature generation berjalan dari split raw,
- leakage guard test hijau.

### Gate 2 — Model baseline locked

Sudah terpenuhi:

- parameter terbaik tersimpan,
- metrik final tersimpan di `models/metrics.json`,
- kalibrasi tersimpan di `models/calibration.json`.

### Gate 3 — XAI complete

Sudah terpenuhi secara pipeline:

- explainability SHAP tersedia,
- figure SHAP tersedia,
- narasi Bahasa Indonesia tersedia,
- counterfactual fallback tersedia,
- jalur export `.onnx` dan `.ubj` tersedia untuk inferensi lokal.

### Gate 4 — Notebook complete

Dipenuhi melalui dua notebook terpisah:

- `training.ipynb`
- `inference.ipynb`

Notebook ini dirancang untuk mengeksekusi pipeline training dan inference secara lokal.

### Gate 5 — Submission ready

Komponen submission-ready difokuskan pada:

- proposal Bab 1–4,
- proposal final markdown,
- proposal final PDF,
- notebook training/inference,
- model hasil training,
- requirements dengan exact pins.

## 3.4 Pengendalian Risiko Teknis

Beberapa kill-switch dari rencana awal tetap dipertahankan:

1. Jika path focal loss tidak stabil, sistem tetap aman memakai class-weighted XGBoost.
2. Jika review calibration high-confidence kurang dari 80 sampel, temperature scaling dapat dimatikan.
3. Jika DiCE terlalu berat atau timebox terlampaui, counterfactual SHAP tetap tersedia.
4. Jika proposal dan implementasi berbeda, artefak implementasi menjadi sumber kebenaran.

## 3.5 Bukti Verifikasi

Verifikasi terkini pada branch kerja menunjukkan:

- `pytest -q` → 106 passed, 5 skipped
- `python3 -m compileall src tests scripts` → passed
- `git diff --check` → passed

Bukti ini menegaskan bahwa basis implementasi stabil untuk dibawa ke tahap finalisasi submission.


## 3.6 Posisi Ilmiah yang Jujur

Secara engineering, sistem sudah memenuhi kebutuhan Phase 2: pipeline lengkap, explainability tersedia, notebook dapat dieksekusi, dan inferensi berjalan offline. Namun, secara ilmiah ada dua pembatas utama yang harus dinyatakan secara eksplisit kepada juri:

1. dataset kerja saat ini masih sintetis,
2. label target masih berupa heuristic risk labels.

Karena itu, kontribusi utama LPSE-X pada tahap ini adalah **pembuktian arsitektur dan explainability pipeline**, bukan klaim final akurasi terhadap kasus korupsi riil. Robustness audit tambahan dipakai untuk memperkuat kejujuran posisi ini.


---

# Bab 4 — Hasil dan Pembahasan

# BAB 4: HASIL DAN PEMBAHASAN

## 4.1 Ringkasan Hasil Utama

Berdasarkan artefak evaluasi pada `models/metrics.json`, model LPSE-X mencapai performa yang sangat tinggi pada test split heuristik:

- Accuracy: **0,997**
- Macro-F1: **0,995**
- Weighted-F1: **0,997**
- Log loss: **0,0165**
- Jumlah sampel test: **997**

Performa ini menunjukkan bahwa pipeline mampu mempelajari pola red flag yang terdefinisi pada weak labels dengan sangat baik pada lingkungan data yang digunakan untuk Phase 2.

## 4.2 Analisis per Kelas

Nilai F1 per kelas adalah sebagai berikut:

- Low Risk: **0,9978**
- Medium Risk: **0,9978**
- High Risk: **0,9894**

Interpretasinya:

1. Kelas Low dan Medium ditangani sangat stabil.
2. Kelas High sedikit lebih menantang, tetapi recall kelas high tetap tinggi (0,9789).
3. Tidak ada indikasi penurunan performa yang drastis pada kelas paling penting dari sisi prioritisasi audit.

Figure pendukung:

- `proposal/figures/per_class_f1.png`
- `proposal/figures/confusion_matrix.png`

## 4.3 Confusion Matrix

Confusion matrix final menunjukkan mayoritas prediksi tepat pada diagonal utama:

- Low Risk: 225/225 terklasifikasi benar
- Medium Risk: 676/677 terklasifikasi benar
- High Risk: 93/95 terklasifikasi benar

Dua sumber kesalahan utama muncul pada batas Medium–High, yang memang wajar karena indikator risiko heuristik di kedua kelas dapat tumpang tindih secara parsial.

## 4.4 Kalibrasi Probabilitas

Model akhir menggunakan temperature scaling dengan parameter `T = 9,999901` berdasarkan 95 sampel calibration yang usable, 94 di antaranya high confidence. Nilai temperatur yang tinggi menandakan probabilitas awal model terlalu tajam dan perlu dilunakkan.

Dari sisi praktik audit, kalibrasi penting karena membantu skor probabilitas lebih jujur ketika dipakai untuk prioritisasi investigasi, bukan sekadar klasifikasi keras.

Figure pendukung:

- `proposal/figures/calibration_curve.png`

## 4.5 Explainability dan Faktor Risiko

Global explanation menggunakan SHAP menunjukkan bahwa beberapa fitur paling dominan untuk kelas berisiko tinggi mencakup:

- indikator single bidder,
- jumlah tenderers,
- rasio deviasi harga,
- nilai tender/award,
- pola histori buyer-supplier.

Hal ini konsisten dengan hipotesis domain pengadaan publik: risiko meningkat ketika kompetisi rendah, harga terlihat tidak normal, atau relasi buyer-supplier terlalu sering berulang.

Figure pendukung:

- `proposal/figures/shap_summary.png`

## 4.6 Kegunaan untuk Auditor

Nilai praktis utama sistem ini adalah explainable prioritization. Auditor dapat:

1. melihat skor risiko,
2. melihat faktor top yang mendorong prediksi,
3. membaca narasi Bahasa Indonesia,
4. melihat saran counterfactual untuk menurunkan risiko.

Dengan pendekatan ini, sistem tidak hanya memberi label, tetapi juga memberi konteks yang dapat ditindaklanjuti.

## 4.7 Audit Kelemahan Model

Audit tambahan pada `models/robustness.json` memberi insight yang sangat penting. Ketika model hanya diberi 9 fitur yang langsung beririsan dengan aturan labeling (proxy-only), Macro-F1 tetap **0,9990**. Namun ketika fitur-fitur proksi langsung tersebut dihapus (proxy-reduced), Macro-F1 turun tajam menjadi **0,3911**.

Artinya, model saat ini memang sangat kuat, tetapi kekuatan itu sebagian besar berasal dari kemampuannya memulihkan struktur aturan heuristik yang dipakai untuk membuat label. Dari perspektif Phase 2, ini tetap bernilai karena membuktikan explainable screening pipeline. Namun dari perspektif validitas ilmiah, hasil ini menegaskan bahwa model belum boleh diposisikan sebagai detektor fraud operasional yang independen dari rubric labeling.

## 4.8 Keterbatasan

Walaupun hasil metrik sangat tinggi, ada beberapa keterbatasan penting:

1. Label yang dipakai adalah **heuristik risiko**, bukan ground-truth fraud outcome.
2. Provenance audit menunjukkan dataset kerja saat ini **bersifat sintetis** (`synthetic_ocid_ratio = 1.0`).
3. Audit ablation menunjukkan adanya circularity risk yang kuat antara aturan labeling dan fitur prediktif utama.
4. Artefak data kerja merepresentasikan snapshot pipeline yang terstruktur; generalisasi ke data lapangan mentah tetap perlu validasi tambahan.
5. Counterfactual yang tersedia masih berbasis SHAP fallback, bukan sistem optimasi tindakan penuh.

## 4.9 Kesimpulan Bab

Secara keseluruhan, LPSE-X berhasil memenuhi tujuan Phase 2: membangun pipeline explainable AI berbasis XGBoost + SHAP yang bekerja offline, menjaga prinsip anti-leakage, menghasilkan metrik kuat pada test split, dan menyajikan penjelasan yang dapat dibaca auditor non-teknis.

Hasil ini menunjukkan bahwa pendekatan tabular gradient boosting dengan explainability deterministik adalah strategi yang efektif untuk skrining awal risiko pengadaan pada setting kompetisi Track C.
