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

Kalibrasi probabilitas dilakukan dengan temperature scaling menggunakan subset `val_calibration` yang telah melalui clean-label review. Protokol review disimpan pada `data/processed/clean_labels_protocol.md`, sementara iterasi review yang lebih besar saat ini tersedia di `data/processed/clean_labels_300.csv`.

Konfigurasi kalibrasi akhir (`models/calibration.json`) pada benchmark riil multi-tahun mengikuti artefak yang tersimpan di repo dan tetap dipakai sebagai pelunak probabilitas untuk inferensi. Iterasi terbaru menggunakan **287 reviewed rows** yang valid untuk temperature scaling.

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

- **baseline_all_features (30 fitur)** → Macro-F1 0,9833
- **proxy_core_removed (19 fitur)** → Macro-F1 0,5215
- **proxy_broad_removed (13 fitur)** → Macro-F1 0,5204

Hasil ini menunjukkan bahwa ketergantungan pada fitur proksi langsung tetap kuat, bahkan setelah benchmark diperluas menjadi multi-tahun. Jadi, migrasi ke data riil multi-tahun memperbaiki kredibilitas eksternal, tetapi tidak menghapus circularity risk.

## 2.11 Perbandingan Benchmark Sintetis vs Riil

Artefak `models/benchmark_comparison.json` membandingkan benchmark sintetis sebelumnya dengan benchmark riil multi-tahun saat ini.

Ringkasan utama:

- Macro-F1 sintetis: 0,9950
- Macro-F1 riil 2021-2023: 0,9833
- Delta: -0,0117

Kesimpulan metodologisnya jelas: benchmark sintetis tetap terlalu optimistis, tetapi benchmark riil multi-tahun menunjukkan bahwa pipeline mentransfer lebih baik daripada benchmark riil satu tahun yang sebelumnya dipakai. Ini memperkuat validitas Phase 2 tanpa kembali ke klaim yang berlebihan.

## 2.12 Jalur Import Row-Level Reviewed Labels

Repo kini menyediakan jalur eksplisit untuk mengimpor reviewed labels tingkat baris melalui:

- `scripts/import_reviewed_row_level.py`
- path standar hasil impor: `data/processed/review_benchmark_500_reviewed.csv`

Setelah file row-level tersebut tersedia, `scripts/run_diagnostics.py` akan memprioritaskan bukti row-level di atas summary import. Dengan demikian, transisi dari summary-level evidence ke reviewed benchmark yang lebih kuat dapat dilakukan tanpa mengubah arsitektur pipeline.

## 2.13 Track Validasi Proxy-Reduced

Untuk menegaskan sisi ilmiah evaluasi, repo sekarang juga menyimpan satu track validasi yang lebih ketat:

- `models/proxy_reduced_validation.json`
- `proposal/figures/proxy_reduced_validation.png`

Track ini menggunakan hasil `proxy_core_removed`, yaitu evaluasi setelah fitur-fitur yang paling dekat dengan aturan labeling dihapus. Hasil saat ini:

- Macro-F1 full model: **0,9833**
- Macro-F1 proxy-reduced: **0,5215**
- Delta: **-0,4618**

Maknanya jelas: model operasional sangat kuat, tetapi sebagian besar kekuatan tersebut masih datang dari sinyal yang dekat dengan heuristic rules.


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
- `src/diagnostics.py` — provenance + circularity audit

Struktur ini membantu pemisahan tanggung jawab serta memudahkan verifikasi oleh panel software engineering dan architect.

## 3.3 Status Gate Implementasi

### Gate 0 — Foundation

Sudah terpenuhi:

- scaffold proyek tersedia,
- marker pytest tersedia,
- import package berhasil,
- `pytest -m p0` dan `pytest -q` berjalan pada environment proyek.

### Gate 1 — Data freeze dan compliance split

Sudah terpenuhi pada benchmark riil multi-tahun:

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

Keduanya sudah dieksekusi ulang setelah perluasan benchmark riil multi-tahun.

### Gate 5 — Submission ready

Komponen submission-ready saat ini mencakup:

- proposal Bab 1–4,
- proposal final markdown,
- proposal final PDF,
- notebook training/inference,
- model hasil training,
- requirements dengan exact pins,
- provenance dan benchmark comparison artifacts.

## 3.4 Pengendalian Risiko Teknis

Beberapa kill-switch dari rencana awal tetap dipertahankan:

1. Jika path focal loss tidak stabil, sistem tetap aman memakai class-weighted XGBoost.
2. Jika review calibration high-confidence kurang dari 80 sampel, temperature scaling dapat dimatikan.
3. Jika DiCE terlalu berat atau timebox terlampaui, counterfactual SHAP tetap tersedia.
4. Jika proposal dan implementasi berbeda, artefak implementasi menjadi sumber kebenaran.

## 3.5 Bukti Verifikasi

Verifikasi terkini pada branch kerja menunjukkan:

- `pytest -q` → suite hijau setelah perluasan benchmark riil
- `python3 -m compileall src tests scripts` → passed
- `git diff --check` → passed
- notebook training dan inference dapat dieksekusi dengan `nbconvert`

## 3.6 Posisi Ilmiah yang Jujur

Perluasan ke data riil multi-tahun memperbaiki kredibilitas submission lebih jauh dibanding benchmark riil satu tahun. Namun, ada tiga pembatas utama yang tetap harus dinyatakan secara eksplisit kepada juri:

1. benchmark riil saat ini masih berupa **slice 2021-2023**, bukan seluruh histori LPSE/OCDS,
2. label target masih berupa **heuristic risk labels**,
3. audit ablation menunjukkan circularity risk yang tetap kuat.

Karena itu, kontribusi utama LPSE-X pada tahap ini adalah **pembuktian arsitektur dan explainability pipeline pada data riil multi-tahun yang tidak sempurna**, bukan klaim final akurasi terhadap kasus korupsi terverifikasi.


---

# Bab 4 — Hasil dan Pembahasan

# BAB 4: HASIL DAN PEMBAHASAN

## 4.1 Ringkasan Hasil Utama

Berdasarkan artefak evaluasi pada `models/metrics.json`, model LPSE-X pada benchmark riil 2021-2023 mencapai:

- Accuracy: **0,9899**
- Macro-F1: **0,9830**
- Weighted-F1: **0,9898**
- Log loss: **0,0553**
- Jumlah sampel test: **93.034**

Nilai ini tetap lebih rendah dibanding benchmark sintetis sebelumnya, tetapi jauh lebih kredibel daripada benchmark sintetis maupun benchmark riil satu tahun. Dengan kata lain, hardening pada benchmark riil 2021-2023 mempertahankan performa sangat tinggi sambil meningkatkan validitas eksternal dan kejujuran ilmiah sistem.

## 4.2 Analisis per Kelas

Nilai F1 per kelas pada benchmark riil multi-tahun adalah sebagai berikut:

- Low Risk: **0,9921**
- Medium Risk: **0,9911**
- High Risk: **0,9668**

Interpretasi utama:

1. Kelas Low dan Medium tetap sangat kuat.
2. Kelas High masih paling sulit, tetapi jauh lebih baik dibanding benchmark riil satu tahun.
3. Performa ini menunjukkan bahwa penambahan cakupan tahun riil memberi histori yang lebih kaya dan memperbaiki stabilitas prediksi.

Figure pendukung:

- `proposal/figures/per_class_f1.png`
- `proposal/figures/confusion_matrix.png`

## 4.3 Confusion Matrix

Confusion matrix final menunjukkan:

- Low Risk: 34.802/34.806 terklasifikasi benar
- Medium Risk: 51.848/52.427 terklasifikasi benar
- High Risk: 5.453/5.801 terklasifikasi benar

Kesalahan utama tetap terjadi ketika kelas High diprediksi sebagai Medium, tetapi tingkat deteksi kelas High sudah jauh lebih baik dibanding benchmark riil 2023 saja.

## 4.4 Kalibrasi Probabilitas

Model akhir tetap menggunakan temperature scaling berdasarkan clean-label review subset. Temperatur saat ini berada pada **7,697482**, dengan **287 reviewed rows** yang valid untuk fitting. Ini menandakan probabilitas mentah model masih cukup tajam dan perlu dilunakkan.

Figure pendukung:

- `proposal/figures/calibration_curve.png`

## 4.5 Explainability dan Faktor Risiko

Global explanation berbasis SHAP tetap menunjukkan bahwa fitur nilai, timing, dan histori buyer-supplier berperan penting. Namun, audit robustness memperlihatkan bahwa sebagian besar kekuatan model tetap bergantung pada fitur yang sangat dekat dengan aturan labeling.

Figure pendukung:

- `proposal/figures/shap_summary.png`
- `proposal/figures/robustness_ablation.png`

## 4.6 Perbandingan Benchmark Sintetis vs Riil

Perbandingan pada `models/benchmark_comparison.json` menunjukkan:

- Macro-F1 benchmark sintetis: **0,9950**
- Macro-F1 benchmark riil 2021-2023: **0,9831**
- Delta: **-0,0117**

Ini adalah hasil yang jauh lebih sehat secara ilmiah. Benchmark sintetis sebelumnya jelas terlalu optimistis. Namun setelah benchmark diperluas ke data riil multi-tahun, performa kembali naik dibanding benchmark riil satu tahun dan tetap berada pada level yang kuat untuk Phase 2.

## 4.7 Audit Kelemahan Model

Audit tambahan pada `models/robustness.json` dan `models/proxy_reduced_validation.json` menunjukkan:

- full model → Macro-F1 **0,9831**
- proxy_core_removed → Macro-F1 **0,5047**
- drop vs full → **0,4784**

Artinya, model masih sangat bergantung pada fitur yang berdekatan dengan heuristic labeling rules. Namun, `models/feature_health.json` menunjukkan feature catalog tetap sehat dan aktif, sehingga kelemahan utama yang tersisa benar-benar berada pada circularity risk, bukan lagi pada feature engineering yang rusak.

## 4.8 Operational Review Metrics

Artefak `models/operational_metrics.json` dan `proposal/figures/operational_metrics.png` mengukur seberapa baik model memprioritaskan baris High Risk pada budget review auditor yang terbatas.

Hasil utama:

- Precision@50 = **1,00**
- Precision@100 = **1,00**
- Precision@250 = **1,00**
- Precision@500 = **1,00**
- Precision@1000 = **1,00**

Artinya, pada benchmark saat ini, daftar prioritas tertinggi hampir sepenuhnya terisi oleh kasus High Risk. Ini adalah sinyal operasional yang kuat untuk workflow audit berbasis antrean review.

## 4.9 External Validation

Artefak `models/external_validation.json` dan `proposal/figures/external_validation.png` mengevaluasi model dengan skema holdout-year pada rentang 2019-2023.

Ringkasan:

- mean Macro-F1 = **0,9151**
- min Macro-F1 = **0,6956** (holdout 2019)
- max Macro-F1 = **0,9934** (holdout 2023)
- mean High Risk F1 = **0,8972**

Interpretasi:

1. Generalisasi pada tahun-tahun terbaru sangat kuat.
2. Fold 2019 paling lemah karena histori latih sebelum 2019 sangat terbatas.
3. Validasi ini memberi bukti temporal yang lebih kuat daripada hanya satu split train/test.

## 4.10 Manual Review Summary

Artefak `data/processed/manual_review_summary.csv`, `models/reviewed_subset_metrics.json`, dan `models/explanation_validation.json` mengimpor hasil review manual 500 baris benchmark.

Ringkasan utama:

- overall agreement model vs review: **95,8%**
- reviewed-subset Macro-F1: **0,9679**
- reviewed High Risk F1: **0,9603**
- explanation agreement: **95,8%**
- explanation clarity mean: **3,48 / 5**
- explanation actionability mean: **4,03 / 5**

Temuan penting:

1. Seluruh disagreement tetap berada pada batas **Medium ↔ High**.
2. Tidak ada flip ekstrem **Low ↔ High**.
3. Reviewer cenderung menaikkan sebagian kasus Medium menjadi High pada kelompok `high_uncertainty`.

Ini memperkuat klaim bahwa model secara umum selaras dengan penilaian manual, sambil tetap menunjukkan area terlemah yang memang berada pada boundary uncertainty.

## 4.11 What Manual Review Changed

Manual review mengubah posisi ilmiah proyek secara nyata:

1. Validasi tidak lagi hanya bergantung pada metric terhadap heuristic labels.
2. Kini ada bukti bahwa prediksi model selaras dengan review manual pada **95,8%** kasus.
3. Area lemah model dapat diidentifikasi dengan lebih spesifik, yaitu boundary **Medium ↔ High** pada baris ber-entropy tinggi.
4. Explainability tidak hanya tersedia, tetapi juga dinilai cukup membantu, dengan actionability mean **4,03 / 5**.

Dengan kata lain, manual review mengubah klaim proyek dari sekadar “model cocok dengan weak labels” menjadi “model juga cukup konsisten dengan penilaian manual pada sampel audit terbatas”.

## 4.12 Keterbatasan

Walaupun hasil benchmark riil multi-tahun jauh lebih kredibel, ada beberapa keterbatasan penting:

1. Label yang dipakai tetap **heuristik risiko**, bukan ground-truth fraud outcome.
2. Bukti manual review yang terimpor saat ini masih berbentuk **summary-level evidence**, belum berupa row-level reviewed sheet penuh di repo.
3. `tender_numberOfTenderers`, `contracts`, dan `procurementMethod` masih memiliki coverage yang lemah pada sumber riil.
4. Audit ablation menunjukkan circularity risk yang tetap kuat antara aturan labeling dan fitur utama.
5. Counterfactual yang tersedia masih berbasis SHAP fallback, bukan sistem optimasi tindakan penuh.
6. External validation 2019 masih lemah, menandakan adanya sensitivitas pada fold dengan histori sangat pendek.

## 4.14 Evidence-Backed Risk Lane

Upgrade terbaru menambahkan lane evidence resmi yang terpisah dari model heuristik. Artefak utama berada pada:

- `data/processed/evidence/evidence_records.parquet`
- `data/processed/evidence/linked_label_records.parquet`
- `data/processed/evidence/evidence_match_summary.json`
- `proposal/judge_casebook.md`

Saat ini snapshot evidence lane menunjukkan:

- total evidence rows: **5**
- matched to procurement rows: **4**
- still needing reviewer confirmation: **1**
- link confidence threshold: **0,55**

Makna ilmiahnya penting: LPSE-X tidak lagi hanya mengandalkan weak-label heuristik, tetapi sudah memiliki jalur terpisah untuk mengimpor bukti resmi, menghubungkannya ke `ocid`, dan menandai mana kasus yang cukup kuat untuk eskalasi investigatif.

Artefak `proposal/official_evidence_showcase.md` sekarang juga menunjukkan bagaimana model berperilaku pada kasus official-evidence yang benar-benar terhubung. Pada snapshot saat ini:

- terdapat **3 kasus linked** yang ditopang oleh **4 baris bukti resmi**,
- **1 dari 3** kasus sekarang sudah memiliki **multi-source corroboration** dari dua sumber resmi berbeda,
- **2 dari 3** kasus official evidence diprediksi langsung sebagai **High Risk** oleh model,
- **1 dari 3** kasus official evidence hanya diprediksi **Medium Risk**, tetapi tetap dinaikkan menjadi **Risiko Kritis** oleh evidence lane.

Ini adalah argumen demo yang sangat kuat: evidence lane bukan kosmetik, tetapi benar-benar memperbaiki blind spot model-only triage ketika bukti resmi tersedia.

## 4.15 Judge-Facing 4-Level Rating

Untuk kebutuhan demo dan komunikasi ke juri, hasil sekarang tidak hanya berhenti pada tiga kelas model (`Low`, `Medium`, `High`). Sistem sekarang juga memiliki lapisan presentasi 4-level yang lebih selaras dengan workflow audit:

- **Aman** → sinyal model rendah
- **Perlu Pantauan** → butuh monitoring atau review manual
- **Risiko Tinggi** → triase model kuat, tetapi belum ada bukti resmi final
- **Risiko Kritis** → ada bukti resmi terhubung, misalnya `confirmed_fraud`, `sanctioned_supplier`, atau irregularity resmi yang cukup kuat

Dengan desain ini, LPSE-X lebih jujur dan lebih defensible di depan juri: model tidak otomatis mengklaim fraud, dan status kritis hanya muncul bila ada evidence lane yang mendukung.

## 4.16 Demo Packaging dan Casebook

Artefak `proposal/judge_casebook.md` merangkum tiga komponen yang penting untuk demo:

1. definisi skala rating 4-level,
2. daftar kasus dengan official evidence yang sudah linked ke procurement rows,
3. top review/demo rows lengkap dengan business rating, faktor SHAP utama, dan narasi investigator-facing,
4. archetype demo yang membedakan critical case, review-needed case, dan model-only triage case.

Artefak tambahan `proposal/official_evidence_showcase.md` memperlihatkan output model pada kasus official evidence yang benar-benar linked. Pada snapshot sekarang, ada **3 kasus linked** yang ditopang oleh **4 supporting evidence rows**, dengan **1 kasus** yang sudah dikonfirmasi oleh **dua sumber resmi berbeda**. Dari tiga kasus itu, **2 kasus** diprediksi langsung sebagai High Risk, sedangkan **1 kasus** hanya diprediksi Medium Risk dan perlu dikoreksi oleh evidence lane menjadi Risiko Kritis.

Casebook ini meningkatkan kualitas presentasi karena juri tidak hanya melihat angka evaluasi, tetapi juga melihat bagaimana sistem dipakai sebagai alat triase investigatif yang realistis. Ini membantu menjembatani gap antara model tabular, explanation output, dan cerita demo yang lebih mudah dipahami.

## 4.17 Kesimpulan Bab

Secara keseluruhan, LPSE-X berhasil menunjukkan bahwa pipeline explainable AI berbasis XGBoost + SHAP tetap bekerja pada data riil multi-tahun yang lebih noisy dan tidak lengkap. Dibanding benchmark riil satu tahun, hasil sekarang lebih stabil; dibanding benchmark sintetis, hasil sekarang jauh lebih kredibel.

Kesimpulan praktisnya: LPSE-X sudah layak diposisikan sebagai **prototype explainable procurement-risk screening** yang berjalan offline dan patuh constraint. Bukti sekarang sudah lebih luas karena mencakup kalibrasi review yang lebih besar, operational review metrics, dan external validation lintas tahun, tetapi sistem ini tetap belum boleh diklaim sebagai sistem fraud detection operasional final sampai tersedia reviewed labels yang benar-benar diisi manusia.
