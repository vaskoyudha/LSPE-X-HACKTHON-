# BAB 2: METODOLOGI

## 2.1 Gambaran Umum Pipeline

LPSE-X dibangun sebagai pipeline offline untuk mengubah data pengadaan publik menjadi **skor risiko yang dapat dijelaskan**. Alur kerjanya terdiri atas enam tahap inti:

1. ingest dan pembersihan data OCDS;
2. pemisahan `train_data` dan `test_data` pada level raw data;
3. rekayasa fitur yang split-aware;
4. pembentukan heuristic risk labels;
5. pelatihan model XGBoost dan kalibrasi probabilitas;
6. generasi explanation berbasis SHAP dan narasi Bahasa Indonesia.

Untuk kebutuhan penilai, metodologi inti LPSE-X diringkas pada diagram berikut agar hubungan antara data, model, dan keluaran explainability dapat dibaca dengan cepat dan jelas.

![Arsitektur end-to-end LPSE-X](figures/pipeline-architecture.png)

## 2.2 Sumber Data dan Kualitas

Pipeline bekerja di atas artefak lokal yang dibangun dari publikasi data OCDS Indonesia. Provenance utamanya tersimpan pada `data/processed/data_provenance.json`, sedangkan metadata split ada di `data/processed/split_metadata.json`.

Ringkasan benchmark saat ini:

- total baris usable: **465.184**
- buyer unik: **618**
- supplier unik: **60.976**
- train rows: **372.150**
- test rows: **93.034**
- rentang train: **2015-07-09 s.d. 2023-03-10 07:27:51 UTC**
- rentang test: **2023-03-10 07:38:45 s.d. 2023-12-20 23:00:00 UTC**

Temuan kualitas data yang paling penting untuk dibaca penilai adalah sebagai berikut.

1. `award_value_amount` relatif kuat sehingga tetap berguna untuk sinyal nilai.
2. Beberapa field seperti `tender.value.amount` dan `tender.description` memerlukan fallback karena coverage tidak konsisten.
3. Coverage `tender_numberOfTenderers`, `contracts`, dan `procurementMethod` masih lemah pada slice ini.
4. Karena kualitas field tidak merata, proposal ini lebih jujur bila memposisikan sistem sebagai **prototype risk screening** daripada solusi operasional nasional yang matang.

## 2.3 Strategi Split Data dan Anti-Leakage

Constraint terpenting pada Track C adalah bukti bahwa **tidak ada data leakage** antara train dan test. Karena itu, LPSE-X menerapkan aturan berikut.

1. `src/split.py` memisahkan raw data terlebih dahulu menjadi `train_data/raw.parquet` dan `test_data/raw.parquet`.
2. Feature engineering dilakukan **setelah** pemisahan tersebut, bukan sebelumnya.
3. Di dalam train split, data masih dipecah lagi menjadi `train_fit`, `val_hpo`, dan `val_calibration` untuk menjaga disiplin eksperimen.
4. `test_data/` tidak dipakai untuk hyperparameter optimization, threshold tuning, ataupun temperature scaling.
5. Fitur historis dibangun dengan prinsip expanding-window, sehingga hanya memakai histori masa lalu.

Konsekuensi desain ini adalah setiap angka evaluasi di Bab 4 berasal dari pemisahan yang defensible terhadap kebocoran data.

![Alur anti-leakage dan pemisahan train/test LPSE-X](figures/anti-leakage-flow.png)

## 2.4 Pelabelan Risiko dan Implikasi Ilmiahnya

Karena tidak tersedia label fraud terverifikasi untuk seluruh populasi data, LPSE-X menggunakan **heuristic risk labeling** dengan tiga kelas:

- **Low Risk**
- **Medium Risk**
- **High Risk**

Distribusi label pada artefak saat ini adalah:

| Split | Low | Medium | High |
| --- | ---: | ---: | ---: |
| Train | 124.351 | 223.427 | 24.372 |
| Test | 26.358 | 58.425 | 8.251 |

Label ini dibentuk dari kombinasi red flag yang relevan untuk pengadaan, misalnya sinyal peserta tunggal, deviasi harga, timing pengadaan, dan pola hubungan buyer–supplier. Secara metodologis, ini berarti model belajar **mendekati struktur sinyal risiko** yang didefinisikan oleh aturan tersebut. Karena itu, performa tinggi wajib dibaca bersama audit circularity; model ini tidak boleh diklaim sebagai estimator sempurna atas fraud yang telah dibuktikan secara hukum.

## 2.5 Rekayasa Fitur

Manifest fitur pada `data/processed/feature_manifest.json` menunjukkan bahwa model akhir memakai **34 fitur**. Fitur-fitur ini dibagi ke dalam dua lapisan besar.

### Tier 1 — Fitur langsung dari paket pengadaan

Contoh sinyal yang digunakan:

- nilai tender dan nilai award (dengan fallback yang terdokumentasi),
- deviasi harga,
- durasi dan timing tender,
- panjang judul/deskripsi,
- jumlah item,
- indikator musiman seperti Q4 dan Desember.

### Tier 2 — Fitur historis dan relasional

Contoh sinyal yang digunakan:

- rata-rata historis nilai buyer,
- frekuensi kemenangan supplier,
- pengulangan pasangan buyer–supplier,
- z-score nilai tender relatif terhadap histori buyer,
- intensitas aktivitas buyer atau supplier pada jendela waktu tertentu.

Seluruh artefak fiturnya dimaterialisasi ke `train_data/features.parquet` dan `test_data/features.parquet` agar proses training maupun audit dapat diulang tanpa langkah tersembunyi.

## 2.6 Pemodelan dan Alasan Pemilihan Model

Model utama yang dipakai adalah **XGBoost multiclass** dengan objective `multi:softprob`. Pemilihan ini disengaja karena XGBoost memenuhi kebutuhan Track C secara lebih natural dibanding model yang lebih opaque:

1. cocok untuk data tabular terstruktur,
2. efisien pada CPU-only,
3. mudah dihubungkan ke SHAP untuk explainability global maupun lokal,
4. bisa diekspor ke artefak ringan untuk demo lokal.

Dibanding pendekatan deep learning murni, XGBoost memberi trade-off yang lebih rasional untuk fase seleksi ini: performanya kuat pada data tabular, biaya komputasinya rendah, dan jalur interpretasinya jauh lebih mudah dijelaskan kepada penilai maupun calon pengguna institusional. Dibanding rule-only system, model ini juga memberi fleksibilitas lebih baik untuk menangkap kombinasi sinyal risiko yang tidak selalu terlihat dari satu aturan tunggal.

Artefak model yang saat ini tersedia juga kecil dan praktis untuk submission:

- `models/xgb_model.ubj` ≈ **1,1 MB**
- `models/xgb_model.onnx` ≈ **423 KB**

Ukuran ini mendukung narasi bahwa solusi dapat dibawa dan dijalankan secara offline tanpa kebutuhan infrastruktur berat.

## 2.7 Kalibrasi Probabilitas dan Explainability

LPSE-X tidak berhenti pada skor mentah. Pipeline juga melapisi model dengan:

1. **temperature scaling** untuk melunakkan probabilitas,
2. **SHAP** untuk faktor pendorong prediksi,
3. **narasi Bahasa Indonesia** agar output bisa dibaca auditor.

Parameter kalibrasi saat ini tersimpan di `models/calibration.json` dengan ringkasan:

- temperature: **7,697482**
- calibration samples: **287**
- method: **temperature scaling**

Pada level output, fungsi `explain_single(...)` dirancang untuk memenuhi kebutuhan Track C: setiap prediksi dapat diterjemahkan menjadi minimal tiga faktor teratas, lengkap dengan arah pengaruhnya, lalu dirender ke penjelasan Bahasa Indonesia yang bisa dipakai dalam notebook inference maupun casebook demo.

## 2.8 Artefak Submission dan Reproducibility

Struktur submission yang disiapkan untuk penilai mengikuti constraint umum kompetisi.

| Artefak | Peran |
| --- | --- |
| `training.ipynb` | menunjukkan pelatihan model dan log yang terlihat |
| `inference.ipynb` | menunjukkan alur inferensi yang bersih dan siap demo |
| `train_data/` | artefak data latih hasil split raw |
| `test_data/` | artefak data uji yang terpisah |
| `models/xgb_model.ubj` / `models/xgb_model.onnx` | model final untuk deployment lokal |
| `requirements.txt` | dependency agar eksperimen dapat dijalankan ulang |
| `proposal/figures/` | visual evaluasi dan bukti presentasi |

Dengan desain ini, penilai tidak perlu menebak alur kerja proyek: semua komponen utama tersedia sebagai artefak lokal yang eksplisit.

## 2.9 Filosofi Evaluasi

Agar hasil mudah dibaca secara profesional, evaluasi pada proposal ini dibagi menjadi empat lapis:

1. **headline benchmark** pada test split heuristik,
2. **calibration dan confusion analysis** untuk membaca trade-off operasional,
3. **robustness / proxy-reduced validation** untuk mengukur circularity risk,
4. **manual review, external validation, dan official evidence lane** untuk memperkaya bukti di luar sekadar angka terhadap weak labels.

Dengan demikian, evaluasi pada proposal ini tidak berhenti pada klaim “akurasi tinggi”, tetapi menunjukkan **apa yang benar-benar dibuktikan oleh setiap lapisan pengujian**. Pendekatan ini membuat metodologi LPSE-X lebih kuat secara kompetitif karena tidak hanya menonjolkan angka, tetapi juga kedewasaan desain evaluasi.
