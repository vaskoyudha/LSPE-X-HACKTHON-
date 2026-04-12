# BAB 4: HASIL DAN PEMBAHASAN

## 4.1 Ringkasan Hasil Utama

Pada benchmark heuristik yang saat ini dipakai di repo, LPSE-X mencapai performa berikut pada `models/metrics.json`:

| Metrik | Nilai |
| --- | ---: |
| Accuracy | **0,9899** |
| Macro-F1 | **0,9830** |
| Weighted-F1 | **0,9898** |
| Log loss | **0,0553** |
| Test rows | **93.034** |

Untuk juri, angka ini berarti model memiliki kemampuan ranking dan klasifikasi yang sangat kuat **terhadap label risiko heuristik yang digunakan pada benchmark saat ini**. Interpretasi ini penting: hasil tinggi tidak otomatis berarti sistem siap menggeneralisasi ke seluruh kasus nyata di lapangan.

![Perbandingan benchmark sintetis dan benchmark data riil saat ini](figures/benchmark_comparison.png)

## 4.2 Analisis per Kelas dan Confusion Matrix

F1 per kelas pada test split adalah sebagai berikut:

| Kelas | F1 |
| --- | ---: |
| Low Risk | **0,9932** |
| Medium Risk | **0,9920** |
| High Risk | **0,9639** |

![Skor F1 per kelas](figures/per_class_f1.png)

Confusion matrix menunjukkan bahwa kesalahan terbesar tetap terkonsentrasi pada batas **Medium Risk ↔ High Risk**, bukan pada lompatan ekstrem Low ↔ High.

- Low Risk benar: **26.354 / 26.358**
- Medium Risk benar: **58.015 / 58.425**
- High Risk benar: **7.726 / 8.251**

![Confusion matrix pada test split](figures/confusion_matrix.png)

Secara operasional, pola ini cukup masuk akal: batas ketidakpastian terbesar memang terjadi pada kasus yang sama-sama menampilkan sinyal risiko, tetapi belum cukup kuat untuk diposisikan pada kelas tertinggi.

## 4.3 Kalibrasi Probabilitas

LPSE-X tidak hanya mengejar klasifikasi yang tepat, tetapi juga probabilitas yang lebih dapat dipercaya. Temperature scaling dijalankan dengan **287 calibration samples** dan menghasilkan temperatur **7,697482**. Ini menunjukkan bahwa probabilitas mentah perlu dilunakkan sebelum dipakai sebagai dasar prioritas review.

![Kurva kalibrasi probabilitas LPSE-X](figures/calibration_curve.png)

Bagi juri, poin utamanya adalah: sistem tidak berhenti pada label kelas, tetapi juga memperhatikan kualitas skor probabilitas yang akan dipakai dalam workflow nyata.

## 4.4 Explainability dan Human-Readable Output

Track C menuntut penjelasan yang bisa dibaca manusia. Pada LPSE-X, SHAP digunakan untuk menunjukkan faktor global dan lokal, lalu hasilnya diterjemahkan menjadi narasi Bahasa Indonesia. Faktor yang sering muncul pada explanation review antara lain:

- `f_buyer_supplier_repeat_count`,
- `f_is_q4`,
- `f_title_length`,
- `f_supplier_recent_90d_award_count`,
- `f_tender_value_log`.

![Ringkasan SHAP global untuk model final](figures/shap_summary.png)

Pada evaluasi review manual, kualitas explanation menunjukkan:

- agreement explanation: **95,8%**
- clarity mean: **3,48 / 5**
- actionability mean: **4,03 / 5**

Artinya, explanation yang dihasilkan belum sempurna dari sisi kejelasan, tetapi sudah cukup membantu untuk actionability review.

> **Placeholder visual untuk PDF final:** tambahkan satu kartu contoh kasus (*single-case explanation card*) yang menampilkan skor, tiga faktor teratas, arah pengaruh, dan rekomendasi tindak lanjut.

## 4.5 Manual Review dan Validasi Tambahan

Review manual 500 baris menambah lapisan bukti penting di luar metrik terhadap weak labels. Artefak `models/reviewed_subset_metrics.json` menunjukkan:

- reviewed rows: **500**
- overall agreement: **95,8%**
- reviewed-subset Macro-F1: **0,9679**
- reviewed High Risk F1: **0,9603**

![Ringkasan manual review dan kualitas explanation](figures/manual_review_summary.png)

Temuan utamanya adalah disagreement terkonsentrasi pada area **Medium Risk ↔ High Risk**, sedangkan flip ekstrem **Low Risk ↔ High Risk** tidak muncul. Ini memperkuat posisi bahwa model cukup stabil sebagai alat triase, meskipun belum dapat disebut penyelesai akhir.

## 4.6 Robustness, Circularity Risk, dan Kejujuran Ilmiah

Bagian ini adalah alasan mengapa proposal kami tetap ilmiah dan tidak overclaim. Saat fitur-fitur yang paling dekat dengan aturan labeling dihapus, performa turun tajam:

| Track evaluasi | Macro-F1 |
| --- | ---: |
| Full model | **0,9831** |
| Proxy-core-removed | **0,5047** |
| Penurunan | **0,4784** |

![Audit robustness terhadap fitur proksi](figures/robustness_ablation.png)

![Validasi proxy-reduced sebagai track evaluasi yang lebih ketat](figures/proxy_reduced_validation.png)

Interpretasinya jelas: model saat ini sangat efektif sebagai **interpreter dan accelerator untuk risk rules yang ada**, tetapi masih menyimpan circularity gap yang besar. Inilah sebabnya LPSE-X diposisikan sebagai *explainable procurement-risk screening*, bukan penentu akhir atas outcome hukum atau investigatif.

## 4.7 Validasi Eksternal Lintas Waktu

LPSE-X juga diuji dengan skema holdout-year pada 2019–2023. Ringkasan `models/external_validation.json` menunjukkan:

- mean Macro-F1: **0,9151**
- min Macro-F1: **0,6956**
- max Macro-F1: **0,9934**
- mean High Risk F1: **0,8972**

![External validation lintas tahun](figures/external_validation.png)

Pesan penting untuk juri: generalisasi pada tahun-tahun terbaru terlihat kuat, tetapi fold awal seperti 2019 jauh lebih berat. Ini membuat klaim performa menjadi lebih seimbang dan realistis.

## 4.8 Metrik Operasional untuk Budget Review Terbatas

Dari sudut pandang pengguna nyata, auditor sering hanya punya kapasitas untuk meninjau sebagian kecil paket. Pada `models/operational_metrics.json`, LPSE-X menunjukkan bahwa daftar teratas sangat kaya sinyal High Risk:

- Precision@50 = **1,00**
- Precision@100 = **1,00**
- Precision@250 = **1,00**
- Precision@500 = **1,00**
- Precision@1000 = **1,00**

![Metrik operasional pada berbagai budget review](figures/operational_metrics.png)

Interpretasi yang benar adalah: pada benchmark saat ini, model sangat baik dalam mengurutkan paket yang dianggap berisiko tinggi oleh sistem label yang digunakan. Sekali lagi, ini adalah kekuatan besar untuk workflow triase, tetapi tetap perlu dibaca bersama batasan label heuristik.

## 4.9 Evidence Lane dan Casebook Demo

Salah satu penguatan terbaru pada proyek ini adalah **evidence-backed risk lane** yang terpisah dari model heuristik. Artefak `data/processed/evidence/evidence_match_summary.json` menunjukkan:

- total evidence rows: **5**
- matched rows: **4**
- needs review rows: **1**

Sementara artefak `proposal/official_evidence_showcase.md` menunjukkan:

- official evidence-linked cases: **3**
- supporting official evidence rows: **4**
- cases with multi-source corroboration: **1**
- model predicted High Risk: **2**
- model predicted Medium Risk: **1**
- cases escalated to **Risiko Kritis** after evidence linkage: **3**

Bagi juri, ini menambah kualitas demo secara signifikan. Sistem tidak lagi hanya menampilkan angka model, tetapi juga menunjukkan bagaimana kasus dengan bukti resmi dapat dinaikkan ke status yang lebih kritis secara investigatif.

> **Placeholder visual untuk PDF final:** tambahkan diagram decision flow 4-level (*Aman → Perlu Pantauan → Risiko Tinggi → Risiko Kritis*) agar alur presentasi demo lebih intuitif.

## 4.10 Keterbatasan

Keterbatasan utama LPSE-X saat ini harus dibaca secara terbuka.

1. Benchmark utama masih menggunakan **heuristic risk labels**.
2. Audit ablation menunjukkan circularity risk yang masih kuat.
3. Coverage beberapa field sumber masih lemah.
4. Bukti manual review dan official evidence sudah berguna, tetapi belum setara dengan ground truth komprehensif untuk seluruh populasi data.
5. Sistem ini adalah **alat prioritisasi audit**, bukan pengganti investigator manusia.

## 4.11 Kesimpulan Bab

Secara keseluruhan, hasil LPSE-X cukup kuat untuk Tahap 2 karena memenuhi tiga hal yang paling penting bagi Track C:

1. **prediksi yang kuat terhadap benchmark yang digunakan**,
2. **explainability yang nyata dan human-readable**,
3. **kejujuran ilmiah terhadap batasan sistem**.

Itulah alasan proposal ini memosisikan LPSE-X sebagai solusi yang **siap dinilai, siap didemokan, dan patuh constraint**, sambil tetap jujur bahwa pekerjaan lanjutan terbesar berada pada penguatan label, validasi lapangan, dan pengurangan circularity risk.
