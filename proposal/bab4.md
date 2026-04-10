# BAB 4: HASIL DAN PEMBAHASAN

## 4.1 Ringkasan Hasil Utama

Berdasarkan artefak evaluasi pada `models/metrics.json`, model LPSE-X pada benchmark riil 2021-2023 mencapai:

- Accuracy: **0,991**
- Macro-F1: **0,9833**
- Weighted-F1: **0,9899**
- Log loss: **0,0584**
- Jumlah sampel test: **93.034**

Nilai ini tetap lebih rendah dibanding benchmark sintetis sebelumnya, tetapi kini jauh lebih dekat dan jauh lebih kredibel daripada benchmark sintetis maupun benchmark riil satu tahun. Dengan kata lain, hardening pada benchmark riil 2021-2023 meningkatkan performa sambil mempertahankan validitas eksternal yang lebih baik.

## 4.2 Analisis per Kelas

Nilai F1 per kelas pada benchmark riil multi-tahun adalah sebagai berikut:

- Low Risk: **0,9921**
- Medium Risk: **0,9911**
- High Risk: **0,9668**

Interpretasi utama:

1. Kelas Low dan Medium tetap sangat kuat.
2. Kelas High masih paling sulit, tetapi kini sudah sangat kuat dibanding benchmark riil satu tahun maupun versi sebelum hardening.
3. Performa ini menunjukkan bahwa kombinasi multi-year real benchmark, feature refresh, dan label redesign memberi histori yang lebih kaya serta memperbaiki stabilitas prediksi.

Figure pendukung:

- `proposal/figures/per_class_f1.png`
- `proposal/figures/confusion_matrix.png`

## 4.3 Confusion Matrix

Confusion matrix final menunjukkan:

- Low Risk: 34.802/34.806 terklasifikasi benar
- Medium Risk: 51.848/52.427 terklasifikasi benar
- High Risk: 5.453/5.801 terklasifikasi benar

Kesalahan utama tetap terjadi ketika kelas High diprediksi sebagai Medium, tetapi tingkat deteksi kelas High sekarang jauh lebih tinggi daripada versi benchmark riil sebelumnya.

## 4.4 Kalibrasi Probabilitas

Model akhir tetap menggunakan temperature scaling berdasarkan clean-label review subset. Temperatur saat ini berada pada **7,836317**, dengan **287 reviewed rows** yang valid untuk fitting. Ini menandakan probabilitas mentah model masih cukup tajam dan perlu dilunakkan sebelum dipakai sebagai skor prioritas.

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
- Macro-F1 benchmark riil 2021-2023: **0,9833**
- Delta: **-0,0117**

Ini adalah hasil yang jauh lebih sehat secara ilmiah. Benchmark sintetis sebelumnya jelas terlalu optimistis. Namun setelah benchmark diperluas ke data riil multi-tahun, feature slots dibersihkan, dan label direka ulang mengikuti sinyal riil yang tersedia, performa kembali naik sampai mendekati benchmark sintetis sekaligus tetap berada pada level yang kuat untuk Phase 2.

## 4.7 Audit Kelemahan Model

Audit tambahan pada `models/robustness.json` menunjukkan:

- full model → Macro-F1 **0,9833**
- proxy_core_removed → Macro-F1 **0,5215**
- proxy_broad_removed → Macro-F1 **0,5204**

Artinya, model masih sangat bergantung pada fitur yang berdekatan dengan heuristic labeling rules. Namun, `models/feature_health.json` kini menunjukkan **0 active dead features**, sehingga kelemahan utama tersisa benar-benar berada pada circularity risk, bukan lagi pada feature catalog yang rusak.

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

Artefak `data/processed/manual_review_summary.csv`, `models/reviewed_subset_metrics.json`, dan `models/explanation_validation.json` mengimpor hasil review manual 500 baris benchmark yang Anda lakukan di luar repo.

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

## 4.11 Keterbatasan

Walaupun hasil benchmark riil multi-tahun jauh lebih kredibel, ada beberapa keterbatasan penting:

1. Label yang dipakai tetap **heuristik risiko**, bukan ground-truth fraud outcome.
2. Bukti manual review yang terimpor saat ini masih berbentuk **summary-level evidence**, belum berupa row-level reviewed sheet penuh di repo.
3. `tender_numberOfTenderers`, `contracts`, dan `procurementMethod` masih memiliki coverage yang lemah pada sumber riil.
4. Audit ablation menunjukkan circularity risk yang tetap kuat antara aturan labeling dan fitur utama.
5. Counterfactual yang tersedia masih berbasis SHAP fallback, bukan sistem optimasi tindakan penuh.
6. External validation 2019 masih lemah, menandakan adanya sensitivitas pada fold dengan histori sangat pendek.

## 4.12 Kesimpulan Bab

Secara keseluruhan, LPSE-X berhasil menunjukkan bahwa pipeline explainable AI berbasis XGBoost + SHAP tetap bekerja pada data riil multi-tahun yang lebih noisy dan tidak lengkap. Dibanding benchmark riil satu tahun, hasil sekarang lebih stabil; dibanding benchmark sintetis, hasil sekarang jauh lebih kredibel.

Kesimpulan praktisnya: LPSE-X sudah layak diposisikan sebagai **prototype explainable procurement-risk screening** yang berjalan offline dan patuh constraint. Bukti sekarang sudah lebih luas karena mencakup kalibrasi review yang lebih besar, operational review metrics, dan external validation lintas tahun, tetapi sistem ini tetap belum boleh diklaim sebagai fraud detection operasional final sampai tersedia reviewed labels yang benar-benar diisi manusia.
