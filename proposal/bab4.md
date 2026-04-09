# BAB 4: HASIL DAN PEMBAHASAN

## 4.1 Ringkasan Hasil Utama

Berdasarkan artefak evaluasi pada `models/metrics.json`, model LPSE-X pada benchmark riil 2023 mencapai:

- Accuracy: **0,9561**
- Macro-F1: **0,8309**
- Weighted-F1: **0,9549**
- Log loss: **0,1662**
- Jumlah sampel test: **26.494**

Nilai ini jelas lebih rendah dibanding benchmark sintetis sebelumnya, tetapi jauh lebih kredibel untuk dibawa ke evaluasi Phase 2 karena model diuji pada data riil yang coverage field-nya tidak sempurna.

## 4.2 Analisis per Kelas

Nilai F1 per kelas pada benchmark riil adalah sebagai berikut:

- Low Risk: **0,9493**
- Medium Risk: **0,9654**
- High Risk: **0,5780**

Interpretasi utama:

1. Kelas Low dan Medium tetap tertangani kuat.
2. Kelas High jauh lebih sulit pada data riil.
3. Ini adalah sinyal realistis bahwa kasus berisiko tinggi pada data nyata lebih sulit dipisahkan dari kasus medium dibanding pada benchmark sintetis.

Figure pendukung:

- `proposal/figures/per_class_f1.png`
- `proposal/figures/confusion_matrix.png`

## 4.3 Confusion Matrix

Confusion matrix final menunjukkan:

- Low Risk: 8.987/8.987 terklasifikasi benar
- Medium Risk: 16.206/17.165 terklasifikasi benar
- High Risk: 139/342 terklasifikasi benar

Kesalahan utama terjadi ketika kelas High diprediksi sebagai Medium. Ini konsisten dengan fakta bahwa signal coverage pada sumber riil tidak selengkap benchmark sintetis, terutama untuk field yang berkaitan dengan kompetisi tender dan metode pengadaan.

## 4.4 Kalibrasi Probabilitas

Model akhir tetap menggunakan temperature scaling dengan parameter `T = 9,999995` berdasarkan 95 sampel calibration yang usable, 94 di antaranya high confidence. Seperti sebelumnya, temperatur yang tinggi menunjukkan bahwa probabilitas mentah model masih terlalu tajam dan perlu dilunakkan.

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
- Macro-F1 benchmark riil: **0,8309**
- Delta: **-0,1641**

Ini adalah hasil yang sangat penting. Secara praktis, benchmark sintetis sebelumnya berguna untuk membuktikan bahwa pipeline bisa dibangun. Namun benchmark riil menunjukkan tingkat kesulitan yang sebenarnya jauh lebih tinggi. Dengan kata lain, migrasi ke data riil membuat performa turun, tetapi meningkatkan kejujuran ilmiah submission.

## 4.7 Audit Kelemahan Model

Audit tambahan pada `models/robustness.json` menunjukkan:

- full model → Macro-F1 **0,8299**
- proxy_core_removed → Macro-F1 **0,3466**
- proxy_broad_removed → Macro-F1 **0,3371**

Artinya, model masih sangat bergantung pada fitur yang berdekatan dengan heuristic labeling rules. Pada benchmark riil, ketergantungan itu sedikit lebih rendah daripada benchmark sintetis, tetapi tetap besar. Jadi model ini masih lebih tepat diposisikan sebagai **explainable risk-screening engine** daripada detektor fraud yang sepenuhnya bebas dari proxy-label effects.

## 4.8 Keterbatasan

Walaupun hasil benchmark riil jauh lebih kredibel, ada beberapa keterbatasan penting:

1. Label yang dipakai tetap **heuristik risiko**, bukan ground-truth fraud outcome.
2. Benchmark riil saat ini hanya memakai **slice 2023**, bukan histori penuh.
3. `tender_numberOfTenderers`, `contracts`, dan `procurementMethod` memiliki coverage yang lemah pada benchmark riil saat ini.
4. Audit ablation menunjukkan circularity risk yang tetap kuat antara aturan labeling dan fitur utama.
5. Counterfactual yang tersedia masih berbasis SHAP fallback, bukan sistem optimasi tindakan penuh.

## 4.9 Kesimpulan Bab

Secara keseluruhan, LPSE-X berhasil menunjukkan bahwa pipeline explainable AI berbasis XGBoost + SHAP tetap bekerja pada data riil yang lebih noisy dan tidak lengkap. Meskipun performa turun dibanding benchmark sintetis, hasil ini jauh lebih kredibel untuk Phase 2 karena merefleksikan tingkat kesulitan yang lebih dekat ke situasi nyata.

Kesimpulan praktisnya: LPSE-X sudah layak diposisikan sebagai **prototype explainable procurement-risk screening** yang berjalan offline dan patuh constraint, tetapi belum boleh diklaim sebagai sistem fraud detection operasional final.
