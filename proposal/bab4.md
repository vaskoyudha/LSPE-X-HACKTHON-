# BAB 4: HASIL DAN PEMBAHASAN

## 4.1 Ringkasan Hasil Utama

Berdasarkan artefak evaluasi pada `models/metrics.json`, model LPSE-X pada benchmark riil 2021-2023 mencapai:

- Accuracy: **0,991**
- Macro-F1: **0,9432**
- Weighted-F1: **0,9909**
- Log loss: **0,0735**
- Jumlah sampel test: **93.034**

Nilai ini tetap lebih rendah dibanding benchmark sintetis sebelumnya, tetapi jauh lebih kredibel daripada benchmark sintetis maupun benchmark riil satu tahun. Dengan kata lain, perluasan ke 2021-2023 menaikkan kembali performa sambil mempertahankan validitas eksternal yang lebih baik.

## 4.2 Analisis per Kelas

Nilai F1 per kelas pada benchmark riil multi-tahun adalah sebagai berikut:

- Low Risk: **0,9912**
- Medium Risk: **0,9916**
- High Risk: **0,8468**

Interpretasi utama:

1. Kelas Low dan Medium tetap sangat kuat.
2. Kelas High masih paling sulit, tetapi jauh lebih baik dibanding benchmark riil satu tahun.
3. Performa ini menunjukkan bahwa penambahan cakupan tahun riil memberi histori yang lebih kaya dan memperbaiki stabilitas prediksi.

Figure pendukung:

- `proposal/figures/per_class_f1.png`
- `proposal/figures/confusion_matrix.png`

## 4.3 Confusion Matrix

Confusion matrix final menunjukkan:

- Low Risk: 42.611/42.616 terklasifikasi benar
- Medium Risk: 49.313/50.100 terklasifikasi benar
- High Risk: 234/318 terklasifikasi benar

Kesalahan utama tetap terjadi ketika kelas High diprediksi sebagai Medium, tetapi tingkat deteksi kelas High sudah jauh lebih baik dibanding benchmark riil 2023 saja.

## 4.4 Kalibrasi Probabilitas

Model akhir tetap menggunakan temperature scaling berdasarkan clean-label review subset. Temperatur tetap tinggi, yang menandakan probabilitas mentah model masih terlalu tajam dan perlu dilunakkan.

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
- Macro-F1 benchmark riil 2021-2023: **0,9349**
- Delta: **-0,0518**

Ini adalah hasil yang jauh lebih sehat secara ilmiah. Benchmark sintetis sebelumnya jelas terlalu optimistis. Namun setelah benchmark diperluas ke data riil multi-tahun, performa kembali naik dibanding benchmark riil satu tahun dan tetap berada pada level yang kuat untuk Phase 2.

## 4.7 Audit Kelemahan Model

Audit tambahan pada `models/robustness.json` menunjukkan:

- full model → Macro-F1 **0,9432**
- proxy_core_removed → Macro-F1 **0,3854**
- proxy_broad_removed → Macro-F1 **0,3781**

Artinya, model masih sangat bergantung pada fitur yang berdekatan dengan heuristic labeling rules. Perluasan benchmark ke 2021-2023 memperbaiki kualitas data dan performa umum, tetapi tidak menghilangkan circularity risk.

## 4.8 Keterbatasan

Walaupun hasil benchmark riil multi-tahun jauh lebih kredibel, ada beberapa keterbatasan penting:

1. Label yang dipakai tetap **heuristik risiko**, bukan ground-truth fraud outcome.
2. Benchmark riil saat ini hanya memakai **slice 2021-2023**, bukan histori penuh.
3. `tender_numberOfTenderers`, `contracts`, dan `procurementMethod` masih memiliki coverage yang lemah pada sumber riil.
4. Audit ablation menunjukkan circularity risk yang tetap kuat antara aturan labeling dan fitur utama.
5. Counterfactual yang tersedia masih berbasis SHAP fallback, bukan sistem optimasi tindakan penuh.

## 4.9 Kesimpulan Bab

Secara keseluruhan, LPSE-X berhasil menunjukkan bahwa pipeline explainable AI berbasis XGBoost + SHAP tetap bekerja pada data riil multi-tahun yang lebih noisy dan tidak lengkap. Dibanding benchmark riil satu tahun, hasil sekarang lebih stabil; dibanding benchmark sintetis, hasil sekarang jauh lebih kredibel.

Kesimpulan praktisnya: LPSE-X sudah layak diposisikan sebagai **prototype explainable procurement-risk screening** yang berjalan offline dan patuh constraint, tetapi belum boleh diklaim sebagai sistem fraud detection operasional final.
