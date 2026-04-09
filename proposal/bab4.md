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

## 4.7 Keterbatasan

Walaupun hasil metrik sangat tinggi, ada beberapa keterbatasan penting:

1. Label yang dipakai adalah **heuristik risiko**, bukan ground-truth fraud outcome.
2. Sebagian fitur dan label dibangun dari keluarga red flag yang berdekatan, sehingga ada risiko circularity.
3. Artefak data kerja saat ini merepresentasikan snapshot pipeline yang sangat terstruktur; generalisasi ke data lapangan mentah tetap perlu validasi tambahan.
4. Counterfactual yang tersedia masih berbasis SHAP fallback, bukan sistem optimasi tindakan penuh.

## 4.8 Kesimpulan Bab

Secara keseluruhan, LPSE-X berhasil memenuhi tujuan Phase 2: membangun pipeline explainable AI berbasis XGBoost + SHAP yang bekerja offline, menjaga prinsip anti-leakage, menghasilkan metrik kuat pada test split, dan menyajikan penjelasan yang dapat dibaca auditor non-teknis.

Hasil ini menunjukkan bahwa pendekatan tabular gradient boosting dengan explainability deterministik adalah strategi yang efektif untuk skrining awal risiko pengadaan pada setting kompetisi Track C.
