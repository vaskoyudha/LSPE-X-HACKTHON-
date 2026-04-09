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
