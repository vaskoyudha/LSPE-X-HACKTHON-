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
