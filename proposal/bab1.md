# BAB 1: PENDAHULUAN

## 1.1 Latar Belakang

Pengadaan barang dan jasa pemerintah adalah domain dengan nilai transaksi besar, kompleksitas proses tinggi, dan konsekuensi publik yang langsung terasa. Dalam praktiknya, tantangan utama bukan sekadar ketersediaan data, tetapi ketiadaan **mekanisme triase yang mampu menyaring ribuan paket secara konsisten, cepat, dan dapat dijelaskan**. Tanpa alat bantu yang explainable, pengawasan cenderung kembali pada pemeriksaan manual yang mahal, lambat, dan sulit diprioritaskan.

LPSE-X dikembangkan untuk menjawab kebutuhan tersebut pada subtema **Smart Governance & Public Service**. Fokus kami bukan membangun mesin keputusan hukum, melainkan **sistem penyaringan risiko pengadaan** yang dapat membantu auditor menentukan paket mana yang layak ditelaah lebih dulu. Posisi ini penting karena pada konteks publik, akurasi yang tinggi saja belum cukup; alasan di balik skor juga harus dapat dipahami dan dipertanggungjawabkan.

Secara teknis, Track C menuntut dua hal sekaligus: model prediktif yang presisi **dan** transparan. Karena itu, solusi kami dirancang dari awal sebagai pipeline offline berbasis data terstruktur, dengan pemisahan train/test sebelum preprocessing, model tabular yang dapat diinspeksi, serta lapisan explainability yang menghasilkan narasi Bahasa Indonesia untuk setiap prediksi.

## 1.2 Rumusan Masalah

Rumusan masalah yang dijawab proposal ini adalah sebagai berikut.

1. Bagaimana mengubah data pengadaan publik berbasis OCDS menjadi pipeline analitik yang siap dipakai untuk triase risiko?
2. Bagaimana membangun model prediktif yang tetap patuh pada prinsip **anti-data-leakage** dan dapat dijalankan sepenuhnya secara offline?
3. Bagaimana menghasilkan penjelasan prediksi yang tidak berhenti pada probabilitas, tetapi dapat dibaca manusia dan berguna bagi reviewer non-teknis?
4. Bagaimana menyajikan hasil model secara jujur, termasuk keterbatasan weak labels dan circularity risk, agar solusi tetap ilmiah dan defensible di depan juri?

## 1.3 Posisi Solusi dan Nilai Utama

LPSE-X diposisikan sebagai **explainable procurement-risk screening system** dengan empat nilai utama.

1. **Praktis untuk triase** — sistem memprioritaskan paket yang layak ditinjau lebih dulu, bukan menggantikan investigator.
2. **Patuh Track C** — setiap prediksi disertai alasan, arah pengaruh, dan bukti bahwa pipeline tidak melanggar aturan anti-leakage serta offline total.
3. **Siap didemokan** — model diekspor ke artefak ringan (`.ubj` dan `.onnx`) dan dilengkapi notebook training serta inference.
4. **Jujur secara ilmiah** — metrik dilaporkan terhadap label heuristik risiko, bukan diklaim sebagai tingkat keberhasilan mendeteksi korupsi yang sudah terverifikasi.

## 1.4 Tujuan

Tujuan pengembangan LPSE-X pada Tahap 2 adalah:

1. menyusun pipeline data pengadaan yang rapi, split-aware, dan reproducible;
2. melatih model prediktif berbasis XGBoost untuk klasifikasi risiko pengadaan;
3. menghasilkan output explainability yang memenuhi kebutuhan Track C, termasuk minimal tiga faktor teratas beserta arah pengaruhnya;
4. menyiapkan paket submission yang mudah diperiksa juri: proposal, notebook, model, dan folder data terpisah;
5. menunjukkan posisi ilmiah proyek secara seimbang: cukup kuat untuk demo, tetapi tetap eksplisit mengenai keterbatasannya.

## 1.5 Batasan dan Kejujuran Ilmiah

Agar tidak terjadi overclaim, proposal ini menetapkan batasan berikut.

1. Data kerja berasal dari artefak OCDS yang telah diproses lokal di repo ini; benchmark saat ini adalah **slice data riil** dengan total 465.184 baris usable, bukan seluruh histori pengadaan nasional.
2. Label target adalah **heuristic risk labels**, bukan putusan hukum atau ground truth fraud outcome untuk seluruh dataset.
3. Model digunakan untuk **risk screening**, sehingga outputnya harus dipahami sebagai prioritas review, bukan keputusan final atau tuduhan pelanggaran.
4. Explainability yang dipakai adalah SHAP dan narasi deterministik, bukan generative AI atau cloud explanation service.
5. Hasil evaluasi yang sangat tinggi tetap harus dibaca bersama audit circularity agar tidak disalahartikan sebagai bukti final validitas lapangan.

## 1.6 Manfaat

### Bagi juri dan pengguna akhir

- Memberi contoh solusi AI yang relevan langsung dengan tata kelola publik.
- Menunjukkan bagaimana model prediktif dapat tetap transparan dan audit-friendly.
- Menawarkan alur kerja yang realistis: model membantu memprioritaskan review, lalu reviewer manusia mengambil keputusan.

### Bagi auditor atau pengawas

- Menyediakan daftar prioritas paket dengan alasan utama di balik skor.
- Mengurangi beban pemeriksaan awal pada kumpulan data yang besar.
- Membuka jalan menuju casebook investigatif yang lebih mudah dikomunikasikan.

### Bagi pengembangan riset lanjut

- Menyediakan baseline pipeline yang reproducible untuk eksperimen data, label review, dan validasi yang lebih kuat di masa depan.
- Menunjukkan secara terbuka area yang masih lemah, terutama circularity risk dan keterbatasan weak labels.

## 1.7 Sistematika Penulisan

- **Bab 1 — Pendahuluan**: konteks masalah, posisi solusi, tujuan, manfaat, dan batasan ilmiah.
- **Bab 2 — Metodologi**: sumber data, split anti-leakage, fitur, model, explainability, dan artefak reproduksibilitas.
- **Bab 3 — Kepatuhan dan Implementasi**: pembuktian langsung terhadap setiap constraint Track C dan kesiapan paket submission.
- **Bab 4 — Rancangan Sistem & Bisnis (Integrasi Phase 3 & Analisis Dampak)**: rancangan arsitektur target, alur operasional, model adopsi, KPI implementasi, analisis dampak, dan mitigasi risiko.
