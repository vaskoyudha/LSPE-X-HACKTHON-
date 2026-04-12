# BAB 4: RANCANGAN SISTEM & BISNIS (INTEGRASI PHASE 3 & ANALISIS DAMPAK)

## 4.1 Tujuan Integrasi Phase 3

Tahap 2 menempatkan LPSE-X sebagai **prototype prediktif yang dapat dijelaskan**. Tahap 3 tidak dimaknai sebagai perubahan menjadi mesin keputusan otomatis, melainkan sebagai **integrasi terkontrol ke workflow review** agar manfaat model benar-benar terasa bagi pengguna institusional. Dengan posisi tersebut, tujuan integrasi Phase 3 adalah:

1. memasukkan skor risiko dan penjelasan model ke alur kerja review yang sudah ada,
2. memastikan setiap rekomendasi tetap dapat ditelusuri kembali ke fitur, narasi, dan bukti pendukung,
3. menambah lapisan evidence-backed review agar kasus prioritas tinggi dapat diperkaya dengan sumber resmi,
4. mengubah output model dari sekadar artefak notebook menjadi **alat bantu operasional** bagi tim audit/pengawasan.

Secara prinsip, Phase 3 harus mempertahankan tiga batas etik yang sudah dijaga sejak awal: **offline-first**, **human-in-the-loop**, dan **tidak overclaim**. Artinya, LPSE-X memberi rekomendasi prioritas pemeriksaan, tetapi keputusan investigatif, administratif, maupun hukum tetap berada pada reviewer manusia.

## 4.2 Rancangan Sistem Target untuk Phase 3

Arsitektur target LPSE-X dirancang sebagai sistem modular agar mudah diadopsi bertahap. Komponen intinya adalah sebagai berikut.

1. **Data ingestion layer** — menerima ekspor OCDS atau snapshot data pengadaan dari sumber lokal.
2. **Feature preparation layer** — melakukan pembersihan, normalisasi, dan rekayasa fitur secara split-aware serta terdokumentasi.
3. **Risk scoring engine** — memuat model XGBoost final dan menghasilkan probabilitas kelas risiko.
4. **Explainability layer** — menerjemahkan kontribusi fitur menjadi alasan yang bisa dibaca manusia.
5. **Evidence lane** — menghubungkan paket prioritas dengan bukti resmi atau casebook pendukung bila tersedia.
6. **Reviewer workspace** — menyediakan daftar prioritas, kartu penjelasan kasus, dan status tindak lanjut.
7. **Audit trail & governance layer** — menyimpan log prediksi, alasan, artefak model, dan umpan balik reviewer.

![Arsitektur target end-to-end LPSE-X untuk integrasi Phase 3](figures/pipeline-architecture.png)

Dalam rancangan ini, LPSE-X tidak menuntut infrastruktur cloud atau komponen yang sulit diaudit. Model, notebook, dan data split yang sudah disiapkan pada Tahap 2 cukup menjadi fondasi untuk menjalankan scoring batch lokal, lalu menyalurkan hasilnya ke dashboard internal atau laporan prioritas review.

## 4.3 Alur Operasional yang Diusulkan

Agar integrasi tidak berhenti pada demo teknis, berikut alur operasional yang kami usulkan untuk Phase 3.

1. **Snapshot data berkala** diambil dari sistem pengadaan atau folder ekspor resmi.
2. Pipeline lokal menjalankan preprocessing dan scoring terhadap paket yang masuk.
3. Sistem menghasilkan **risk score, kelas risiko, tiga faktor utama, dan narasi Bahasa Indonesia**.
4. Paket dengan skor tertinggi masuk ke antrean reviewer beserta justifikasi model.
5. Jika ada kecocokan dengan evidence lane, kasus dapat dinaikkan ke prioritas pemeriksaan lebih tinggi.
6. Reviewer memberi status lanjut: dipantau, direview mendalam, atau ditutup.
7. Hasil review disimpan sebagai jejak audit sekaligus bahan penguatan label pada iterasi berikutnya.

![Alur inference dan rekomendasi review pada LPSE-X](figures/inference-flow.png)

Secara praktis, alur ini cocok untuk dua mode operasi:

- **batch harian/mingguan** untuk membantu tim pengawasan dengan volume data besar,
- **case-based review** untuk menunjukkan penjelasan satu paket tertentu ketika dibutuhkan presentasi atau investigasi manual.

Nilai penting dari rancangan ini adalah keterhubungan langsung antara **skor → alasan → tindakan review**. Tanpa hubungan itu, model hanya menjadi dashboard angka; dengan hubungan itu, LPSE-X berubah menjadi alat bantu kerja yang benar-benar bisa dipakai.

## 4.4 Integrasi Phase 3 ke Workflow Pengguna

Target pengguna utama LPSE-X adalah fungsi pengawasan dan audit, misalnya tim inspektorat, analis risiko pengadaan, atau unit pengendalian internal yang perlu menyaring banyak paket secara cepat. Karena itu, desain integrasinya harus mengikuti kebiasaan kerja mereka, bukan memaksa pengguna belajar proses baru yang terlalu teknis.

### A. Antarmuka kerja yang disarankan

Phase 3 idealnya menyediakan tiga tampilan inti.

1. **Daftar prioritas** — menampilkan paket dengan skor tertinggi, kelas risiko, dan alasan ringkas.
2. **Kartu kasus** — menampilkan detail paket, faktor dominan, narasi penjelasan, dan tautan bukti pendukung.
3. **Panel tindak lanjut** — mencatat keputusan reviewer dan alasan penutupan atau eskalasi.

### B. Integrasi peran pengguna

| Peran | Kebutuhan utama | Dukungan LPSE-X |
| --- | --- | --- |
| Auditor / reviewer | Menentukan paket mana yang perlu dilihat lebih dulu | daftar prioritas berbasis skor dan explanation |
| Supervisor | Melihat gambaran backlog dan prioritas unit | ringkasan jumlah kasus per level risiko dan status tindak lanjut |
| Tim tata kelola data | Memeriksa konsistensi input dan jejak prediksi | audit trail, manifest fitur, dan artefak model |
| Pengambil keputusan | Melihat dampak dan kualitas sistem secara agregat | dashboard KPI operasional dan ringkasan dampak |

### C. Tahapan integrasi yang realistis

Kami mengusulkan integrasi bertahap berikut agar adopsi tetap rendah risiko.

- **Tahap 3A — pilot internal terbatas**: scoring batch lokal untuk satu instansi atau satu rentang waktu, fokus pada validasi workflow.
- **Tahap 3B — evidence-assisted review**: evidence lane aktif untuk kasus prioritas tinggi dan casebook demonstratif mulai dipakai.
- **Tahap 3C — feedback-enabled governance**: hasil review manual disimpan terstruktur sebagai sumber pembelajaran untuk iterasi model selanjutnya.

Pendekatan bertahap ini lebih defensible dibanding langsung mengklaim kesiapan produksi penuh, karena memberi ruang untuk validasi organisasi, kalibrasi SOP, dan penyesuaian peran manusia di lapangan.

## 4.5 Rancangan Bisnis dan Keberlanjutan Solusi

Pada konteks **Smart Governance & Public Service**, makna “bisnis” tidak selalu identik dengan monetisasi komersial jangka pendek. Untuk LPSE-X, rancangan bisnis yang paling kuat justru berbentuk **model adopsi nilai publik**: solusi memberi penghematan waktu review, meningkatkan kualitas prioritisasi, dan memperkuat akuntabilitas audit.

### A. Value proposition

LPSE-X menawarkan tiga nilai utama bagi institusi.

1. **Efisiensi review** — tim tidak perlu memulai dari seluruh populasi paket, tetapi dari shortlist yang sudah diberi alasan.
2. **Transparansi keputusan awal** — setiap prioritas disertai faktor pendorong dan narasi, sehingga lebih mudah dipertanggungjawabkan.
3. **Fondasi pembelajaran institusional** — hasil review dapat dikumpulkan kembali sebagai umpan balik untuk memperkuat sistem ke depan.

### B. Opsi model implementasi

| Opsi | Bentuk | Kesesuaian untuk LPSE-X |
| --- | --- | --- |
| Internal decision-support tool | dipakai langsung oleh unit pengawasan | **paling realistis** untuk fase awal karena offline-first dan mudah diaudit |
| Managed analytics pilot | dijalankan tim kecil lintas fungsi untuk beberapa instansi | cocok untuk demonstrasi manfaat dan penguatan SOP |
| Platform layanan lebih luas | integrasi ke dashboard pengawasan multi-unit | layak dipertimbangkan setelah validasi pilot dan feedback loop matang |

Untuk tahap kompetisi dan fase awal implementasi, opsi terbaik adalah **internal decision-support tool**. Pilihan ini sejalan dengan ukuran model yang ringan, dependensi lokal, dan kebutuhan kontrol data yang ketat.

### C. Struktur biaya dan keberlanjutan

LPSE-X dirancang hemat infrastruktur karena:

- model akhir berukuran kecil,
- inferensi dapat berjalan di CPU biasa,
- tidak memerlukan API cloud,
- artefak submission sudah mendekati bentuk operasional minimum.

Konsekuensinya, hambatan adopsi awal lebih banyak berada pada **SOP, kualitas data, dan kesiapan reviewer**, bukan pada biaya komputasi. Ini penting untuk narasi bisnis karena menunjukkan bahwa solusi relatif murah untuk diuji, tetapi tetap membutuhkan disiplin tata kelola agar bernilai dalam jangka panjang.

## 4.6 Analisis Dampak Operasional dan Publik

Analisis dampak Phase 3 perlu dibangun dari manfaat yang realistis, bukan klaim bombastis. Dengan melihat karakter output LPSE-X saat ini, kami memproyeksikan empat kelompok dampak utama.

### A. Dampak operasional

- mempercepat penentuan prioritas review,
- mengurangi waktu yang dihabiskan untuk menyaring paket berisiko rendah,
- membantu reviewer fokus pada shortlist yang lebih kaya sinyal risiko.

Pada artefak operasional saat ini, precision pada shortlist teratas terlihat sangat kuat terhadap benchmark heuristik yang digunakan. Ini mendukung hipotesis bahwa sistem efektif sebagai alat **ranking untuk review terbatas**, meskipun tetap harus dibaca dalam konteks weak labels.

![Metrik operasional LPSE-X untuk budget review terbatas](figures/operational_metrics.png)

### B. Dampak kualitas keputusan awal

LPSE-X tidak hanya memberi skor, tetapi juga alasan. Ini berpotensi meningkatkan kualitas diskusi awal reviewer karena prioritas tidak muncul sebagai “angka misterius”. Temuan review manual yang sudah ada menunjukkan bahwa explanation cukup membantu dari sisi actionability, meskipun kualitas kejelasan narasi masih perlu ditingkatkan.

![Ringkasan review manual dan kualitas explanation](figures/manual_review_summary.png)

### C. Dampak tata kelola dan akuntabilitas

Bila diterapkan dengan audit trail yang benar, LPSE-X dapat memperkuat:

- dokumentasi alasan prioritas pemeriksaan,
- konsistensi antar reviewer,
- kemampuan institusi menjelaskan mengapa suatu paket masuk antrean tinjau.

Dampak ini sangat relevan untuk sektor publik karena tujuan utamanya bukan sekadar efisiensi, tetapi juga **keputusan yang dapat dipertanggungjawabkan**.

### D. Dampak publik jangka menengah

Secara lebih luas, sistem seperti LPSE-X dapat membantu mendorong budaya pengawasan yang lebih proaktif. Jika proses triase menjadi lebih cepat dan lebih transparan, institusi berpeluang menangani sinyal risiko lebih dini. Namun, klaim ini tetap harus dibingkai sebagai **potensi dampak**, bukan hasil yang sudah terbukti penuh pada Tahap 2.

## 4.7 Evidence Lane sebagai Penguat Nilai Bisnis

Salah satu elemen pembeda LPSE-X adalah adanya **evidence-backed risk lane**. Dalam desain bisnis dan sistem, komponen ini penting karena meningkatkan nilai solusi dari sekadar “predictive scoring” menjadi **review support yang lebih kredibel**. Ketika bukti resmi tersedia, reviewer tidak hanya melihat skor model, tetapi juga konteks eksternal yang memperkuat prioritas kasus.

![Decision flow empat level untuk prioritisasi dan eskalasi kasus](figures/risk-decision-flow.png)

Pada Phase 3, komponen ini dapat berfungsi sebagai:

1. mekanisme eskalasi untuk kasus tertentu,
2. jembatan antara screening kuantitatif dan investigasi kualitatif,
3. dasar untuk menyusun casebook atau portofolio kasus prioritas.

Dari perspektif bisnis/adopsi, evidence lane membuat solusi lebih mudah diterima karena reviewer cenderung lebih percaya pada sistem yang tidak hanya mengeluarkan skor, tetapi juga memberi konteks dan jalur verifikasi tambahan.

## 4.8 Risiko Implementasi dan Strategi Mitigasi

Agar rancangan Phase 3 tetap kredibel, proposal ini perlu menjabarkan risiko implementasi secara terbuka.

| Risiko | Dampak | Mitigasi yang diusulkan |
| --- | --- | --- |
| Weak labels dan circularity risk | model terlihat sangat baik pada benchmark internal tetapi belum identik dengan outcome lapangan | gunakan hasil model sebagai prioritas review, bukan keputusan final; lanjutkan label strengthening |
| Data quality tidak konsisten | explanation atau skor dapat turun kualitasnya pada instansi tertentu | siapkan data-quality checks dan fallback field sebelum scoring |
| False positive pada kasus ambigu | reviewer bisa kehilangan waktu pada kasus yang ternyata tidak kritis | gunakan threshold berbasis kapasitas review dan wajibkan review manusia |
| Drift lintas waktu | performa dapat berubah ketika pola pengadaan berubah | lakukan monitoring berkala dan retraining terkontrol |
| Resistensi organisasi | sistem tidak dipakai walau secara teknis baik | mulai dari pilot kecil, KPI jelas, dan integrasi ke workflow yang sudah dikenal |

Mitigasi ini menegaskan kembali posisi ilmiah LPSE-X: solusi ini menjanjikan, tetapi hanya akan bernilai bila dioperasikan dengan governance yang benar.

## 4.9 KPI Keberhasilan Phase 3

Keberhasilan integrasi sebaiknya diukur dengan kombinasi KPI teknis dan operasional. Kami mengusulkan indikator berikut.

1. **Precision@K pada subset yang direview manusia** untuk mengukur kualitas shortlist.
2. **Turnaround time review** sebelum dan sesudah LPSE-X dipakai.
3. **Explanation usefulness score** dari reviewer untuk mengukur apakah narasi benar-benar membantu.
4. **Evidence match rate** pada kasus prioritas tinggi.
5. **Jumlah kasus yang terdokumentasi lengkap** dari skor hingga tindak lanjut.

Dengan KPI tersebut, Phase 3 tidak dinilai hanya dari akurasi model, tetapi dari seberapa baik sistem membantu pekerjaan nyata dan menghasilkan jejak akuntabel.

## 4.10 Kesimpulan Bab

Bab ini menempatkan LPSE-X secara eksplisit sebagai **rancangan sistem dan bisnis untuk integrasi Phase 3**, bukan sekadar eksperimen model. Rancangan yang kami usulkan bersifat modular, offline-first, human-in-the-loop, dan berorientasi pada nilai publik. Nilai utamanya bukan mengganti auditor, melainkan membantu mereka bekerja lebih cepat, lebih konsisten, dan lebih mampu menjelaskan alasan prioritas review.

Secara bisnis, bentuk adopsi yang paling realistis adalah decision-support tool internal dengan pilot bertahap. Secara dampak, manfaat utamanya terletak pada efisiensi triase, transparansi keputusan awal, dan penguatan akuntabilitas. Secara ilmiah, proposal ini tetap menjaga kejujuran: LPSE-X sudah cukup matang untuk didemokan dan diintegrasikan secara terbatas, tetapi masih memerlukan penguatan label, evaluasi lapangan, dan governance implementasi sebelum dapat diklaim sebagai sistem yang sepenuhnya matang.
