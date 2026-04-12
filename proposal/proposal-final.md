# Proposal LPSE-X — Find IT! 2026 Tahap 2

## Identitas Dokumen

- **Nama Tim:** BismillahFirstTry-Phase2
- **Nama Solusi:** LPSE-X
- **Track:** Track C — *The Explainable Oracle (Predictive Analytics)*
- **Subtema:** Smart Governance & Public Service
- **Posisi Solusi:** explainable procurement-risk screening system untuk membantu prioritisasi audit pengadaan publik

## Ringkasan Eksekutif

LPSE-X adalah sistem prediktif offline untuk membantu menyaring paket pengadaan yang layak diperiksa lebih dulu. Sistem ini dibangun dengan model XGBoost tabular, dilengkapi SHAP untuk explainability, dan menghasilkan penjelasan Bahasa Indonesia agar hasil prediksi dapat dibaca manusia. Seluruh pipeline dirancang patuh terhadap constraint Track C: explainability wajib, human-readable explanation, anti-black-box, validasi anti-leakage, dan offline total.

Secara ilmiah, proposal ini mengambil posisi yang konservatif dan defensible. Metrik utama memang tinggi pada benchmark saat ini, tetapi tetap dilaporkan sebagai performa terhadap **heuristic risk labels**, bukan sebagai bukti final keberhasilan mendeteksi kasus korupsi yang sudah terverifikasi untuk seluruh dataset. Karena itu, kekuatan utama LPSE-X pada Tahap 2 adalah kombinasi antara **kepatuhan teknis, kualitas presentasi, dan kejujuran evaluasi**.

## Daftar Isi

1. Bab 1 — Pendahuluan
2. Bab 2 — Metodologi
3. Bab 3 — Kepatuhan dan Implementasi
4. Bab 4 — Rancangan Sistem & Bisnis (Integrasi Phase 3 & Analisis Dampak)


---

# BAB 1: PENDAHULUAN

## 1.1 Latar Belakang

Pengadaan barang dan jasa pemerintah adalah domain dengan nilai transaksi besar, kompleksitas proses tinggi, dan konsekuensi publik yang langsung terasa. Pada praktiknya, auditor dan pengawas tidak kekurangan data; yang kurang adalah **alat triase yang mampu menyaring ribuan paket secara konsisten, cepat, dan dapat dijelaskan**. Tanpa alat bantu yang explainable, pengawasan cenderung kembali pada pemeriksaan manual yang mahal, lambat, dan sulit diprioritaskan.

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


---

# BAB 2: METODOLOGI

## 2.1 Gambaran Umum Pipeline

LPSE-X dibangun sebagai pipeline offline untuk mengubah data pengadaan publik menjadi **skor risiko yang dapat dijelaskan**. Alur kerjanya terdiri atas enam tahap inti:

1. ingest dan pembersihan data OCDS;
2. pemisahan `train_data` dan `test_data` pada level raw data;
3. rekayasa fitur yang split-aware;
4. pembentukan heuristic risk labels;
5. pelatihan model XGBoost dan kalibrasi probabilitas;
6. generasi explanation berbasis SHAP dan narasi Bahasa Indonesia.

Untuk kebutuhan juri, metodologi inti LPSE-X diringkas pada diagram berikut agar hubungan antara data, model, dan keluaran explainability dapat dibaca dengan cepat dan jelas.

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

Temuan kualitas data yang paling penting untuk dibaca juri adalah sebagai berikut.

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

Konsekuensi desain ini adalah setiap angka evaluasi yang dipakai sebagai landasan integrasi pada Bab 4 tetap berasal dari pemisahan yang defensible terhadap kebocoran data.

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

Struktur submission yang disiapkan untuk juri mengikuti constraint umum kompetisi.

| Artefak | Peran |
| --- | --- |
| `training.ipynb` | menunjukkan pelatihan model dan log yang terlihat |
| `inference.ipynb` | menunjukkan alur inferensi yang bersih dan siap demo |
| `train_data/` | artefak data latih hasil split raw |
| `test_data/` | artefak data uji yang terpisah |
| `models/xgb_model.ubj` / `models/xgb_model.onnx` | model final untuk deployment lokal |
| `requirements.txt` | dependency agar eksperimen dapat dijalankan ulang |
| `proposal/figures/` | visual evaluasi dan bukti presentasi |

Dengan desain ini, juri tidak perlu menebak alur kerja proyek: semua komponen utama tersedia sebagai artefak lokal yang eksplisit.

## 2.9 Filosofi Evaluasi

Agar hasil mudah dibaca secara profesional, evaluasi pada proposal ini dibagi menjadi empat lapis:

1. **headline benchmark** pada test split heuristik,
2. **calibration dan confusion analysis** untuk membaca trade-off operasional,
3. **robustness / proxy-reduced validation** untuk mengukur circularity risk,
4. **manual review, external validation, dan official evidence lane** untuk memperkaya bukti di luar sekadar angka terhadap weak labels.

Dengan demikian, Bab 4 tidak hanya mengusulkan integrasi sistem, tetapi juga menautkannya pada bukti evaluasi yang dibaca secara hati-hati dan tidak berlebihan.


---

# BAB 3: KEPATUHAN DAN IMPLEMENTASI

## 3.1 Matriks Kepatuhan Track C

Bab ini ditulis khusus untuk memenuhi ketentuan panitia bahwa **Bab 3 harus menjelaskan secara rinci bagaimana solusi mematuhi setiap constraint track**. Tabel berikut menjadi ringkasan paling langsung untuk juri.

| Kode | Constraint resmi | Implementasi pada LPSE-X | Bukti utama |
| --- | --- | --- | --- |
| C-C1 | Explainability wajib | Prediksi dijelaskan dengan SHAP global dan lokal | `src/explain.py`, `figures/shap_summary.png` |
| C-C2 | Output penjelasan harus human-readable | Inference menghasilkan narasi Bahasa Indonesia dengan faktor utama dan arah pengaruh | `src/narrative.py`, `inference.ipynb` |
| C-C3 | Anti-black-box | Model utama adalah XGBoost tabular yang dapat diinspeksi; explainability bukan tempelan kosmetik | `src/model.py`, `models/xgb_model.ubj`, `models/xgb_model.onnx` |
| C-C4 | Wajib membuktikan tidak ada data leakage | Raw split dilakukan sebelum preprocessing; test tidak dipakai untuk tuning atau kalibrasi | `src/split.py`, `train_data/raw.parquet`, `test_data/raw.parquet`, `data/processed/split_metadata.json` |
| C-C5 | Offline total | Training, inferensi, dan explainability berjalan lokal tanpa API cloud | `training.ipynb`, `inference.ipynb`, `requirements.txt` |

Tabel di atas adalah checklist kepatuhan utama untuk juri: setiap constraint Track C dipetakan langsung ke artefak implementasi yang bisa diperiksa pada repo submission.

## 3.2 Pembuktian per Constraint

### C-C1 — Explainability wajib

LPSE-X memenuhi constraint explainability dengan menjadikan SHAP sebagai bagian inti pipeline, bukan sekadar lampiran presentasi. Model tidak hanya mengeluarkan probabilitas kelas, tetapi juga daftar faktor yang paling mendorong hasil prediksi. Ini memungkinkan reviewer mengetahui **mengapa** sebuah paket diprioritaskan.

![Ringkasan faktor global berbasis SHAP](figures/shap_summary.png)

### C-C2 — Output penjelasan yang dapat dibaca manusia

Track C menuntut penjelasan yang dapat dibaca manusia untuk setiap prediksi. Karena itu, LPSE-X menyediakan dua lapisan output pada jalur inference:

1. daftar minimal tiga faktor teratas,
2. arah pengaruh masing-masing faktor terhadap skor,
3. narasi Bahasa Indonesia yang menjelaskan hasil dalam bentuk kalimat operasional.

Dengan desain ini, reviewer tidak perlu menginterpretasi angka mentah sendiri. Outputnya sudah siap dipakai sebagai bahan prioritisasi atau diskusi awal.

### C-C3 — Anti-black-box

Kami sengaja tidak menggunakan arsitektur yang sepenuhnya opaque untuk submission Tahap 2. XGBoost dipilih karena lebih cocok untuk data tabular dan lebih mudah dipertanggungjawabkan pada konteks kebijakan publik. Bila juri menelusuri artefaknya, mereka dapat memeriksa:

- fitur input yang digunakan,
- manifest fitur dan split,
- model akhir yang diekspor,
- metrik evaluasi,
- visual explainability.

Dengan demikian, sistem tetap bisa diaudit dari input hingga output.

### C-C4 — Validasi data leakage

Ini adalah constraint paling kritis, dan LPSE-X memenuhinya secara eksplisit.

1. Folder `train_data/` dan `test_data/` dibentuk dari raw split terlebih dahulu.
2. Feature engineering dilakukan terpisah setelah raw split sudah final.
3. Hyperparameter optimization, thresholding, dan temperature scaling hanya memakai data di sisi train/dev.
4. Fitur historis tidak diizinkan melihat masa depan.

Ringkasan split yang dipakai pada repo saat ini adalah:

- train: **372.150** baris
- test: **93.034** baris
- split boundary: **2023-03-10 07:27:51 UTC**

Desain ini adalah inti kepatuhan C-C4 dan menjadi salah satu alasan utama solusi kami tetap defensible.

### C-C5 — Offline total

Seluruh komponen inti berjalan lokal:

- training model,
- inference,
- SHAP explanation,
- narasi Bahasa Indonesia,
- ekspor model `.ubj` dan `.onnx`.

LPSE-X **tidak** menggunakan API inferensi cloud, API explainability, maupun layanan generative AI eksternal di pipeline utama. Kepatuhan ini penting bukan hanya untuk aturan kompetisi, tetapi juga selaras dengan semangat digital sovereignty pada tema hackathon.

## 3.3 Kesiapan Paket Submission

Selain constraint Track C, panitia juga mensyaratkan struktur artefak yang jelas. Paket Tahap 2 untuk LPSE-X disusun agar juri mudah memeriksa ulang komponen utama berikut.

| Artefak submission | Status peran |
| --- | --- |
| `proposal-final.md` / PDF final | narasi proposal yang siap diekspor ke PDF |
| `training.ipynb` | jalur pelatihan dengan log yang terlihat |
| `inference.ipynb` | jalur inferensi yang lebih bersih dan demo-friendly |
| `train_data/` dan `test_data/` | bukti split fisik yang terpisah |
| file model final | model siap dipakai secara lokal |
| `requirements.txt` | reproduksibilitas environment |

Di lane submission resmi, kami juga mengunci tiga aturan tambahan agar paket tidak menimbulkan ambiguitas saat dinilai.

1. **Single-model submission** — hanya satu model utama yang dikirim, yaitu XGBoost multiclass yang diekspor ke `model_risk.ubj` dan `model_risk.onnx`; tidak ada ensemble terpisah atau model cloud pendamping pada paket final.
2. **Penamaan resmi panitia** — folder/repo final dikunci ke `BismillahFirstTry-Phase2_Tahap2_FindIT2026`, sedangkan proposal PDF final dikunci ke `Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf`.
3. **Bundle judge-safe** — isi folder dibatasi pada proposal, notebook, model, data split, source code inti, dan figures yang benar-benar dipakai juri.

Pendekatan ini sengaja dibuat judge-safe: yang ditampilkan adalah artefak yang benar-benar dibutuhkan untuk evaluasi, bukan seluruh histori eksperimen internal.

## 3.4 Pengendalian Risiko Teknis

Agar implementasi tetap stabil di bawah tekanan waktu kompetisi, beberapa prinsip pengendalian risiko dipertahankan.

1. **Fallback tetap tersedia** — bila jalur counterfactual yang lebih berat tidak stabil, sistem tetap bisa menjelaskan hasil melalui SHAP dan narasi deterministik.
2. **Model artefak ganda** — ekspor `.ubj` dan `.onnx` mengurangi risiko kegagalan demo pada satu format saja.
3. **Sumber kebenaran artefak** — bila ada perbedaan antara narasi proposal dan implementasi, artefak repo menjadi acuan utama.
4. **No overclaim policy** — output disebut sebagai risk screening, bukan putusan fraud.

## 3.5 Posisi Ilmiah yang Jujur

Kepatuhan teknis tidak boleh membuat proposal kehilangan kejujuran ilmiah. Karena itu, LPSE-X secara eksplisit menyatakan bahwa:

1. metrik utama masih dievaluasi terhadap **heuristic risk labels**,
2. audit robustness menunjukkan circularity risk yang signifikan,
3. evidence lane dan manual review menambah bukti yang berguna, tetapi belum mengubah sistem menjadi oracle hukum final.

Justru dengan menyatakan batasan ini secara terbuka, proposal menjadi lebih kuat di hadapan juri: solusi terlihat serius, patuh constraint, dan tidak menjual klaim berlebihan.

## 3.6 Bukti Verifikasi Implementasi

Status implementasi yang relevan untuk Tahap 2 dapat diringkas sebagai berikut.

- pipeline split-aware tersedia dan terdokumentasi;
- model akhir sudah tersimpan dan dapat diekspor untuk inferensi lokal;
- visual evaluasi utama sudah tersedia di `proposal/figures/`;
- notebook training dan inference sudah menjadi artefak submission;
- proposal kini memetakan setiap constraint Track C ke artefak nyata.

![Peta artefak yang diterima juri pada paket submission](figures/submission-package-map.png)


---


---

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
