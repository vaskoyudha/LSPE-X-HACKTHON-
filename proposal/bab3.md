# BAB 3: KEPATUHAN DAN IMPLEMENTASI

## 3.1 Kepatuhan terhadap Constraint Kompetisi

Bab ini menyajikan pemetaan langsung antara implementasi LPSE-X dengan constraint wajib pada Track C.

### C-C1 — Explainability wajib

Constraint ini dipenuhi melalui penggunaan SHAP pada `src/explain.py`. Sistem menghasilkan faktor utama yang memengaruhi prediksi, baik untuk analisis satu baris maupun ringkasan global. Artefak bukti yang relevan adalah `proposal/figures/shap_summary.png` dan fungsi `explain_single(...)`.

### C-C2 — Output penjelasan yang dapat dibaca manusia

Constraint ini dipenuhi melalui dua lapisan output:

1. daftar faktor SHAP dengan arah pengaruh (`factors`),
2. narasi Bahasa Indonesia pada `src/narrative.py`.

Dengan demikian, auditor non-teknis dapat melihat bukan hanya skor risiko, tetapi juga alasan utama di balik prediksi.

### C-C3 — Anti-black-box

Model yang digunakan adalah XGBoost, bukan model opaque tanpa kontrol explainability. Selain itu, seluruh jalur inferensi utama dapat diaudit melalui:

- fitur input yang eksplisit,
- file parameter terbaik,
- metrik final,
- penjelasan SHAP,
- narasi terstruktur.

### C-C4 — Validasi data leakage

Constraint ini merupakan titik paling kritis dari desain sistem. Implementasi menggunakan kebijakan split-aware:

- `src/split.py` melakukan pemisahan raw train/test sebelum feature engineering,
- `test_data/` tidak dipakai untuk HPO,
- `test_data/` tidak dipakai untuk temperature scaling,
- fitur Tier 2 hanya memakai histori masa lalu.

Validasi ini diperkuat oleh test suite leakage guard dan hasil temporal split yang tidak overlap.

### C-C5 — Offline total

Seluruh komponen inti dapat berjalan lokal:

- pelatihan XGBoost,
- explainability SHAP,
- narasi Bahasa Indonesia,
- model `.ubj`,
- model `.onnx`,
- notebook training dan inference.

Tidak ada cloud inference API pada jalur training maupun inferensi.

## 3.2 Arsitektur Implementasi

Arsitektur kode dibagi ke beberapa modul yang memiliki kontrak jelas:

- `src/data.py` — akuisisi, flattening, cleaning, quality report
- `src/split.py` — split temporal eksternal dan internal
- `src/features.py` — 30 feature families split-aware
- `src/labels.py` — red-flag heuristic labeling + calibration helpers
- `src/model.py` — training, evaluation, calibration, ONNX/export helpers
- `src/explain.py` — SHAP, explain_single, counterfactual path
- `src/narrative.py` — render narasi Bahasa Indonesia

Struktur ini membantu pemisahan tanggung jawab serta memudahkan verifikasi oleh panel software engineering dan architect.

## 3.3 Status Gate Implementasi

### Gate 0 — Foundation

Sudah terpenuhi:

- scaffold proyek tersedia,
- marker pytest tersedia,
- import package berhasil,
- `pytest -m p0` dan `pytest -q` berjalan pada environment proyek.

### Gate 1 — Data freeze dan compliance split

Sudah terpenuhi secara implementasi internal:

- raw split train/test tersedia,
- metadata split tersedia,
- feature generation berjalan dari split raw,
- leakage guard test hijau.

### Gate 2 — Model baseline locked

Sudah terpenuhi:

- parameter terbaik tersimpan,
- metrik final tersimpan di `models/metrics.json`,
- kalibrasi tersimpan di `models/calibration.json`.

### Gate 3 — XAI complete

Sudah terpenuhi secara pipeline:

- explainability SHAP tersedia,
- figure SHAP tersedia,
- narasi Bahasa Indonesia tersedia,
- counterfactual fallback tersedia,
- jalur export `.onnx` dan `.ubj` tersedia untuk inferensi lokal.

### Gate 4 — Notebook complete

Dipenuhi melalui dua notebook terpisah:

- `training.ipynb`
- `inference.ipynb`

Notebook ini dirancang untuk mengeksekusi pipeline training dan inference secara lokal.

### Gate 5 — Submission ready

Komponen submission-ready difokuskan pada:

- proposal Bab 1–4,
- proposal final markdown,
- proposal final PDF,
- notebook training/inference,
- model hasil training,
- requirements dengan exact pins.

## 3.4 Pengendalian Risiko Teknis

Beberapa kill-switch dari rencana awal tetap dipertahankan:

1. Jika path focal loss tidak stabil, sistem tetap aman memakai class-weighted XGBoost.
2. Jika review calibration high-confidence kurang dari 80 sampel, temperature scaling dapat dimatikan.
3. Jika DiCE terlalu berat atau timebox terlampaui, counterfactual SHAP tetap tersedia.
4. Jika proposal dan implementasi berbeda, artefak implementasi menjadi sumber kebenaran.

## 3.5 Bukti Verifikasi

Verifikasi terkini pada branch kerja menunjukkan:

- `pytest -q` → 106 passed, 5 skipped
- `python3 -m compileall src tests scripts` → passed
- `git diff --check` → passed

Bukti ini menegaskan bahwa basis implementasi stabil untuk dibawa ke tahap finalisasi submission.
