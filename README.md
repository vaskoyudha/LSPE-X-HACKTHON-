# LPSE-X — Explainable AI for Procurement Risk Triage

LPSE-X adalah proyek **Explainable AI untuk triase risiko pengadaan publik** yang dikembangkan untuk **Find IT! 2026 Track C — The Explainable Oracle (Predictive Analytics)**. Fokus utamanya adalah membantu proses **prioritisasi audit pengadaan pemerintah Indonesia** melalui model tabular yang transparan, dapat dijelaskan, dan berjalan sepenuhnya secara lokal.

Repositori ini dirapikan untuk menampilkan artefak utama secara langsung: proposal final, notebook training dan inference, model akhir, serta dataset train/test yang dipisahkan untuk menjaga kontrol anti-data-leakage.

---

## 1. Apa yang Dikerjakan LPSE-X

LPSE-X tidak diposisikan sebagai mesin keputusan hukum final. Sistem ini dirancang sebagai **alat triase risiko** yang membantu reviewer menjawab pertanyaan awal yang paling mahal dalam workflow audit:

> paket mana yang perlu diperiksa lebih dulu, dan mengapa?

Untuk itu, LPSE-X menggabungkan:
- **model XGBoost** untuk prediksi risiko pada data tabular,
- **SHAP** untuk menunjukkan faktor utama yang mendorong skor,
- **narasi Bahasa Indonesia** agar hasil model dapat dibaca manusia,
- **split train/test yang terpisah** agar evaluasi tetap defensible dari sisi kebocoran data.

---

## 2. Artefak Utama di Repo Ini

### Proposal
- `proposal/proposal-final.pdf` — proposal final lengkap Bab 1–4
- `proposal/proposal-final.docx` — versi DOCX proposal final
- `proposal/proposal-final.md` — sumber proposal final

### Notebook
- `training.ipynb` — notebook pelatihan model dengan log/output yang terlihat
- `inference.ipynb` — notebook inferensi dengan hasil prediksi, penjelasan, narasi, dan output ONNX yang terlihat

### Model
- `models/xgb_model.ubj`
- `models/xgb_model.onnx`

### Dataset
- `train_data/`
- `test_data/`

---

## 3. Ringkasan Data dan Benchmark

Benchmark utama LPSE-X menggunakan **slice data riil multi-tahun OCDS Indonesia** yang telah diproses lokal.

Ringkasan data:
- total baris usable: **465.184**
- train rows: **372.150**
- test rows: **93.034**
- buyer unik: **618**
- supplier unik: **60.976**

Ringkasan performa utama (`models/metrics.json`):
- Accuracy: **0,9899**
- Macro-F1: **0,9830**
- Weighted-F1: **0,9898**

Ringkasan nilai operasional (`models/operational_metrics.json`):
- Precision@100: **1,00**

Artinya, pada benchmark saat ini, LPSE-X sangat kuat dalam **mengurutkan shortlist prioritas** berdasarkan sinyal risiko heuristik yang digunakan.

---

## 4. Struktur Dataset

### `train_data/`
- `raw.parquet` — data mentah hasil split train
- `features.parquet` — hasil rekayasa fitur untuk training
- `labels.parquet` — label risiko untuk pembelajaran dan evaluasi internal

### `test_data/`
- `raw.parquet` — data mentah hasil split test
- `features.parquet` — hasil rekayasa fitur untuk evaluasi held-out
- `labels.parquet` — label risiko untuk evaluasi akhir

Pemisahan `train_data` dan `test_data` dipertahankan sebagai bukti kontrol **anti-data-leakage**.

---

## 5. Struktur Proyek

```text
src/            modul inti pipeline
models/         artefak model dan metrik
train_data/     split data latih
test_data/      split data uji
proposal/       proposal final dan visual pendukung
tests/          pengujian terfokus
scripts/        utilitas build, export, dan evaluasi
```

---

## 6. Cara Menjalankan dengan Cepat

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m jupyter nbconvert --to notebook --execute training.ipynb --output /tmp/training-check.ipynb
python -m jupyter nbconvert --to notebook --execute inference.ipynb --output /tmp/inference-check.ipynb
```

Jika ingin mengecek pipeline tambahan:

```bash
python scripts/run_diagnostics.py
```

---

## 7. Catatan Metodologis Penting

Beberapa hal penting untuk dipahami saat membaca hasil LPSE-X:

1. Sistem ini adalah **triase risiko pengadaan**, bukan vonis fraud final.
2. Seluruh pipeline berjalan **sepenuhnya secara lokal** tanpa layanan cloud.
3. Proposal final memuat **Bab 1–4 lengkap**.
4. Bab 3 memetakan kepatuhan terhadap setiap constraint Track C.
5. Bab 4 menjelaskan integrasi Phase 3, model adopsi, KPI, dan analisis dampak.

---

## 8. Keterbatasan yang Diakui

Repositori ini secara eksplisit mengakui bahwa:
- benchmark masih menggunakan **heuristic risk labels**,
- masih ada **circularity risk** yang perlu dibaca dengan hati-hati,
- validasi lapangan dan penguatan label tetap dibutuhkan untuk tahap berikutnya.

Dengan kata lain, kekuatan utama LPSE-X ada pada kombinasi antara:
- relevansi masalah,
- kepatuhan teknis,
- transparansi model,
- dan nilai operasional untuk prioritisasi audit awal.
