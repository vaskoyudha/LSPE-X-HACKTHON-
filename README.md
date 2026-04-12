# LPSE-X

**LPSE-X** adalah sistem triase risiko pengadaan publik berbasis Explainable AI untuk **Find IT! 2026 Track C — The Explainable Oracle**.

Repositori ini disusun agar juri atau reviewer dapat langsung menemukan artefak utama tanpa harus menelusuri seluruh histori eksperimen.

## Ringkasan Singkat

LPSE-X membantu reviewer memprioritaskan paket pengadaan yang paling layak diperiksa lebih dulu. Sistem ini berjalan sepenuhnya secara lokal, menggunakan model tabular berbasis **XGBoost**, menyediakan penjelasan **SHAP**, dan menghasilkan narasi Bahasa Indonesia agar alasan di balik skor risiko dapat dibaca manusia.

Posisi ilmiahnya sengaja dibuat tegas tetapi jujur:
- ini adalah **alat triase risiko pengadaan**,
- bukan mesin keputusan hukum,
- dan metrik utamanya dibaca terhadap **heuristic risk labels**, bukan outcome fraud final.

## File yang Paling Penting untuk Dinilai

### Proposal
- `proposal/proposal-final.pdf` — proposal final lengkap Bab 1–4
- `proposal/proposal-final.docx` — versi DOCX proposal final
- `proposal/proposal-final.md` — sumber utama proposal

### Notebook
- `training.ipynb` — notebook training dengan output/log terlihat
- `inference.ipynb` — notebook inference dengan prediksi dan penjelasan yang terlihat

### Model
- `models/xgb_model.ubj`
- `models/xgb_model.onnx`

### Dataset
- `train_data/`
- `test_data/`

## Ringkasan Benchmark

Benchmark saat ini menggunakan **slice data riil multi-tahun OCDS Indonesia**.

Ringkasan utama:
- total baris usable: **465.184**
- train rows: **372.150**
- test rows: **93.034**
- buyer unik: **618**
- supplier unik: **60.976**

Metrik utama pada held-out test (`models/metrics.json`):
- Accuracy: **0,9899**
- Macro-F1: **0,9830**
- Weighted-F1: **0,9898**

Nilai operasional utama (`models/operational_metrics.json`):
- Precision@100: **1,00**

## Struktur Repo

```text
src/            modul inti pipeline
models/         artefak model dan metrik
train_data/     split data latih
test_data/      split data uji
proposal/       proposal final dan visual pendukung
tests/          pengujian terfokus
```

## Cara Menjalankan Secara Singkat

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m jupyter nbconvert --to notebook --execute training.ipynb --output /tmp/training-check.ipynb
python -m jupyter nbconvert --to notebook --execute inference.ipynb --output /tmp/inference-check.ipynb
```

## Catatan Penting untuk Reviewer

- `train_data` dan `test_data` dipisahkan untuk menjaga kontrol **anti-data-leakage**
- model dieksekusi **tanpa layanan cloud**
- proposal Bab 3 secara khusus memetakan kepatuhan terhadap setiap constraint Track C
- proposal Bab 4 menjelaskan integrasi Phase 3, dampak, dan model adopsi

## Keterbatasan yang Diakui

LPSE-X adalah proposal yang kuat untuk **triase risiko** dan **prioritisasi audit awal**, tetapi belum boleh dibaca sebagai sistem fraud detection final yang menyelesaikan seluruh masalah lapangan. Proposal ini secara eksplisit mengakui:

- penggunaan **heuristic risk labels**
- masih adanya **circularity risk**
- kebutuhan validasi lapangan dan penguatan label lebih lanjut
