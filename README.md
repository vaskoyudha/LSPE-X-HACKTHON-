# LPSE-X — Find IT! 2026 Track C

LPSE-X adalah **sistem triase risiko pengadaan publik berbasis Explainable AI** untuk membantu prioritisasi audit pengadaan pemerintah Indonesia. Repositori ini telah dirapikan agar pembaca dapat langsung menemukan artefak utama tanpa harus menelusuri dokumen kerja internal.

## Artefak Utama

### Proposal
- `Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf` — **file utama yang siap diunggah**
- `Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.docx` — versi DOCX proposal final
- `proposal/proposal-final.md` — sumber proposal final Bab 1–4

### Notebook
- `training.ipynb` — notebook training dengan log/output terlihat
- `inference.ipynb` — notebook inference dengan prediksi, explanation, dan output ONNX terlihat

### Model
- `models/xgb_model.ubj`
- `models/xgb_model.onnx`

### Dataset
- `train_data/`
- `test_data/`

## Ringkasan Cepat Model

- Data riil multi-tahun OCDS Indonesia
- Total baris usable: **465.184**
- Train rows: **372.150**
- Test rows: **93.034**
- Accuracy: **0,9899**
- Macro-F1: **0,9830**
- Precision@100: **1,00**

## Struktur Data

### `train_data/`
- `raw.parquet` — data mentah hasil split train
- `features.parquet` — fitur hasil rekayasa fitur
- `labels.parquet` — label risiko untuk training/evaluasi internal

### `test_data/`
- `raw.parquet` — data mentah hasil split test
- `features.parquet` — fitur hasil rekayasa fitur
- `labels.parquet` — label risiko untuk evaluasi held-out

Pemisahan `train_data` dan `test_data` dipertahankan sebagai bukti kontrol **anti-data-leakage**.

## Cara Cek Cepat

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python -m jupyter nbconvert --to notebook --execute training.ipynb --output /tmp/training-check.ipynb
python -m jupyter nbconvert --to notebook --execute inference.ipynb --output /tmp/inference-check.ipynb
```

## Catatan Penting

- LPSE-X adalah **alat triase risiko pengadaan**, bukan mesin keputusan hukum final.
- Pipeline berjalan **sepenuhnya secara lokal** tanpa layanan cloud.
- Proposal final memuat **Bab 1–4 lengkap**.
- Bab 3 memetakan kepatuhan terhadap setiap constraint Track C.
- Bab 4 menjelaskan integrasi Phase 3, model adopsi, KPI, dan analisis dampak.

## Keterbatasan yang Diakui

- benchmark masih menggunakan **heuristic risk labels**
- masih ada **circularity risk** yang diakui secara eksplisit
- masih diperlukan validasi lapangan dan penguatan label lebih lanjut
