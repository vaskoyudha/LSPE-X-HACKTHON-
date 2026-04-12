# BismillahFirstTry-Phase2 — Find IT! 2026 Tahap 2 Submission Bundle

Repositori ini adalah **bundle submission judge-safe** untuk Tahap 2 Find IT! 2026.
Sistem yang diajukan adalah **LPSE-X**, yaitu prototipe *offline* untuk skrining risiko pengadaan publik dengan **single-model XGBoost**, **explainability berbasis SHAP**, dan **narasi penjelasan yang dapat dibaca manusia**.

## Isi bundle

- `Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf` — proposal final untuk diunggah ke Google Form
- `proposal_preview.md` — versi markdown agar isi proposal cepat dibaca di GitHub
- `training.ipynb` — notebook pelatihan model
- `inference.ipynb` — notebook inferensi/penggunaan model
- `model_risk.ubj` dan `model_risk.onnx` — artefak model untuk inferensi lokal
- `train_data/` dan `test_data/` — data mentah/fitur/label yang sudah dipisah untuk mencegah *data leakage*
- `src/` — modul Python inti untuk fitur, model, penjelasan, dan evaluasi
- `figures/` — grafik dan diagram yang dipakai di proposal
- `requirements.txt` — dependensi Python

## Struktur dataset

Folder dataset sengaja dipisahkan menjadi dua bagian utama agar sesuai dengan aturan submission dan kontrol *anti-leakage*:

- `train_data/` — seluruh artefak data untuk proses pelatihan
- `test_data/` — seluruh artefak data untuk evaluasi akhir

Di masing-masing folder terdapat tiga file:

- `raw.parquet` — data hasil split mentah sebelum preprocessing lanjutan
- `features.parquet` — data fitur hasil feature engineering
- `labels.parquet` — label risiko untuk eksperimen/modeling

Dengan struktur ini, juri dapat melihat dengan jelas bahwa data pelatihan dan data pengujian dipisahkan sebagai folder yang berbeda, lalu setiap tahap penting (*raw*, *features*, *labels*) tetap tersedia di dalam masing-masing folder.

## Cara menjalankan secara lokal

1. Buat virtual environment Python 3.11+.
2. Install dependensi:
   ```bash
   pip install -r requirements.txt
   ```
3. Buka `inference.ipynb` untuk menjalankan inferensi lokal.
4. Bila ingin meninjau proses pelatihan, buka `training.ipynb`.

## Catatan untuk juri

- Submission ini menggunakan **single-model track**.
- Seluruh alur inferensi dan explainability dirancang untuk berjalan **tanpa layanan cloud**.
- Penjelasan prediksi disediakan dalam bentuk faktor-faktor utama yang memengaruhi skor risiko.
- Pemisahan `train_data/` dan `test_data/` dipertahankan sebagai bukti kontrol *anti-leakage*.

## Penamaan resmi

- Nama folder/repo: `BismillahFirstTry-Phase2_Tahap2_FindIT2026`
- Nama proposal PDF: `Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf`
