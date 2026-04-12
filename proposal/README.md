# Proposal Workspace

Folder ini menyimpan sumber proposal Tahap 2 untuk LPSE-X.

## File utama

- `bab1.md` — pendahuluan dan framing judge-facing
- `bab2.md` — metodologi, artefak, dan reproducibility
- `bab3.md` — pembuktian kepatuhan Track C
- `bab4.md` — hasil, visual evaluasi, dan keterbatasan
- `proposal-final.md` — dokumen gabungan yang siap diekspor menjadi PDF
- `figures/` — seluruh visual lokal yang dipakai proposal

## Aturan kerja untuk proposal

1. **Sumber kebenaran** proposal final ada di markdown, bukan di PDF hasil ekspor.
2. Semua referensi gambar lokal harus memakai path relatif `figures/...` agar tetap valid dari dalam folder `proposal/`.
3. Placeholder visual untuk diagram baru sengaja ditulis sebagai catatan teks agar penyempurnaan desain tidak memblokir finalisasi isi proposal.
4. Narasi proposal harus menyebut LPSE-X sebagai **risk screening / triage system**, bukan alat keputusan hukum final.
5. Bab 3 wajib tetap eksplisit memetakan semua constraint Track C ke artefak implementasi.

## Ekspor

Setelah isi final disetujui, `proposal-final.md` dapat diekspor ke PDF menggunakan script repo berikut agar styling dan penamaannya konsisten:

```bash
python3 scripts/export_proposal_pdf.py
```

PDF tetap menjadi artefak submission utama. Bila dibutuhkan, DOCX dapat dibuat kemudian sebagai artefak pendamping tanpa mengubah markdown sumber.
