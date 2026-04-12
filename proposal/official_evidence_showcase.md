# Official Evidence Showcase

This artifact shows how LPSE-X behaves on procurement rows that already have official linked evidence.

## Summary

- Official evidence-linked cases found in current split artifacts: 3
- Supporting official evidence rows behind those cases: 4
- Cases with multi-source official corroboration: 1
- Model predicted High Risk: 2
- Model predicted Medium Risk: 1
- Model predicted Low Risk: 0
- Final business rating Risiko Kritis after evidence escalation: 3
- Cases where evidence lane corrected a non-High-Risk model output: 1

## Why this matters

The showcase demonstrates the value of the hybrid design: the model provides triage, while the evidence lane prevents known official cases from being understated when model-only risk is insufficient.

## Case 1: ocds-20h3g7-3317469

- Split: train
- Tender title: Pengadaan Peralatan Pendeteksi Korban Reruntuhan
- Buyer: Badan Nasional Pencarian dan Pertolongan
- Supplier: PT INTERTEKNO GRAFIKASEJATI
- Evidence families: confirmed_fraud
- Evidence sources: kpk_ppid_report, kpk_procurement_case
- Case stages: final_outcome
- Supporting evidence rows: 2
- Official source count: 2
- Model prediction: Medium Risk (58.58%)
- Probability vector: low=14.66%, medium=58.58%, high=26.76%
- Final business rating: Risiko Kritis [official_evidence]
- Rating reason: Dinaikkan ke Risiko Kritis karena ada bukti resmi terhubung (confirmed_fraud) dari sumber kpk_ppid_report, kpk_procurement_case.
- Top factors: f_tender_value_log=-1.4122 | f_is_q4=0.9832 | f_buyer_supplier_repeat_count=0.8988
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Kritis**.
  Dinaikkan ke Risiko Kritis karena ada bukti resmi terhubung (confirmed_fraud) dari sumber kpk_ppid_report, kpk_procurement_case.
  Model mengklasifikasikan paket ini sebagai **Medium Risk** dengan probabilitas 58.58%.
  Catatan penting: rating kritis didukung bukti resmi terhubung, tetapi kecocokan entitas dan konteks kasus tetap harus diverifikasi investigator.
  Faktor yang paling memengaruhi prediksi adalah:
  - Nilai tender bernilai 23.0258 dan menurunkan skor risiko dengan kontribusi SHAP sekitar 1.4122.
  - Waktu publikasi pada kuartal iv bernilai 0 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 0.9832.
  - Frekuensi hubungan buyer-supplier berulang bernilai 0 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 0.8988.

## Case 2: ocds-20h3g7-3282469

- Split: train
- Tender title: Pengadaan Public Safety Diving Equipment
- Buyer: Badan Nasional Pencarian dan Pertolongan
- Supplier: PT. KINDAH ABADI UTAMA
- Evidence families: confirmed_fraud
- Evidence sources: kpk_procurement_case
- Case stages: final_outcome
- Supporting evidence rows: 1
- Official source count: 1
- Model prediction: High Risk (86.63%)
- Probability vector: low=6.26%, medium=7.11%, high=86.63%
- Final business rating: Risiko Kritis [official_evidence]
- Rating reason: Dinaikkan ke Risiko Kritis karena ada bukti resmi terhubung (confirmed_fraud) dari sumber kpk_procurement_case.
- Top factors: f_tender_value_log=3.4595 | f_buyer_supplier_repeat_count=3.2338 | f_is_q4=3.2271
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Kritis**.
  Dinaikkan ke Risiko Kritis karena ada bukti resmi terhubung (confirmed_fraud) dari sumber kpk_procurement_case.
  Model mengklasifikasikan paket ini sebagai **High Risk** dengan probabilitas 86.63%.
  Catatan penting: rating kritis didukung bukti resmi terhubung, tetapi kecocokan entitas dan konteks kasus tetap harus diverifikasi investigator.
  Faktor yang paling memengaruhi prediksi adalah:
  - Nilai tender bernilai 23.5855 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.4595.
  - Frekuensi hubungan buyer-supplier berulang bernilai 3 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.2338.
  - Waktu publikasi pada kuartal iv bernilai 1 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.2271.

## Case 3: ocds-20h3g7-3372469

- Split: train
- Tender title: Pengadaan ROV untuk KN SAR Ganesha (Multiyears 2023-2024)
- Buyer: Badan Nasional Pencarian dan Pertolongan
- Supplier: PT. KINDAH ABADI UTAMA
- Evidence families: confirmed_fraud
- Evidence sources: kpk_procurement_case
- Case stages: final_outcome
- Supporting evidence rows: 1
- Official source count: 1
- Model prediction: High Risk (84.19%)
- Probability vector: low=7.30%, medium=8.52%, high=84.19%
- Final business rating: Risiko Kritis [official_evidence]
- Rating reason: Dinaikkan ke Risiko Kritis karena ada bukti resmi terhubung (confirmed_fraud) dari sumber kpk_procurement_case.
- Top factors: f_tender_value_log=4.3621 | f_tender_value_zscore_buyer=4.2313 | f_buyer_supplier_repeat_count=3.1992
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Kritis**.
  Dinaikkan ke Risiko Kritis karena ada bukti resmi terhubung (confirmed_fraud) dari sumber kpk_procurement_case.
  Model mengklasifikasikan paket ini sebagai **High Risk** dengan probabilitas 84.19%.
  Catatan penting: rating kritis didukung bukti resmi terhubung, tetapi kecocokan entitas dan konteks kasus tetap harus diverifikasi investigator.
  Faktor yang paling memengaruhi prediksi adalah:
  - Nilai tender bernilai 25.223 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 4.3621.
  - F tender value zscore buyer bernilai 12.4994 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 4.2313.
  - Frekuensi hubungan buyer-supplier berulang bernilai 4 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.1992.
