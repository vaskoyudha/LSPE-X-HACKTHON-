# LPSE-X Judge Casebook

LPSE-X is positioned as an evidence-backed procurement risk triage and investigation support system, not a pure confirmed-fraud predictor.

## 4-Level Judge-Facing Rating

- Aman: model sees low immediate procurement-risk signal.
- Perlu Pantauan: package deserves monitoring or manual follow-up.
- Risiko Tinggi: strong triage signal from the model, but not yet official proof.
- Risiko Kritis: official linked evidence exists, such as confirmed fraud, sanctions, or formal irregularity findings.

## Evidence Lane Snapshot

- Total evidence rows: 5
- Matched to procurement rows: 4
- Still needing reviewer confirmation: 1
- Linking confidence threshold: 0.55

## Official Evidence-Linked Cases

### Case 1: ocds-20h3g7-3282469

- Tender title: Pengadaan Public Safety Diving Equipment
- Buyer: Badan Nasional Pencarian dan Pertolongan
- Supplier: PT. KINDAH ABADI UTAMA
- Label families: confirmed_fraud
- Source(s): kpk_procurement_case
- Supporting evidence rows: 1
- Official source count: 1
- Case stage(s): final_outcome
- Decision date(s): 2023-12-20
- Confidence score: 1.0
- Tender value: Rp 17.499.969.180
- Award value: Rp 17.447.468.400
- Provenance note: KPK merilis perkara pengadaan barang dan jasa di Basarnas yang melibatkan Direktur Utama PT Kindah Abadi Utama. Halaman resmi perkara memuat pengadaan Public Safety Diving Equipment sebagai salah satu paket yang relevan. | suspect_names=Roni Aidil | package_names=Pengadaan Public Safety Diving Equipment

### Case 2: ocds-20h3g7-3317469

- Tender title: Pengadaan Peralatan Pendeteksi Korban Reruntuhan
- Buyer: Badan Nasional Pencarian dan Pertolongan
- Supplier: PT INTERTEKNO GRAFIKASEJATI
- Label families: confirmed_fraud
- Source(s): kpk_ppid_report, kpk_procurement_case
- Supporting evidence rows: 2
- Official source count: 2
- Case stage(s): final_outcome
- Decision date(s): 2023-12-20, 2023-12-21
- Confidence score: 1.0
- Tender value: Rp 9.999.738.030
- Award value: Rp 9.997.104.000
- Provenance note: KPK mencatat perkara suap pengadaan barang dan jasa di Basarnas. Pada 20 Des 2023, Majelis Hakim Tipikor Jakarta Pusat menjatuhkan vonis terhadap pihak pemberi suap. Halaman perkara resmi juga merujuk pengadaan peralatan pendeteksi korban reruntuhan di lingkungan Basarnas. | suspect_names=Marilya; Mulsunadi Gunawan | package_names=Pengadaan Peralatan Pendeteksi Korban Reruntuhan

### Case 3: ocds-20h3g7-3372469

- Tender title: Pengadaan ROV untuk KN SAR Ganesha (Multiyears 2023-2024)
- Buyer: Badan Nasional Pencarian dan Pertolongan
- Supplier: PT. KINDAH ABADI UTAMA
- Label families: confirmed_fraud
- Source(s): kpk_procurement_case
- Supporting evidence rows: 1
- Official source count: 1
- Case stage(s): final_outcome
- Decision date(s): 2023-12-20
- Confidence score: 1.0
- Tender value: Rp 89.996.922.102
- Award value: Rp 89.959.950.000
- Provenance note: Halaman perkara resmi KPK untuk perkara Basarnas memuat pengadaan ROV untuk KN SAR Ganesha sebagai salah satu paket yang terkait dengan pengadaan 2021 s/d 2023. | suspect_names=Roni Aidil | package_names=Pengadaan ROV untuk KN SAR Ganesha (Multiyears 2023-2024)

## Evidence Rows Still Needing Review

- vuGffsXDKSMt6SXMI1fQQ2VMajcDlv: sanctioned_supplier from lkpp_inaproc_blacklist (reviewer_needed=True)

## Judge Demo Archetypes

### Archetype 1: Official evidence-linked critical case

- OCID: ocds-20h3g7-3282469
- Tender title: Pengadaan Public Safety Diving Equipment
- Buyer: Badan Nasional Pencarian dan Pertolongan
- Supplier: PT. KINDAH ABADI UTAMA
- Recommended business rating: Risiko Kritis
- Evidence source(s): kpk_procurement_case
- Evidence families: confirmed_fraud

### Archetype 2: Evidence row still needing reviewer confirmation

- Source record: vuGffsXDKSMt6SXMI1fQQ2VMajcDlv
- Label family: sanctioned_supplier
- Source: lkpp_inaproc_blacklist
- Reviewer needed: True
- Provenance note: Seed record curated from the official INAPROC blacklist detail page. Sensitive supplier identifiers should be re-verified before public demo use.

### Archetype 3: Model-only high-risk triage case

- OCID: ocds-20h3g7-5076434
- Tender title: Pengadaan Sapi Lokal
- Buyer: Pemerintah Daerah Kabupaten Lahat
- Supplier: CV. Raje  Bungkuk
- Predicted label: High Risk
- Business rating: Risiko Tinggi [model_only]
- Sampling reason: priority_mix

## Top Review / Demo Rows

### Demo Row 1: ocds-20h3g7-5076434

- Tender title: Pengadaan Sapi Lokal
- Buyer: Pemerintah Daerah Kabupaten Lahat
- Supplier: CV. Raje  Bungkuk
- Predicted label: High Risk (0.50372)
- Business rating: Risiko Tinggi [model_only]
- Evidence families: none linked yet
- Review priority: 0.746742 via priority_mix
- Top factors: f_title_length=4.1907 | f_buyer_supplier_repeat_count=3.0983 | f_is_q4=-1.7404
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Tinggi**.
  Diturunkan dari triase model: High Risk dengan probabilitas 50.37%.
  Model mengklasifikasikan paket ini sebagai **High Risk** dengan probabilitas 50.37%.
  Catatan penting: ini adalah triase risiko, bukan bukti fraud final. Status kritis hanya boleh dinaikkan bila ada bukti resmi yang terhubung.
  Faktor yang paling memengaruhi prediksi adalah:
  - Panjang judul tender bernilai 20 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 4.1907.
  - Frekuensi hubungan buyer-supplier berulang bernilai 5 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.0983.
  - Waktu publikasi pada kuartal iv bernilai 0 dan menurunkan skor risiko dengan kontribusi SHAP sekitar 1.7404.

### Demo Row 2: ocds-20h3g7-1926669

- Tender title: Pembangunan Restoran
- Buyer: Pemerintah Daerah Kabupaten Nias Barat
- Supplier: AHAR KONSTRUKSI
- Predicted label: High Risk (0.508396)
- Business rating: Risiko Tinggi [model_only]
- Evidence families: none linked yet
- Review priority: 0.746541 via priority_mix
- Top factors: f_title_length=4.1946 | f_buyer_supplier_repeat_count=3.1777 | f_is_q4=-1.7507
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Tinggi**.
  Diturunkan dari triase model: High Risk dengan probabilitas 50.84%.
  Model mengklasifikasikan paket ini sebagai **High Risk** dengan probabilitas 50.84%.
  Catatan penting: ini adalah triase risiko, bukan bukti fraud final. Status kritis hanya boleh dinaikkan bila ada bukti resmi yang terhubung.
  Faktor yang paling memengaruhi prediksi adalah:
  - Panjang judul tender bernilai 20 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 4.1946.
  - Frekuensi hubungan buyer-supplier berulang bernilai 2 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.1777.
  - Waktu publikasi pada kuartal iv bernilai 0 dan menurunkan skor risiko dengan kontribusi SHAP sekitar 1.7507.

### Demo Row 3: ocds-20h3g7-3665437

- Tender title: Belanja Bibit Durian
- Buyer: Pemerintah Daerah Kabupaten Sigi
- Supplier: CV. Agrokultura Mandiri
- Predicted label: High Risk (0.503734)
- Business rating: Risiko Tinggi [model_only]
- Evidence families: none linked yet
- Review priority: 0.745544 via priority_mix
- Top factors: f_title_length=4.1715 | f_buyer_supplier_repeat_count=3.2196 | f_is_q4=-1.7325
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Tinggi**.
  Diturunkan dari triase model: High Risk dengan probabilitas 50.37%.
  Model mengklasifikasikan paket ini sebagai **High Risk** dengan probabilitas 50.37%.
  Catatan penting: ini adalah triase risiko, bukan bukti fraud final. Status kritis hanya boleh dinaikkan bila ada bukti resmi yang terhubung.
  Faktor yang paling memengaruhi prediksi adalah:
  - Panjang judul tender bernilai 20 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 4.1715.
  - Frekuensi hubungan buyer-supplier berulang bernilai 2 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.2196.
  - Waktu publikasi pada kuartal iv bernilai 0 dan menurunkan skor risiko dengan kontribusi SHAP sekitar 1.7325.

### Demo Row 4: ocds-20h3g7-2679340

- Tender title: Pembangunan Drainase
- Buyer: Pemerintah Daerah Kabupaten Bangka Barat
- Supplier: CV. MANUNGGAL PERTAMA
- Predicted label: High Risk (0.533988)
- Business rating: Risiko Tinggi [model_only]
- Evidence families: none linked yet
- Review priority: 0.745343 via priority_mix
- Top factors: f_title_length=4.0558 | f_buyer_supplier_repeat_count=3.3202 | f_is_q4=-1.7855
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Tinggi**.
  Diturunkan dari triase model: High Risk dengan probabilitas 53.40%.
  Model mengklasifikasikan paket ini sebagai **High Risk** dengan probabilitas 53.40%.
  Catatan penting: ini adalah triase risiko, bukan bukti fraud final. Status kritis hanya boleh dinaikkan bila ada bukti resmi yang terhubung.
  Faktor yang paling memengaruhi prediksi adalah:
  - Panjang judul tender bernilai 20 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 4.0558.
  - Frekuensi hubungan buyer-supplier berulang bernilai 2 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.3202.
  - Waktu publikasi pada kuartal iv bernilai 0 dan menurunkan skor risiko dengan kontribusi SHAP sekitar 1.7855.

### Demo Row 5: ocds-20h3g7-1489603

- Tender title: Har Gedung Satfaskon
- Buyer: Kementerian Pertahanan
- Supplier: PT TORSI CIPTA KREASI
- Predicted label: High Risk (0.50219)
- Business rating: Risiko Tinggi [model_only]
- Evidence families: none linked yet
- Review priority: 0.745272 via priority_mix
- Top factors: f_title_length=4.1750 | f_buyer_supplier_repeat_count=3.2493 | f_is_q4=-1.7873
- Narrative:
  Peringkat investigatif paket ini adalah **Risiko Tinggi**.
  Diturunkan dari triase model: High Risk dengan probabilitas 50.22%.
  Model mengklasifikasikan paket ini sebagai **High Risk** dengan probabilitas 50.22%.
  Catatan penting: ini adalah triase risiko, bukan bukti fraud final. Status kritis hanya boleh dinaikkan bila ada bukti resmi yang terhubung.
  Faktor yang paling memengaruhi prediksi adalah:
  - Panjang judul tender bernilai 20 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 4.1750.
  - Frekuensi hubungan buyer-supplier berulang bernilai 12 dan meningkatkan skor risiko dengan kontribusi SHAP sekitar 3.2493.
  - Waktu publikasi pada kuartal iv bernilai 0 dan menurunkan skor risiko dengan kontribusi SHAP sekitar 1.7873.

## Judge Notes

- The live model is still a 3-class XGBoost procurement risk model trained on heuristic risk labels.
- The business-facing 4-level scale is a transparent presentation layer: only official linked evidence can escalate a package to Risiko Kritis.
- This casebook is meant to support demo storytelling, reviewer calibration, and investigation handoff.
