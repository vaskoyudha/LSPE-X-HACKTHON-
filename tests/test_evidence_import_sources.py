import json
from pathlib import Path

import pandas as pd
import pytest

from scripts.import_official_evidence import import_official_evidence


@pytest.mark.p1
def test_import_official_evidence_supports_lkpp_blacklist_source_mode(tmp_path: Path):
    input_path = tmp_path / "lkpp_blacklist.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "id": "vuGffsXDKSMt6SXMI1fQQ2VMajcDlv",
                    "detail_url": "https://daftar-hitam.inaproc.id/blacklist/vuGffsXDKSMt6SXMI1fQQ2VMajcDlv",
                    "nama_penyedia": "PT Contoh Hitam Sejahtera",
                    "npwp_penyedia": "01.234.567.8-901.000",
                    "nib_penyedia": "1234567890123",
                    "alamat_penyedia": "Jl. Merdeka No. 1",
                    "provinsi": "Jawa Timur",
                    "kota_kabupaten": "Kab. Blitar",
                    "nomor_sk": "SK-01/DH/2024",
                    "jenis_pelanggaran": "Peraturan LKPP No. 4 Tahun 2021 Lampiran II angka 3.1 huruf g",
                    "deskripsi_pelanggaran": "Memberikan data kualifikasi yang tidak benar",
                    "tanggal_berlaku": "2024-03-01",
                    "tanggal_selesai": "2026-02-28",
                    "id_rup_tender": "10868203",
                    "nama_paket": "Pembangunan Gedung Perpustakaan Daerah",
                    "jenis_pengadaan": "Pekerjaan Konstruksi",
                    "kl_pd": "Kab. Blitar",
                    "satker": "Dinas Perpustakaan dan Kearsipan",
                    "hps": 9999012000,
                    "pagu": 10000000000,
                    "tahun_anggaran": 2024,
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = import_official_evidence(
        input_path,
        output_dir=tmp_path / "processed",
        source="lkpp_inaproc_blacklist",
        raw_archive_dir=tmp_path / "raw_archive",
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    label_df = pd.read_parquet(outputs["label_records"])

    assert len(evidence_df) == 1
    evidence = evidence_df.iloc[0].to_dict()
    assert evidence["source_name"] == "lkpp_inaproc_blacklist"
    assert evidence["source_type"] == "sanction_list"
    assert evidence["label_family"] == "sanctioned_supplier"
    assert evidence["case_stage"] == "administrative_sanction"
    assert evidence["supplier_name"] == "PT Contoh Hitam Sejahtera"
    assert evidence["buyer_name"] == "Kab. Blitar"
    assert evidence["package_name"] == "Pembangunan Gedung Perpustakaan Daerah"
    assert evidence["package_id"] == "10868203"
    assert evidence["package_value_amount"] == pytest.approx(9999012000.0)
    assert evidence["raw_file_path"]
    assert Path(evidence["raw_file_path"]).exists()
    assert "nomor_sk=SK-01/DH/2024" in evidence["provenance_note"]

    assert len(label_df) == 1
    label = label_df.iloc[0].to_dict()
    assert label["label_family"] == "sanctioned_supplier"
    assert label["source_name"] == "lkpp_inaproc_blacklist"
    assert bool(label["reviewer_needed"]) is True


@pytest.mark.p1
def test_import_official_evidence_lkpp_mode_uses_formatted_hps_when_package_value_is_blank(tmp_path: Path):
    input_path = tmp_path / "lkpp_blacklist_formatted_hps.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "id": "lkpp-formatted-hps",
                    "detail_url": "https://daftar-hitam.inaproc.id/blacklist/lkpp-formatted-hps",
                    "nama_penyedia": "PT Contoh Hitam Sejahtera",
                    "nomor_sk": "SK-02/DH/2024",
                    "nama_paket": "Pembangunan Gedung Perpustakaan Daerah",
                    "kl_pd": "Kab. Blitar",
                    "package_value_amount": "",
                    "hps": "9.999.012.000",
                    "tahun_anggaran": 2024,
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = import_official_evidence(
        input_path,
        output_dir=tmp_path / "processed",
        source="lkpp_inaproc_blacklist",
        raw_archive_dir=tmp_path / "raw_archive",
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    assert evidence_df.iloc[0]["package_value_amount"] == pytest.approx(9999012000.0)


@pytest.mark.p1
def test_import_official_evidence_supports_kpk_procurement_case_source_mode(tmp_path: Path):
    input_path = tmp_path / "kpk_case.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "basarnas-2023",
                    "url": "https://www.kpk.go.id/id/publikasi-data/penanganan-perkara/tpk-berupa-suap-pengadaan-barang-dan-jasa-di-basarnas-badan-nasional-pencarian-dan-pertolongan-tahun-2021-sd-2023",
                    "title": "TPK Berupa Suap Pengadaan Barang dan Jasa Di Basarnas (Badan Nasional Pencarian Dan Pertolongan) Tahun 2021 s/d 2023",
                    "publication_date": "2023-07-25",
                    "decision_date": "2023-12-20",
                    "case_stage": "final_outcome",
                    "agency_name": "Basarnas",
                    "supplier_name": "PT Multi Grafika Cipta Sejati",
                    "package_name": "Pengadaan peralatan pendeteksi korban reruntuhan",
                    "package_value_amount": 9900000000,
                    "case_summary": "Majelis Hakim Tipikor Jakarta Pusat menjatuhkan vonis dalam perkara suap pengadaan barang dan jasa di Basarnas.",
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = import_official_evidence(
        input_path,
        output_dir=tmp_path / "processed",
        source="kpk_procurement_case",
        raw_archive_dir=tmp_path / "raw_archive",
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    label_df = pd.read_parquet(outputs["label_records"])

    assert len(evidence_df) == 1
    evidence = evidence_df.iloc[0].to_dict()
    assert evidence["source_name"] == "kpk_procurement_case"
    assert evidence["organization"] == "Komisi Pemberantasan Korupsi (KPK)"
    assert evidence["source_type"] == "case_press_release"
    assert evidence["label_family"] == "confirmed_fraud"
    assert evidence["case_stage"] == "final_outcome"
    assert evidence["buyer_name"] == "Basarnas"
    assert evidence["supplier_name"] == "PT Multi Grafika Cipta Sejati"
    assert evidence["package_name"] == "Pengadaan peralatan pendeteksi korban reruntuhan"
    assert "Tipikor Jakarta Pusat" in evidence["provenance_note"]

    assert len(label_df) == 1
    label = label_df.iloc[0].to_dict()
    assert label["label_family"] == "confirmed_fraud"
    assert label["source_name"] == "kpk_procurement_case"
    assert bool(label["reviewer_needed"]) is True


@pytest.mark.p1
def test_kpk_source_mode_normalizes_finished_status_to_final_outcome(tmp_path: Path):
    input_path = tmp_path / "kpk_status_case.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "case_id": "basarnas-finished-status",
                    "url": "https://www.kpk.go.id/id/publikasi-data/penanganan-perkara/tpk-berupa-suap-pengadaan-barang-dan-jasa-di-basarnas-badan-nasional-pencarian-dan-pertolongan-tahun-2021-sd-2023",
                    "title": "TPK Berupa Suap Pengadaan Barang dan Jasa Di Basarnas (Badan Nasional Pencarian Dan Pertolongan) Tahun 2021 s/d 2023",
                    "status": "Finished 2023",
                    "supplier_name": "PT Multi Grafika Cipta Sejati",
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = import_official_evidence(
        input_path,
        output_dir=tmp_path / "processed",
        source="kpk_procurement_case",
        raw_archive_dir=tmp_path / "raw_archive",
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    evidence = evidence_df.iloc[0].to_dict()
    assert evidence["case_stage"] == "final_outcome"
    assert evidence["label_family"] == "confirmed_fraud"


@pytest.mark.p1
def test_import_official_evidence_supports_kpk_ppid_report_source_mode(tmp_path: Path):
    input_path = tmp_path / "kpk_ppid_report.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "source_record_id": "kpk-ppid-dec2023-intertekno-reruntuhan",
                    "source_url": "https://ppid.kpk.go.id/auo/api/private-file?filename=sample.pdf",
                    "report_title": "Laporan Pelaksanaan Kegiatan Deputi Bidang Penindakan dan Eksekusi Desember 2023",
                    "document_id": "ppid-202406-sample",
                    "case_stage": "final_outcome",
                    "decision_date": "2023-12-21",
                    "agency_name": "Badan Nasional Pencarian dan Pertolongan",
                    "supplier_name": "PT Intertekno Grafika Sejati",
                    "package_name": "Pengadaan Peralatan Pendeteksi Korban Reruntuhan",
                    "package_year": "2023",
                    "matched_ocid": "ocds-20h3g7-3317469",
                    "provenance_note": "Official PPID report references Putusan PN 87/Pid.Sus-TPK/2023/PN.Jkt.Pst.",
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = import_official_evidence(
        input_path,
        output_dir=tmp_path / "processed",
        source="kpk_ppid_report",
        raw_archive_dir=tmp_path / "raw_archive",
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    label_df = pd.read_parquet(outputs["label_records"])

    assert len(evidence_df) == 1
    evidence = evidence_df.iloc[0].to_dict()
    assert evidence["source_name"] == "kpk_ppid_report"
    assert evidence["source_type"] == "ppid_activity_report"
    assert evidence["organization"] == "Komisi Pemberantasan Korupsi (KPK) / PPID"
    assert evidence["label_family"] == "confirmed_fraud"
    assert evidence["case_stage"] == "final_outcome"
    assert evidence["matched_ocid"] == "ocds-20h3g7-3317469"
    assert evidence["package_name"] == "Pengadaan Peralatan Pendeteksi Korban Reruntuhan"

    assert len(label_df) == 1
    label = label_df.iloc[0].to_dict()
    assert label["source_name"] == "kpk_ppid_report"
    assert label["label_family"] == "confirmed_fraud"
    assert label["ocid"] == "ocds-20h3g7-3317469"
    assert bool(label["reviewer_needed"]) is False


@pytest.mark.p1
def test_import_official_evidence_generic_mode_still_accepts_existing_normalized_records(tmp_path: Path):
    input_path = tmp_path / "generic.json"
    input_path.write_text(
        json.dumps(
            [
                {
                    "source_record_id": "generic-1",
                    "source_name": "manual-audit-import",
                    "source_type": "audit_report",
                    "source_url": "https://example.test/audit/1",
                    "title": "Audit finding",
                    "organization": "BPK",
                    "label_family": "confirmed_irregularity",
                    "label_value": "audit_finding",
                    "case_stage": "audit_finding",
                    "decision_date": "2024-05-01",
                }
            ]
        ),
        encoding="utf-8",
    )

    outputs = import_official_evidence(
        input_path,
        output_dir=tmp_path / "processed",
        source="generic",
        raw_archive_dir=tmp_path / "raw_archive",
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    assert evidence_df.iloc[0]["label_family"] == "confirmed_irregularity"
    assert evidence_df.iloc[0]["source_name"] == "manual-audit-import"


@pytest.mark.p1
def test_import_official_evidence_preserves_string_identifiers_from_csv(tmp_path: Path):
    input_path = tmp_path / "generic.csv"
    input_path.write_text(
        "source_record_id,source_name,source_type,label_family,label_value,case_stage,supplier_id,buyer_id,package_value_amount\n"
        "00123,manual-audit-import,audit_report,confirmed_irregularity,audit_finding,audit_finding,000456,00789,1000\n",
        encoding="utf-8",
    )

    outputs = import_official_evidence(
        input_path,
        output_dir=tmp_path / "processed",
        source="generic",
        raw_archive_dir=tmp_path / "raw_archive",
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    label_df = pd.read_parquet(outputs["label_records"])

    assert evidence_df.iloc[0]["source_record_id"] == "00123"
    assert evidence_df.iloc[0]["supplier_id"] == "000456"
    assert evidence_df.iloc[0]["buyer_id"] == "00789"
    assert label_df.iloc[0]["source_record_id"] == "00123"


@pytest.mark.p1
def test_import_official_evidence_appends_existing_outputs_and_preserves_unique_raw_archives(tmp_path: Path):
    output_dir = tmp_path / "processed"
    raw_archive_dir = tmp_path / "raw_archive"

    batch_a = tmp_path / "batch_a"
    batch_b = tmp_path / "batch_b"
    batch_a.mkdir()
    batch_b.mkdir()

    input_a = batch_a / "records.json"
    input_b = batch_b / "records.json"
    input_a.write_text(
        json.dumps(
            [
                {
                    "source_record_id": "generic-a",
                    "source_name": "manual-audit-import",
                    "source_type": "audit_report",
                    "label_family": "confirmed_irregularity",
                    "label_value": "audit_finding",
                    "case_stage": "audit_finding",
                }
            ]
        ),
        encoding="utf-8",
    )
    input_b.write_text(
        json.dumps(
            [
                {
                    "source_record_id": "generic-b",
                    "source_name": "manual-audit-import",
                    "source_type": "audit_report",
                    "label_family": "reviewed_risk",
                    "label_value": "manual_review",
                    "case_stage": "human_review",
                }
            ]
        ),
        encoding="utf-8",
    )

    import_official_evidence(
        input_a,
        output_dir=output_dir,
        source="generic",
        raw_archive_dir=raw_archive_dir,
    )
    outputs = import_official_evidence(
        input_b,
        output_dir=output_dir,
        source="generic",
        raw_archive_dir=raw_archive_dir,
    )

    evidence_df = pd.read_parquet(outputs["evidence_records"])
    label_df = pd.read_parquet(outputs["label_records"])
    archived_files = sorted((raw_archive_dir / "generic").glob("records*.json"))

    assert set(evidence_df["source_record_id"].tolist()) == {"generic-a", "generic-b"}
    assert set(label_df["source_record_id"].tolist()) == {"generic-a", "generic-b"}
    assert len(archived_files) == 2
    assert archived_files[0].name == "records--1.json"
    assert archived_files[1].name == "records.json"
