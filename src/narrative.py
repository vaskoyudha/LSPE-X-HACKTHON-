"""Bahasa Indonesia explanation rendering utilities."""

from __future__ import annotations

from typing import Any, Iterable

RISK_LABEL_ID = {
    0: "Risiko Rendah",
    1: "Risiko Sedang",
    2: "Risiko Tinggi",
}

BUSINESS_RATING_ID = {
    "aman": "Aman",
    "perlu_pantauan": "Perlu Pantauan",
    "risiko_tinggi": "Risiko Tinggi",
    "risiko_kritis": "Risiko Kritis",
}

FEATURE_LABELS = {
    "f_single_bidder": "indikasi peserta tunggal",
    "f_num_tenderers": "jumlah peserta tender",
    "f_price_deviation_ratio": "rasio deviasi harga terhadap nilai tender",
    "f_procurement_method_enc": "metode pengadaan",
    "f_is_q4": "waktu publikasi pada kuartal IV",
    "f_is_december": "publikasi pada bulan Desember",
    "f_tender_value_log": "nilai tender",
    "f_award_value_log": "nilai pemenang",
    "f_contract_value_log": "nilai kontrak",
    "f_title_length": "panjang judul tender",
    "f_description_length": "kelengkapan deskripsi tender",
    "f_buyer_supplier_repeat_count": "frekuensi hubungan buyer-supplier berulang",
}

CRITICAL_EVIDENCE_FAMILIES = {
    "confirmed_fraud",
    "sanctioned_supplier",
}

OFFICIAL_IRREGULARITY_STAGES = {
    "final_outcome",
    "audit_finding",
    "administrative_sanction",
}

WATCHLIST_EVIDENCE_FAMILIES = {
    "reviewed_risk",
    "candidate_review_queue",
}


def _format_value(value) -> str:
    if value is None:
        return "tidak tersedia"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _base_business_rating(predicted_class: int) -> tuple[str, str]:
    mapping = {
        0: ("aman", BUSINESS_RATING_ID["aman"]),
        1: ("perlu_pantauan", BUSINESS_RATING_ID["perlu_pantauan"]),
        2: ("risiko_tinggi", BUSINESS_RATING_ID["risiko_tinggi"]),
    }
    return mapping.get(int(predicted_class), ("perlu_pantauan", BUSINESS_RATING_ID["perlu_pantauan"]))


def derive_business_rating(
    explanation: dict,
    evidence_records: Iterable[dict] | None = None,
) -> dict[str, str]:
    """Map the 3-class model output plus evidence lane into a 4-level judge-facing rating."""
    predicted_class = int(explanation.get("predicted_class", 1))
    probability = float(explanation.get("probability", 0.0))
    predicted_label = _normalize_text(explanation.get("predicted_label")) or RISK_LABEL_ID.get(
        predicted_class, str(predicted_class)
    )

    evidence_list = list(evidence_records or [])
    critical_records: list[dict] = []
    watchlist_records: list[dict] = []
    for record in evidence_list:
        family = _normalize_text(record.get("label_family"))
        stage = _normalize_text(record.get("case_stage"))
        reviewer_needed = bool(record.get("reviewer_needed", False))
        if not family:
            continue
        if not reviewer_needed and (
            family in CRITICAL_EVIDENCE_FAMILIES
            or (family == "confirmed_irregularity" and stage in OFFICIAL_IRREGULARITY_STAGES)
        ):
            critical_records.append(record)
        elif family in WATCHLIST_EVIDENCE_FAMILIES or reviewer_needed:
            watchlist_records.append(record)

    if critical_records:
        critical_families = sorted(
            {_normalize_text(record.get("label_family")) for record in critical_records if record.get("label_family")}
        )
        sources = sorted(
            {_normalize_text(record.get("source_name")) for record in critical_records if record.get("source_name")}
        )
        return {
            "rating_code": "risiko_kritis",
            "rating_label": BUSINESS_RATING_ID["risiko_kritis"],
            "rating_source": "official_evidence",
            "rating_reason": (
                "Dinaikkan ke Risiko Kritis karena ada bukti resmi terhubung "
                f"({', '.join(critical_families)}) dari sumber {', '.join(sources) or 'resmi'}."
            ),
        }

    base_code, base_label = _base_business_rating(predicted_class)
    if watchlist_records and base_code == "aman":
        return {
            "rating_code": "perlu_pantauan",
            "rating_label": BUSINESS_RATING_ID["perlu_pantauan"],
            "rating_source": "review_queue",
            "rating_reason": (
                "Dinaikkan ke Perlu Pantauan karena sudah ada sinyal evidence/review queue, "
                "meski model dasar belum menempatkan paket pada kelas tinggi."
            ),
        }

    return {
        "rating_code": base_code,
        "rating_label": base_label,
        "rating_source": "model_only",
        "rating_reason": (
            f"Diturunkan dari triase model: {predicted_label} dengan probabilitas {probability:.2%}."
        ),
    }


def render_factor_sentence(factor: dict) -> str:
    """Render one explanation factor into a short Indonesian sentence."""
    label = _feature_label(factor["feature"])
    value = _format_value(factor.get("value", factor.get("feature_value")))
    shap_value = abs(float(factor.get("shap_value", 0.0)))
    direction = factor.get("direction")
    if direction == "decreases_risk" or float(factor.get("shap_value", 0.0)) < 0:
        verb = "menurunkan"
    else:
        verb = "meningkatkan"
    return (
        f"{label.capitalize()} bernilai {value} dan {verb} skor risiko "
        f"dengan kontribusi SHAP sekitar {shap_value:.4f}."
    )


def render_counterfactuals(counterfactuals: Iterable[dict]) -> list[str]:
    """Render SHAP-based counterfactual suggestions in Indonesian."""
    rendered: list[str] = []
    for item in counterfactuals:
        if "message" in item:
            rendered.append(str(item["message"]))
            continue
        feature = _feature_label(item.get("feature", "fitur"))
        suggestion = item.get("suggestion", "tinjau kembali faktor ini")
        impact = item.get("impact")
        if impact is None:
            rendered.append(f"- {feature.capitalize()}: {suggestion}.")
        else:
            rendered.append(
                f"- {feature.capitalize()}: {suggestion} (perkiraan dampak {float(impact):.4f})."
            )
    return rendered


def render_explanation_narrative(
    explanation: dict,
    counterfactuals: Iterable[dict] | None = None,
    *,
    evidence_records: Iterable[dict] | None = None,
    business_rating: dict[str, str] | None = None,
) -> str:
    """Convert an explain_single payload into a Bahasa Indonesia narrative."""
    predicted_class = explanation.get("predicted_class")
    label = explanation.get("predicted_label") or RISK_LABEL_ID.get(predicted_class, str(predicted_class))
    probability = float(explanation.get("probability", 0.0))
    factors = explanation.get("factors", [])
    business_rating = business_rating or derive_business_rating(explanation, evidence_records=evidence_records)

    lines = [
        f"Peringkat investigatif paket ini adalah **{business_rating['rating_label']}**.",
        business_rating["rating_reason"],
        f"Model mengklasifikasikan paket ini sebagai **{label}** dengan probabilitas {probability:.2%}.",
    ]

    if business_rating["rating_source"] != "official_evidence":
        lines.append(
            "Catatan penting: ini adalah triase risiko, bukan bukti fraud final. "
            "Status kritis hanya boleh dinaikkan bila ada bukti resmi yang terhubung."
        )
    else:
        lines.append(
            "Catatan penting: rating kritis didukung bukti resmi terhubung, tetapi kecocokan entitas dan konteks kasus tetap harus diverifikasi investigator."
        )

    if factors:
        lines.append("Faktor yang paling memengaruhi prediksi adalah:")
        for factor in factors:
            lines.append(f"- {render_factor_sentence(factor)}")

    if counterfactuals:
        rendered_cf = render_counterfactuals(counterfactuals)
        if rendered_cf:
            lines.append("Saran tindak lanjut untuk menurunkan risiko:")
            lines.extend(rendered_cf)

    return "\n".join(lines)
