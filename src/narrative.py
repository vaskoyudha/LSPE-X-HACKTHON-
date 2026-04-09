"""Bahasa Indonesia explanation rendering utilities."""

from __future__ import annotations

from typing import Iterable

RISK_LABEL_ID = {
    0: "Risiko Rendah",
    1: "Risiko Sedang",
    2: "Risiko Tinggi",
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


def _format_value(value) -> str:
    if value is None:
        return "tidak tersedia"
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return str(value)


def _feature_label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " "))


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
) -> str:
    """Convert an explain_single payload into a Bahasa Indonesia narrative."""
    predicted_class = explanation.get("predicted_class")
    label = explanation.get("predicted_label") or RISK_LABEL_ID.get(predicted_class, str(predicted_class))
    probability = float(explanation.get("probability", 0.0))
    factors = explanation.get("factors", [])

    lines = [
        f"Model mengklasifikasikan paket ini sebagai **{label}** dengan probabilitas {probability:.2%}.",
    ]

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
