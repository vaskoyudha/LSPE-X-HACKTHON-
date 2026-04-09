"""Bahasa Indonesia explanation rendering utilities.

Converts the output of explain_single() into competition-grade
Bahasa Indonesia narratives for the proposal and inference notebook.

Contract:
  - Input: explain_single() output dict with keys:
      predicted_class, predicted_label, probability, probabilities, factors
  - Output: Bahasa Indonesia narrative string
  - Reads from `factors` contract, NOT any legacy key
  - Includes mandatory disclaimer language
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Feature name translations
# ---------------------------------------------------------------------------

FEATURE_LABELS_ID: dict[str, str] = {
    "f_tender_value_log": "Nilai tender (log)",
    "f_award_value_log": "Nilai kontrak (log)",
    "f_price_deviation_ratio": "Rasio deviasi harga",
    "f_num_tenderers": "Jumlah peserta tender",
    "f_single_bidder": "Penawar tunggal",
    "f_title_length": "Panjang judul tender",
    "f_description_length": "Panjang deskripsi tender",
    "f_procurement_method_enc": "Metode pengadaan",
    "f_is_q4": "Pengadaan di kuartal 4",
    "f_month_sin": "Bulan (sin)",
    "f_month_cos": "Bulan (cos)",
    "f_day_of_week": "Hari dalam minggu",
    "f_value_per_char": "Nilai per karakter judul",
    "f_tenderer_value_ratio": "Rasio jumlah peserta/nilai",
    "f_has_description": "Memiliki deskripsi",
    "f_buyer_contract_count": "Jumlah kontrak pembeli",
    "f_buyer_total_value": "Total nilai kontrak pembeli",
    "f_buyer_avg_value": "Rata-rata nilai kontrak pembeli",
    "f_buyer_supplier_diversity": "Keberagaman pemasok pembeli",
    "f_buyer_single_bid_ratio": "Rasio penawar tunggal pembeli",
    "f_supplier_contract_count": "Jumlah kontrak pemasok",
    "f_supplier_total_value": "Total nilai kontrak pemasok",
    "f_supplier_avg_value": "Rata-rata nilai kontrak pemasok",
    "f_supplier_buyer_diversity": "Keberagaman pembeli pemasok",
    "f_supplier_distinct_buyers": "Jumlah pembeli unik pemasok",
    "f_buyer_method_diversity": "Keberagaman metode pembeli",
    "f_buyer_q4_ratio": "Rasio pengadaan Q4 pembeli",
    "f_supplier_q4_ratio": "Rasio pengadaan Q4 pemasok",
    "f_buyer_price_dev_std": "Std deviasi harga pembeli",
    "f_supplier_price_dev_std": "Std deviasi harga pemasok",
}

CLASS_LABELS_ID: dict[int, str] = {
    0: "Risiko Rendah",
    1: "Risiko Sedang",
    2: "Risiko Tinggi",
}

# ---------------------------------------------------------------------------
# Narrative templates
# ---------------------------------------------------------------------------

DISCLAIMER = (
    "Catatan: Analisis ini didasarkan pada indikator risiko heuristik, "
    "bukan hasil investigasi forensik. Label risiko bersifat indikatif dan "
    "tidak menunjukkan adanya kecurangan yang terkonfirmasi. Pengguna "
    "diharapkan melakukan verifikasi lebih lanjut sebelum mengambil tindakan."
)


def _get_feature_label(feature_name: str) -> str:
    """Get Indonesian label for a feature, falling back to raw name."""
    return FEATURE_LABELS_ID.get(feature_name, feature_name)


def _describe_direction(direction: str) -> str:
    """Convert direction to Indonesian description."""
    if direction == "increases_risk":
        return "meningkatkan risiko"
    return "menurunkan risiko"


def _format_value(value: Any) -> str:
    """Format a feature value for display."""
    if value is None:
        return "tidak tersedia"
    if isinstance(value, float):
        if abs(value) >= 1000:
            return f"{value:,.0f}"
        return f"{value:.4f}"
    return str(value)


# ---------------------------------------------------------------------------
# Main narrative generator
# ---------------------------------------------------------------------------


def generate_narrative(explanation: dict) -> str:
    """Generate a full Bahasa Indonesia narrative from explain_single() output.

    Args:
        explanation: Output dict from explain_single() with keys:
            predicted_class, predicted_label, probability, probabilities, factors

    Returns:
        Multi-paragraph Bahasa Indonesia narrative string.
    """
    pred_class = explanation["predicted_class"]
    pred_label = CLASS_LABELS_ID.get(pred_class, explanation["predicted_label"])
    probability = explanation["probability"]
    factors = explanation["factors"]

    # Opening paragraph
    paragraphs = []

    confidence_word = _confidence_word(probability)
    paragraphs.append(
        f"Proses pengadaan ini diklasifikasikan sebagai **{pred_label}** "
        f"dengan tingkat keyakinan {confidence_word} ({probability:.1%})."
    )

    # Factor analysis paragraph
    if factors:
        risk_factors = [f for f in factors if f["direction"] == "increases_risk"]
        protective_factors = [f for f in factors if f["direction"] == "decreases_risk"]

        if risk_factors:
            risk_lines = []
            for i, f in enumerate(risk_factors, 1):
                label = _get_feature_label(f["feature"])
                value = _format_value(f["value"])
                risk_lines.append(
                    f"{i}. **{label}** (nilai: {value}) — {_describe_direction(f['direction'])}"
                )
            paragraphs.append(
                "**Faktor-faktor yang meningkatkan risiko:**\n" + "\n".join(risk_lines)
            )

        if protective_factors:
            prot_lines = []
            for i, f in enumerate(protective_factors, 1):
                label = _get_feature_label(f["feature"])
                value = _format_value(f["value"])
                prot_lines.append(
                    f"{i}. **{label}** (nilai: {value}) — {_describe_direction(f['direction'])}"
                )
            paragraphs.append(
                "**Faktor-faktor yang menurunkan risiko:**\n" + "\n".join(prot_lines)
            )

    # Summary paragraph based on risk level
    if pred_class == 2:
        paragraphs.append(
            "Pengadaan ini menunjukkan beberapa indikator risiko tinggi yang memerlukan "
            "perhatian lebih lanjut. Disarankan untuk melakukan pemeriksaan mendalam "
            "terhadap proses tender dan dokumentasi terkait."
        )
    elif pred_class == 1:
        paragraphs.append(
            "Pengadaan ini memiliki beberapa indikator risiko sedang. Meskipun tidak "
            "memerlukan tindakan segera, pemantauan berkala disarankan untuk memastikan "
            "kepatuhan terhadap prosedur pengadaan."
        )
    else:
        paragraphs.append(
            "Pengadaan ini menunjukkan profil risiko rendah berdasarkan indikator "
            "yang tersedia. Tidak ditemukan anomali signifikan pada proses tender."
        )

    # Mandatory disclaimer
    paragraphs.append(f"---\n\n*{DISCLAIMER}*")

    return "\n\n".join(paragraphs)


def _confidence_word(probability: float) -> str:
    """Convert probability to Indonesian confidence descriptor."""
    if probability >= 0.90:
        return "sangat tinggi"
    elif probability >= 0.75:
        return "tinggi"
    elif probability >= 0.50:
        return "sedang"
    else:
        return "rendah"


# ---------------------------------------------------------------------------
# Counterfactual narrative (Task 21 integration)
# ---------------------------------------------------------------------------


def generate_counterfactual_narrative(suggestions: list[dict]) -> str:
    """Generate a Bahasa Indonesia narrative for counterfactual suggestions.

    Args:
        suggestions: Output from shap_counterfactual() — list of dicts with
            feature, current_value, suggestion, impact

    Returns:
        Bahasa Indonesia recommendation narrative.
    """
    if not suggestions:
        return "Tidak ada rekomendasi perubahan yang tersedia untuk pengadaan ini."

    if "message" in suggestions[0]:
        return "Pengadaan ini sudah berada pada tingkat risiko target."

    paragraphs = ["**Rekomendasi untuk menurunkan tingkat risiko:**\n"]

    for i, s in enumerate(suggestions, 1):
        label = _get_feature_label(s["feature"])
        current = _format_value(s["current_value"])
        suggestion_text = s.get("suggestion", "Tinjau parameter ini")
        impact = s.get("impact", 0)

        paragraphs.append(
            f"{i}. **{label}** (saat ini: {current})\n"
            f"   - Rekomendasi: {suggestion_text}\n"
            f"   - Dampak potensial: {impact:.4f}"
        )

    paragraphs.append(
        f"\n---\n\n*{DISCLAIMER}*"
    )

    return "\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Full report generator
# ---------------------------------------------------------------------------


def generate_full_report(
    explanation: dict,
    counterfactual_suggestions: list[dict] | None = None,
) -> str:
    """Generate a complete Bahasa Indonesia report combining
    explanation narrative and counterfactual recommendations.

    Args:
        explanation: Output from explain_single()
        counterfactual_suggestions: Optional output from shap_counterfactual()

    Returns:
        Complete Bahasa Indonesia report string.
    """
    sections = []

    # Header
    pred_label = CLASS_LABELS_ID.get(
        explanation["predicted_class"], explanation["predicted_label"]
    )
    sections.append(f"# Laporan Analisis Risiko Pengadaan\n")
    sections.append(f"**Klasifikasi:** {pred_label}")
    sections.append(f"**Probabilitas:** {explanation['probability']:.1%}\n")

    # Explanation narrative
    sections.append("## Analisis\n")
    sections.append(generate_narrative(explanation))

    # Counterfactual recommendations
    if counterfactual_suggestions:
        sections.append("\n## Rekomendasi\n")
        sections.append(generate_counterfactual_narrative(counterfactual_suggestions))

    return "\n".join(sections)
