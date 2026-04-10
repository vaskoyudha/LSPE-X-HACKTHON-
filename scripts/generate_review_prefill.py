"""Generate a draft review prefill file for the human review benchmark.

IMPORTANT:
- This file is a draft to help human reviewers move faster.
- It must not be presented as real human-reviewed validation.
- It intentionally writes to a separate CSV so the original human-review
  template remains untouched.
"""

from __future__ import annotations

import math
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR

SOURCE_PATH = PROCESSED_DIR / "review_benchmark_500.csv"
OUTPUT_PATH = PROCESSED_DIR / "review_benchmark_500_draft.csv"

LABEL_NAMES = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def _to_int(value, default: int = 1) -> int:
    try:
        return int(value)
    except Exception:
        return default


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _extract_factor_names(top_factors: str) -> list[str]:
    if not isinstance(top_factors, str) or not top_factors.strip():
        return []
    parts = [segment.strip() for segment in top_factors.split("|")]
    names: list[str] = []
    for part in parts:
        match = re.match(r"([A-Za-z0-9_]+)=", part)
        if match:
            names.append(match.group(1))
    return names


def _feature_phrase(feature_name: str) -> str:
    mapping = {
        "f_buyer_supplier_repeat_count": "hubungan buyer-supplier yang berulang",
        "f_supplier_recent_90d_award_count": "lonjakan kemenangan supplier dalam 90 hari",
        "f_tender_value_zscore_buyer": "lonjakan nilai tender relatif terhadap histori buyer",
        "f_price_deviation_ratio": "deviasi harga tender-award",
        "f_title_length": "judul tender yang terlalu singkat",
        "f_description_length": "deskripsi tender yang terlalu singkat",
        "f_title_token_count": "judul tender yang minim detail",
        "f_description_token_count": "deskripsi tender yang minim detail",
        "f_is_q4": "timing akhir tahun anggaran",
        "f_is_december": "publikasi pada bulan Desember",
        "f_tender_value_log": "nilai tender yang besar",
        "f_award_value_log": "nilai pemenang yang besar",
    }
    return mapping.get(feature_name, feature_name.replace("f_", "").replace("_", " "))


def _decide_review_label(row: pd.Series) -> tuple[int, str, str]:
    predicted = _to_int(row.get("predicted_label"), _to_int(row.get("heuristic_risk_label"), 1))
    heuristic = _to_int(row.get("heuristic_risk_label"), predicted)
    prob_low = _to_float(row.get("prob_low"))
    prob_medium = _to_float(row.get("prob_medium"))
    prob_high = _to_float(row.get("prob_high"))
    entropy = _to_float(row.get("prediction_entropy"), 0.0)

    # Conservative draft logic: align with strong probabilities, otherwise
    # keep the central class unless heuristic and prediction agree.
    if prob_high >= 0.85:
        label = 2
    elif prob_low >= 0.85:
        label = 0
    elif predicted == heuristic:
        label = predicted
    elif entropy >= 0.8:
        label = 1
    else:
        label = predicted

    if max(prob_low, prob_medium, prob_high) >= 0.9 and predicted == heuristic:
        confidence = "high"
    elif max(prob_low, prob_medium, prob_high) >= 0.7:
        confidence = "medium"
    else:
        confidence = "low"

    factors = _extract_factor_names(str(row.get("top_factors", "")))
    factor_phrases = [_feature_phrase(name) for name in factors[:2]]
    if factor_phrases:
        notes = "Draft review: risiko didukung oleh " + " dan ".join(factor_phrases) + "."
    else:
        notes = "Draft review: gunakan konteks paket dan probabilitas model untuk verifikasi manual."

    return label, confidence, notes


def _decide_explanation_fields(row: pd.Series, reviewed_label: int, confidence: str) -> tuple[str, int, str, str]:
    narrative = str(row.get("narrative_id", "") or "")
    top_factors = str(row.get("top_factors", "") or "")
    entropy = _to_float(row.get("prediction_entropy"), 0.0)

    has_narrative = bool(narrative.strip())
    has_factors = bool(top_factors.strip())

    if has_narrative and has_factors and confidence != "low":
        agrees = "yes"
    elif has_narrative or has_factors:
        agrees = "yes"
    else:
        agrees = "no"

    if not has_narrative:
        clarity = 2
    elif confidence == "high" and entropy < 0.5:
        clarity = 4
    elif confidence == "medium":
        clarity = 3
    else:
        clarity = 2

    actionable = "yes" if reviewed_label in {1, 2} and has_factors else "no"
    if actionable == "yes":
        notes = "Draft review: penjelasan tampak cukup membantu untuk prioritisasi audit awal."
    else:
        notes = "Draft review: penjelasan masih perlu verifikasi manual sebelum dipakai untuk keputusan."

    return agrees, clarity, actionable, notes


def main() -> None:
    if not SOURCE_PATH.exists():
        raise FileNotFoundError(
            f"{SOURCE_PATH} not found. Generate the human review benchmark first."
        )

    df = pd.read_csv(SOURCE_PATH)

    reviewed_labels: list[int] = []
    review_confidences: list[str] = []
    review_notes: list[str] = []
    explanation_agrees: list[str] = []
    explanation_clarity: list[int] = []
    explanation_actionable: list[str] = []
    explanation_notes: list[str] = []

    for _, row in df.iterrows():
        label, confidence, notes = _decide_review_label(row)
        agrees, clarity, actionable, explanation_note = _decide_explanation_fields(
            row,
            label,
            confidence,
        )
        reviewed_labels.append(label)
        review_confidences.append(confidence)
        review_notes.append(notes)
        explanation_agrees.append(agrees)
        explanation_clarity.append(clarity)
        explanation_actionable.append(actionable)
        explanation_notes.append(explanation_note)

    prefill = df.copy()
    prefill["reviewed_label"] = reviewed_labels
    prefill["review_confidence"] = review_confidences
    prefill["review_notes"] = review_notes
    prefill["explanation_agrees"] = explanation_agrees
    prefill["explanation_clarity"] = explanation_clarity
    prefill["explanation_actionable"] = explanation_actionable
    prefill["explanation_notes"] = explanation_notes
    prefill["review_source"] = "draft_prefill"
    prefill["review_warning"] = "Draft only. Manual verification required before use."

    prefill.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved draft review prefill file to {OUTPUT_PATH} ({len(prefill)} rows)")


if __name__ == "__main__":
    main()
