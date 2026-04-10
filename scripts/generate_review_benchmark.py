"""Generate a larger reviewed benchmark template from the held-out test split.

This script prepares a human-review sheet that can be used to collect:
  1. stronger reviewed labels on real held-out rows
  2. reviewer judgments about explanation quality/actionability

The output is intentionally a template: reviewers should fill
`reviewed_label`, `review_confidence`, `review_notes`,
`explanation_agrees`, `explanation_clarity`, `explanation_actionable`,
and `explanation_notes`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR
from src.explain import get_explainer, load_model as load_xai_model, explain_single
from src.model import (
    apply_temperature,
    load_decision_thresholds,
    load_model,
    load_test_artifacts,
    predict_with_thresholds,
)
from src.narrative import render_explanation_narrative
from src.split import load_raw_split

OUTPUT_PATH = PROCESSED_DIR / "review_benchmark_500.csv"
LABEL_TEXT = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)


def _select_review_rows(
    probs: np.ndarray,
    preds: np.ndarray,
    n_rows: int,
) -> tuple[np.ndarray, list[str]]:
    """Select a review mix: high-risk, uncertain, and stratified by class."""
    p_high = probs[:, 2]
    entropy = _entropy(probs)
    selected: list[int] = []
    reasons: list[str] = []

    def add_indices(indices: np.ndarray, reason: str, limit: int) -> None:
        for idx in indices:
            if idx not in selected:
                selected.append(int(idx))
                reasons.append(reason)
                if len([r for r in reasons if r == reason]) >= limit:
                    break

    high_n = max(1, int(n_rows * 0.40))
    uncertain_n = max(1, int(n_rows * 0.30))
    stratified_n = max(1, n_rows - high_n - uncertain_n)

    add_indices(np.argsort(p_high)[::-1], "high_risk_probability", high_n)
    add_indices(np.argsort(entropy)[::-1], "high_uncertainty", uncertain_n)

    remaining = [idx for idx in range(len(preds)) if idx not in selected]
    remaining_arr = np.array(remaining, dtype=int)
    if len(remaining_arr) > 0:
        per_class = max(1, stratified_n // 3)
        for cls in [2, 1, 0]:
            cls_idx = remaining_arr[preds[remaining_arr] == cls]
            add_indices(cls_idx, f"stratified_predicted_{LABEL_TEXT[cls].lower().replace(' ', '_')}", per_class)

    if len(selected) < n_rows:
        add_indices(np.argsort(p_high)[::-1], "top_up", n_rows)

    return np.array(selected[:n_rows], dtype=int), reasons[:n_rows]


def main(n_rows: int = 500) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    raw = load_raw_split("test").reset_index(drop=True)
    features, labels = load_test_artifacts()
    features = features.reset_index(drop=True)
    labels = labels.reset_index(drop=True)

    model = load_model()
    calibration_path = ROOT / "models" / "calibration.json"
    calibration = json.loads(calibration_path.read_text()) if calibration_path.exists() else None
    thresholds = load_decision_thresholds() or {"high_risk": 0.5, "low_risk": 0.5}

    probs = model.predict(__import__("xgboost").DMatrix(features))
    if calibration and calibration.get("enabled"):
        probs = apply_temperature(probs, calibration["temperature"])
    preds = predict_with_thresholds(probs, thresholds)

    selected_idx, reasons = _select_review_rows(probs, preds, min(n_rows, len(features)))

    xai_model = load_xai_model()
    explainer = get_explainer(xai_model)
    feature_names = list(features.columns)

    rows: list[dict[str, object]] = []
    for review_order, (idx, reason) in enumerate(zip(selected_idx, reasons), start=1):
        feature_row = features.iloc[idx]
        explanation = explain_single(
            feature_row.to_dict(),
            feature_names,
            model=xai_model,
            explainer=explainer,
            top_k=3,
        )
        narrative = render_explanation_narrative(explanation)
        factor_summary = " | ".join(
            f"{factor['feature']}={factor['shap_value']:.4f}"
            for factor in explanation["factors"]
        )

        rows.append(
            {
                "review_order": review_order,
                "source_partition": "test",
                "source_row_idx": int(idx),
                "sampling_reason": reason,
                "ocid": raw.iloc[idx].get("ocid", ""),
                "tender_title": raw.iloc[idx].get("tender_title", ""),
                "tender_datePublished": raw.iloc[idx].get("tender_datePublished", ""),
                "buyer_name": raw.iloc[idx].get("buyer_name", ""),
                "supplier_name": raw.iloc[idx].get("supplier_name", ""),
                "tender_value_amount": raw.iloc[idx].get("tender_value_amount", np.nan),
                "award_value_amount": raw.iloc[idx].get("award_value_amount", np.nan),
                "heuristic_risk_label": int(labels.iloc[idx]),
                "predicted_label": int(preds[idx]),
                "predicted_label_name": LABEL_TEXT[int(preds[idx])],
                "predicted_probability": round(float(probs[idx, preds[idx]]), 6),
                "prob_low": round(float(probs[idx, 0]), 6),
                "prob_medium": round(float(probs[idx, 1]), 6),
                "prob_high": round(float(probs[idx, 2]), 6),
                "prediction_entropy": round(float(_entropy(probs[idx:idx + 1])[0]), 6),
                "top_factors": factor_summary,
                "narrative_id": narrative,
                "reviewed_label": "",
                "review_confidence": "",
                "review_notes": "",
                "explanation_agrees": "",
                "explanation_clarity": "",
                "explanation_actionable": "",
                "explanation_notes": "",
            }
        )

    review_df = pd.DataFrame(rows)
    review_df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
    print(f"Saved review benchmark template to {OUTPUT_PATH} ({len(review_df)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-rows", type=int, default=500)
    args = parser.parse_args()
    main(n_rows=args.n_rows)
