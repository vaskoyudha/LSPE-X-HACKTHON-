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
from src.narrative import derive_business_rating, render_explanation_narrative
from src.split import load_raw_split

OUTPUT_PATH = PROCESSED_DIR / "review_benchmark_500.csv"
EVIDENCE_LABEL_PATH = PROCESSED_DIR / "evidence" / "linked_label_records.parquet"
LABEL_TEXT = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}


def _load_linked_evidence_by_ocid(path: Path = EVIDENCE_LABEL_PATH) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {}
    evidence_df = pd.read_parquet(path)
    if "ocid" not in evidence_df.columns or evidence_df.empty:
        return {}

    grouped: dict[str, list[dict[str, object]]] = {}
    for ocid, group in evidence_df.dropna(subset=["ocid"]).groupby("ocid", sort=False):
        grouped[str(ocid)] = group.to_dict(orient="records")
    return grouped


def _summarize_evidence_records(evidence_records: list[dict[str, object]]) -> tuple[str, str, bool]:
    if not evidence_records:
        return "", "", False

    families = sorted(
        {str(record.get("label_family", "")).strip() for record in evidence_records if record.get("label_family")}
    )
    sources = sorted(
        {str(record.get("source_name", "")).strip() for record in evidence_records if record.get("source_name")}
    )
    has_official_evidence = any(not bool(record.get("reviewer_needed", False)) for record in evidence_records)
    return "|".join(families), "|".join(sources), has_official_evidence



def _prioritize_evidence_rows(
    raw: pd.DataFrame,
    evidence_by_ocid: dict[str, list[dict[str, object]]],
    *,
    limit: int,
) -> tuple[list[int], list[str]]:
    if limit <= 0 or raw.empty or not evidence_by_ocid:
        return [], []

    official_indices: list[int] = []
    review_indices: list[int] = []
    seen: set[int] = set()

    for idx, ocid in enumerate(raw.get("ocid", pd.Series([], dtype=object)).astype(str).tolist()):
        evidence_records = evidence_by_ocid.get(ocid, [])
        if not evidence_records:
            continue
        has_official = any(not bool(record.get("reviewer_needed", False)) for record in evidence_records)
        if has_official and idx not in seen:
            official_indices.append(idx)
            seen.add(idx)
        elif idx not in seen:
            review_indices.append(idx)
            seen.add(idx)

    selected = (official_indices + review_indices)[:limit]
    reasons = ["official_evidence_linked"] * min(len(official_indices), len(selected))
    reasons.extend(["evidence_needs_review"] * (len(selected) - len(reasons)))
    return selected, reasons



def _entropy(probs: np.ndarray) -> np.ndarray:
    eps = 1e-12
    return -np.sum(probs * np.log(probs + eps), axis=1)



def _nearest_threshold_margin(
    probs: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Distance to the nearest low/high decision threshold."""
    probs = np.asarray(probs, dtype=float)
    if probs.ndim != 2 or probs.shape[1] != 3:
        raise ValueError("Expected probability matrix with shape (n_samples, 3)")

    high_threshold = float(thresholds["high_risk"])
    low_threshold = float(thresholds["low_risk"])
    high_margin = np.abs(probs[:, 2] - high_threshold)
    low_margin = np.abs(probs[:, 0] - low_threshold)
    return np.minimum(high_margin, low_margin)



def _review_priority_score(
    probs: np.ndarray,
    preds: np.ndarray,
    heuristic_labels: np.ndarray,
    thresholds: dict[str, float],
) -> np.ndarray:
    """Composite queue score for manual review prioritization.

    Higher values mean the row is more useful for review collection: strong high-risk
    signal, disagreement between heuristic and model, high uncertainty, and proximity
    to a decision threshold.
    """
    probs = np.asarray(probs, dtype=float)
    preds = np.asarray(preds, dtype=int)
    heuristic = np.asarray(heuristic_labels, dtype=int)

    p_high = probs[:, 2]
    entropy = _entropy(probs)
    entropy_norm = entropy / np.log(probs.shape[1])
    disagreement = (preds != heuristic).astype(float)
    margin = _nearest_threshold_margin(probs, thresholds)
    boundary_focus = 1.0 - np.clip(margin / 0.25, 0.0, 1.0)

    score = (
        0.45 * p_high
        + 0.25 * entropy_norm
        + 0.20 * disagreement
        + 0.10 * boundary_focus
    )
    return np.clip(score, 0.0, 1.0)



def _select_review_rows(
    probs: np.ndarray,
    preds: np.ndarray,
    heuristic_labels: np.ndarray,
    thresholds: dict[str, float],
    n_rows: int,
) -> tuple[np.ndarray, list[str]]:
    """Select a review mix: high-risk, disagreement, uncertainty, threshold-boundary, and stratified rows."""
    p_high = probs[:, 2]
    entropy = _entropy(probs)
    disagreement_mask = preds != heuristic_labels
    threshold_margin = _nearest_threshold_margin(probs, thresholds)
    priority = _review_priority_score(probs, preds, heuristic_labels, thresholds)

    selected: list[int] = []
    reasons: list[str] = []

    def add_indices(indices: np.ndarray, reason: str, limit: int) -> None:
        added = 0
        for idx in indices:
            idx_int = int(idx)
            if idx_int not in selected:
                selected.append(idx_int)
                reasons.append(reason)
                added += 1
                if added >= limit:
                    break

    high_n = max(1, int(round(n_rows * 0.30)))
    disagreement_n = max(1, int(round(n_rows * 0.25)))
    uncertain_n = max(1, int(round(n_rows * 0.20)))
    boundary_n = max(1, int(round(n_rows * 0.15)))
    stratified_n = max(1, n_rows - high_n - disagreement_n - uncertain_n - boundary_n)

    add_indices(np.argsort(priority)[::-1], "priority_mix", high_n)

    disagreement_idx = np.flatnonzero(disagreement_mask)
    if len(disagreement_idx) > 0:
        disagreement_sorted = disagreement_idx[np.argsort(priority[disagreement_idx])[::-1]]
        add_indices(disagreement_sorted, "model_heuristic_disagreement", disagreement_n)

    add_indices(np.argsort(entropy)[::-1], "high_uncertainty", uncertain_n)
    add_indices(np.argsort(threshold_margin), "near_decision_threshold", boundary_n)

    remaining = np.array([idx for idx in range(len(preds)) if idx not in selected], dtype=int)
    if len(remaining) > 0:
        per_class = max(1, stratified_n // 3)
        for cls in [2, 1, 0]:
            cls_idx = remaining[preds[remaining] == cls]
            if len(cls_idx) == 0:
                continue
            cls_sorted = cls_idx[np.argsort(priority[cls_idx])[::-1]]
            add_indices(
                cls_sorted,
                f"stratified_predicted_{LABEL_TEXT[cls].lower().replace(' ', '_')}",
                per_class,
            )

    if len(selected) < n_rows:
        add_indices(np.argsort(priority)[::-1], "top_up", n_rows)

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
    heuristic_labels = labels.to_numpy(dtype=int)
    evidence_by_ocid = _load_linked_evidence_by_ocid()

    priority_evidence_limit = min(
        len(features),
        min(25, max(5, int(round(min(n_rows, len(features)) * 0.10)))),
    )
    priority_selected, priority_reasons = _prioritize_evidence_rows(
        raw,
        evidence_by_ocid,
        limit=priority_evidence_limit,
    )

    base_selected, base_reasons = _select_review_rows(
        probs,
        preds,
        heuristic_labels,
        thresholds,
        min(n_rows, len(features)),
    )
    selected_idx: list[int] = list(priority_selected)
    reasons: list[str] = list(priority_reasons)
    for idx, reason in zip(base_selected.tolist(), base_reasons):
        if len(selected_idx) >= min(n_rows, len(features)):
            break
        if idx not in selected_idx:
            selected_idx.append(int(idx))
            reasons.append(reason)

    priority_scores = _review_priority_score(probs, preds, heuristic_labels, thresholds)
    threshold_margins = _nearest_threshold_margin(probs, thresholds)

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
            calibration=calibration,
            top_k=3,
        )
        ocid = str(raw.iloc[idx].get("ocid", "") or "")
        evidence_records = evidence_by_ocid.get(ocid, [])
        business_rating = derive_business_rating(explanation, evidence_records=evidence_records)
        narrative = render_explanation_narrative(
            explanation,
            evidence_records=evidence_records,
            business_rating=business_rating,
        )
        factor_summary = " | ".join(
            f"{factor['feature']}={factor['shap_value']:.4f}"
            for factor in explanation["factors"]
        )
        evidence_families, evidence_sources, has_official_evidence = _summarize_evidence_records(evidence_records)

        rows.append(
            {
                "review_order": review_order,
                "source_partition": "test",
                "source_row_idx": int(idx),
                "sampling_reason": reason,
                "review_priority_score": round(float(priority_scores[idx]), 6),
                "heuristic_pred_disagree": bool(preds[idx] != heuristic_labels[idx]),
                "nearest_threshold_margin": round(float(threshold_margins[idx]), 6),
                "high_risk_threshold": round(float(thresholds["high_risk"]), 6),
                "low_risk_threshold": round(float(thresholds["low_risk"]), 6),
                "high_risk_margin": round(float(probs[idx, 2] - thresholds["high_risk"]), 6),
                "low_risk_margin": round(float(probs[idx, 0] - thresholds["low_risk"]), 6),
                "ocid": ocid,
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
                "business_rating_label": business_rating["rating_label"],
                "business_rating_source": business_rating["rating_source"],
                "business_rating_reason": business_rating["rating_reason"],
                "evidence_label_families": evidence_families,
                "evidence_sources": evidence_sources,
                "has_official_evidence": bool(has_official_evidence),
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
