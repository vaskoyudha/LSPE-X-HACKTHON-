"""Generate a showcase of official evidence-linked procurement cases with live model outputs."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.explain import explain_single, get_explainer
from src.model import (
    CLASS_NAMES,
    apply_temperature,
    load_decision_thresholds,
    load_model,
    predict_with_thresholds,
)
from src.narrative import derive_business_rating, render_explanation_narrative

EVIDENCE_DIR = ROOT / "data" / "processed" / "evidence"
LINKED_LABELS_PATH = EVIDENCE_DIR / "linked_label_records.parquet"
SHOWCASE_CSV_PATH = EVIDENCE_DIR / "official_evidence_showcase.csv"
SHOWCASE_MD_PATH = ROOT / "proposal" / "official_evidence_showcase.md"
CALIBRATION_PATH = ROOT / "models" / "calibration.json"

RAW_COLUMNS = [
    "ocid",
    "tender_title",
    "tender_datePublished",
    "buyer_name",
    "supplier_name",
    "tender_value_amount",
    "award_value_amount",
]


def _load_official_evidence_groups(path: Path = LINKED_LABELS_PATH) -> dict[str, list[dict[str, object]]]:
    if not path.exists():
        return {}
    linked = pd.read_parquet(path)
    if linked.empty or "ocid" not in linked.columns:
        return {}

    official = linked.dropna(subset=["ocid"]).copy()
    official = official[official["reviewer_needed"].fillna(True) == False]
    groups: dict[str, list[dict[str, object]]] = {}
    for ocid, group in official.groupby("ocid", sort=False):
        groups[str(ocid)] = group.to_dict(orient="records")
    return groups


def _load_split_artifacts() -> dict[str, tuple[pd.DataFrame, pd.DataFrame]]:
    artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for split in ["train", "test"]:
        raw = pd.read_parquet(ROOT / f"{split}_data" / "raw.parquet", columns=RAW_COLUMNS).reset_index(drop=True)
        features = pd.read_parquet(ROOT / f"{split}_data" / "features.parquet").reset_index(drop=True)
        if len(raw) != len(features):
            raise ValueError(f"Row mismatch between {split} raw and features artifacts")
        artifacts[split] = (raw, features)
    return artifacts


def _find_case_row(
    ocid: str,
    split_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
) -> tuple[str, pd.Series, pd.Series] | None:
    for split in ["test", "train"]:
        raw, features = split_artifacts[split]
        matches = raw.index[raw["ocid"].astype(str) == str(ocid)].tolist()
        if matches:
            idx = int(matches[0])
            return split, raw.iloc[idx], features.iloc[idx]
    return None


def _summarize_evidence(records: list[dict[str, object]]) -> tuple[str, str, str, float | None, int, int]:
    families = sorted({str(r.get("label_family", "")).strip() for r in records if r.get("label_family")})
    sources = sorted({str(r.get("source_name", "")).strip() for r in records if r.get("source_name")})
    stages = sorted({str(r.get("case_stage", "")).strip() for r in records if r.get("case_stage")})
    confidence_scores = [float(r.get("confidence_score")) for r in records if r.get("confidence_score") not in (None, "")]
    return (
        ", ".join(families),
        ", ".join(sources),
        ", ".join(stages),
        (max(confidence_scores) if confidence_scores else None),
        len(records),
        len(sources),
    )


def _score_case(
    model,
    explainer,
    calibration: dict | None,
    thresholds: dict[str, float],
    feature_row: pd.Series,
    evidence_records: list[dict[str, object]],
) -> dict[str, object]:
    feature_df = feature_row.to_frame().T
    probs = model.predict(__import__("xgboost").DMatrix(feature_df))
    if calibration and calibration.get("enabled"):
        probs = apply_temperature(probs, calibration["temperature"])
    pred = int(predict_with_thresholds(probs, thresholds)[0])

    explanation = explain_single(
        feature_row,
        feature_names=list(feature_df.columns),
        model=model,
        explainer=explainer,
        calibration=calibration,
        top_k=3,
    )
    business_rating = derive_business_rating(explanation, evidence_records=evidence_records)
    narrative = render_explanation_narrative(
        explanation,
        evidence_records=evidence_records,
        business_rating=business_rating,
    )
    factor_summary = " | ".join(
        f"{factor['feature']}={factor['shap_value']:.4f}" for factor in explanation.get("factors", [])
    )
    return {
        "predicted_label_name": CLASS_NAMES.get(pred, str(pred)),
        "predicted_probability": float(probs[0, pred]),
        "prob_low": float(probs[0, 0]),
        "prob_medium": float(probs[0, 1]),
        "prob_high": float(probs[0, 2]),
        "business_rating_label": business_rating["rating_label"],
        "business_rating_source": business_rating["rating_source"],
        "business_rating_reason": business_rating["rating_reason"],
        "top_factors": factor_summary,
        "narrative_id": narrative,
    }


def _build_showcase_rows(
    official_groups: dict[str, list[dict[str, object]]],
    split_artifacts: dict[str, tuple[pd.DataFrame, pd.DataFrame]],
    *,
    model,
    explainer,
    calibration: dict | None,
    thresholds: dict[str, float],
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ocid, records in official_groups.items():
        located = _find_case_row(ocid, split_artifacts)
        if located is None:
            continue
        split, raw_row, feature_row = located
        label_families, evidence_sources, case_stages, confidence_score, evidence_record_count, evidence_source_count = _summarize_evidence(records)
        scored = _score_case(
            model,
            explainer,
            calibration,
            thresholds,
            feature_row,
            records,
        )
        rows.append(
            {
                "ocid": ocid,
                "source_partition": split,
                "tender_title": raw_row.get("tender_title", ""),
                "tender_datePublished": raw_row.get("tender_datePublished", ""),
                "buyer_name": raw_row.get("buyer_name", ""),
                "supplier_name": raw_row.get("supplier_name", ""),
                "tender_value_amount": raw_row.get("tender_value_amount"),
                "award_value_amount": raw_row.get("award_value_amount"),
                "evidence_label_families": label_families,
                "evidence_sources": evidence_sources,
                "evidence_case_stages": case_stages,
                "evidence_confidence_score": confidence_score,
                "evidence_record_count": evidence_record_count,
                "evidence_source_count": evidence_source_count,
                **scored,
            }
        )
    return pd.DataFrame(rows)


def _build_showcase_markdown(showcase_df: pd.DataFrame) -> str:
    lines: list[str] = []
    lines.append("# Official Evidence Showcase")
    lines.append("")
    if showcase_df.empty:
        lines.append("No official evidence-linked cases were available in the current artifact set.")
        return "\n".join(lines)

    total_cases = len(showcase_df)
    high_risk_cases = int((showcase_df["predicted_label_name"] == "High Risk").sum())
    medium_risk_cases = int((showcase_df["predicted_label_name"] == "Medium Risk").sum())
    low_risk_cases = int((showcase_df["predicted_label_name"] == "Low Risk").sum())
    critical_cases = int((showcase_df["business_rating_label"] == "Risiko Kritis").sum())
    total_supporting_rows = int(showcase_df["evidence_record_count"].fillna(0).sum()) if "evidence_record_count" in showcase_df else total_cases
    corroborated_cases = int((showcase_df.get("evidence_source_count", pd.Series(dtype=float)).fillna(0) >= 2).sum())
    evidence_override_cases = int(
        ((showcase_df["predicted_label_name"] != "High Risk") & (showcase_df["business_rating_label"] == "Risiko Kritis")).sum()
    )

    lines.append("This artifact shows how LPSE-X behaves on procurement rows that already have official linked evidence.")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    lines.append(f"- Official evidence-linked cases found in current split artifacts: {total_cases}")
    lines.append(f"- Supporting official evidence rows behind those cases: {total_supporting_rows}")
    lines.append(f"- Cases with multi-source official corroboration: {corroborated_cases}")
    lines.append(f"- Model predicted High Risk: {high_risk_cases}")
    lines.append(f"- Model predicted Medium Risk: {medium_risk_cases}")
    lines.append(f"- Model predicted Low Risk: {low_risk_cases}")
    lines.append(f"- Final business rating Risiko Kritis after evidence escalation: {critical_cases}")
    lines.append(f"- Cases where evidence lane corrected a non-High-Risk model output: {evidence_override_cases}")
    lines.append("")
    lines.append("## Why this matters")
    lines.append("")
    lines.append(
        "The showcase demonstrates the value of the hybrid design: the model provides triage, while the evidence lane prevents known official cases from being understated when model-only risk is insufficient."
    )
    lines.append("")

    for idx, row in showcase_df.iterrows():
        lines.append(f"## Case {idx + 1}: {row['ocid']}")
        lines.append("")
        lines.append(f"- Split: {row['source_partition']}")
        lines.append(f"- Tender title: {row['tender_title']}")
        lines.append(f"- Buyer: {row['buyer_name']}")
        lines.append(f"- Supplier: {row['supplier_name']}")
        lines.append(f"- Evidence families: {row['evidence_label_families']}")
        lines.append(f"- Evidence sources: {row['evidence_sources']}")
        lines.append(f"- Case stages: {row['evidence_case_stages']}")
        lines.append(f"- Supporting evidence rows: {int(row.get('evidence_record_count', 0))}")
        lines.append(f"- Official source count: {int(row.get('evidence_source_count', 0))}")
        lines.append(f"- Model prediction: {row['predicted_label_name']} ({row['predicted_probability']:.2%})")
        lines.append(
            f"- Probability vector: low={row['prob_low']:.2%}, medium={row['prob_medium']:.2%}, high={row['prob_high']:.2%}"
        )
        lines.append(
            f"- Final business rating: {row['business_rating_label']} [{row['business_rating_source']}]"
        )
        lines.append(f"- Rating reason: {row['business_rating_reason']}")
        lines.append(f"- Top factors: {row['top_factors']}")
        lines.append("- Narrative:")
        for line in str(row['narrative_id']).splitlines():
            lines.append(f"  {line}")
        lines.append("")

    return "\n".join(lines)


def generate_official_evidence_showcase(
    csv_path: Path = SHOWCASE_CSV_PATH,
    markdown_path: Path = SHOWCASE_MD_PATH,
) -> tuple[Path, Path]:
    official_groups = _load_official_evidence_groups()
    split_artifacts = _load_split_artifacts()
    model = load_model()
    explainer = get_explainer(model)
    calibration = json.loads(CALIBRATION_PATH.read_text(encoding="utf-8")) if CALIBRATION_PATH.exists() else None
    thresholds = load_decision_thresholds() or {"high_risk": 0.5, "low_risk": 0.5}

    showcase_df = _build_showcase_rows(
        official_groups,
        split_artifacts,
        model=model,
        explainer=explainer,
        calibration=calibration,
        thresholds=thresholds,
    )
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    showcase_df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    markdown_path.write_text(_build_showcase_markdown(showcase_df), encoding="utf-8")
    return csv_path, markdown_path


if __name__ == "__main__":
    csv_path, markdown_path = generate_official_evidence_showcase()
    print(f"Saved official evidence showcase CSV to {csv_path}")
    print(f"Saved official evidence showcase markdown to {markdown_path}")
