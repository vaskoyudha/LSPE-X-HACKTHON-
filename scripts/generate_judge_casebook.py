"""Generate a judge-facing markdown casebook from evidence and review artifacts."""

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import FLAT_PARQUET

EVIDENCE_DIR = ROOT / "data" / "processed" / "evidence"
EVIDENCE_SUMMARY_PATH = EVIDENCE_DIR / "evidence_match_summary.json"
LINKED_LABELS_PATH = EVIDENCE_DIR / "linked_label_records.parquet"
REVIEW_BENCHMARK_PATH = ROOT / "data" / "processed" / "review_benchmark_500.csv"
OUTPUT_PATH = ROOT / "proposal" / "judge_casebook.md"

PROCUREMENT_COLUMNS = [
    "ocid",
    "tender_title",
    "tender_datePublished",
    "buyer_name",
    "supplier_name",
    "tender_value_amount",
    "award_value_amount",
]


def _format_currency(value: object) -> str:
    if value is None or value == "" or pd.isna(value):
        return "n/a"
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"Rp {numeric:,.0f}".replace(",", ".")


def _load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _load_review_benchmark(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, encoding="utf-8-sig")


def _load_procurement_lookup(ocids: list[str]) -> pd.DataFrame:
    if not ocids or not FLAT_PARQUET.exists():
        return pd.DataFrame(columns=PROCUREMENT_COLUMNS)
    procurement = pd.read_parquet(FLAT_PARQUET, columns=PROCUREMENT_COLUMNS)
    return procurement[procurement["ocid"].astype(str).isin(ocids)].drop_duplicates(subset=["ocid"]).reset_index(drop=True)


def _build_official_case_rows(linked_labels: pd.DataFrame) -> pd.DataFrame:
    if linked_labels.empty or "ocid" not in linked_labels.columns:
        return pd.DataFrame()

    matched = linked_labels.dropna(subset=["ocid"]).copy()
    if matched.empty:
        return matched

    grouped = (
        matched.groupby("ocid", dropna=False)
        .agg(
            evidence_record_count=("source_record_id", "count"),
            official_source_count=("source_name", lambda s: len({str(v) for v in s if pd.notna(v) and str(v).strip()})),
            label_families=("label_family", lambda s: ", ".join(sorted({str(v) for v in s if pd.notna(v) and str(v).strip()}))),
            source_names=("source_name", lambda s: ", ".join(sorted({str(v) for v in s if pd.notna(v) and str(v).strip()}))),
            case_stages=("case_stage", lambda s: ", ".join(sorted({str(v) for v in s if pd.notna(v) and str(v).strip()}))),
            decision_dates=("decision_date", lambda s: ", ".join(sorted({str(v) for v in s if pd.notna(v) and str(v).strip()}))),
            confidence_score=("confidence_score", "max"),
            reviewer_needed=("reviewer_needed", "max"),
            package_name=("package_name", lambda s: next((str(v) for v in s if pd.notna(v) and str(v).strip()), "")),
            provenance_note=("provenance_note", lambda s: next((str(v) for v in s if pd.notna(v) and str(v).strip()), "")),
        )
        .reset_index()
    )

    procurement_lookup = _load_procurement_lookup(grouped["ocid"].astype(str).tolist())
    if not procurement_lookup.empty:
        grouped = grouped.merge(procurement_lookup, on="ocid", how="left")
    return grouped


def generate_judge_casebook(output_path: Path = OUTPUT_PATH) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    evidence_summary = _load_json(EVIDENCE_SUMMARY_PATH)
    linked_labels = _load_parquet(LINKED_LABELS_PATH)
    review_benchmark = _load_review_benchmark(REVIEW_BENCHMARK_PATH)
    official_cases = _build_official_case_rows(linked_labels)
    unmatched_cases = linked_labels[linked_labels["ocid"].isna()].copy() if not linked_labels.empty and "ocid" in linked_labels.columns else pd.DataFrame()

    lines: list[str] = []
    lines.append("# LPSE-X Judge Casebook")
    lines.append("")
    lines.append("LPSE-X is positioned as an evidence-backed procurement risk triage and investigation support system, not a pure confirmed-fraud predictor.")
    lines.append("")
    lines.append("## 4-Level Judge-Facing Rating")
    lines.append("")
    lines.append("- Aman: model sees low immediate procurement-risk signal.")
    lines.append("- Perlu Pantauan: package deserves monitoring or manual follow-up.")
    lines.append("- Risiko Tinggi: strong triage signal from the model, but not yet official proof.")
    lines.append("- Risiko Kritis: official linked evidence exists, such as confirmed fraud, sanctions, or formal irregularity findings.")
    lines.append("")

    if evidence_summary:
        lines.append("## Evidence Lane Snapshot")
        lines.append("")
        lines.append(f"- Total evidence rows: {evidence_summary.get('total_evidence_rows', 0)}")
        lines.append(f"- Matched to procurement rows: {evidence_summary.get('matched_rows', 0)}")
        lines.append(f"- Still needing reviewer confirmation: {evidence_summary.get('needs_review_rows', 0)}")
        lines.append(f"- Linking confidence threshold: {evidence_summary.get('min_confidence', 0.0)}")
        lines.append("")

    lines.append("## Official Evidence-Linked Cases")
    lines.append("")
    if official_cases.empty:
        lines.append("No linked official-evidence cases are available yet.")
        lines.append("")
    else:
        for idx, row in official_cases.head(10).iterrows():
            lines.append(f"### Case {idx + 1}: {row.get('ocid', 'unknown ocid')}")
            lines.append("")
            lines.append(f"- Tender title: {row.get('tender_title', row.get('package_name', 'n/a'))}")
            lines.append(f"- Buyer: {row.get('buyer_name', 'n/a')}")
            lines.append(f"- Supplier: {row.get('supplier_name', 'n/a')}")
            lines.append(f"- Label families: {row.get('label_families', 'n/a')}")
            lines.append(f"- Source(s): {row.get('source_names', 'n/a')}")
            lines.append(f"- Supporting evidence rows: {row.get('evidence_record_count', 'n/a')}")
            lines.append(f"- Official source count: {row.get('official_source_count', 'n/a')}")
            lines.append(f"- Case stage(s): {row.get('case_stages', 'n/a')}")
            lines.append(f"- Decision date(s): {row.get('decision_dates', 'n/a')}")
            lines.append(f"- Confidence score: {row.get('confidence_score', 'n/a')}")
            lines.append(f"- Tender value: {_format_currency(row.get('tender_value_amount'))}")
            lines.append(f"- Award value: {_format_currency(row.get('award_value_amount'))}")
            provenance_note = str(row.get('provenance_note', '') or '').strip()
            if provenance_note:
                lines.append(f"- Provenance note: {provenance_note}")
            lines.append("")

    lines.append("## Evidence Rows Still Needing Review")
    lines.append("")
    if unmatched_cases.empty:
        lines.append("No unmatched evidence rows remain in the current artifact set.")
        lines.append("")
    else:
        for idx, row in unmatched_cases.head(10).iterrows():
            lines.append(f"- {row.get('source_record_id', f'row-{idx}')}: {row.get('label_family', 'unknown')} from {row.get('source_name', 'unknown source')} (reviewer_needed={row.get('reviewer_needed', True)})")
        lines.append("")

    lines.append("## Judge Demo Archetypes")
    lines.append("")
    archetype_count = 0
    if not official_cases.empty:
        critical = official_cases.iloc[0]
        archetype_count += 1
        lines.append(f"### Archetype {archetype_count}: Official evidence-linked critical case")
        lines.append("")
        lines.append(f"- OCID: {critical.get('ocid', 'n/a')}")
        lines.append(f"- Tender title: {critical.get('tender_title', critical.get('package_name', 'n/a'))}")
        lines.append(f"- Buyer: {critical.get('buyer_name', 'n/a')}")
        lines.append(f"- Supplier: {critical.get('supplier_name', 'n/a')}")
        lines.append(f"- Recommended business rating: Risiko Kritis")
        lines.append(f"- Evidence source(s): {critical.get('source_names', 'n/a')}")
        lines.append(f"- Evidence families: {critical.get('label_families', 'n/a')}")
        lines.append("")

    if not unmatched_cases.empty:
        review_needed = unmatched_cases.iloc[0]
        archetype_count += 1
        lines.append(f"### Archetype {archetype_count}: Evidence row still needing reviewer confirmation")
        lines.append("")
        lines.append(f"- Source record: {review_needed.get('source_record_id', 'n/a')}")
        lines.append(f"- Label family: {review_needed.get('label_family', 'n/a')}")
        lines.append(f"- Source: {review_needed.get('source_name', 'n/a')}")
        lines.append(f"- Reviewer needed: {review_needed.get('reviewer_needed', True)}")
        provenance_note = str(review_needed.get('provenance_note', '') or '').strip()
        if provenance_note:
            lines.append(f"- Provenance note: {provenance_note}")
        lines.append("")

    if not review_benchmark.empty:
        model_only_rows = review_benchmark[
            review_benchmark.get("business_rating_source", pd.Series([], dtype=object)).fillna("") != "official_evidence"
        ]
        if not model_only_rows.empty:
            model_only = model_only_rows.iloc[0]
            archetype_count += 1
            lines.append(f"### Archetype {archetype_count}: Model-only high-risk triage case")
            lines.append("")
            lines.append(f"- OCID: {model_only.get('ocid', 'n/a')}")
            lines.append(f"- Tender title: {model_only.get('tender_title', 'n/a')}")
            lines.append(f"- Buyer: {model_only.get('buyer_name', 'n/a')}")
            lines.append(f"- Supplier: {model_only.get('supplier_name', 'n/a')}")
            lines.append(f"- Predicted label: {model_only.get('predicted_label_name', 'n/a')}")
            lines.append(f"- Business rating: {model_only.get('business_rating_label', 'n/a')} [{model_only.get('business_rating_source', 'n/a')}]")
            lines.append(f"- Sampling reason: {model_only.get('sampling_reason', 'n/a')}")
            lines.append("")

    if archetype_count == 0:
        lines.append("No demo archetypes are available yet.")
        lines.append("")

    lines.append("## Top Review / Demo Rows")
    lines.append("")
    if review_benchmark.empty:
        lines.append("review_benchmark_500.csv is not available yet.")
        lines.append("")
    else:
        preview = review_benchmark.head(5)
        for idx, row in preview.iterrows():
            lines.append(f"### Demo Row {idx + 1}: {row.get('ocid', 'unknown ocid')}")
            lines.append("")
            lines.append(f"- Tender title: {row.get('tender_title', 'n/a')}")
            lines.append(f"- Buyer: {row.get('buyer_name', 'n/a')}")
            lines.append(f"- Supplier: {row.get('supplier_name', 'n/a')}")
            lines.append(f"- Predicted label: {row.get('predicted_label_name', 'n/a')} ({row.get('predicted_probability', 'n/a')})")
            lines.append(f"- Business rating: {row.get('business_rating_label', 'n/a')} [{row.get('business_rating_source', 'n/a')}]")
            evidence_families = row.get('evidence_label_families', '')
            if pd.isna(evidence_families) or not str(evidence_families).strip() or str(evidence_families).strip().lower() == 'nan':
                evidence_families = 'none linked yet'
            lines.append(f"- Evidence families: {evidence_families}")
            lines.append(f"- Review priority: {row.get('review_priority_score', 'n/a')} via {row.get('sampling_reason', 'n/a')}")
            lines.append(f"- Top factors: {row.get('top_factors', 'n/a')}")
            narrative = str(row.get('narrative_id', '') or '').strip()
            if narrative:
                lines.append("- Narrative:")
                for line in narrative.splitlines():
                    lines.append(f"  {line}")
            lines.append("")

    lines.append("## Judge Notes")
    lines.append("")
    lines.append("- The live model is still a 3-class XGBoost procurement risk model trained on heuristic risk labels.")
    lines.append("- The business-facing 4-level scale is a transparent presentation layer: only official linked evidence can escalate a package to Risiko Kritis.")
    lines.append("- This casebook is meant to support demo storytelling, reviewer calibration, and investigation handoff.")
    lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


if __name__ == "__main__":
    output = generate_judge_casebook()
    print(f"Saved judge casebook to {output}")
