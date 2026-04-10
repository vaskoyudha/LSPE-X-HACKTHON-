"""Canonical evidence-label helpers for reviewed and confirmed-outcome data."""

from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd

from src.data import PROCESSED_DIR

REVIEWED_LABELS_PATH = PROCESSED_DIR / "reviewed_row_labels.parquet"
CONFIRMED_OUTCOMES_PATH = PROCESSED_DIR / "fraud_outcomes.parquet"

ALLOWED_LABEL_FAMILIES = {
    "reviewed_risk",
    "confirmed_irregularity",
    "confirmed_fraud",
}
ALLOWED_REVIEWED_LABELS = {0: "low", 1: "medium", 2: "high"}
OUTCOME_TO_FAMILY = {
    "fraud": "confirmed_fraud",
    "corruption": "confirmed_fraud",
    "confirmed_fraud": "confirmed_fraud",
    "irregularity": "confirmed_irregularity",
    "sanction": "confirmed_irregularity",
    "confirmed_irregularity": "confirmed_irregularity",
}

REQUIRED_EVIDENCE_COLUMNS = [
    "ocid",
    "label_family",
    "label_value",
    "source_name",
    "source_type",
    "source_record_id",
    "decision_date",
    "confidence_score",
    "review_notes",
    "reviewer_id",
    "ingested_at",
]


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO timestamp."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def normalize_confidence_scores(values: pd.Series | object) -> pd.Series:
    """Normalize review confidence values into the 0.0-1.0 range."""
    confidence = pd.to_numeric(values, errors="coerce")
    if confidence.notna().any():
        max_conf = float(confidence.max(skipna=True) or 1.0)
        if max_conf > 1.0:
            confidence = confidence.clip(lower=0.0, upper=5.0) / 5.0
    return confidence.fillna(1.0).clip(lower=0.0, upper=1.0)


def _optional_series(df: pd.DataFrame, column: str, default: object = "") -> pd.Series:
    if column in df.columns:
        return df[column]
    return pd.Series([default] * len(df), index=df.index, dtype="object")


def validate_evidence_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize a canonical evidence-label dataframe."""
    missing = sorted(set(REQUIRED_EVIDENCE_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    normalized = df[REQUIRED_EVIDENCE_COLUMNS].copy()
    normalized["ocid"] = normalized["ocid"].fillna("").astype(str).str.strip()
    normalized["label_family"] = (
        normalized["label_family"].fillna("").astype(str).str.strip().str.lower()
    )
    normalized["label_value"] = (
        normalized["label_value"].fillna("").astype(str).str.strip().str.lower()
    )
    normalized["source_name"] = normalized["source_name"].fillna("").astype(str).str.strip()
    normalized["source_type"] = (
        normalized["source_type"].fillna("").astype(str).str.strip().str.lower()
    )
    normalized["source_record_id"] = (
        normalized["source_record_id"].fillna("").astype(str).str.strip()
    )
    normalized["review_notes"] = normalized["review_notes"].fillna("").astype(str)
    normalized["reviewer_id"] = normalized["reviewer_id"].fillna("").astype(str)
    normalized["decision_date"] = pd.to_datetime(
        normalized["decision_date"], errors="coerce"
    ).dt.strftime("%Y-%m-%d")
    normalized["ingested_at"] = pd.to_datetime(
        normalized["ingested_at"], errors="coerce", utc=True
    ).dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    normalized["confidence_score"] = pd.to_numeric(
        normalized["confidence_score"], errors="coerce"
    )

    if normalized["ocid"].eq("").any():
        raise ValueError("ocid must be non-empty for every evidence label row")
    if normalized["source_name"].eq("").any():
        raise ValueError("source_name must be non-empty for every evidence label row")
    if normalized["source_type"].eq("").any():
        raise ValueError("source_type must be non-empty for every evidence label row")
    if normalized["source_record_id"].eq("").any():
        raise ValueError("source_record_id must be non-empty for every evidence label row")
    if not normalized["label_family"].isin(ALLOWED_LABEL_FAMILIES).all():
        invalid = sorted(
            normalized.loc[
                ~normalized["label_family"].isin(ALLOWED_LABEL_FAMILIES),
                "label_family",
            ].unique()
        )
        raise ValueError(f"Invalid label_family values: {invalid}")
    if normalized["decision_date"].isna().any():
        raise ValueError("decision_date must parse as YYYY-MM-DD")
    if normalized["ingested_at"].isna().any():
        raise ValueError("ingested_at must parse as an ISO timestamp")
    if normalized["confidence_score"].isna().any():
        raise ValueError("confidence_score must be numeric between 0.0 and 1.0")
    if ((normalized["confidence_score"] < 0) | (normalized["confidence_score"] > 1)).any():
        raise ValueError("confidence_score must stay within 0.0 and 1.0")

    return normalized


def add_evidence_strength(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a simple evidence-strength score by label family."""

    weighted = df.copy()
    families = (
        weighted["label_family"].fillna("").astype(str).str.strip().str.lower()
    )
    weighted["label_family"] = families
    weighted["evidence_strength"] = families.map(
        {
            "reviewed_risk": 0.5,
            "confirmed_irregularity": 0.8,
            "confirmed_fraud": 1.0,
        }
    )
    if weighted["evidence_strength"].isna().any():
        invalid = sorted(families[weighted["evidence_strength"].isna()].unique())
        raise ValueError(f"Unsupported label_family values for evidence strength: {invalid}")
    return weighted


def normalize_reviewed_evidence_rows(
    imported: pd.DataFrame,
    review_base: pd.DataFrame,
    *,
    source_name: str,
    decision_date: str | None = None,
    ingested_at: str | None = None,
) -> pd.DataFrame:
    """Map imported reviewed rows into canonical evidence records."""
    imported = imported.copy()
    imported["source_row_idx"] = pd.to_numeric(
        imported["source_row_idx"], errors="coerce"
    )
    imported["reviewed_label"] = pd.to_numeric(
        imported["reviewed_label"], errors="coerce"
    )
    imported = imported[
        imported["source_row_idx"].notna()
        & imported["reviewed_label"].isin(ALLOWED_REVIEWED_LABELS)
    ].copy()
    imported["source_row_idx"] = imported["source_row_idx"].astype(int)
    imported["reviewed_label"] = imported["reviewed_label"].astype(int)

    base_cols = review_base[["source_row_idx", "ocid"]].copy()
    base_cols["source_row_idx"] = pd.to_numeric(base_cols["source_row_idx"], errors="coerce")
    base_cols = base_cols.dropna(subset=["source_row_idx", "ocid"]).drop_duplicates(
        subset=["source_row_idx"]
    )
    base_cols["source_row_idx"] = base_cols["source_row_idx"].astype(int)

    merged = imported.merge(base_cols, on="source_row_idx", how="left", validate="m:1")
    if merged["ocid"].isna().any():
        unresolved = merged.loc[merged["ocid"].isna(), "source_row_idx"].tolist()
        raise ValueError(f"Could not resolve ocid for source_row_idx values: {unresolved}")

    confidence = normalize_confidence_scores(merged.get("review_confidence"))

    now = ingested_at or utc_now_iso()
    review_date = decision_date or now[:10]
    normalized = pd.DataFrame(
        {
            "ocid": merged["ocid"].astype(str),
            "label_family": "reviewed_risk",
            "label_value": merged["reviewed_label"].map(ALLOWED_REVIEWED_LABELS),
            "source_name": source_name,
            "source_type": "human_review",
            "source_record_id": merged["source_row_idx"].astype(str),
            "decision_date": (
                merged["decision_date"]
                if "decision_date" in merged.columns
                else pd.Series(review_date, index=merged.index)
            ),
            "confidence_score": confidence,
            "review_notes": _optional_series(merged, "review_notes", ""),
            "reviewer_id": _optional_series(merged, "reviewer_id", ""),
            "ingested_at": now,
        }
    )
    return validate_evidence_labels(normalized)


def normalize_confirmed_outcome_rows(
    df: pd.DataFrame,
    *,
    source_name: str,
    source_type: str,
    ingested_at: str | None = None,
) -> pd.DataFrame:
    """Map raw confirmed-outcome rows into canonical evidence records."""
    required = {"ocid", "outcome_label", "source_record_id", "decision_date"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    now = ingested_at or utc_now_iso()
    labels = df["outcome_label"].fillna("").astype(str).str.strip().str.lower()
    normalized = pd.DataFrame(
        {
            "ocid": df["ocid"].astype(str).str.strip(),
            "label_family": labels.map(OUTCOME_TO_FAMILY),
            "label_value": labels,
            "source_name": source_name,
            "source_type": source_type,
            "source_record_id": df["source_record_id"].astype(str).str.strip(),
            "decision_date": df["decision_date"],
            "confidence_score": normalize_confidence_scores(
                _optional_series(df, "confidence_score", 1.0)
            ),
            "review_notes": _optional_series(df, "review_notes", ""),
            "reviewer_id": _optional_series(df, "reviewer_id", ""),
            "ingested_at": now,
        }
    )
    if normalized["label_family"].isna().any():
        invalid = sorted(labels[normalized["label_family"].isna()].unique())
        raise ValueError(f"Unsupported outcome_label values: {invalid}")
    return validate_evidence_labels(normalized)
