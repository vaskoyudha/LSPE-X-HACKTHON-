"""Fraud-evidence dataset helpers for the separate stronger-label lane."""

from __future__ import annotations

import pandas as pd

from src.graph_features import build_relationship_features
from src.outcomes import add_evidence_strength

POSITIVE_EVIDENCE_FAMILIES = {"confirmed_fraud", "confirmed_irregularity"}


def build_fraud_evidence_dataset(
    raw_df: pd.DataFrame,
    base_features: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine raw rows, base features, graph features, and evidence labels."""

    merged = raw_df.reset_index(drop=True).copy()
    merged = pd.concat([merged, base_features.reset_index(drop=True)], axis=1)
    merged = pd.concat([merged, build_relationship_features(raw_df)], axis=1)

    evidence = evidence_df.copy()
    if "evidence_strength" not in evidence.columns:
        evidence = add_evidence_strength(evidence)
    evidence = evidence.drop_duplicates(subset=["ocid"], keep="first")

    merged = merged.merge(
        evidence[["ocid", "label_family", "label_value", "evidence_strength"]],
        on="ocid",
        how="left",
    )
    merged["fraud_evidence_target"] = (
        merged["label_family"].isin(POSITIVE_EVIDENCE_FAMILIES).astype(int)
    )
    merged["is_unlabeled"] = (merged["fraud_evidence_target"] == 0).astype(int)
    merged["sample_weight"] = merged["evidence_strength"].fillna(0.2).astype(float)
    return merged
