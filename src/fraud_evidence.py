"""Fraud-evidence dataset helpers for the separate stronger-label lane."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score

from src.graph_features import build_relationship_features
from src.outcomes import add_evidence_strength

POSITIVE_EVIDENCE_FAMILIES = {"confirmed_fraud", "confirmed_irregularity"}
UNLABELED_DEFAULT_WEIGHT = 0.2


def _prepare_evidence_rows(evidence_df: pd.DataFrame) -> pd.DataFrame:
    """Resolve one evidence row per ocid, prioritizing stronger positive labels."""

    evidence = evidence_df.copy()
    if "evidence_strength" not in evidence.columns:
        evidence = add_evidence_strength(evidence)

    evidence["label_family"] = evidence.get("label_family", "").fillna("").astype(str)
    evidence["evidence_strength"] = pd.to_numeric(
        evidence.get("evidence_strength"), errors="coerce"
    )
    evidence["_is_positive"] = evidence["label_family"].isin(POSITIVE_EVIDENCE_FAMILIES)
    evidence["_strength_rank"] = evidence["evidence_strength"].fillna(
        UNLABELED_DEFAULT_WEIGHT
    )

    return (
        evidence.sort_values(
            by=["ocid", "_is_positive", "_strength_rank"],
            ascending=[True, False, False],
            kind="mergesort",
        )
        .drop_duplicates(subset=["ocid"], keep="first")
        .drop(columns=["_is_positive", "_strength_rank"])
    )


def build_fraud_evidence_dataset(
    raw_df: pd.DataFrame,
    base_features: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    """Combine raw rows, base features, graph features, and evidence labels."""

    merged = raw_df.reset_index(drop=True).copy()
    merged = pd.concat([merged, base_features.reset_index(drop=True)], axis=1)
    merged = pd.concat([merged, build_relationship_features(raw_df)], axis=1)

    evidence = _prepare_evidence_rows(evidence_df)

    merged = merged.merge(
        evidence[["ocid", "label_family", "label_value", "evidence_strength"]],
        on="ocid",
        how="left",
    )
    merged["fraud_evidence_target"] = (
        merged["label_family"].isin(POSITIVE_EVIDENCE_FAMILIES).astype(int)
    )
    merged["is_unlabeled"] = (merged["fraud_evidence_target"] == 0).astype(int)
    merged["sample_weight"] = (
        merged["evidence_strength"].fillna(UNLABELED_DEFAULT_WEIGHT).astype(float)
    )
    return merged


def fraud_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the fraud-evidence feature columns."""

    return [col for col in df.columns if col.startswith("f_") or col.startswith("g_")]


def train_fraud_evidence_model(
    dataset: pd.DataFrame,
) -> tuple[xgb.XGBClassifier, dict[str, object]]:
    """Train a binary fraud-evidence model and return metrics."""

    feature_cols = fraud_feature_columns(dataset)
    if not feature_cols:
        raise ValueError("No fraud-evidence feature columns found (expected f_*/g_*).")

    X = dataset[feature_cols].fillna(0)
    y = dataset["fraud_evidence_target"].astype(int)
    if y.nunique() < 2:
        raise ValueError(
            "Fraud-evidence training requires at least one positive and one unlabeled row."
        )
    weights = dataset["sample_weight"].astype(float)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=weights)

    probs = model.predict_proba(X)[:, 1]
    top_k = max(1, int(np.ceil(len(probs) * 0.1)))
    top_idx = np.argsort(probs)[::-1][:top_k]
    top_precision = float(np.asarray(y)[top_idx].mean())

    metrics = {
        "label_type": "fraud_evidence_positive_unlabeled",
        "n_samples": int(len(dataset)),
        "n_positive": int(y.sum()),
        "average_precision": round(float(average_precision_score(y, probs)), 4),
        "roc_auc": round(float(roc_auc_score(y, probs)), 4),
        "precision_at_10pct": round(top_precision, 4),
        "feature_count": len(feature_cols),
    }
    return model, metrics
