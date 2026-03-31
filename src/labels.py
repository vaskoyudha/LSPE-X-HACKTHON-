"""Weak-labeling and calibration-sample utilities.

Implements ICW-style Potential Fraud Analysis (PFA) heuristic red-flag
indicators for procurement risk classification.

IMPORTANT:
- Labels are HEURISTIC risk indicators, NOT confirmed fraud outcomes.
- Expanding-window rules use past-only history (no look-ahead).
- Circularity risk between red-flag features and red-flag labels is
  acknowledged and documented for Bab 2/Bab 3.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.data import PROJECT_ROOT

logger = logging.getLogger(__name__)

TRAIN_DIR = PROJECT_ROOT / "train_data"
TEST_DIR = PROJECT_ROOT / "test_data"


# ---------------------------------------------------------------------------
# Individual red-flag indicators (ICW PFA-based)
# ---------------------------------------------------------------------------


def flag_single_bidder(df: pd.DataFrame) -> pd.Series:
    """Red flag: single bidder (numberOfTenderers == 1).

    Indicates limited competition, a common collusion signal.
    """
    n = df.get("tender_numberOfTenderers")
    if n is None:
        return pd.Series(False, index=df.index, dtype=bool)
    return n.fillna(-1).astype(float) == 1.0


def flag_short_title(df: pd.DataFrame, threshold: int = 20) -> pd.Series:
    """Red flag: tender title shorter than threshold characters.

    Short titles may indicate copy-paste or template fraud.
    """
    title = df.get("tender_title", pd.Series("", index=df.index))
    return title.fillna("").str.len() < threshold


def flag_short_description(df: pd.DataFrame, threshold: int = 60) -> pd.Series:
    """Red flag: tender description shorter than threshold characters.

    Short descriptions suggest inadequate specification.
    """
    desc = df.get("tender_description", pd.Series("", index=df.index))
    return desc.fillna("").str.len() < threshold


def flag_q4_timing(df: pd.DataFrame) -> pd.Series:
    """Red flag: procurement in Q4 (Oct-Dec).

    Year-end fiscal rush correlates with higher irregularity risk.
    """
    date_col = "tender_datePublished"
    if date_col not in df.columns:
        return pd.Series(False, index=df.index, dtype=bool)

    month = pd.to_datetime(df[date_col], errors="coerce").dt.month
    return month.isin([10, 11, 12]).fillna(False)


def flag_price_deviation(
    df: pd.DataFrame,
    low_threshold: float = 0.7,
    high_threshold: float = 1.0,
) -> pd.Series:
    """Red flag: award value deviates significantly from tender estimate.

    A ratio very close to 1.0 (ceiling price) or very low (<0.7) is suspicious.
    """
    tender_val = pd.to_numeric(df.get("tender_value_amount"), errors="coerce")
    award_val = pd.to_numeric(df.get("award_value_amount"), errors="coerce")

    ratio = award_val / tender_val.replace(0, np.nan)

    # Flag if ratio >= high_threshold (suspiciously close to ceiling)
    # or ratio <= low_threshold (suspiciously low)
    suspicious = (ratio >= high_threshold) | (ratio <= low_threshold)
    return suspicious.fillna(False)


def flag_high_value(df: pd.DataFrame, percentile: float = 0.9) -> pd.Series:
    """Red flag: contract value above the given percentile.

    High-value contracts attract more corruption risk.
    """
    val = pd.to_numeric(df.get("tender_value_amount"), errors="coerce")
    threshold = val.quantile(percentile)
    return (val >= threshold).fillna(False)


def flag_direct_procurement(df: pd.DataFrame) -> pd.Series:
    """Red flag: non-competitive procurement method.

    Direct / limited procurement bypasses open competition.
    """
    method = df.get("tender_procurementMethod", pd.Series("", index=df.index))
    method_lower = method.fillna("").str.lower()
    # In Indonesian OCDS: "direct", "limited", "penunjukan langsung"
    return method_lower.isin(["direct", "limited", "selective"])


# ---------------------------------------------------------------------------
# Composite risk labeling
# ---------------------------------------------------------------------------

# All available red-flag functions
RED_FLAG_FUNCTIONS = {
    "single_bidder": flag_single_bidder,
    "short_title": flag_short_title,
    "short_description": flag_short_description,
    "q4_timing": flag_q4_timing,
    "price_deviation": flag_price_deviation,
    "high_value": flag_high_value,
    "direct_procurement": flag_direct_procurement,
}


def compute_red_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all individual red-flag columns.

    Returns a DataFrame with boolean columns, one per flag.
    """
    flags = pd.DataFrame(index=df.index)
    for name, func in RED_FLAG_FUNCTIONS.items():
        flags[f"flag_{name}"] = func(df)
    return flags


def compute_risk_labels(
    df: pd.DataFrame,
    low_max: int = 0,
    high_min: int = 3,
) -> pd.DataFrame:
    """Assign heuristic risk labels based on red-flag count.

    Risk classes:
        0 = Low Risk    (0 flags triggered)
        1 = Medium Risk (1-2 flags triggered)
        2 = High Risk   (3+ flags triggered)

    Parameters
    ----------
    df : pd.DataFrame
        Raw data with the required fields.
    low_max : int
        Max flag count for "low risk" class.
    high_min : int
        Min flag count for "high risk" class.

    Returns
    -------
    pd.DataFrame with columns: all flag columns + 'flag_count' + 'risk_label'
    """
    flags = compute_red_flags(df)
    flags["flag_count"] = flags.sum(axis=1)

    flags["risk_label"] = np.where(
        flags["flag_count"] <= low_max,
        0,  # Low Risk
        np.where(
            flags["flag_count"] >= high_min,
            2,  # High Risk
            1,  # Medium Risk
        ),
    )

    logger.info(
        "Label distribution: Low=%d, Medium=%d, High=%d",
        (flags["risk_label"] == 0).sum(),
        (flags["risk_label"] == 1).sum(),
        (flags["risk_label"] == 2).sum(),
    )

    return flags


# ---------------------------------------------------------------------------
# Save labels
# ---------------------------------------------------------------------------


def save_labels(labels: pd.DataFrame, partition: str) -> Path:
    """Save label artifacts.

    Parameters
    ----------
    partition : str
        Either 'train' or 'test'.
    """
    if partition == "train":
        out_dir = TRAIN_DIR
    elif partition == "test":
        out_dir = TEST_DIR
    else:
        raise ValueError(f"Unknown partition: '{partition}'")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "labels.parquet"
    labels.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d labels to %s", len(labels), path)
    return path


# ---------------------------------------------------------------------------
# Calibration sample helpers (Task 14)
# ---------------------------------------------------------------------------


def select_calibration_samples(
    labels: pd.DataFrame,
    raw_df: pd.DataFrame,
    n_samples: int = 100,
    seed: int = 42,
) -> pd.DataFrame:
    """Select a stratified sample from val_calibration for human review.

    Samples are drawn proportionally from each risk class to maintain
    label distribution representation.
    """
    # Join labels with raw data for context
    combined = labels.copy()
    # Add relevant raw columns for reviewer context
    context_cols = ["ocid", "tender_title", "tender_value_amount", "buyer_name", "supplier_name"]
    for col in context_cols:
        if col in raw_df.columns:
            combined[col] = raw_df[col].values[:len(combined)]

    # Stratified sampling
    samples = combined.groupby("risk_label", group_keys=False).apply(
        lambda x: x.sample(
            n=min(len(x), max(1, int(n_samples * len(x) / len(combined)))),
            random_state=seed,
        )
    )

    # Add empty review columns
    samples["verified_label"] = np.nan
    samples["confidence"] = ""
    samples["notes"] = ""

    logger.info("Selected %d calibration samples", len(samples))
    return samples
