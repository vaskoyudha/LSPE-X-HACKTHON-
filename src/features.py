"""Feature engineering utilities for split-aware procurement features.

All features are computed from split raw inputs only.
No feature may look ahead in time (expanding-window only for Tier 2).
All output columns are numeric-safe for downstream ONNX path.

Feature catalog:
  Tier 1 (15 families): direct from raw fields
  Tier 2 (15 families): temporal and aggregated, past-only windows
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

# Canonical feature list — frozen after Task 9
FEATURE_CATALOG: list[str] = []


# ---------------------------------------------------------------------------
# Helper: safe numeric conversion
# ---------------------------------------------------------------------------


def _to_numeric(series: pd.Series | None) -> pd.Series:
    """Convert to numeric, coercing errors to NaN."""
    if series is None:
        return pd.Series(dtype="float64")
    return pd.to_numeric(series, errors="coerce")


def _safe_log1p(series: pd.Series | None) -> pd.Series:
    """Log1p transform, handling NaN, negatives, and None."""
    if series is None:
        return pd.Series(dtype="float64")
    vals = _to_numeric(series)
    return np.log1p(vals.clip(lower=0))


def _safe_len(series: pd.Series) -> pd.Series:
    """String length, handling NaN."""
    return series.fillna("").astype(str).str.len()


def _parse_dates(series: pd.Series) -> pd.Series:
    """Parse dates to datetime, coercing errors."""
    return pd.to_datetime(series, errors="coerce", utc=True)


# ---------------------------------------------------------------------------
# Tier 1: Direct features from raw fields (15 families)
# ---------------------------------------------------------------------------


def tier1_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Tier 1 features directly from raw procurement fields.

    Returns a DataFrame with 15 numeric feature columns.
    """
    feats = pd.DataFrame(index=df.index)

    # 1. Tender value (log-scaled)
    feats["f_tender_value_log"] = _safe_log1p(df.get("tender_value_amount"))

    # 2. Award value (log-scaled)
    feats["f_award_value_log"] = _safe_log1p(df.get("award_value_amount"))

    # 3. Price deviation ratio (award / tender estimate)
    tender_val = _to_numeric(df.get("tender_value_amount"))
    award_val = _to_numeric(df.get("award_value_amount"))
    feats["f_price_deviation_ratio"] = award_val / tender_val.replace(0, np.nan)

    # 4. Tender duration (days between period start and end)
    start = _parse_dates(df.get("tender_tenderPeriod_startDate"))
    end = _parse_dates(df.get("tender_tenderPeriod_endDate"))
    feats["f_tender_duration_days"] = (end - start).dt.total_seconds() / 86400

    # 5. Award duration (days from tender start to award)
    award_date = _parse_dates(df.get("award_date"))
    feats["f_award_duration_days"] = (award_date - start).dt.total_seconds() / 86400

    # 6. Number of tenderers
    feats["f_num_tenderers"] = _to_numeric(df.get("tender_numberOfTenderers"))

    # 7. Single bidder flag (binary)
    feats["f_single_bidder"] = (
        _to_numeric(df.get("tender_numberOfTenderers")).fillna(-1) == 1
    ).astype(float)

    # 8. Title length
    feats["f_title_length"] = _safe_len(df.get("tender_title")).astype(float)

    # 9. Description length
    feats["f_description_length"] = _safe_len(
        df.get("tender_description")
    ).astype(float)

    # 10. Procurement method (encoded: open=0, selective/limited=1, direct=2)
    method = df.get("tender_procurementMethod", pd.Series("", index=df.index))
    method_lower = method.fillna("").str.lower()
    method_map = {"open": 0, "selective": 1, "limited": 1, "direct": 2}
    feats["f_procurement_method_enc"] = method_lower.map(method_map).fillna(-1).astype(float)

    # 11. Is Q4 (October-December)
    pub_date = _parse_dates(df.get("tender_datePublished"))
    feats["f_is_q4"] = pub_date.dt.month.isin([10, 11, 12]).astype(float)

    # 12. Is December specifically
    feats["f_is_december"] = (pub_date.dt.month == 12).astype(float)

    # 13. Contract value (log-scaled)
    feats["f_contract_value_log"] = _safe_log1p(df.get("contract_value_amount"))

    # 14. Contract-to-award ratio
    contract_val = _to_numeric(df.get("contract_value_amount"))
    feats["f_contract_award_ratio"] = contract_val / award_val.replace(0, np.nan)

    # 15. Days to contract signing (from award)
    contract_date = _parse_dates(df.get("contract_dateSigned"))
    feats["f_days_to_contract"] = (
        (contract_date - award_date).dt.total_seconds() / 86400
    )

    return feats


# ---------------------------------------------------------------------------
# Tier 2: Temporal and aggregated features (15 families)
# ---------------------------------------------------------------------------


def tier2_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute Tier 2 features using expanding-window (past-only) aggregations.

    CRITICAL: For each row, only data from BEFORE that row's tender date
    is used. This is the anti-leakage guarantee.

    Returns a DataFrame with 15 numeric feature columns.
    """
    feats = pd.DataFrame(index=df.index)

    # Sort by date for expanding-window correctness
    date_col = "tender_datePublished"
    dates = _parse_dates(df.get(date_col))
    tender_val = _to_numeric(df.get("tender_value_amount"))
    award_val = _to_numeric(df.get("award_value_amount"))
    buyer_id = df.get("buyer_id", pd.Series("", index=df.index)).fillna("")
    supplier_id = df.get("supplier_id", pd.Series("", index=df.index)).fillna("")

    # We need sorted order for expanding-window calculations
    sort_idx = dates.argsort()
    df_sorted = df.iloc[sort_idx].copy()
    dates_sorted = dates.iloc[sort_idx]

    # Pre-compute sorted series
    buyer_sorted = buyer_id.iloc[sort_idx]
    supplier_sorted = supplier_id.iloc[sort_idx]
    tender_val_sorted = tender_val.iloc[sort_idx]
    award_val_sorted = award_val.iloc[sort_idx]

    # Initialize result arrays (will be filled in sorted order, then reindexed)
    n = len(df)

    # Helper: expanding window stats per group
    def _expanding_group_stat(group_col, value_col, stat="mean"):
        """Compute expanding-window stat per group using past-only data."""
        result = pd.Series(np.nan, index=range(n))
        group_history: dict[str, list[float]] = {}

        for i in range(n):
            g = group_col.iloc[i]
            v = value_col.iloc[i]

            if g and g in group_history and len(group_history[g]) > 0:
                hist = group_history[g]
                if stat == "mean":
                    result.iloc[i] = np.nanmean(hist)
                elif stat == "std":
                    result.iloc[i] = np.nanstd(hist) if len(hist) > 1 else 0
                elif stat == "count":
                    result.iloc[i] = len(hist)
                elif stat == "max":
                    result.iloc[i] = np.nanmax(hist)

            # Add current value to history AFTER computing (past-only)
            if g and pd.notna(v):
                group_history.setdefault(g, []).append(v)

        return result

    def _expanding_pair_count(g1_col, g2_col):
        """Count past interactions between a pair."""
        result = pd.Series(0.0, index=range(n))
        pair_history: dict[tuple, int] = {}

        for i in range(n):
            key = (g1_col.iloc[i], g2_col.iloc[i])
            if key[0] and key[1]:
                result.iloc[i] = pair_history.get(key, 0)
                pair_history[key] = pair_history.get(key, 0) + 1

        return result

    # 16. Buyer historical average tender value
    feats_sorted_16 = _expanding_group_stat(buyer_sorted, tender_val_sorted, "mean")

    # 17. Buyer historical value std (spending volatility)
    feats_sorted_17 = _expanding_group_stat(buyer_sorted, tender_val_sorted, "std")

    # 18. Supplier historical win count
    feats_sorted_18 = _expanding_group_stat(supplier_sorted, award_val_sorted, "count")

    # 19. Buyer-supplier repeat interaction count
    feats_sorted_19 = _expanding_pair_count(buyer_sorted, supplier_sorted)

    # 20. Buyer total past tender count
    feats_sorted_20 = _expanding_group_stat(buyer_sorted, tender_val_sorted, "count")

    # 21. Supplier historical max award value
    feats_sorted_21 = _expanding_group_stat(supplier_sorted, award_val_sorted, "max")

    # 22. Tender value z-score vs buyer history
    feats_sorted_22 = pd.Series(np.nan, index=range(n))
    buyer_history_vals: dict[str, list[float]] = {}
    for i in range(n):
        b = buyer_sorted.iloc[i]
        v = tender_val_sorted.iloc[i]
        if b and b in buyer_history_vals and len(buyer_history_vals[b]) > 1:
            hist = buyer_history_vals[b]
            mean_h = np.nanmean(hist)
            std_h = np.nanstd(hist)
            if std_h > 0 and pd.notna(v):
                feats_sorted_22.iloc[i] = (v - mean_h) / std_h
        if b and pd.notna(v):
            buyer_history_vals.setdefault(b, []).append(v)

    # 23. Days since buyer's last tender
    feats_sorted_23 = pd.Series(np.nan, index=range(n))
    buyer_last_date: dict[str, pd.Timestamp] = {}
    for i in range(n):
        b = buyer_sorted.iloc[i]
        d = dates_sorted.iloc[i]
        if b and b in buyer_last_date and pd.notna(d):
            delta = (d - buyer_last_date[b]).total_seconds() / 86400
            feats_sorted_23.iloc[i] = delta
        if b and pd.notna(d):
            buyer_last_date[b] = d

    # 24. Buyer historical procurement method diversity (unique methods / count)
    feats_sorted_24 = pd.Series(np.nan, index=range(n))
    method_col = df_sorted.get(
        "tender_procurementMethod", pd.Series("", index=df_sorted.index)
    ).fillna("")
    buyer_methods: dict[str, list[str]] = {}
    for i in range(n):
        b = buyer_sorted.iloc[i]
        m = method_col.iloc[i]
        if b and b in buyer_methods and len(buyer_methods[b]) > 0:
            unique = len(set(buyer_methods[b]))
            total = len(buyer_methods[b])
            feats_sorted_24.iloc[i] = unique / total
        if b and m:
            buyer_methods.setdefault(b, []).append(m)

    # 25. Supplier distinct buyer count (how many unique buyers this supplier has served)
    feats_sorted_25 = pd.Series(np.nan, index=range(n))
    supplier_buyer_sets: dict[str, set[str]] = {}
    for i in range(n):
        s = supplier_sorted.iloc[i]
        b = buyer_sorted.iloc[i]
        if s and s in supplier_buyer_sets and len(supplier_buyer_sets[s]) > 0:
            feats_sorted_25.iloc[i] = len(supplier_buyer_sets[s])
        # Add current buyer AFTER computing (past-only)
        if s and b:
            supplier_buyer_sets.setdefault(s, set()).add(b)

    # 26. Value growth rate for buyer (current / historical mean)
    feats_sorted_26 = pd.Series(np.nan, index=range(n))
    buyer_val_hist: dict[str, list[float]] = {}
    for i in range(n):
        b = buyer_sorted.iloc[i]
        v = tender_val_sorted.iloc[i]
        if b and b in buyer_val_hist and len(buyer_val_hist[b]) > 0 and pd.notna(v):
            hist_mean = np.nanmean(buyer_val_hist[b])
            if hist_mean > 0:
                feats_sorted_26.iloc[i] = v / hist_mean
        if b and pd.notna(v):
            buyer_val_hist.setdefault(b, []).append(v)

    # 27. Supplier capacity ratio (current award / historical max)
    feats_sorted_27 = pd.Series(np.nan, index=range(n))
    supplier_max_hist: dict[str, float] = {}
    for i in range(n):
        s = supplier_sorted.iloc[i]
        v = award_val_sorted.iloc[i]
        if s and s in supplier_max_hist and pd.notna(v):
            if supplier_max_hist[s] > 0:
                feats_sorted_27.iloc[i] = v / supplier_max_hist[s]
        if s and pd.notna(v):
            supplier_max_hist[s] = max(supplier_max_hist.get(s, 0), v)

    # 28-29. Price deviation statistics per buyer
    feats_sorted_28 = _expanding_group_stat(buyer_sorted, award_val_sorted, "mean")
    feats_sorted_29 = _expanding_group_stat(buyer_sorted, award_val_sorted, "std")

    # 30. Supplier historical average award value
    feats_sorted_30 = _expanding_group_stat(supplier_sorted, award_val_sorted, "mean")

    # Map sorted results back to original index
    inverse_idx = sort_idx.argsort()

    feats["f_buyer_hist_avg_value"] = feats_sorted_16.values[inverse_idx]
    feats["f_buyer_hist_value_std"] = feats_sorted_17.values[inverse_idx]
    feats["f_supplier_hist_win_count"] = feats_sorted_18.values[inverse_idx]
    feats["f_buyer_supplier_repeat_count"] = feats_sorted_19.values[inverse_idx]
    feats["f_buyer_hist_tender_count"] = feats_sorted_20.values[inverse_idx]
    feats["f_supplier_hist_max_award"] = feats_sorted_21.values[inverse_idx]
    feats["f_tender_value_zscore_buyer"] = feats_sorted_22.values[inverse_idx]
    feats["f_days_since_last_buyer_tender"] = feats_sorted_23.values[inverse_idx]
    feats["f_buyer_method_diversity"] = feats_sorted_24.values[inverse_idx]
    feats["f_supplier_unique_buyers"] = feats_sorted_25.values[inverse_idx]
    feats["f_buyer_value_growth_rate"] = feats_sorted_26.values[inverse_idx]
    feats["f_supplier_capacity_ratio"] = feats_sorted_27.values[inverse_idx]
    feats["f_buyer_hist_avg_award"] = feats_sorted_28.values[inverse_idx]
    feats["f_buyer_hist_award_std"] = feats_sorted_29.values[inverse_idx]
    feats["f_supplier_hist_avg_award"] = feats_sorted_30.values[inverse_idx]

    return feats


# ---------------------------------------------------------------------------
# Combined feature pipeline
# ---------------------------------------------------------------------------


def compute_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all 30 feature families (Tier 1 + Tier 2).

    Returns a DataFrame with 30 numeric columns, all ONNX-safe.
    """
    t1 = tier1_features(df)
    t2 = tier2_features(df)
    combined = pd.concat([t1, t2], axis=1)

    # Update the frozen catalog
    global FEATURE_CATALOG
    FEATURE_CATALOG = list(combined.columns)

    logger.info("Computed %d features: %s", len(combined.columns), list(combined.columns))

    # Verify all numeric
    for col in combined.columns:
        if not pd.api.types.is_numeric_dtype(combined[col]):
            raise TypeError(f"Feature '{col}' is not numeric: {combined[col].dtype}")

    return combined


def save_features(features: pd.DataFrame, partition: str) -> Path:
    """Save feature artifacts for a partition."""
    if partition == "train":
        out_dir = TRAIN_DIR
    elif partition == "test":
        out_dir = TEST_DIR
    else:
        raise ValueError(f"Unknown partition: '{partition}'")

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "features.parquet"
    features.to_parquet(path, index=False, engine="pyarrow")
    logger.info("Saved %d features (%d rows) to %s", len(features.columns), len(features), path)
    return path
