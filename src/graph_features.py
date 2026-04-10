"""Point-in-time buyer/supplier relationship features for the fraud-evidence lane."""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


def _ordered_history_view(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Return a stable time-ordered view with original row positions preserved."""

    ordered = raw_df.reset_index(drop=False).rename(columns={"index": "_original_index"}).copy()

    if "tender_datePublished" in ordered.columns:
        ordered["_sort_tender_date"] = pd.to_datetime(
            ordered["tender_datePublished"], errors="coerce", utc=True
        )
    else:
        ordered["_sort_tender_date"] = pd.NaT

    if "award_date" in ordered.columns:
        ordered["_sort_award_date"] = pd.to_datetime(
            ordered["award_date"], errors="coerce", utc=True
        )
    else:
        ordered["_sort_award_date"] = pd.NaT

    return ordered.sort_values(
        by=["_sort_tender_date", "_sort_award_date", "_original_index"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_relationship_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build leakage-safe relationship features using only past rows."""

    ordered = _ordered_history_view(raw_df)

    pair_counts: dict[tuple[str, str], int] = defaultdict(int)
    supplier_buyers: dict[str, set[str]] = defaultdict(set)
    buyer_suppliers: dict[str, set[str]] = defaultdict(set)
    pair_award_sums: dict[tuple[str, str], float] = defaultdict(float)

    prev_pair_count: list[int] = []
    prev_supplier_buyer_count: list[int] = []
    prev_buyer_supplier_count: list[int] = []
    prev_pair_award_sum: list[float] = []

    award_values = pd.to_numeric(ordered.get("award_value_amount"), errors="coerce").fillna(0.0)

    for idx, row in ordered.iterrows():
        buyer_id = str(row.get("buyer_id", "") or "")
        supplier_id = str(row.get("supplier_id", "") or "")
        pair_key = (buyer_id, supplier_id)

        prev_pair_count.append(pair_counts[pair_key])
        prev_supplier_buyer_count.append(len(supplier_buyers[supplier_id]))
        prev_buyer_supplier_count.append(len(buyer_suppliers[buyer_id]))
        prev_pair_award_sum.append(float(pair_award_sums[pair_key]))

        pair_counts[pair_key] += 1
        supplier_buyers[supplier_id].add(buyer_id)
        buyer_suppliers[buyer_id].add(supplier_id)
        pair_award_sums[pair_key] += float(award_values.iloc[idx])

    features = pd.DataFrame(
        {
            "_original_index": ordered["_original_index"].astype(int),
            "g_buyer_supplier_prev_contract_count": prev_pair_count,
            "g_supplier_prev_buyer_count": prev_supplier_buyer_count,
            "g_buyer_prev_supplier_count": prev_buyer_supplier_count,
            "g_pair_prev_award_value_sum": prev_pair_award_sum,
        }
    )

    return (
        features.sort_values("_original_index", kind="mergesort")
        .drop(columns="_original_index")
        .reset_index(drop=True)
    )
