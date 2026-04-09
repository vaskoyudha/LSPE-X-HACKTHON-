"""Diagnostics and robustness helpers for Phase 2 evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, f1_score, log_loss

from src.model import BEST_PARAMS_PATH, compute_sample_weights

SYNTHETIC_OCID_PREFIX = "ocds-synth-"

# Direct feature proxies to the heuristic risk rules in src.labels.
PROXY_CORE_FEATURES = [
    "f_single_bidder",
    "f_num_tenderers",
    "f_title_length",
    "f_description_length",
    "f_is_q4",
    "f_is_december",
    "f_price_deviation_ratio",
    "f_tender_value_log",
    "f_procurement_method_enc",
]

# Broader set including near-proxies tied to value and contract amounts.
PROXY_BROAD_FEATURES = PROXY_CORE_FEATURES + [
    "f_award_value_log",
    "f_contract_value_log",
    "f_contract_award_ratio",
]


def _iso(value) -> str | None:
    if pd.isna(value):
        return None
    return str(value)


def summarize_data_provenance(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> dict:
    """Summarize dataset provenance and whether the working data is synthetic."""
    combined = pd.concat([train_raw, test_raw], ignore_index=True)
    ocids = combined.get("ocid", pd.Series(dtype="object")).astype(str)
    synthetic_ratio = float(ocids.str.startswith(SYNTHETIC_OCID_PREFIX).mean()) if len(ocids) else 0.0
    is_synthetic = bool(len(ocids) and synthetic_ratio == 1.0)

    return {
        "data_kind": "synthetic_structured_benchmark" if is_synthetic else "real_or_mixed_ocds",
        "synthetic_ratio": round(synthetic_ratio, 4),
        "all_ocids_use_synthetic_prefix": is_synthetic,
        "row_count_total": int(len(combined)),
        "row_count_train": int(len(train_raw)),
        "row_count_test": int(len(test_raw)),
        "date_range_train": {
            "min": _iso(train_raw["tender_datePublished"].min()),
            "max": _iso(train_raw["tender_datePublished"].max()),
        },
        "date_range_test": {
            "min": _iso(test_raw["tender_datePublished"].min()),
            "max": _iso(test_raw["tender_datePublished"].max()),
        },
        "unique_buyers": int(combined.get("buyer_id", pd.Series(dtype="object")).nunique()),
        "unique_suppliers": int(combined.get("supplier_id", pd.Series(dtype="object")).nunique()),
        "procurement_methods": combined.get("tender_procurementMethod", pd.Series(dtype="object")).value_counts().to_dict(),
        "warning": (
            "Current tracked benchmark is synthetic, so metrics should be interpreted as pipeline validation "
            "rather than proof of real-world fraud-detection performance."
            if is_synthetic
            else "Dataset provenance does not appear fully synthetic from OCID prefixes."
        ),
    }


def resolve_proxy_feature_sets(feature_names: Iterable[str]) -> dict[str, list[str]]:
    """Return feature subsets for circularity ablations."""
    feature_names = list(feature_names)
    return {
        "full": feature_names,
        "proxy_core_removed": [name for name in feature_names if name not in PROXY_CORE_FEATURES],
        "proxy_broad_removed": [name for name in feature_names if name not in PROXY_BROAD_FEATURES],
    }


def _load_best_params() -> tuple[dict, int]:
    params = json.loads(BEST_PARAMS_PATH.read_text())
    n_rounds = int(params.pop("n_rounds", 449))
    return params, n_rounds


def evaluate_feature_subset(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
    feature_names: list[str],
    *,
    seed: int = 42,
) -> dict:
    """Train and evaluate a feature subset with the current best params."""
    params, n_rounds = _load_best_params()
    model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        seed=seed,
        n_estimators=n_rounds,
        n_jobs=-1,
        **params,
    )
    weights = compute_sample_weights(train_y)
    model.fit(train_X[feature_names], train_y, sample_weight=weights)
    probs = model.predict_proba(test_X[feature_names])
    preds = probs.argmax(axis=1)

    return {
        "feature_count": len(feature_names),
        "accuracy": round(float(accuracy_score(test_y, preds)), 4),
        "macro_f1": round(float(f1_score(test_y, preds, average="macro")), 4),
        "weighted_f1": round(float(f1_score(test_y, preds, average="weighted")), 4),
        "log_loss": round(float(log_loss(test_y, probs, labels=[0, 1, 2])), 4),
    }


def run_circularity_ablation(
    train_X: pd.DataFrame,
    train_y: pd.Series,
    test_X: pd.DataFrame,
    test_y: pd.Series,
) -> dict:
    """Measure how much performance depends on direct heuristic-label proxies."""
    feature_sets = resolve_proxy_feature_sets(train_X.columns)
    results = {
        name: evaluate_feature_subset(train_X, train_y, test_X, test_y, cols)
        for name, cols in feature_sets.items()
    }
    baseline = results["full"]
    for name in ["proxy_core_removed", "proxy_broad_removed"]:
        results[name]["macro_f1_drop_vs_full"] = round(
            baseline["macro_f1"] - results[name]["macro_f1"], 4
        )
        results[name]["dropped_features"] = [
            feature for feature in train_X.columns if feature not in feature_sets[name]
        ]
    results["interpretation"] = (
        "Large performance drops after removing direct heuristic proxies indicate the current benchmark acts more as "
        "an interpretable risk-rule accelerator than a validated real-world anomaly detector."
    )
    return results
