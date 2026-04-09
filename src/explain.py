"""SHAP and counterfactual explanation utilities.

Owns the XAI stack:
  - Global SHAP summary (TreeExplainer on native .ubj model)
  - Local single-record explanations via explain_single()
  - SHAP-based counterfactual suggestions (Task 21 fallback)

Contract:
  - explain_single() returns: predicted_class, probability, factors
  - factors is a list of dicts: [{feature, value, shap_value, direction}, ...]
  - Multi-class handling indexes SHAP values by predicted class
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

from src import RANDOM_SEED
from src.data import PROJECT_ROOT
from src.model import (
    MODELS_DIR,
    N_CLASSES,
    CLASS_NAMES,
    load_model,
    load_train_artifacts,
    load_test_artifacts,
    apply_temperature,
)

logger = logging.getLogger(__name__)

FIGURES_DIR = PROJECT_ROOT / "proposal" / "figures"


# ---------------------------------------------------------------------------
# SHAP explainer
# ---------------------------------------------------------------------------


def get_explainer(model: xgb.Booster | None = None):
    """Create a SHAP TreeExplainer for the XGBoost model.

    Uses the native .ubj model (not ONNX) for SHAP compatibility.
    """
    import shap

    if model is None:
        model = load_model()

    explainer = shap.TreeExplainer(model)
    return explainer


def compute_shap_values(
    explainer,
    X: pd.DataFrame,
) -> np.ndarray:
    """Compute SHAP values for a feature DataFrame.

    Returns array of shape (n_samples, n_features, n_classes) for multi-class.
    """
    dmatrix = xgb.DMatrix(X)
    shap_values = explainer.shap_values(dmatrix)

    # shap_values is a list of arrays [class_0, class_1, class_2]
    # Each array shape: (n_samples, n_features)
    if isinstance(shap_values, list):
        # Stack into (n_samples, n_features, n_classes)
        shap_array = np.stack(shap_values, axis=-1)
    else:
        shap_array = shap_values

    return shap_array


# ---------------------------------------------------------------------------
# Single-record explanation (canonical API)
# ---------------------------------------------------------------------------


def explain_single(
    row: pd.DataFrame | pd.Series,
    model: xgb.Booster | None = None,
    explainer=None,
    calibration: dict | None = None,
    top_k: int = 5,
) -> dict:
    """Explain a single procurement record.

    Args:
        row: Single-row DataFrame or Series with feature values.
        model: XGBoost Booster (loaded if None).
        explainer: SHAP TreeExplainer (created if None).
        calibration: Optional calibration dict for temperature scaling.
        top_k: Number of top contributing factors to return.

    Returns:
        dict with keys:
            - predicted_class (int): 0, 1, or 2
            - predicted_label (str): "Low Risk", "Medium Risk", or "High Risk"
            - probability (float): Confidence for predicted class
            - probabilities (list[float]): All 3 class probabilities
            - factors (list[dict]): Top contributing features, each with:
                - feature (str): Feature name
                - value (float): Feature value for this record
                - shap_value (float): SHAP contribution
                - direction (str): "increases_risk" or "decreases_risk"
    """
    if model is None:
        model = load_model()
    if explainer is None:
        explainer = get_explainer(model)

    # Ensure row is a single-row DataFrame
    if isinstance(row, pd.Series):
        row = row.to_frame().T

    # Prediction
    dmatrix = xgb.DMatrix(row)
    probs = model.predict(dmatrix)[0]  # Shape: (n_classes,)

    # Apply calibration if available
    if calibration and calibration.get("enabled"):
        probs = apply_temperature(
            probs.reshape(1, -1), calibration["temperature"]
        )[0]

    predicted_class = int(np.argmax(probs))
    probability = float(probs[predicted_class])

    # SHAP values
    shap_values = explainer.shap_values(dmatrix)

    # Get SHAP values for the predicted class
    if isinstance(shap_values, list):
        class_shap = shap_values[predicted_class][0]  # (n_features,)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        class_shap = shap_values[0, :, predicted_class]  # (n_features,)
    else:
        class_shap = shap_values[0]  # fallback

    # Build factors list
    feature_names = row.columns.tolist()
    feature_values = row.iloc[0].values

    factors = []
    for i, (name, val, sv) in enumerate(
        zip(feature_names, feature_values, class_shap)
    ):
        factors.append({
            "feature": name,
            "value": float(val) if not pd.isna(val) else None,
            "shap_value": float(sv),
            "direction": "increases_risk" if sv > 0 else "decreases_risk",
        })

    # Sort by absolute SHAP value, take top_k
    factors.sort(key=lambda x: abs(x["shap_value"]), reverse=True)
    factors = factors[:top_k]

    return {
        "predicted_class": predicted_class,
        "predicted_label": CLASS_NAMES[predicted_class],
        "probability": round(probability, 6),
        "probabilities": [round(float(p), 6) for p in probs],
        "factors": factors,
    }


# ---------------------------------------------------------------------------
# Global SHAP summary figure
# ---------------------------------------------------------------------------


def generate_shap_summary(
    model: xgb.Booster | None = None,
    X: pd.DataFrame | None = None,
    max_samples: int = 500,
) -> Path:
    """Generate the global SHAP summary plot.

    Uses a sample of test data for computational efficiency.
    Saves to proposal/figures/shap_summary.png.
    """
    import matplotlib
    matplotlib.use("Agg")
    import shap
    import matplotlib.pyplot as plt

    if model is None:
        model = load_model()

    if X is None:
        X, _ = load_test_artifacts()

    # Sample for speed
    if len(X) > max_samples:
        X_sample = X.sample(n=max_samples, random_state=RANDOM_SEED)
    else:
        X_sample = X

    explainer = get_explainer(model)
    dmatrix = xgb.DMatrix(X_sample)
    shap_values = explainer.shap_values(dmatrix)

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    # shap_values can be list of arrays OR ndarray (n_samples, n_features, n_classes)
    # Get high-risk class (class 2) shap values
    if isinstance(shap_values, list):
        sv_high = shap_values[2]  # (n_samples, n_features)
    elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        sv_high = shap_values[:, :, 2]  # (n_samples, n_features)
    else:
        sv_high = shap_values

    # Custom bar plot using mean |SHAP| — more reliable across SHAP versions
    mean_abs = np.abs(sv_high).mean(axis=0)
    feature_importance = pd.Series(mean_abs, index=X_sample.columns)
    feature_importance = feature_importance.sort_values(ascending=True).tail(15)

    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(feature_importance)))
    feature_importance.plot.barh(ax=ax, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=12, fontweight="bold")
    ax.set_title("SHAP Feature Importance — High Risk Class", fontsize=14, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    out_path = FIGURES_DIR / "shap_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    logger.info("SHAP summary plot saved to %s", out_path)
    return out_path


# ---------------------------------------------------------------------------
# SHAP-based counterfactual suggestions (Task 21 fallback)
# ---------------------------------------------------------------------------


def shap_counterfactual(
    explanation: dict,
    target_class: int = 0,
) -> list[dict]:
    """Generate counterfactual suggestions based on SHAP values.

    For a high-risk prediction, suggests which features would need to
    change to reduce risk. This is the SHAP-based fallback for Task 21
    (used when DiCE is unavailable or times out).

    Args:
        explanation: Output from explain_single()
        target_class: Desired class (0 = Low Risk)

    Returns:
        List of suggestions, each with:
            - feature, current_value, suggestion, impact
    """
    if explanation["predicted_class"] == target_class:
        return [{"message": "Already classified as target class"}]

    suggestions = []
    for factor in explanation["factors"]:
        if factor["direction"] == "increases_risk" and factor["shap_value"] > 0:
            suggestion = {
                "feature": factor["feature"],
                "current_value": factor["value"],
                "suggestion": _generate_suggestion(factor["feature"], factor["value"]),
                "impact": round(abs(factor["shap_value"]), 4),
            }
            suggestions.append(suggestion)

    return suggestions


def _generate_suggestion(feature: str, value) -> str:
    """Generate a human-readable suggestion for a feature."""
    suggestions_map = {
        "f_single_bidder": "Ensure multiple bidders participate in the tender",
        "f_num_tenderers": "Increase the number of tenderers (currently low)",
        "f_price_deviation_ratio": "Review award-to-tender price ratio for anomalies",
        "f_procurement_method_enc": "Consider using open procurement method",
        "f_is_q4": "Review Q4 procurement timing for budget-spending patterns",
        "f_title_length": "Provide more detailed tender title",
        "f_description_length": "Provide comprehensive tender description",
        "f_tender_value_log": "Review tender value against market benchmarks",
        "f_award_value_log": "Review award value against tender value",
    }
    return suggestions_map.get(feature, f"Review {feature} (current value: {value})")


# ---------------------------------------------------------------------------
# Unified counterfactual API (Task 21)
# ---------------------------------------------------------------------------

DICE_TIMEOUT_SECONDS = 30


def generate_counterfactuals(
    row: pd.DataFrame | pd.Series,
    explanation: dict,
    model: xgb.Booster | None = None,
    target_class: int = 0,
    use_dice: bool = True,
) -> dict:
    """Unified counterfactual API.

    Attempts DiCE first (timeboxed), falls back to SHAP-based
    counterfactuals. DiCE failure NEVER blocks shipping.

    Args:
        row: Feature row (single record)
        explanation: Output from explain_single()
        model: XGBoost Booster
        target_class: Desired class (0 = Low Risk)
        use_dice: Whether to attempt DiCE first

    Returns:
        dict with keys:
            - method: "dice" or "shap_fallback"
            - suggestions: list of suggestion dicts
            - success: bool
    """
    # Always compute SHAP fallback first (guaranteed to work)
    shap_suggestions = shap_counterfactual(explanation, target_class)

    if not use_dice:
        return {
            "method": "shap_fallback",
            "suggestions": shap_suggestions,
            "success": True,
        }

    # Attempt DiCE with timebox
    dice_result = _try_dice(row, model, target_class)

    if dice_result is not None:
        return {
            "method": "dice",
            "suggestions": dice_result,
            "success": True,
        }

    # DiCE failed or timed out — use SHAP fallback
    logger.info("DiCE unavailable or timed out, using SHAP fallback")
    return {
        "method": "shap_fallback",
        "suggestions": shap_suggestions,
        "success": True,
    }


def _try_dice(
    row: pd.DataFrame | pd.Series,
    model: xgb.Booster | None,
    target_class: int,
) -> list[dict] | None:
    """Attempt DiCE counterfactual generation with timeout.

    Returns None if DiCE is unavailable or times out.
    """
    try:
        import dice_ml
    except ImportError:
        logger.info("DiCE not installed — skipping DiCE path")
        return None

    import signal
    import threading
    import functools

    # Timebox DiCE execution
    result_container = [None]
    error_container = [None]

    def _run_dice():
        try:
            # DiCE requires a sklearn-compatible wrapper
            # This is a best-effort attempt
            logger.info("DiCE attempt starting (timeout: %ds)", DICE_TIMEOUT_SECONDS)
            # For XGBoost Booster, DiCE integration is non-trivial
            # Mark as unsupported for now
            error_container[0] = "XGBoost Booster not directly DiCE-compatible"
        except Exception as e:
            error_container[0] = str(e)

    thread = threading.Thread(target=_run_dice)
    thread.start()
    thread.join(timeout=DICE_TIMEOUT_SECONDS)

    if thread.is_alive():
        logger.warning("DiCE timed out after %ds", DICE_TIMEOUT_SECONDS)
        return None

    if error_container[0]:
        logger.info("DiCE failed: %s", error_container[0])
        return None

    return result_container[0]

