"""Training, evaluation, ONNX export, and calibration utilities.

Owns the full model lifecycle:
  - HPO via Optuna on internal dev splits (train_fit / val_hpo)
  - Final model training on train_fit + val_hpo
  - Evaluation metrics (per-class F1, confusion matrix)
  - Model save/load (.ubj format)
  - ONNX export (Task 19)
  - Temperature-scaled calibration (Task 16)

HARD RULES:
  - HPO uses train_fit + val_hpo ONLY, never test_data
  - test_data is used ONLY for final reported metrics
  - models/metrics.json is the canonical metrics file
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import (
    f1_score,
    classification_report,
    confusion_matrix,
    accuracy_score,
    log_loss,
)

from src import RANDOM_SEED
from src.data import PROJECT_ROOT, PROCESSED_DIR
from src.split import TRAIN_DIR, TEST_DIR

logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_CLASSES = 3
CLASS_NAMES = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}

# Default XGBoost base params (non-tunable)
BASE_PARAMS = {
    "objective": "multi:softprob",
    "num_class": N_CLASSES,
    "eval_metric": "mlogloss",
    "tree_method": "hist",
    "seed": RANDOM_SEED,
    "verbosity": 0,
}


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_train_artifacts() -> tuple[pd.DataFrame, pd.Series]:
    """Load training features and labels."""
    features = pd.read_parquet(TRAIN_DIR / "features.parquet")
    labels = pd.read_parquet(TRAIN_DIR / "labels.parquet")
    return features, labels["risk_label"]


def load_test_artifacts() -> tuple[pd.DataFrame, pd.Series]:
    """Load test features and labels."""
    features = pd.read_parquet(TEST_DIR / "features.parquet")
    labels = pd.read_parquet(TEST_DIR / "labels.parquet")
    return features, labels["risk_label"]


def load_dev_split_indices(
    train_features: pd.DataFrame,
    train_raw: pd.DataFrame | None = None,
) -> dict[str, np.ndarray]:
    """Load dev split boundaries and return index arrays.

    Uses the dev_split_manifest to partition train data into
    train_fit, val_hpo, and val_calibration by temporal ordering.
    """
    manifest_path = PROCESSED_DIR / "dev_split_manifest.json"
    manifest = json.loads(manifest_path.read_text())

    n = len(train_features)
    n_fit = manifest["train_fit"]["count"]
    n_hpo = manifest["val_hpo"]["count"]
    n_cal = manifest["val_calibration"]["count"]

    # The dev splits were created by temporal ordering, so we can
    # reconstruct index ranges from counts
    indices = {
        "train_fit": np.arange(0, n_fit),
        "val_hpo": np.arange(n_fit, n_fit + n_hpo),
        "val_calibration": np.arange(n_fit + n_hpo, n_fit + n_hpo + n_cal),
    }

    return indices


# ---------------------------------------------------------------------------
# Class weighting
# ---------------------------------------------------------------------------


def compute_class_weights(y: pd.Series) -> dict[int, float]:
    """Compute balanced class weights (inverse frequency)."""
    counts = y.value_counts()
    total = len(y)
    n_classes = len(counts)
    weights = {}
    for cls in range(n_classes):
        if cls in counts.index:
            weights[cls] = total / (n_classes * counts[cls])
        else:
            weights[cls] = 1.0
    logger.info("Class weights: %s", weights)
    return weights


def compute_sample_weights(y: pd.Series) -> np.ndarray:
    """Convert class weights to per-sample weights for XGBoost."""
    class_weights = compute_class_weights(y)
    return np.array([class_weights[int(label)] for label in y])


# ---------------------------------------------------------------------------
# HPO with Optuna
# ---------------------------------------------------------------------------


def run_hpo(
    X_fit: pd.DataFrame,
    y_fit: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    n_trials: int = 50,
    timeout: int = 300,
) -> dict:
    """Run Optuna HPO on internal train_fit / val_hpo splits.

    Returns the best hyperparameter dict.

    IMPORTANT: This function NEVER sees test_data.
    """
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # Sample weights for class imbalance
    w_fit = compute_sample_weights(y_fit)
    w_val = compute_sample_weights(y_val)

    dtrain = xgb.DMatrix(X_fit, label=y_fit, weight=w_fit)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    def objective(trial: optuna.Trial) -> float:
        params = {
            **BASE_PARAMS,
            "max_depth": trial.suggest_int("max_depth", 3, 8),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
        }

        n_rounds = trial.suggest_int("n_rounds", 50, 500)

        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_rounds,
            evals=[(dval, "val")],
            early_stopping_rounds=20,
            verbose_eval=False,
        )

        # Evaluate with macro F1 on validation
        preds_prob = model.predict(dval)
        preds_class = np.argmax(preds_prob, axis=1)
        macro_f1 = f1_score(y_val, preds_class, average="macro")

        return macro_f1

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED),
    )
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best = study.best_params.copy()
    best_f1 = study.best_value

    logger.info("HPO complete: best macro F1 = %.4f", best_f1)
    logger.info("Best params: %s", best)

    return best


# ---------------------------------------------------------------------------
# Model training
# ---------------------------------------------------------------------------


def train_final_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    hpo_params: dict,
) -> xgb.Booster:
    """Train the final model on train_fit + val_hpo with best HPO params.

    Uses early stopping on val to determine optimal boosting rounds.
    """
    # Extract n_rounds from HPO params (separate from XGB params)
    n_rounds = hpo_params.pop("n_rounds", 300)

    # Combine train_fit + val_hpo for final training
    X_combined = pd.concat([X_train, X_val], axis=0).reset_index(drop=True)
    y_combined = pd.concat([y_train, y_val], axis=0).reset_index(drop=True)

    w_combined = compute_sample_weights(y_combined)
    w_val = compute_sample_weights(y_val)

    dtrain = xgb.DMatrix(X_combined, label=y_combined, weight=w_combined)
    dval = xgb.DMatrix(X_val, label=y_val, weight=w_val)

    params = {**BASE_PARAMS, **hpo_params}

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_rounds,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )

    logger.info(
        "Final model trained: %d trees, best iteration: %d",
        model.num_boosted_rounds(),
        model.best_iteration,
    )

    return model


# ---------------------------------------------------------------------------
# Save / Load
# ---------------------------------------------------------------------------


def save_model(model: xgb.Booster, params: dict) -> None:
    """Save model as .ubj and best params as JSON."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    model_path = MODELS_DIR / "xgb_model.ubj"
    model.save_model(str(model_path))
    logger.info("Model saved to %s", model_path)

    params_path = MODELS_DIR / "best_params.json"
    # Ensure all values are JSON serializable
    serializable = {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in params.items()}
    params_path.write_text(json.dumps(serializable, indent=2), encoding="utf-8")
    logger.info("Best params saved to %s", params_path)


def load_model() -> xgb.Booster:
    """Load the saved XGBoost model."""
    model_path = MODELS_DIR / "xgb_model.ubj"
    if not model_path.exists():
        raise FileNotFoundError(f"{model_path} not found. Train the model first.")
    model = xgb.Booster()
    model.load_model(str(model_path))
    return model


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate(
    model: xgb.Booster,
    X: pd.DataFrame,
    y: pd.Series,
    partition_name: str = "test",
) -> dict:
    """Evaluate model and return metrics dict.

    Returns the canonical metrics structure for models/metrics.json.
    """
    dmatrix = xgb.DMatrix(X)
    probs = model.predict(dmatrix)
    preds = np.argmax(probs, axis=1)

    macro_f1 = f1_score(y, preds, average="macro")
    weighted_f1 = f1_score(y, preds, average="weighted")
    per_class_f1 = f1_score(y, preds, average=None).tolist()
    acc = accuracy_score(y, preds)
    cm = confusion_matrix(y, preds).tolist()

    # Log loss (if probabilities available)
    try:
        ll = log_loss(y, probs, labels=[0, 1, 2])
    except Exception:
        ll = None

    report = classification_report(
        y, preds, target_names=[CLASS_NAMES[i] for i in range(N_CLASSES)], output_dict=True
    )

    metrics = {
        "partition": partition_name,
        "label_type": "heuristic_risk_labels",
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "weighted_f1": round(weighted_f1, 4),
        "per_class_f1": {CLASS_NAMES[i]: round(f, 4) for i, f in enumerate(per_class_f1)},
        "log_loss": round(ll, 4) if ll is not None else None,
        "confusion_matrix": cm,
        "classification_report": report,
        "n_samples": len(y),
    }

    logger.info(
        "[%s] Accuracy=%.4f, Macro-F1=%.4f, Weighted-F1=%.4f",
        partition_name, acc, macro_f1, weighted_f1,
    )
    for i in range(N_CLASSES):
        logger.info(
            "  %s F1=%.4f", CLASS_NAMES[i], per_class_f1[i]
        )

    return metrics


def save_metrics(metrics: dict) -> None:
    """Save to the canonical models/metrics.json."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / "metrics.json"
    path.write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    logger.info("Metrics saved to %s", path)


# ---------------------------------------------------------------------------
# Full training pipeline
# ---------------------------------------------------------------------------


def run_training_pipeline(
    n_trials: int = 50,
    hpo_timeout: int = 300,
) -> dict:
    """Execute the complete training pipeline.

    1. Load train artifacts
    2. Split into dev sub-splits
    3. Run HPO on train_fit / val_hpo
    4. Train final model on train_fit + val_hpo
    5. Evaluate on val_hpo (internal) and test (final)
    6. Save model + params + metrics

    Returns metrics dict.
    """
    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE START")
    logger.info("=" * 60)

    # Load artifacts
    train_features, train_labels = load_train_artifacts()
    test_features, test_labels = load_test_artifacts()

    # Dev split indices
    dev_idx = load_dev_split_indices(train_features)

    X_fit = train_features.iloc[dev_idx["train_fit"]].reset_index(drop=True)
    y_fit = train_labels.iloc[dev_idx["train_fit"]].reset_index(drop=True)
    X_hpo = train_features.iloc[dev_idx["val_hpo"]].reset_index(drop=True)
    y_hpo = train_labels.iloc[dev_idx["val_hpo"]].reset_index(drop=True)

    logger.info("Train_fit: %d rows, Val_hpo: %d rows", len(X_fit), len(X_hpo))

    # Step 1: HPO
    logger.info("--- Running HPO ---")
    best_params = run_hpo(
        X_fit, y_fit, X_hpo, y_hpo,
        n_trials=n_trials, timeout=hpo_timeout,
    )

    # Step 2: Train final model
    logger.info("--- Training final model ---")
    model = train_final_model(X_fit, y_fit, X_hpo, y_hpo, best_params.copy())

    # Step 3: Save
    save_model(model, best_params)

    # Step 4: Evaluate on val_hpo (internal validation)
    logger.info("--- Internal validation metrics ---")
    val_metrics = evaluate(model, X_hpo, y_hpo, "val_hpo")

    # Step 5: Evaluate on test (final held-out metrics)
    logger.info("--- Final test metrics ---")
    test_metrics = evaluate(model, test_features, test_labels, "test")

    # Save canonical metrics
    full_metrics = {
        "note": "Metrics against heuristic risk labels, NOT confirmed fraud outcomes",
        "internal_validation": val_metrics,
        "final_test": test_metrics,
    }
    save_metrics(full_metrics)

    logger.info("=" * 60)
    logger.info("TRAINING PIPELINE COMPLETE")
    logger.info("=" * 60)

    return full_metrics
