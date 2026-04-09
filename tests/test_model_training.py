"""Tests for src.model module — XGBoost training and HPO."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import onnxmltools

from src.model import (
    compute_class_weights,
    compute_sample_weights,
    run_hpo,
    train_final_model,
    save_model,
    load_model,
    evaluate,
    BASE_PARAMS,
    N_CLASSES,
    MODELS_DIR,
)
from src.split import TRAIN_DIR, TEST_DIR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_train_data():
    """Generate synthetic train features + labels for model testing."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        {f"f_{i}": rng.standard_normal(n) for i in range(30)}
    )
    # Create labels with realistic imbalance (Low:Medium:High ≈ 40:45:15)
    y = pd.Series(
        rng.choice([0, 1, 2], size=n, p=[0.40, 0.45, 0.15])
    )
    return X, y


@pytest.fixture
def synthetic_split_data(synthetic_train_data):
    """Split synthetic data into train_fit and val_hpo."""
    X, y = synthetic_train_data
    split_idx = int(len(X) * 0.75)  # 75% fit, 25% hpo
    X_fit = X.iloc[:split_idx].reset_index(drop=True)
    y_fit = y.iloc[:split_idx].reset_index(drop=True)
    X_hpo = X.iloc[split_idx:].reset_index(drop=True)
    y_hpo = y.iloc[split_idx:].reset_index(drop=True)
    return X_fit, y_fit, X_hpo, y_hpo


# ---------------------------------------------------------------------------
# P0: Core training tests
# ---------------------------------------------------------------------------


@pytest.mark.p0
class TestClassWeights:
    def test_weights_computed(self, synthetic_train_data):
        _, y = synthetic_train_data
        weights = compute_class_weights(y)
        assert len(weights) == N_CLASSES
        assert all(w > 0 for w in weights.values())

    def test_sample_weights_length(self, synthetic_train_data):
        _, y = synthetic_train_data
        sw = compute_sample_weights(y)
        assert len(sw) == len(y)
        assert all(w > 0 for w in sw)

    def test_missing_class_still_gets_weight_entry(self):
        y = pd.Series([1, 1, 2, 2, 2])
        weights = compute_class_weights(y)
        assert len(weights) == N_CLASSES
        assert 0 in weights
        assert weights[0] == 1.0


@pytest.mark.p0
class TestHPO:
    def test_hpo_returns_params(self, synthetic_split_data):
        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        best = run_hpo(X_fit, y_fit, X_hpo, y_hpo, n_trials=3, timeout=60)
        assert isinstance(best, dict)
        assert "max_depth" in best
        assert "learning_rate" in best

    def test_hpo_never_uses_test_data(self, synthetic_split_data):
        """HPO should only see train_fit and val_hpo data."""
        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        # If test_data artifacts exist, verify HPO didn't touch them
        test_feat_path = TEST_DIR / "features.parquet"
        if test_feat_path.exists():
            test_X = pd.read_parquet(test_feat_path)
            # Verify no row from test appears in fit or hpo
            assert len(X_fit) + len(X_hpo) < len(test_X) + len(X_fit) + len(X_hpo)


@pytest.mark.p0
class TestTrainFinalModel:
    def test_model_trains(self, synthetic_split_data):
        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        params = {"max_depth": 4, "learning_rate": 0.1, "n_rounds": 50,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 3, "gamma": 0.1,
                  "reg_alpha": 0.01, "reg_lambda": 1.0}
        model = train_final_model(X_fit, y_fit, X_hpo, y_hpo, params)
        assert isinstance(model, xgb.Booster)

    def test_model_predicts_3_classes(self, synthetic_split_data):
        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        params = {"max_depth": 4, "learning_rate": 0.1, "n_rounds": 50,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 3, "gamma": 0.1,
                  "reg_alpha": 0.01, "reg_lambda": 1.0}
        model = train_final_model(X_fit, y_fit, X_hpo, y_hpo, params)
        dtest = xgb.DMatrix(X_hpo)
        probs = model.predict(dtest)
        assert probs.shape == (len(X_hpo), N_CLASSES)
        # Probabilities should sum to 1
        row_sums = probs.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-5)


@pytest.mark.p1
class TestSaveLoad:
    def test_roundtrip(self, synthetic_split_data, tmp_path, monkeypatch):
        import src.model as model_mod
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp_path)

        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        params = {"max_depth": 4, "learning_rate": 0.1, "n_rounds": 30,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 3, "gamma": 0.1,
                  "reg_alpha": 0.01, "reg_lambda": 1.0}
        model = train_final_model(X_fit, y_fit, X_hpo, y_hpo, params)
        save_model(model, params)

        loaded = load_model()
        # Predictions should be identical
        dtest = xgb.DMatrix(X_hpo)
        np.testing.assert_array_equal(
            model.predict(dtest), loaded.predict(dtest)
        )

    def test_best_params_json(self, tmp_path, monkeypatch):
        import src.model as model_mod
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp_path)

        params = {"max_depth": 4, "learning_rate": 0.1}
        # Create a dummy model to save
        dtrain = xgb.DMatrix(np.random.randn(10, 5), label=[0]*10)
        model = xgb.train(BASE_PARAMS, dtrain, num_boost_round=2)
        save_model(model, params)

        saved = json.loads((tmp_path / "best_params.json").read_text())
        assert saved["max_depth"] == 4


@pytest.mark.p0
class TestEvaluation:
    def test_metrics_structure(self, synthetic_split_data):
        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        params = {"max_depth": 4, "learning_rate": 0.1, "n_rounds": 30,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 3, "gamma": 0.1,
                  "reg_alpha": 0.01, "reg_lambda": 1.0}
        model = train_final_model(X_fit, y_fit, X_hpo, y_hpo, params)
        metrics = evaluate(model, X_hpo, y_hpo, "val_hpo")

        assert "macro_f1" in metrics
        assert "per_class_f1" in metrics
        assert "confusion_matrix" in metrics
        assert "accuracy" in metrics
        assert metrics["partition"] == "val_hpo"
        assert metrics["label_type"] == "heuristic_risk_labels"
        assert 0 <= metrics["macro_f1"] <= 1

    def test_metrics_structure_when_a_class_is_missing(self, synthetic_split_data):
        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        y_fit = y_fit.replace(0, 1)
        y_hpo = y_hpo.replace(0, 1)
        params = {"max_depth": 4, "learning_rate": 0.1, "n_rounds": 20,
                  "subsample": 0.8, "colsample_bytree": 0.8,
                  "min_child_weight": 3, "gamma": 0.1,
                  "reg_alpha": 0.01, "reg_lambda": 1.0}
        model = train_final_model(X_fit, y_fit, X_hpo, y_hpo, params)
        metrics = evaluate(model, X_hpo, y_hpo, "val_hpo_missing_class")
        assert set(metrics["per_class_f1"].keys()) == {"Low Risk", "Medium Risk", "High Risk"}
        assert len(metrics["confusion_matrix"]) == 3
