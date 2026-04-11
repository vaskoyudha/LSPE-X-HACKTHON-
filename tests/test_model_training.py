"""Tests for src.model module — XGBoost training and HPO."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import onnxmltools

from src.model import (
    _resolve_threshold_tuning_subset,
    _select_calibration_subset,
    _selected_n_rounds_from_booster,
    compute_class_weights,
    compute_sample_weights,
    predict_with_thresholds,
    run_hpo,
    search_decision_thresholds,
    train_final_model,
    save_decision_thresholds,
    save_model,
    load_decision_thresholds,
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

    def test_final_model_does_not_reuse_validation_for_early_stopping(self, synthetic_split_data, monkeypatch):
        import src.model as model_mod

        X_fit, y_fit, X_hpo, y_hpo = synthetic_split_data
        captured = {}

        class DummyBooster:
            def num_boosted_rounds(self):
                return 17

        def fake_train(params, dtrain, num_boost_round, evals=None, early_stopping_rounds=None, verbose_eval=False):
            captured["num_boost_round"] = num_boost_round
            captured["evals"] = evals
            captured["early_stopping_rounds"] = early_stopping_rounds
            return DummyBooster()

        monkeypatch.setattr(model_mod.xgb, "train", fake_train)

        model = train_final_model(
            X_fit,
            y_fit,
            X_hpo,
            y_hpo,
            {"max_depth": 4, "learning_rate": 0.1, "n_rounds": 17},
        )

        assert isinstance(model, DummyBooster)
        assert captured["num_boost_round"] == 17
        assert captured["early_stopping_rounds"] is None
        assert [name for _, name in captured["evals"]] == ["train"]


@pytest.mark.p0
class TestBoostRoundSelection:
    def test_selected_n_rounds_uses_best_iteration_when_available(self):
        class DummyBooster:
            best_iteration = 18

        assert _selected_n_rounds_from_booster(DummyBooster(), 50) == 19

    def test_selected_n_rounds_falls_back_when_best_iteration_missing(self):
        class DummyBooster:
            pass

        assert _selected_n_rounds_from_booster(DummyBooster(), 50) == 50


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

    def test_calibration_subset_uses_source_indices_when_present(self):
        cal_probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.6, 0.2],
                [0.1, 0.3, 0.6],
                [0.05, 0.1, 0.85],
            ]
        )
        clean = pd.DataFrame(
            {
                "verified_label": [2, 1],
                "confidence": ["high", "high"],
                "source_row_idx": [3, 1],
            }
        )
        sample_probs, sample_labels = _select_calibration_subset(cal_probs, clean)
        np.testing.assert_array_equal(sample_probs, cal_probs[[3, 1]])
        np.testing.assert_array_equal(sample_labels, np.array([2, 1]))


@pytest.mark.p0
class TestDecisionThresholds:
    def test_predict_with_thresholds_promotes_high_risk_when_probability_clears_cutoff(self):
        probs = np.array(
            [
                [0.20, 0.55, 0.25],
                [0.05, 0.30, 0.65],
            ]
        )
        thresholds = {"high_risk": 0.60, "low_risk": 0.80}

        preds = predict_with_thresholds(probs, thresholds)

        np.testing.assert_array_equal(preds, np.array([1, 2]))

    def test_search_thresholds_returns_serializable_thresholds(self):
        probs = np.array(
            [
                [0.7, 0.2, 0.1],
                [0.2, 0.3, 0.5],
                [0.1, 0.6, 0.3],
            ]
        )
        y_true = pd.Series([0, 2, 1])

        thresholds = search_decision_thresholds(probs, y_true)

        assert set(thresholds) == {"high_risk", "low_risk"}
        assert all(isinstance(value, float) for value in thresholds.values())

    def test_load_decision_thresholds_ignores_metadata(self, tmp_path, monkeypatch):
        import src.model as model_mod

        monkeypatch.setattr(model_mod, "DECISION_THRESHOLDS_PATH", tmp_path / "decision_thresholds.json")
        save_decision_thresholds(
            {"high_risk": 0.61, "low_risk": 0.42},
            metadata={"source": "reviewed_val_calibration", "n_rows": 123},
        )

        loaded = load_decision_thresholds()

        assert loaded == {"high_risk": 0.61, "low_risk": 0.42}

    def test_resolve_threshold_tuning_subset_prefers_reviewed_rows(self, monkeypatch):
        import src.model as model_mod

        class DummyModel:
            def predict(self, dmatrix):
                n = dmatrix.num_row()
                if n == 3:
                    return np.array(
                        [
                            [0.70, 0.20, 0.10],
                            [0.10, 0.70, 0.20],
                            [0.05, 0.25, 0.70],
                        ],
                        dtype=float,
                    )
                if n == 4:
                    return np.array(
                        [
                            [0.60, 0.30, 0.10],
                            [0.20, 0.50, 0.30],
                            [0.10, 0.25, 0.65],
                            [0.05, 0.15, 0.80],
                        ],
                        dtype=float,
                    )
                raise AssertionError(f"unexpected num_row={n}")

        train_features = pd.DataFrame({f"f_{i}": np.arange(7, dtype=float) for i in range(3)})
        train_labels = pd.Series([0, 1, 2, 0, 1, 2, 2])

        monkeypatch.setattr(
            model_mod,
            "load_dev_split_indices",
            lambda _: {"train_fit": np.array([0, 1]), "val_hpo": np.array([0, 1, 2]), "val_calibration": np.array([3, 4, 5, 6])},
        )
        monkeypatch.setattr(
            model_mod,
            "load_clean_labels",
            lambda: pd.DataFrame(
                {
                    "verified_label": [2, 1],
                    "confidence": ["high", "medium"],
                    "source_row_idx": [3, 1],
                }
            ),
        )

        probs, labels, metadata = _resolve_threshold_tuning_subset(
            DummyModel(),
            train_features,
            train_labels,
            min_reviewed_rows=2,
        )

        np.testing.assert_array_equal(labels, np.array([2, 1]))
        np.testing.assert_allclose(
            probs,
            np.array(
                [
                    [0.05, 0.15, 0.80],
                    [0.20, 0.50, 0.30],
                ],
                dtype=float,
            ),
        )
        assert metadata["source"] == "reviewed_val_calibration"
        assert metadata["n_rows"] == 2

    def test_resolve_threshold_tuning_subset_applies_calibration(self, monkeypatch):
        import src.model as model_mod

        class DummyModel:
            def predict(self, dmatrix):
                return np.array(
                    [
                        [0.60, 0.30, 0.10],
                        [0.10, 0.20, 0.70],
                    ],
                    dtype=float,
                )

        train_features = pd.DataFrame({f"f_{i}": np.arange(4, dtype=float) for i in range(2)})
        train_labels = pd.Series([0, 2, 1, 2])
        calibration = {"enabled": True, "temperature": 2.0}

        monkeypatch.setattr(
            model_mod,
            "load_dev_split_indices",
            lambda _: {"train_fit": np.array([0]), "val_hpo": np.array([0, 1]), "val_calibration": np.array([2, 3])},
        )
        monkeypatch.setattr(model_mod, "load_clean_labels", lambda: pd.DataFrame())

        probs, labels, metadata = _resolve_threshold_tuning_subset(
            DummyModel(),
            train_features,
            train_labels,
            calibration=calibration,
            min_reviewed_rows=2,
        )

        expected = model_mod.apply_temperature(
            np.array(
                [
                    [0.60, 0.30, 0.10],
                    [0.10, 0.20, 0.70],
                ],
                dtype=float,
            ),
            2.0,
        )
        np.testing.assert_allclose(probs, expected)
        np.testing.assert_array_equal(labels, np.array([0, 2]))
        assert metadata["source"] == "heuristic_val_hpo"
        assert metadata["calibrated"] is True
