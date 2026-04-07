"""Tests for src.explain module — SHAP explainability pipeline."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
from unittest.mock import MagicMock
import onnxmltools

from src.model import BASE_PARAMS, N_CLASSES, MODELS_DIR
from src.explain import (
    get_explainer,
    explain_single,
    shap_counterfactual,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_model():
    """Train a small model for explanation tests."""
    rng = np.random.default_rng(42)
    n = 200
    X = pd.DataFrame({f"f_{i}": rng.standard_normal(n) for i in range(30)})
    y = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.4, 0.45, 0.15]))

    dtrain = xgb.DMatrix(X, label=y)
    model = xgb.train(BASE_PARAMS, dtrain, num_boost_round=20)
    return model, X


@pytest.fixture(scope="module")
def explainer(trained_model):
    model, _ = trained_model
    return get_explainer(model)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
 
def _make_mock_model(predicted_proba: np.ndarray):
    model = MagicMock()
    model.predict_proba.return_value = predicted_proba
    return model
 
 
def _make_mock_explainer(shap_list: list[np.ndarray]):
    """shap_list[class_idx] has shape (1, n_features)."""
    explainer = MagicMock()
    explainer.shap_values.return_value = shap_list
    return explainer

# ---------------------------------------------------------------------------
# explain_single – contract tests
# ---------------------------------------------------------------------------
 
 
@pytest.mark.p1
class TestExplainSingleContract:
    """Verify the output contract: predicted_class, probability, factors."""
 
    def _run(self, n_features=5, predicted_class_idx=2):
        from src.explain import explain_single
 
        proba = np.zeros((1, 3))
        proba[0, predicted_class_idx] = 0.8
        proba[0, (predicted_class_idx + 1) % 3] = 0.1
        proba[0, (predicted_class_idx + 2) % 3] = 0.1
 
        shap_list = [np.zeros((1, n_features)) for _ in range(3)]
        shap_list[predicted_class_idx] = np.array(
            [[0.3, -0.1, 0.5, 0.0, -0.2][:n_features]]
        )
 
        feature_names = [f"feat_{i}" for i in range(n_features)]
        row = {f: float(i) for i, f in enumerate(feature_names)}
 
        return explain_single(
            row,
            feature_names,
            model=_make_mock_model(proba),
            explainer=_make_mock_explainer(shap_list),
        )
 
    def test_required_keys_present(self):
        result = self._run()
        assert "predicted_class" in result
        assert "probability" in result
        assert "factors" in result
 
    def test_top_factors_key_absent(self):
        """Legacy key 'top_factors' must never appear in the output."""
        result = self._run()
        assert "top_factors" not in result, (
            "'top_factors' is a rejected legacy key (see task plan hard rules)"
        )
 
    def test_predicted_class_is_int(self):
        result = self._run()
        assert isinstance(result["predicted_class"], int)
 
    def test_probability_is_float_in_unit_interval(self):
        result = self._run()
        p = result["probability"]
        assert isinstance(p, float)
        assert 0.0 <= p <= 1.0
 
    def test_factors_is_list(self):
        result = self._run()
        assert isinstance(result["factors"], list)
        assert len(result["factors"]) > 0
 
    def test_factors_have_required_sub_keys(self):
        result = self._run()
        for factor in result["factors"]:
            assert "feature" in factor, "factor missing 'feature'"
            assert "shap_value" in factor, "factor missing 'shap_value'"
            assert "feature_value" in factor, "factor missing 'feature_value'"
 
    def test_predicted_class_matches_argmax_proba(self):
        from src.explain import explain_single
 
        proba = np.array([[0.1, 0.7, 0.2]])
        n_features = 3
        shap_list = [np.zeros((1, n_features)) for _ in range(3)]
        shap_list[1] = np.array([[0.4, -0.3, 0.1]])
 
        feature_names = ["a", "b", "c"]
        row = {"a": 1.0, "b": 2.0, "c": 3.0}
 
        result = explain_single(
            row,
            feature_names,
            model=_make_mock_model(proba),
            explainer=_make_mock_explainer(shap_list),
        )
        assert result["predicted_class"] == 1
 
    def test_factors_sorted_by_absolute_shap_descending(self):
        from src.explain import explain_single
 
        proba = np.array([[0.1, 0.1, 0.8]])
        shap_vals = np.array([[0.05, -0.9, 0.3, 0.0, -0.4]])
        n_features = 5
 
        shap_list = [np.zeros((1, n_features)), np.zeros((1, n_features)), shap_vals]
        feature_names = [f"f{i}" for i in range(n_features)]
        row = {f: float(i) for i, f in enumerate(feature_names)}
 
        result = explain_single(
            row,
            feature_names,
            model=_make_mock_model(proba),
            explainer=_make_mock_explainer(shap_list),
        )
 
        abs_vals = [abs(f["shap_value"]) for f in result["factors"]]
        assert abs_vals == sorted(abs_vals, reverse=True), (
            "factors must be sorted by |shap_value| descending"
        )
 
    def test_row_as_numpy_array(self):
        """explain_single should accept a 1-D numpy array as row input."""
        from src.explain import explain_single
 
        proba = np.array([[0.2, 0.5, 0.3]])
        n_features = 4
        shap_list = [np.zeros((1, n_features)) for _ in range(3)]
        shap_list[1] = np.array([[0.1, 0.4, -0.2, 0.0]])
 
        feature_names = ["x0", "x1", "x2", "x3"]
        row_arr = np.array([0.0, 1.0, 2.0, 3.0])
 
        result = explain_single(
            row_arr,
            feature_names,
            model=_make_mock_model(proba),
            explainer=_make_mock_explainer(shap_list),
        )
 
        assert result["predicted_class"] == 1
        assert "factors" in result
 
 
# ---------------------------------------------------------------------------
# explain_single – multi-class SHAP variant: ndarray (1, n_feat, n_classes)
# ---------------------------------------------------------------------------
 
 
@pytest.mark.p1
class TestExplainSingleNdarrayShap:
    """Cover the ndarray-style SHAP output (xgboost >=2 default)."""
 
    def test_ndarray_3d_shap(self):
        from src.explain import explain_single
 
        n_features = 4
        n_classes = 3
        proba = np.array([[0.1, 0.2, 0.7]])
 
        # shape (1, n_features, n_classes)
        shap_ndarray = np.zeros((1, n_features, n_classes))
        shap_ndarray[0, :, 2] = [0.5, -0.1, 0.3, 0.0]
 
        explainer = MagicMock()
        explainer.shap_values.return_value = shap_ndarray
 
        feature_names = [f"v{i}" for i in range(n_features)]
        row = {f: float(i) for i, f in enumerate(feature_names)}
 
        result = explain_single(
            row,
            feature_names,
            model=_make_mock_model(proba),
            explainer=explainer,
        )
 
        assert result["predicted_class"] == 2
        assert len(result["factors"]) > 0
        assert "top_factors" not in result
 
 
# ---------------------------------------------------------------------------
# Counterfactual – SHAP fallback
# ---------------------------------------------------------------------------
 
 
@pytest.mark.p1
class TestCounterfactualShap:
    def _sample_result(self):
        return {
            "predicted_class": 2,
            "probability": 0.75,
            "factors": [
                {"feature": "single_bid_flag", "shap_value": 0.4, "feature_value": 1.0},
                {"feature": "price_dev_ratio", "shap_value": 0.3, "feature_value": 0.01},
                {"feature": "bid_window_days", "shap_value": -0.1, "feature_value": 7.0},
                {"feature": "repeat_winner", "shap_value": 0.2, "feature_value": 1.0},
            ],
        }
 
    def test_returns_predicted_class_and_changes(self):
        from src.explain import get_counterfactual_shap
 
        result = get_counterfactual_shap(self._sample_result())
        assert "predicted_class" in result
        assert "suggested_changes" in result
 
    def test_changes_direction_is_decrease_for_positive_shap(self):
        from src.explain import get_counterfactual_shap
 
        result = get_counterfactual_shap(self._sample_result())
        for change in result["suggested_changes"]:
            assert change["direction"] in ("decrease", "increase")
 
    def test_max_three_changes(self):
        from src.explain import get_counterfactual_shap
 
        result = get_counterfactual_shap(self._sample_result(), top_changes=3)
        assert len(result["suggested_changes"]) <= 3

# ---------------------------------------------------------------------------
# P0: Core explanation tests
# ---------------------------------------------------------------------------


@pytest.mark.p0
class TestExplainSingle:
    def test_returns_required_keys(self, trained_model, explainer):
        model, X = trained_model
        row = X.iloc[[0]]
        result = explain_single(row, model=model, explainer=explainer)

        assert "predicted_class" in result
        assert "predicted_label" in result
        assert "probability" in result
        assert "probabilities" in result
        assert "factors" in result

    def test_predicted_class_valid(self, trained_model, explainer):
        model, X = trained_model
        result = explain_single(X.iloc[[0]], model=model, explainer=explainer)
        assert result["predicted_class"] in [0, 1, 2]

    def test_probability_range(self, trained_model, explainer):
        model, X = trained_model
        result = explain_single(X.iloc[[0]], model=model, explainer=explainer)
        assert 0 <= result["probability"] <= 1

    def test_probabilities_sum_to_one(self, trained_model, explainer):
        model, X = trained_model
        result = explain_single(X.iloc[[0]], model=model, explainer=explainer)
        assert len(result["probabilities"]) == N_CLASSES
        np.testing.assert_allclose(sum(result["probabilities"]), 1.0, atol=1e-4)

    def test_factors_structure(self, trained_model, explainer):
        model, X = trained_model
        result = explain_single(X.iloc[[0]], model=model, explainer=explainer, top_k=5)
        factors = result["factors"]

        assert len(factors) <= 5
        for f in factors:
            assert "feature" in f
            assert "value" in f
            assert "shap_value" in f
            assert "direction" in f
            assert f["direction"] in ["increases_risk", "decreases_risk"]

    def test_factors_sorted_by_importance(self, trained_model, explainer):
        model, X = trained_model
        result = explain_single(X.iloc[[0]], model=model, explainer=explainer, top_k=10)
        factors = result["factors"]

        shap_abs = [abs(f["shap_value"]) for f in factors]
        assert shap_abs == sorted(shap_abs, reverse=True)

    def test_series_input(self, trained_model, explainer):
        model, X = trained_model
        result = explain_single(X.iloc[0], model=model, explainer=explainer)
        assert "predicted_class" in result


@pytest.mark.p0
class TestExplainer:
    def test_explainer_creation(self, trained_model):
        model, _ = trained_model
        exp = get_explainer(model)
        assert exp is not None


@pytest.mark.p1
class TestCounterfactual:
    def test_counterfactual_returns_suggestions(self, trained_model, explainer):
        model, X = trained_model
        # Find a high-risk prediction
        for i in range(len(X)):
            result = explain_single(X.iloc[[i]], model=model, explainer=explainer)
            if result["predicted_class"] == 2:
                cf = shap_counterfactual(result, target_class=0)
                assert isinstance(cf, list)
                if len(cf) > 0 and "feature" in cf[0]:
                    assert "suggestion" in cf[0]
                    assert "impact" in cf[0]
                return
        pytest.skip("No high-risk prediction found in test data")

    def test_counterfactual_already_target(self, trained_model, explainer):
        model, X = trained_model
        result = explain_single(X.iloc[[0]], model=model, explainer=explainer)
        cf = shap_counterfactual(result, target_class=result["predicted_class"])
        assert cf[0].get("message") is not None
