"""Tests for src.explain module — SHAP explainability pipeline."""

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

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
