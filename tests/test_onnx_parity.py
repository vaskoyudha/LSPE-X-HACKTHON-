"""Tests for ONNX export and parity — Task 19."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb

from src.model import (
    BASE_PARAMS,
    N_CLASSES,
    MODELS_DIR,
    compute_imputation_values,
    export_onnx,
    load_onnx_model,
    verify_onnx_parity,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def trained_artifacts(tmp_path_factory):
    """Train a small model and create all needed artifacts."""
    tmp = tmp_path_factory.mktemp("onnx_test")

    rng = np.random.default_rng(42)
    n = 100
    X_train = pd.DataFrame({f"f_{i}": rng.standard_normal(n) for i in range(30)})
    y_train = pd.Series(rng.choice([0, 1, 2], size=n, p=[0.4, 0.45, 0.15]))

    X_test = pd.DataFrame({f"f_{i}": rng.standard_normal(50) for i in range(30)})
    # Add some NaN to test imputation
    X_test.iloc[0, 0] = np.nan
    X_test.iloc[2, 5] = np.nan

    dtrain = xgb.DMatrix(X_train, label=y_train)
    model = xgb.train(BASE_PARAMS, dtrain, num_boost_round=20)

    return model, X_train, X_test, tmp


# ---------------------------------------------------------------------------
# P0: Core ONNX tests
# ---------------------------------------------------------------------------


@pytest.mark.p0
class TestImputation:
    def test_computes_30_values(self, trained_artifacts, monkeypatch):
        import src.model as model_mod
        model, X_train, _, tmp = trained_artifacts
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp)

        imputation = compute_imputation_values(X_train)
        assert len(imputation) == 30
        assert all(isinstance(v, float) for v in imputation.values())

    def test_values_are_medians(self, trained_artifacts, monkeypatch):
        import src.model as model_mod
        model, X_train, _, tmp = trained_artifacts
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp)

        imputation = compute_imputation_values(X_train)
        for col in X_train.columns:
            expected = float(X_train[col].median())
            assert abs(imputation[col] - expected) < 1e-10


@pytest.mark.p0
class TestONNXExport:
    def test_export_creates_file(self, trained_artifacts, monkeypatch):
        import src.model as model_mod
        model, X_train, _, tmp = trained_artifacts
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp)

        onnx_path = export_onnx(model, X_train)
        assert onnx_path.exists()
        assert onnx_path.stat().st_size > 0

    def test_exported_model_loadable(self, trained_artifacts, monkeypatch):
        import src.model as model_mod
        model, X_train, _, tmp = trained_artifacts
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp)

        export_onnx(model, X_train)
        loaded = load_onnx_model()
        assert isinstance(loaded, xgb.Booster)


@pytest.mark.p0
class TestONNXParity:
    def test_parity_exact(self, trained_artifacts, monkeypatch):
        """JSON export should produce identical predictions."""
        import src.model as model_mod
        model, X_train, X_test, tmp = trained_artifacts
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp)

        export_onnx(model, X_train)
        imputation = compute_imputation_values(X_train)

        parity = verify_onnx_parity(model, X_test, imputation, atol=1e-6)
        assert parity is True, "JSON-exported model should have exact parity"

    def test_predictions_match(self, trained_artifacts, monkeypatch):
        """Predictions from both models must be identical."""
        import src.model as model_mod
        model, X_train, X_test, tmp = trained_artifacts
        monkeypatch.setattr(model_mod, "MODELS_DIR", tmp)

        export_onnx(model, X_train)

        # Fill NaN with imputation values
        imputation = compute_imputation_values(X_train)
        X_clean = X_test.copy()
        for col, val in imputation.items():
            if col in X_clean.columns:
                X_clean[col] = X_clean[col].fillna(val)

        dtest = xgb.DMatrix(X_clean)
        native_preds = np.argmax(model.predict(dtest), axis=1)

        loaded = load_onnx_model()
        onnx_preds = np.argmax(loaded.predict(dtest), axis=1)

        np.testing.assert_array_equal(native_preds, onnx_preds)
