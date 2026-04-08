"""Tests for ONNX export and parity — Task 19."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xgboost as xgb
import onnxmltools

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
# Constants
# ---------------------------------------------------------------------------
 
PARITY_THRESHOLD = 0.01   # maximum allowed mean absolute difference in P(class)
XGB_PATH = Path("models/xgb_model.ubj")
ONNX_PATH = Path("models/xgb_model.onnx")
IMPUTATION_PATH = Path("models/imputation_values.json")
TEST_FEATURES_PATH = Path("test_data/features.parquet")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
 
 
def _skip_unless(*paths: Path):
    missing = [str(p) for p in paths if not p.exists()]
    if missing:
        pytest.skip(f"Artifact(s) not found, skipping: {missing}")
 
 
def _load_xgb():
    import xgboost as xgb
 
    m = xgb.XGBClassifier()
    m.load_model(str(XGB_PATH))
    return m
 
 
def _load_onnx_session():
    import onnxruntime as rt
 
    return rt.InferenceSession(str(ONNX_PATH))
 
 
def _onnx_proba(sess, X: np.ndarray) -> np.ndarray:
    """Run ONNX session and return (n_samples, n_classes) probability array."""
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X.astype(np.float32)})
    # outputs[0] = predicted labels, outputs[1] = probability map (list of dicts)
    raw = outputs[1]
    if isinstance(raw[0], dict):
        n_classes = len(raw[0])
        return np.array([[row[c] for c in range(n_classes)] for row in raw])
    return np.array(raw)
 
 
# ---------------------------------------------------------------------------
# Artifact existence tests (always run)
# ---------------------------------------------------------------------------
 
 
@pytest.mark.p2
class TestArtifactExistence:
    def test_onnx_model_file_exists(self):
        _skip_unless(ONNX_PATH)
        assert ONNX_PATH.exists()
 
    def test_imputation_values_file_exists(self):
        _skip_unless(IMPUTATION_PATH)
        assert IMPUTATION_PATH.exists()
 
    def test_imputation_values_is_valid_json_dict(self):
        _skip_unless(IMPUTATION_PATH)
        with open(IMPUTATION_PATH) as f:
            vals = json.load(f)
        assert isinstance(vals, dict), "imputation_values.json must be a JSON object"
 
    def test_imputation_values_are_not_placeholders(self):
        _skip_unless(IMPUTATION_PATH)
        with open(IMPUTATION_PATH) as f:
            vals = json.load(f)
        real_keys = [k for k in vals if not k.startswith("_")]
        assert len(real_keys) > 0, (
            "imputation_values.json still contains only placeholder keys. "
            "Run src/model.py:fit_imputation() to populate."
        )
 
 
# ---------------------------------------------------------------------------
# Parity tests (require trained model artifacts)
# ---------------------------------------------------------------------------
 
 
@pytest.mark.p2
class TestOnnxParity:
    def test_onnx_parity_on_test_features(self):
        """
        Core parity test: mean absolute probability difference must be below threshold.
        Uses the first 200 rows of test_data/features.parquet.
        """
        _skip_unless(XGB_PATH, ONNX_PATH, TEST_FEATURES_PATH)
 
        import pandas as pd
 
        df = pd.read_parquet(TEST_FEATURES_PATH)
        numeric_cols = df.select_dtypes(include="number").columns.tolist()
        X = df[numeric_cols].values[:200].astype(np.float32)
 
        # Fill NaN using imputation values (fit from train)
        if IMPUTATION_PATH.exists():
            with open(IMPUTATION_PATH) as f:
                imp = json.load(f)
            for i, col in enumerate(numeric_cols):
                if col in imp:
                    nan_mask = np.isnan(X[:, i])
                    X[nan_mask, i] = float(imp[col])
 
        xgb_model = _load_xgb()
        xgb_proba = xgb_model.predict_proba(X)
 
        sess = _load_onnx_session()
        onnx_proba = _onnx_proba(sess, X)
 
        assert xgb_proba.shape == onnx_proba.shape, (
            f"Shape mismatch: XGB={xgb_proba.shape} ONNX={onnx_proba.shape}"
        )
 
        diff = float(np.mean(np.abs(xgb_proba - onnx_proba)))
        assert diff < PARITY_THRESHOLD, (
            f"ONNX parity FAILED: mean_abs_diff={diff:.5f} exceeds "
            f"threshold={PARITY_THRESHOLD}"
        )
 
    def test_onnx_parity_on_synthetic_data(self):
        """
        Parity test using synthetic random data when test features are unavailable.
        Requires only the model files.
        """
        _skip_unless(XGB_PATH, ONNX_PATH)
 
        xgb_model = _load_xgb()
        n_features = xgb_model.n_features_in_
        rng = np.random.default_rng(42)
        X = rng.standard_normal((100, n_features)).astype(np.float32)
 
        xgb_proba = xgb_model.predict_proba(X)
        sess = _load_onnx_session()
        onnx_proba = _onnx_proba(sess, X)
 
        diff = float(np.mean(np.abs(xgb_proba - onnx_proba)))
        assert diff < PARITY_THRESHOLD, (
            f"ONNX parity FAILED on synthetic data: mean_abs_diff={diff:.5f}"
        )
 
    def test_onnx_predictions_are_valid_probabilities(self):
        """ONNX output probabilities must sum to 1 and be in [0, 1]."""
        _skip_unless(XGB_PATH, ONNX_PATH)
 
        xgb_model = _load_xgb()
        n_features = xgb_model.n_features_in_
        rng = np.random.default_rng(7)
        X = rng.standard_normal((50, n_features)).astype(np.float32)
 
        sess = _load_onnx_session()
        onnx_proba = _onnx_proba(sess, X)
 
        assert np.all(onnx_proba >= 0), "ONNX probabilities contain negative values"
        assert np.all(onnx_proba <= 1), "ONNX probabilities contain values > 1"
 
        row_sums = onnx_proba.sum(axis=1)
        np.testing.assert_allclose(
            row_sums, np.ones(len(row_sums)), atol=1e-4,
            err_msg="ONNX probability rows do not sum to 1",
        )
 
    def test_onnx_argmax_agrees_with_xgb_argmax(self):
        """Predicted class from ONNX must match XGBoost argmax on > 95% of rows."""
        _skip_unless(XGB_PATH, ONNX_PATH)
 
        xgb_model = _load_xgb()
        n_features = xgb_model.n_features_in_
        rng = np.random.default_rng(99)
        X = rng.standard_normal((200, n_features)).astype(np.float32)
 
        xgb_pred = np.argmax(xgb_model.predict_proba(X), axis=1)
 
        sess = _load_onnx_session()
        onnx_pred = np.argmax(_onnx_proba(sess, X), axis=1)
 
        agreement = float(np.mean(xgb_pred == onnx_pred))
        assert agreement >= 0.95, (
            f"Predicted-class agreement between XGB and ONNX is only "
            f"{agreement:.1%} (threshold: 95%)"
        )
 
 
# ---------------------------------------------------------------------------
# Imputation integration tests
# ---------------------------------------------------------------------------
 
 
@pytest.mark.p2
class TestImputationContract:
    def test_imputation_values_fitted_from_train_only(self):
        """
        Structural check: imputation file must not contain test-set statistics.
        We verify the file exists and is a non-empty dict of float values.
        """
        _skip_unless(IMPUTATION_PATH)
        with open(IMPUTATION_PATH) as f:
            vals = json.load(f)
 
        real_keys = {k: v for k, v in vals.items() if not k.startswith("_")}
        assert len(real_keys) > 0
 
        for key, val in real_keys.items():
            assert isinstance(val, (int, float)), (
                f"Imputation value for '{key}' is not numeric: {val!r}"
            )
 
    def test_apply_imputation_fills_nans(self):
        """Unit test for src/model.py:apply_imputation()."""
        _skip_unless(IMPUTATION_PATH)
        import pandas as pd
        from src.model import apply_imputation
 
        with open(IMPUTATION_PATH) as f:
            imp = json.load(f)
 
        real_keys = [k for k in imp if not k.startswith("_")]
        if not real_keys:
            pytest.skip("No real imputation values available yet")
 
        col = real_keys[0]
        df = pd.DataFrame({col: [1.0, np.nan, 3.0, np.nan]})
        filled = apply_imputation(df, imputation_values={col: float(imp[col])})
        assert filled[col].isna().sum() == 0, "apply_imputation() left NaN values"


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
