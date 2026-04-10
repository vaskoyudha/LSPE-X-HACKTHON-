# Real Benchmark Model Hardening Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Improve LPSE-X’s real 2021–2023 benchmark quality by replacing dead real-data features, redesigning heuristic labels around supported signals, tuning high-risk decisions, and keeping calibration/benchmark artifacts scientifically honest.

**Architecture:** Keep the current offline pipeline and artifact filenames, but strengthen the weak links in order: enrich flattening with real-supported fields, replace dead features inside the 30-feature catalog, redesign labels to use live signals instead of dead placeholders, then tune decision thresholds and refresh calibration, metrics, diagnostics, and proposal claims. Every stage preserves reproducibility through tracked artifacts and regression tests.

**Tech Stack:** Python 3.12, pandas, XGBoost, Optuna, pytest, onnxmltools, onnxruntime, matplotlib, nbconvert.

---

## File Map

- `src/data.py` — flatten extra real-OCDS fields needed for stronger features (`tender_mainProcurementCategory`, `tender_items_count`, `award_items_count`) and keep fallbacks for sparse fields.
- `src/features.py` — replace dead Tier 1/Tier 2 features with real-supported signals while preserving a 30-column output.
- `src/labels.py` — redesign heuristic flags so they depend on fields/signals that actually exist in the real 2021–2023 slice.
- `src/model.py` — add decision-threshold tuning for High Risk and make the evaluation pipeline save threshold-aware metrics.
- `src/diagnostics.py` — add feature-health and dead-feature audits alongside the existing provenance/circularity diagnostics.
- `scripts/run_diagnostics.py` — emit `models/feature_health.json` and a feature-health figure in addition to robustness artifacts.
- `scripts/rebuild_real_benchmark.py` — keep the multi-year refresh as the one supported end-to-end refresh path.
- `scripts/generate_calibration_sheet.py` / `scripts/simulate_review.py` — continue producing real calibration artifacts after label changes.
- `tests/test_data.py` — regression tests for newly flattened raw fields.
- `tests/test_features.py` — regression tests for the refreshed feature catalog.
- `tests/test_labeling.py` — regression tests for the new real-data-supported flags and label distribution behavior.
- `tests/test_model_training.py` — regression tests for threshold tuning and metric persistence.
- `tests/test_diagnostics.py` — regression tests for feature-health diagnostics.
- `README.md`, `proposal/bab2.md`, `proposal/bab4.md`, `proposal/proposal-final.md`, `training.ipynb`, `inference.ipynb` — update claims only after the new metrics are stable.

## Requirements Summary

1. Replace current dead/constant feature slots such as `f_tender_duration_days`, `f_num_tenderers`, `f_single_bidder`, `f_procurement_method_enc`, `f_contract_value_log`, `f_contract_award_ratio`, `f_days_to_contract`, and `f_buyer_method_diversity` with signals supported by the real 2021–2023 data (`src/features.py:66-135`, `src/features.py:143-339`, `data/processed/quality_report.md`).
2. Stop depending on dead label proxies (`flag_single_bidder`, `flag_direct_procurement`) and redesign heuristic labeling around available signals (`src/labels.py:327-483`).
3. Improve High Risk performance using threshold tuning on validation splits instead of pure argmax decisions (`src/model.py:795-854`, `src/model.py:1233-1366`).
4. Preserve the current offline/XAI bundle and artifact names (`models/metrics.json`, `models/calibration.json`, `models/benchmark_comparison.json`, notebooks, proposal figures).
5. Keep calibration tied to real reviewed rows and rerun it after the label/feature changes (`src/model.py:910-1016`, `data/processed/calibration_sheet_100.csv`, `data/processed/clean_labels_100.csv`).

## Acceptance Criteria

- The refreshed feature catalog still outputs **30 numeric ONNX-safe features**, but no feature is both 100% missing and kept as an active model input.
- `models/feature_health.json` reports the missingness and constant-rate for each feature and shows zero “active dead features” after the refactor.
- The label generator no longer depends on `tender_numberOfTenderers` or blank procurement methods for its core signals on the real benchmark.
- High Risk F1 on the real 2021–2023 test split improves over the current **0.8468** without dropping macro-F1 below **0.9400**.
- Calibration artifacts (`calibration_sheet_100.csv`, `clean_labels_100.csv`, `models/calibration.json`) remain real-data-backed (`ocid` synthetic ratio = 0.0).
- `pytest -q tests/test_data.py tests/test_features.py tests/test_labeling.py tests/test_model_training.py tests/test_diagnostics.py tests/test_split.py tests/test_leakage_guard.py tests/test_onnx_parity.py` passes.
- `training.ipynb` and `inference.ipynb` both execute with `nbconvert` after the new metrics are written.

## Risks and Mitigations

- **Risk:** Replacing dead features breaks notebook/proposal assumptions about the 30-feature schema.  
  **Mitigation:** Keep the feature count at 30, update `data/processed/feature_manifest.json`, and refresh notebooks/docs in the last task only after tests and metrics are green.
- **Risk:** New label rules accidentally create stronger circularity than the current ones.  
  **Mitigation:** Extend diagnostics with a feature-health report and rerun robustness ablations after every label refresh.
- **Risk:** Threshold tuning improves High Risk recall but silently hurts low/medium precision.  
  **Mitigation:** Save both argmax and thresholded metrics and gate acceptance on macro-F1 and High Risk F1 together.
- **Risk:** Real-data refresh runtime becomes too slow for iteration.  
  **Mitigation:** Keep `scripts/rebuild_real_benchmark.py` as the single refresh entry point and cap HPO trials/timeouts during iteration.

## Verification Steps

- Unit/regression tests for raw extraction, feature generation, labeling, threshold tuning, and diagnostics.
- Full targeted suite after the refactor.
- `python -m nbconvert --to notebook --execute training.ipynb --output /tmp/training-executed.ipynb`
- `python -m nbconvert --to notebook --execute inference.ipynb --output /tmp/inference-executed.ipynb`
- `python3 -m compileall src tests scripts`
- `git diff --check`

---

### Task 1: Enrich the flattened real-data contract with fields that can replace dead features

**Files:**
- Modify: `src/data.py:27-59`, `src/data.py:113-207`
- Test: `tests/test_data.py:87-139`

- [ ] **Step 1: Write the failing flattening test for real-supported fields**

```python
# tests/test_data.py

def test_extracts_category_and_item_counts_from_realish_record(self):
    record = {
        "ocid": "ocds-test-real-001",
        "buyer": {"id": "buyer-1", "name": "Kementerian Contoh"},
        "tender": {
            "id": "T-REAL-1",
            "title": "Pengadaan Jasa Konsultansi",
            "description": None,
            "value": {"currency": "IDR"},
            "minValue": {"amount": 125000000, "currency": "IDR"},
            "mainProcurementCategory": "services",
            "items": [{"id": "1"}, {"id": "2"}],
        },
        "awards": [
            {
                "id": "A-REAL-1",
                "value": {"amount": 118000000, "currency": "IDR"},
                "items": [{"id": "1"}],
                "suppliers": [{"id": "sup-1", "name": "PT Contoh"}],
            }
        ],
    }
    row = _flatten_release(record)[0]
    assert row["tender_value_amount"] == 125000000
    assert row["tender_mainProcurementCategory"] == "services"
    assert row["tender_items_count"] == 2
    assert row["award_items_count"] == 1
```

- [ ] **Step 2: Run the flattening test to verify it fails first**

Run: `source .venv/bin/activate && pytest -q tests/test_data.py::TestFlattenRelease::test_extracts_category_and_item_counts_from_realish_record`
Expected: FAIL with missing keys such as `tender_mainProcurementCategory`, `tender_items_count`, or `award_items_count`.

- [ ] **Step 3: Implement minimal flattening support for the new raw fields**

```python
# src/data.py inside _flatten_release(...)
base = {
    ...
    "tender_mainProcurementCategory": tender.get("mainProcurementCategory", ""),
    "tender_items_count": len(tender.get("items", []) or []),
    "award_items_count": len(award.get("items", []) or []),
}
```

Also extend the field audit list so these columns show up in quality reporting:

```python
# src/data.py near REQUIRED_FIELDS
REQUIRED_FIELDS = [
    ...,
    "tender_mainProcurementCategory",
    "tender_items_count",
    "award_items_count",
]
```

- [ ] **Step 4: Run the focused raw-data regression tests**

Run: `source .venv/bin/activate && pytest -q tests/test_data.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/data.py tests/test_data.py
git commit -m "feat: expose real-data category and item-count fields"
```

---

### Task 2: Replace dead features with real-supported signals while preserving a 30-feature catalog

**Files:**
- Modify: `src/features.py:66-135`, `src/features.py:143-339`
- Test: `tests/test_features.py:55-136`

- [ ] **Step 1: Write failing feature tests for the new replacements**

```python
# tests/test_features.py

def test_real_supported_replacement_features_exist(sample_raw_df):
    sample_raw_df["tender_mainProcurementCategory"] = ["services"] * len(sample_raw_df)
    sample_raw_df["tender_items_count"] = [2] * len(sample_raw_df)
    sample_raw_df["award_items_count"] = [1] * len(sample_raw_df)
    feats = compute_all_features(sample_raw_df)
    expected = {
        "f_main_procurement_category_enc",
        "f_tender_items_count",
        "f_award_items_count",
        "f_tender_value_missing",
        "f_award_value_missing",
        "f_buyer_recent_30d_tender_count",
        "f_supplier_recent_90d_award_count",
        "f_buyer_supplier_share_of_history",
    }
    assert expected.issubset(set(feats.columns))
    assert len(feats.columns) == 30
```

- [ ] **Step 2: Run the feature test to confirm the current catalog is missing them**

Run: `source .venv/bin/activate && pytest -q tests/test_features.py::TestCombinedFeatures::test_real_supported_replacement_features_exist`
Expected: FAIL because the new feature names do not exist yet.

- [ ] **Step 3: Replace dead slots with real-supported features**

Implement these Tier 1 replacements in `src/features.py`:

```python
# replace dead Tier 1 slots
feats["f_main_procurement_category_enc"] = (
    df.get("tender_mainProcurementCategory", pd.Series("", index=df.index))
      .fillna("")
      .str.lower()
      .map({"goods": 0, "services": 1, "works": 2})
      .fillna(-1)
      .astype(float)
)
feats["f_tender_items_count"] = _to_numeric(df.get("tender_items_count")).fillna(0)
feats["f_award_items_count"] = _to_numeric(df.get("award_items_count")).fillna(0)
feats["f_tender_value_missing"] = _to_numeric(df.get("tender_value_amount")).isna().astype(float)
feats["f_award_value_missing"] = _to_numeric(df.get("award_value_amount")).isna().astype(float)
```

Implement these Tier 2 replacements for the dead history slot(s):

```python
# real-supported historical replacements
feats["f_buyer_recent_30d_tender_count"] = ...
feats["f_supplier_recent_90d_award_count"] = ...
feats["f_buyer_supplier_share_of_history"] = ...
```

Use the same sorted-date/past-only pattern already established around `tier2_features()`.

- [ ] **Step 4: Run the full feature test file and inspect missingness**

Run:
```bash
source .venv/bin/activate && pytest -q tests/test_features.py
source .venv/bin/activate && python - <<'PY'
import pandas as pd
X = pd.read_parquet('train_data/features.parquet')
print((X.isna().mean().sort_values(ascending=False) * 100).head(12))
PY
```
Expected: feature tests pass and the previous dead slots disappear from the top-missing list.

- [ ] **Step 5: Commit**

```bash
git add src/features.py tests/test_features.py
git commit -m "feat: replace dead real-data feature slots"
```

---

### Task 3: Redesign heuristic labels to use real-supported signals instead of dead proxies

**Files:**
- Modify: `src/labels.py:327-483`
- Test: `tests/test_labeling.py:71-154`

- [ ] **Step 1: Add failing tests for the new real-data-supported flags**

```python
# tests/test_labeling.py

def test_repeat_pair_flag_uses_history_features(sample_procurement_df):
    sample_procurement_df["f_buyer_supplier_repeat_count"] = [0, 1, 3, 0, 4, 2]
    labels = compute_red_flags(sample_procurement_df)
    assert labels["flag_repeat_pair_history"].tolist() == [False, False, True, False, True, True]


def test_recent_supplier_surge_flag(sample_procurement_df):
    sample_procurement_df["f_supplier_recent_90d_award_count"] = [0, 1, 5, 0, 6, 2]
    labels = compute_red_flags(sample_procurement_df)
    assert labels["flag_supplier_recent_surge"].tolist() == [False, False, True, False, True, False]
```

- [ ] **Step 2: Run the new labeling tests so they fail**

Run: `source .venv/bin/activate && pytest -q tests/test_labeling.py -k "repeat_pair_flag or recent_supplier_surge"`
Expected: FAIL because the new flag functions/columns do not exist.

- [ ] **Step 3: Implement the v2 labeling rules**

Add replacement flags in `src/labels.py`:

```python
def flag_repeat_pair_history(df: pd.DataFrame, min_repeat: int = 2) -> pd.Series:
    vals = pd.to_numeric(df.get("f_buyer_supplier_repeat_count"), errors="coerce")
    return (vals >= min_repeat).fillna(False)


def flag_supplier_recent_surge(df: pd.DataFrame, min_recent_awards: int = 3) -> pd.Series:
    vals = pd.to_numeric(df.get("f_supplier_recent_90d_award_count"), errors="coerce")
    return (vals >= min_recent_awards).fillna(False)


def flag_buyer_value_spike(df: pd.DataFrame, z_threshold: float = 2.0) -> pd.Series:
    vals = pd.to_numeric(df.get("f_tender_value_zscore_buyer"), errors="coerce")
    return (vals >= z_threshold).fillna(False)
```

Then replace the dead registry entries with the supported ones:

```python
RED_FLAG_FUNCTIONS = {
    "short_title": flag_short_title,
    "short_description": flag_short_description,
    "q4_timing": flag_q4_timing,
    "price_deviation": flag_price_deviation,
    "high_value": flag_high_value,
    "repeat_pair_history": flag_repeat_pair_history,
    "supplier_recent_surge": flag_supplier_recent_surge,
    "buyer_value_spike": flag_buyer_value_spike,
}
```

When the materialization pipeline calls `compute_risk_labels`, pass a merged view of raw + features:

```python
label_inputs = pd.concat([raw_df.reset_index(drop=True), features_df.reset_index(drop=True)], axis=1)
labels = compute_risk_labels(label_inputs)
```

- [ ] **Step 4: Run labeling tests and inspect class balance on the real benchmark**

Run:
```bash
source .venv/bin/activate && pytest -q tests/test_labeling.py
source .venv/bin/activate && python - <<'PY'
import pandas as pd
labels = pd.read_parquet('train_data/labels.parquet')
print(labels['risk_label'].value_counts().sort_index())
PY
```
Expected: labeling tests pass and all three classes remain populated in train/test.

- [ ] **Step 5: Commit**

```bash
git add src/labels.py tests/test_labeling.py scripts/materialize.py
git commit -m "feat: redesign heuristic labels for real benchmark signals"
```

---

### Task 4: Tune thresholded decision rules to improve High Risk without sacrificing macro-F1

**Files:**
- Modify: `src/model.py:795-1016`, `src/model.py:1233-1366`
- Test: `tests/test_model_training.py:167-218`
- Create: `models/decision_thresholds.json`

- [ ] **Step 1: Write failing tests for threshold tuning helpers**

```python
# tests/test_model_training.py

def test_predict_with_thresholds_promotes_high_risk_when_probability_clears_cutoff():
    probs = np.array([
        [0.20, 0.55, 0.25],
        [0.05, 0.30, 0.65],
    ])
    thresholds = {"high_risk": 0.60, "low_risk": 0.80}
    preds = predict_with_thresholds(probs, thresholds)
    np.testing.assert_array_equal(preds, np.array([1, 2]))


def test_search_thresholds_returns_serializable_thresholds():
    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.2, 0.3, 0.5],
        [0.1, 0.6, 0.3],
    ])
    y_true = pd.Series([0, 2, 1])
    thresholds = search_decision_thresholds(probs, y_true)
    assert set(thresholds) == {"high_risk", "low_risk"}
```

- [ ] **Step 2: Run the threshold tests to verify they fail**

Run: `source .venv/bin/activate && pytest -q tests/test_model_training.py -k "thresholds"`
Expected: FAIL because the helper functions do not exist.

- [ ] **Step 3: Implement threshold tuning on `val_hpo`**

Add helpers in `src/model.py`:

```python
def predict_with_thresholds(probs: np.ndarray, thresholds: dict[str, float]) -> np.ndarray:
    preds = np.argmax(probs, axis=1)
    high_mask = probs[:, 2] >= thresholds["high_risk"]
    low_mask = probs[:, 0] >= thresholds["low_risk"]
    preds[high_mask] = 2
    preds[~high_mask & low_mask] = 0
    return preds


def search_decision_thresholds(probs: np.ndarray, y_true: pd.Series) -> dict[str, float]:
    best = {"high_risk": 0.50, "low_risk": 0.50}
    best_score = -1.0
    for high in np.linspace(0.35, 0.80, 10):
        for low in np.linspace(0.35, 0.80, 10):
            preds = predict_with_thresholds(probs.copy(), {"high_risk": float(high), "low_risk": float(low)})
            score = f1_score(y_true, preds, average="macro")
            if score > best_score:
                best = {"high_risk": float(high), "low_risk": float(low)}
                best_score = score
    return best
```

Wire the saved thresholds into `run_training_pipeline()` / `run_evaluation_pipeline()` and write them to `models/decision_thresholds.json`.

- [ ] **Step 4: Rebuild the real benchmark and compare High Risk F1**

Run:
```bash
source .venv/bin/activate && python scripts/rebuild_real_benchmark.py
source .venv/bin/activate && python - <<'PY'
import json
from pathlib import Path
metrics = json.loads(Path('models/metrics.json').read_text())
print(metrics['final_test'])
print(metrics.get('final_test_thresholded'))
PY
```
Expected: threshold file exists and High Risk F1 improves over the previous tracked value without macro-F1 dropping below 0.9400.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_model_training.py models/decision_thresholds.json
# plus any refreshed metric artifacts written by the rebuild
git commit -m "feat: tune decision thresholds for high-risk recall"
```

---

### Task 5: Extend diagnostics to report feature health and dead-feature removal explicitly

**Files:**
- Modify: `src/diagnostics.py:18-154`, `scripts/run_diagnostics.py:1-62`
- Test: `tests/test_diagnostics.py:1-48`
- Create: `models/feature_health.json`, `proposal/figures/feature_health.png`

- [ ] **Step 1: Write the failing diagnostics test**

```python
# tests/test_diagnostics.py

def test_feature_health_marks_constant_and_all_nan_features():
    df = pd.DataFrame(
        {
            "f_live": [1.0, 2.0, 3.0],
            "f_dead_nan": [np.nan, np.nan, np.nan],
            "f_dead_constant": [0.0, 0.0, 0.0],
        }
    )
    report = summarize_feature_health(df)
    assert report["f_dead_nan"]["all_nan"] is True
    assert report["f_dead_constant"]["constant"] is True
    assert report["f_live"]["constant"] is False
```

- [ ] **Step 2: Run the diagnostics test and confirm it fails**

Run: `source .venv/bin/activate && pytest -q tests/test_diagnostics.py -k feature_health`
Expected: FAIL because `summarize_feature_health` does not exist yet.

- [ ] **Step 3: Implement feature-health diagnostics and emit tracked artifacts**

```python
# src/diagnostics.py

def summarize_feature_health(features: pd.DataFrame) -> dict[str, dict[str, float | bool]]:
    report = {}
    for col in features.columns:
        non_null = features[col].dropna()
        report[col] = {
            "missing_pct": round(float(features[col].isna().mean() * 100), 2),
            "all_nan": bool(features[col].isna().all()),
            "constant": bool(len(non_null) > 0 and non_null.nunique() <= 1),
        }
    return report
```

Then update `scripts/run_diagnostics.py` to save `models/feature_health.json` and a simple bar chart of top missing features.

- [ ] **Step 4: Run diagnostics and review the new artifact**

Run:
```bash
source .venv/bin/activate && python scripts/run_diagnostics.py
source .venv/bin/activate && python - <<'PY'
import json
from pathlib import Path
report = json.loads(Path('models/feature_health.json').read_text())
print({k: v for k, v in report.items() if v['all_nan'] or v['constant']})
PY
```
Expected: the report clearly shows zero active dead features after Task 2 lands.

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics.py scripts/run_diagnostics.py tests/test_diagnostics.py models/feature_health.json proposal/figures/feature_health.png
# plus any refreshed robustness artifacts
git commit -m "feat: add feature-health diagnostics for the real benchmark"
```

---

### Task 6: Refresh calibration, diagnostics, notebooks, and proposal claims from the improved model

**Files:**
- Modify: `scripts/generate_calibration_sheet.py`, `scripts/simulate_review.py`, `scripts/rebuild_real_benchmark.py`
- Modify: `README.md`, `proposal/bab2.md`, `proposal/bab4.md`, `proposal/proposal-final.md`, `training.ipynb`, `inference.ipynb`
- Refresh tracked artifacts: `data/processed/calibration_sheet_100.csv`, `data/processed/clean_labels_100.csv`, `models/calibration.json`, `models/metrics.json`, `models/benchmark_comparison.json`, `proposal/figures/*.png`

- [ ] **Step 1: Write the failing integration check for real calibration provenance**

```python
# tests/test_diagnostics.py

def test_real_calibration_artifacts_are_not_synthetic():
    sheet = pd.read_csv("data/processed/calibration_sheet_100.csv")
    clean = pd.read_csv("data/processed/clean_labels_100.csv")
    assert float(sheet["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0
    assert float(clean["ocid"].astype(str).str.startswith("ocds-synth-").mean()) == 0.0
```

- [ ] **Step 2: Run the integration check and verify it currently fails before the refresh**

Run: `source .venv/bin/activate && pytest -q tests/test_diagnostics.py -k calibration_artifacts_are_not_synthetic`
Expected: FAIL on the pre-refresh branch if the calibration files are stale.

- [ ] **Step 3: Rebuild the end-to-end artifacts with the improved pipeline**

Run the full refresh sequence:

```bash
source .venv/bin/activate && python scripts/rebuild_real_benchmark.py
source .venv/bin/activate && python scripts/generate_calibration_sheet.py
source .venv/bin/activate && python scripts/simulate_review.py
source .venv/bin/activate && python - <<'PY'
from src.model import run_evaluation_pipeline
run_evaluation_pipeline()
PY
source .venv/bin/activate && python scripts/run_diagnostics.py
```

- [ ] **Step 4: Run the full verification bundle**

Run:
```bash
source .venv/bin/activate && pytest -q tests/test_data.py tests/test_features.py tests/test_labeling.py tests/test_model_training.py tests/test_split.py tests/test_leakage_guard.py tests/test_diagnostics.py tests/test_onnx_parity.py
source .venv/bin/activate && python -m nbconvert --to notebook --execute training.ipynb --output /tmp/training-executed.ipynb
source .venv/bin/activate && python -m nbconvert --to notebook --execute inference.ipynb --output /tmp/inference-executed.ipynb
python3 -m compileall src tests scripts
git diff --check
```
Expected: everything passes and the refreshed metrics/figures/docs reflect the new benchmark.

- [ ] **Step 5: Commit**

```bash
git add README.md proposal/bab2.md proposal/bab4.md proposal/proposal-final.md training.ipynb inference.ipynb \
  data/processed/calibration_sheet_100.csv data/processed/clean_labels_100.csv \
  models/calibration.json models/metrics.json models/benchmark_comparison.json models/feature_health.json \
  proposal/figures/calibration_curve.png proposal/figures/confusion_matrix.png proposal/figures/per_class_f1.png \
  proposal/figures/robustness_ablation.png proposal/figures/benchmark_comparison.png proposal/figures/feature_health.png
# plus any other regenerated benchmark artifacts
git commit -m "feat: harden the real benchmark model and refresh submission artifacts"
```

---

## Self-Review

### Spec coverage
- Dead real-data features: covered by Tasks 1-2 and verified in Task 5.
- Weak heuristic labeling: covered by Task 3.
- High Risk performance: covered by Task 4.
- Real calibration correctness: covered by Task 6.
- Proposal/notebook consistency: covered by Task 6.

### Placeholder scan
- No unresolved placeholders remain.
- Every code step includes a concrete code block.
- Every verification step includes concrete commands.

### Type consistency
- Raw-field additions feed feature replacements before label redesign.
- Label redesign merges raw + feature signals before threshold tuning.
- Threshold tuning writes a tracked JSON artifact before docs refresh.

