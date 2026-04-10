# Fraud Evidence Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a separate fraud-evidence lane that uses canonical reviewed / confirmed-outcome labels plus buyer-supplier graph features to produce a defensible, leakage-safe fraud-support score beside the baseline heuristic risk model.

**Architecture:** Keep the current heuristic three-class lane unchanged. Build a second, fully separate binary fraud-evidence pipeline that starts from `src/outcomes.py`, constructs a point-in-time-safe evidence dataset with graph-style relationship features computed from historical procurement activity, trains a positive-unlabeled-friendly XGBoost model using only existing dependencies, and emits its own metrics and diagnostics artifacts. Integrate the new lane into diagnostics and reporting without merging its labels into `train_data/labels.parquet` or `test_data/labels.parquet`.

**Tech Stack:** Python 3.12, pandas, numpy, scikit-learn, xgboost, pytest, matplotlib.

---

## File Map

- Create: `src/graph_features.py` — point-in-time buyer/supplier relationship aggregates without new dependencies.
- Create: `src/fraud_evidence.py` — evidence dataset builder, PU-style label preparation, training, scoring, and metrics helpers for the fraud-evidence lane.
- Modify: `src/outcomes.py` — add evidence-strength metadata normalization for reviewed and confirmed-outcome rows.
- Modify: `src/diagnostics.py` — add fraud-evidence lane summaries and artifact loading helpers.
- Create: `scripts/build_fraud_evidence_dataset.py` — materialize the fraud-evidence dataset from raw/features/outcomes.
- Create: `scripts/train_fraud_evidence_model.py` — train and save the fraud-evidence XGBoost model and metrics.
- Modify: `scripts/run_diagnostics.py` — include the fraud-evidence lane in `models/evaluation_lanes.json` and `.sisyphus/evidence/weakness-diagnostics.json`.
- Create: `tests/test_graph_features.py` — regression tests for graph/relationship features.
- Create: `tests/test_fraud_evidence.py` — regression tests for dataset construction, PU labels, training, and metrics.
- Modify: `tests/test_diagnostics.py` — verify the fraud-evidence lane stays separate from the heuristic and reviewed lanes.
- Modify: `README.md` — document the new fraud-evidence artifacts and claim boundaries.

## Requirements Summary

1. Fraud-evidence labels must remain distinct from heuristic risk labels and reviewed-risk summary metrics.
2. No new dependencies may be introduced; graph features must use pandas/numpy-based aggregates.
3. Feature computation must be point-in-time safe using only historical data available before each contract/tender timestamp.
4. The fraud-evidence lane must be binary and descriptive of stronger evidence (`positive` vs `unlabeled`), not a relabeling of the three-class heuristic task.
5. Metrics and artifacts for the fraud-evidence lane must live in their own files and be reported separately.

## Acceptance Criteria

- `scripts/build_fraud_evidence_dataset.py` writes `models/fraud_evidence_dataset.parquet`.
- `scripts/train_fraud_evidence_model.py` writes:
  - `models/fraud_evidence_model.ubj`
  - `models/fraud_evidence_metrics.json`
  - `models/fraud_evidence_calibration.json`
- `models/evaluation_lanes.json` includes a `fraud_evidence_lane` entry distinct from heuristic and reviewed lanes.
- `pytest -q tests/test_graph_features.py tests/test_fraud_evidence.py tests/test_diagnostics.py` passes.
- `python scripts/train_fraud_evidence_model.py` runs without changing the baseline heuristic artifacts.
- README explicitly says the fraud-evidence lane is stronger than heuristic risk but still not a legal fraud verdict.

## Risks and Mitigations

- **Risk:** Fraud-evidence positives are too sparse for stable supervised learning.  
  **Mitigation:** Treat non-positive rows as unlabeled, keep the training objective binary, and evaluate primarily with ranking-style metrics and matched-positive recall.

- **Risk:** Relationship features leak future activity.  
  **Mitigation:** Build features from prior rows only, ordered by `tender_datePublished` and `award_date`.

- **Risk:** The new lane gets confused with the existing heuristic benchmark.  
  **Mitigation:** Separate artifact names, separate `label_type`, and explicit evaluation-lane reporting in diagnostics.

- **Risk:** Graph feature engineering grows too broad.  
  **Mitigation:** Start with four simple relationship aggregates using existing columns before considering more complex network modeling.

## Verification Steps

- `source .venv/bin/activate && pytest -q tests/test_graph_features.py tests/test_fraud_evidence.py tests/test_diagnostics.py`
- `source .venv/bin/activate && python scripts/build_fraud_evidence_dataset.py`
- `source .venv/bin/activate && python scripts/train_fraud_evidence_model.py`
- `source .venv/bin/activate && python scripts/run_diagnostics.py`
- `source .venv/bin/activate && python -m compileall src tests scripts`
- `git diff --check`

---

### Task 1: Extend evidence labels and create point-in-time graph features

**Files:**
- Modify: `src/outcomes.py`
- Create: `src/graph_features.py`
- Test: `tests/test_graph_features.py`

- [ ] **Step 1: Write failing graph-feature tests**

```python
# tests/test_graph_features.py
from __future__ import annotations

import pandas as pd

from src.graph_features import build_relationship_features


def test_build_relationship_features_uses_only_past_rows():
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-1", "ocds-2", "ocds-3"],
            "buyer_id": ["buyer-a", "buyer-a", "buyer-a"],
            "supplier_id": ["sup-1", "sup-1", "sup-2"],
            "tender_datePublished": pd.to_datetime(
                ["2023-01-01", "2023-02-01", "2023-03-01"], utc=True
            ),
            "award_value_amount": [100.0, 200.0, 300.0],
        }
    )

    feats = build_relationship_features(raw)

    assert feats["g_buyer_supplier_prev_contract_count"].tolist() == [0, 1, 0]
    assert feats["g_supplier_prev_buyer_count"].tolist() == [0, 1, 1]
    assert feats["g_buyer_prev_supplier_count"].tolist() == [0, 1, 1]
```

- [ ] **Step 2: Run the graph-feature test to verify it fails**

Run: `source .venv/bin/activate && pytest -q tests/test_graph_features.py::test_build_relationship_features_uses_only_past_rows`

Expected: FAIL with `ModuleNotFoundError: No module named 'src.graph_features'`.

- [ ] **Step 3: Implement minimal evidence-strength normalization and graph features**

```python
# src/outcomes.py
def add_evidence_strength(df: pd.DataFrame) -> pd.DataFrame:
    weighted = df.copy()
    weighted["evidence_strength"] = weighted["label_family"].map(
        {
            "reviewed_risk": 0.5,
            "confirmed_irregularity": 0.8,
            "confirmed_fraud": 1.0,
        }
    )
    return weighted
```

```python
# src/graph_features.py
from __future__ import annotations

import pandas as pd


def build_relationship_features(raw_df: pd.DataFrame) -> pd.DataFrame:
    ordered = raw_df.sort_values("tender_datePublished").reset_index(drop=True).copy()
    pair_key = ordered["buyer_id"].astype(str) + "||" + ordered["supplier_id"].astype(str)

    ordered["g_buyer_supplier_prev_contract_count"] = pair_key.groupby(pair_key).cumcount()
    ordered["g_supplier_prev_buyer_count"] = (
        ordered.groupby("supplier_id")["buyer_id"]
        .transform(lambda s: s.expanding().apply(lambda x: x[:-1].nunique() if len(x) > 1 else 0))
        .fillna(0)
        .astype(int)
    )
    ordered["g_buyer_prev_supplier_count"] = (
        ordered.groupby("buyer_id")["supplier_id"]
        .transform(lambda s: s.expanding().apply(lambda x: x[:-1].nunique() if len(x) > 1 else 0))
        .fillna(0)
        .astype(int)
    )
    ordered["g_pair_prev_award_value_sum"] = (
        ordered.groupby(pair_key)["award_value_amount"].cumsum()
        - ordered["award_value_amount"].fillna(0)
    )
    return ordered[
        [
            "g_buyer_supplier_prev_contract_count",
            "g_supplier_prev_buyer_count",
            "g_buyer_prev_supplier_count",
            "g_pair_prev_award_value_sum",
        ]
    ]
```

- [ ] **Step 4: Run the graph feature tests**

Run: `source .venv/bin/activate && pytest -q tests/test_graph_features.py`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/outcomes.py src/graph_features.py tests/test_graph_features.py
git commit -F - <<'EOF'
Add point-in-time relationship features for the fraud-evidence lane

This introduces simple buyer-supplier history features and evidence-strength
normalization without adding new dependencies.

Constraint: No new graph library dependencies are allowed
Rejected: Add networkx for richer centrality metrics | unnecessary for the first fraud-evidence iteration
Confidence: medium
Scope-risk: moderate
Directive: Keep graph features strictly historical; do not use future transactions
Tested: pytest -q tests/test_graph_features.py
Not-tested: Full-scale runtime on the entire benchmark
EOF
```

---

### Task 2: Build the fraud-evidence dataset with positive-unlabeled labeling

**Files:**
- Create: `src/fraud_evidence.py`
- Create: `scripts/build_fraud_evidence_dataset.py`
- Test: `tests/test_fraud_evidence.py`

- [ ] **Step 1: Write failing dataset-builder tests**

```python
# tests/test_fraud_evidence.py
from __future__ import annotations

import pandas as pd

from src.fraud_evidence import build_fraud_evidence_dataset


def test_build_fraud_evidence_dataset_marks_positive_and_unlabeled_rows():
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-1", "ocds-2"],
            "buyer_id": ["buyer-a", "buyer-b"],
            "supplier_id": ["sup-1", "sup-2"],
            "tender_datePublished": pd.to_datetime(["2023-01-01", "2023-02-01"], utc=True),
            "award_value_amount": [100.0, 200.0],
        }
    )
    base_features = pd.DataFrame({"f_tender_value_log": [1.0, 2.0]})
    evidence = pd.DataFrame(
        {
            "ocid": ["ocds-2"],
            "label_family": ["confirmed_fraud"],
            "label_value": ["fraud"],
            "evidence_strength": [1.0],
        }
    )

    dataset = build_fraud_evidence_dataset(raw, base_features, evidence)

    assert dataset["fraud_evidence_target"].tolist() == [0, 1]
    assert dataset["is_unlabeled"].tolist() == [1, 0]
```

- [ ] **Step 2: Run the dataset test to confirm it fails first**

Run: `source .venv/bin/activate && pytest -q tests/test_fraud_evidence.py::test_build_fraud_evidence_dataset_marks_positive_and_unlabeled_rows`

Expected: FAIL because `src.fraud_evidence` does not exist yet.

- [ ] **Step 3: Implement the dataset builder**

```python
# src/fraud_evidence.py
from __future__ import annotations

import pandas as pd

from src.graph_features import build_relationship_features
from src.outcomes import add_evidence_strength


def build_fraud_evidence_dataset(
    raw_df: pd.DataFrame,
    base_features: pd.DataFrame,
    evidence_df: pd.DataFrame,
) -> pd.DataFrame:
    graph = build_relationship_features(raw_df)
    evidence = add_evidence_strength(evidence_df)
    merged = raw_df[["ocid", "buyer_id", "supplier_id", "tender_datePublished"]].copy()
    merged = pd.concat([merged.reset_index(drop=True), base_features.reset_index(drop=True), graph], axis=1)
    merged = merged.merge(
        evidence[["ocid", "label_family", "label_value", "evidence_strength"]].drop_duplicates("ocid"),
        on="ocid",
        how="left",
    )
    merged["fraud_evidence_target"] = merged["label_family"].isin(
        ["confirmed_fraud", "confirmed_irregularity"]
    ).astype(int)
    merged["is_unlabeled"] = (merged["fraud_evidence_target"] == 0).astype(int)
    merged["sample_weight"] = merged["evidence_strength"].fillna(0.2)
    return merged
```

```python
# scripts/build_fraud_evidence_dataset.py
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fraud_evidence import build_fraud_evidence_dataset
from src.outcomes import load_confirmed_outcome_labels, load_canonical_reviewed_labels

OUTPUT_PATH = ROOT / "models" / "fraud_evidence_dataset.parquet"


def main() -> None:
    raw = pd.read_parquet(ROOT / "test_data" / "raw.parquet")
    features = pd.read_parquet(ROOT / "test_data" / "features.parquet")
    evidence = pd.concat(
        [load_canonical_reviewed_labels(), load_confirmed_outcome_labels()],
        ignore_index=True,
    )
    dataset = build_fraud_evidence_dataset(raw, features, evidence)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(OUTPUT_PATH, index=False)
    print(f"Wrote {OUTPUT_PATH} ({len(dataset)} rows)")
```

- [ ] **Step 4: Run dataset tests and the builder script**

Run:

```bash
source .venv/bin/activate && pytest -q tests/test_fraud_evidence.py
source .venv/bin/activate && python scripts/build_fraud_evidence_dataset.py
```

Expected:
- tests pass
- `models/fraud_evidence_dataset.parquet` is written

- [ ] **Step 5: Commit**

```bash
git add src/fraud_evidence.py scripts/build_fraud_evidence_dataset.py tests/test_fraud_evidence.py
git commit -F - <<'EOF'
Build a separate positive-unlabeled fraud-evidence dataset

This materializes a second dataset using stronger evidence labels and
historical relationship features without touching the heuristic train/test
targets.

Constraint: The fraud-evidence lane must not overwrite train_data/labels.parquet or test_data/labels.parquet
Rejected: Reuse the heuristic three-class labels as the new fraud target | not scientifically defensible
Confidence: high
Scope-risk: moderate
Directive: Treat non-positive rows as unlabeled support data, not confirmed clean negatives
Tested: pytest -q tests/test_fraud_evidence.py; python scripts/build_fraud_evidence_dataset.py
Not-tested: Large-scale positive scarcity behaviour on the full benchmark
EOF
```

---

### Task 3: Train and evaluate the fraud-evidence lane

**Files:**
- Modify: `src/fraud_evidence.py`
- Create: `scripts/train_fraud_evidence_model.py`
- Test: `tests/test_fraud_evidence.py`

- [ ] **Step 1: Write a failing training test**

```python
# tests/test_fraud_evidence.py
from src.fraud_evidence import train_fraud_evidence_model


def test_train_fraud_evidence_model_returns_binary_metrics():
    dataset = pd.DataFrame(
        {
            "f_tender_value_log": [1.0, 2.0, 3.0, 4.0],
            "g_buyer_supplier_prev_contract_count": [0, 0, 1, 2],
            "fraud_evidence_target": [0, 0, 1, 1],
            "sample_weight": [0.2, 0.2, 1.0, 1.0],
        }
    )
    model, metrics = train_fraud_evidence_model(dataset)
    assert metrics["label_type"] == "fraud_evidence_positive_unlabeled"
    assert "precision_at_10pct" in metrics
    assert "average_precision" in metrics
```

- [ ] **Step 2: Run the focused training test**

Run: `source .venv/bin/activate && pytest -q tests/test_fraud_evidence.py::test_train_fraud_evidence_model_returns_binary_metrics`

Expected: FAIL because `train_fraud_evidence_model` does not exist yet.

- [ ] **Step 3: Implement training and metrics**

```python
# src/fraud_evidence.py
from __future__ import annotations

import json

import numpy as np
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score


def fraud_feature_columns(df: pd.DataFrame) -> list[str]:
    return [
        col
        for col in df.columns
        if col.startswith("f_") or col.startswith("g_")
    ]


def train_fraud_evidence_model(dataset: pd.DataFrame) -> tuple[xgb.XGBClassifier, dict[str, object]]:
    feature_cols = fraud_feature_columns(dataset)
    X = dataset[feature_cols].fillna(0)
    y = dataset["fraud_evidence_target"].astype(int)
    weights = dataset["sample_weight"].astype(float)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X, y, sample_weight=weights)
    probs = model.predict_proba(X)[:, 1]
    top_k = max(1, int(len(probs) * 0.1))
    top_idx = np.argsort(probs)[::-1][:top_k]
    metrics = {
        "label_type": "fraud_evidence_positive_unlabeled",
        "n_samples": int(len(dataset)),
        "n_positive": int(y.sum()),
        "average_precision": round(float(average_precision_score(y, probs)), 4),
        "roc_auc": round(float(roc_auc_score(y, probs)), 4),
        "precision_at_10pct": round(float(y.iloc[top_idx].mean()), 4),
        "feature_count": len(feature_cols),
    }
    return model, metrics
```

```python
# scripts/train_fraud_evidence_model.py
from __future__ import annotations

import json
import sys
from pathlib import Path

import xgboost as xgb
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.fraud_evidence import train_fraud_evidence_model

DATASET_PATH = ROOT / "models" / "fraud_evidence_dataset.parquet"
MODEL_PATH = ROOT / "models" / "fraud_evidence_model.ubj"
METRICS_PATH = ROOT / "models" / "fraud_evidence_metrics.json"
CALIBRATION_PATH = ROOT / "models" / "fraud_evidence_calibration.json"


def main() -> None:
    dataset = pd.read_parquet(DATASET_PATH)
    model, metrics = train_fraud_evidence_model(dataset)
    model.save_model(MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CALIBRATION_PATH.write_text(
        json.dumps({"enabled": False, "note": "calibration pending dedicated holdout"}, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote {MODEL_PATH}")
    print(f"Wrote {METRICS_PATH}")
```

- [ ] **Step 4: Run tests and training**

Run:

```bash
source .venv/bin/activate && pytest -q tests/test_fraud_evidence.py
source .venv/bin/activate && python scripts/train_fraud_evidence_model.py
```

Expected:
- tests pass
- fraud-evidence model and metrics artifacts are written

- [ ] **Step 5: Commit**

```bash
git add src/fraud_evidence.py scripts/train_fraud_evidence_model.py tests/test_fraud_evidence.py
git commit -F - <<'EOF'
Train a separate fraud-evidence model with PU-style labels

This adds a binary model lane that scores stronger evidence of corruption or
fraud without pretending unlabeled procurement rows are verified negatives.

Constraint: Must use existing xgboost/scikit-learn stack only
Rejected: Add a dedicated PU-learning library | violates the no-new-dependency rule for this repo
Confidence: medium
Scope-risk: moderate
Directive: Keep fraud-evidence metrics in separate artifacts from models/metrics.json
Tested: pytest -q tests/test_fraud_evidence.py; python scripts/train_fraud_evidence_model.py
Not-tested: Out-of-sample calibration on a future confirmed-outcome holdout
EOF
```

---

### Task 4: Integrate the fraud-evidence lane into diagnostics and reporting

**Files:**
- Modify: `src/diagnostics.py`
- Modify: `scripts/run_diagnostics.py`
- Modify: `tests/test_diagnostics.py`
- Modify: `tests/test_smoke.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing diagnostics tests for the new lane**

```python
# tests/test_diagnostics.py
from src.diagnostics import summarize_fraud_evidence_lane


def test_summarize_fraud_evidence_lane_reports_binary_metrics():
    metrics = {
        "label_type": "fraud_evidence_positive_unlabeled",
        "n_samples": 100,
        "n_positive": 7,
        "average_precision": 0.41,
        "precision_at_10pct": 0.6,
    }
    summary = summarize_fraud_evidence_lane(metrics)
    assert summary["status"] == "available"
    assert summary["label_family"] == "fraud_evidence"
    assert summary["precision_at_10pct"] == 0.6
```

- [ ] **Step 2: Run the diagnostics test to verify it fails**

Run: `source .venv/bin/activate && pytest -q tests/test_diagnostics.py::test_summarize_fraud_evidence_lane_reports_binary_metrics`

Expected: FAIL because `summarize_fraud_evidence_lane` does not exist yet.

- [ ] **Step 3: Implement lane integration**

```python
# src/diagnostics.py
def summarize_fraud_evidence_lane(metrics: dict[str, object]) -> dict[str, object]:
    if not metrics:
        return {
            "status": "missing_fraud_evidence_metrics",
            "label_family": "fraud_evidence",
            "message": "fraud_evidence_metrics.json not found.",
        }
    return {
        "status": "available",
        "label_family": "fraud_evidence",
        "label_type": metrics.get("label_type"),
        "n_samples": metrics.get("n_samples"),
        "n_positive": metrics.get("n_positive"),
        "average_precision": metrics.get("average_precision"),
        "roc_auc": metrics.get("roc_auc"),
        "precision_at_10pct": metrics.get("precision_at_10pct"),
    }
```

```python
# scripts/run_diagnostics.py
fraud_evidence_metrics_path = MODELS_DIR / "fraud_evidence_metrics.json"
fraud_evidence_metrics = (
    json.loads(fraud_evidence_metrics_path.read_text())
    if fraud_evidence_metrics_path.exists()
    else {}
)
fraud_evidence_lane = summarize_fraud_evidence_lane(fraud_evidence_metrics)
evaluation_lanes["fraud_evidence_lane"] = fraud_evidence_lane
```

Also update `README.md` with:

```markdown
- `models/fraud_evidence_metrics.json` — binary fraud-evidence lane metrics using positive-unlabeled style labels
- `models/fraud_evidence_model.ubj` — separate fraud-evidence model artifact
```

- [ ] **Step 4: Run full diagnostics verification**

Run:

```bash
source .venv/bin/activate && pytest -q tests/test_graph_features.py tests/test_fraud_evidence.py tests/test_diagnostics.py tests/test_smoke.py
source .venv/bin/activate && python scripts/build_fraud_evidence_dataset.py
source .venv/bin/activate && python scripts/train_fraud_evidence_model.py
source .venv/bin/activate && python scripts/run_diagnostics.py
source .venv/bin/activate && python -m compileall src tests scripts
git diff --check
```

Expected:
- all tests pass
- diagnostics include `fraud_evidence_lane`
- no diff-format errors

- [ ] **Step 5: Commit**

```bash
git add src/diagnostics.py scripts/run_diagnostics.py tests/test_diagnostics.py tests/test_smoke.py README.md
git commit -F - <<'EOF'
Report the fraud-evidence lane separately from heuristic risk metrics

This integrates the new binary fraud-evidence metrics into diagnostics while
keeping the existing heuristic and reviewed lanes intact.

Constraint: Fraud-evidence scoring must not overwrite models/metrics.json
Rejected: Merge the fraud-evidence lane into reviewed_risk_lane | would erase the distinction between reviewed-risk and confirmed-outcome evidence
Confidence: high
Scope-risk: moderate
Directive: Preserve separate reporting for heuristic, reviewed, and fraud-evidence lanes
Tested: pytest -q tests/test_graph_features.py tests/test_fraud_evidence.py tests/test_diagnostics.py tests/test_smoke.py; python scripts/build_fraud_evidence_dataset.py; python scripts/train_fraud_evidence_model.py; python scripts/run_diagnostics.py; python -m compileall src tests scripts; git diff --check
Not-tested: Notebook and proposal updates for the new fraud-evidence lane
EOF
```
