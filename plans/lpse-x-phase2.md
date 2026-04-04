# LPSE-X Phase 2 Task Plan

> Source of truth for task-by-task execution.
> This file aligns with `lpse-x-phase2-corrected-skeleton.md`.

---

## Mission

Deliver a complete Find IT! 2026 Track C Phase 2 package with:

- proposal Bab 1-4 in Bahasa Indonesia
- `training.ipynb`
- `inference.ipynb`
- offline XGBoost + SHAP + narrative explanation pipeline
- CPU-safe ONNX inference
- reproducible package with exact-pinned dependencies

---

## Official Timeline Anchor

- Public Phase 2 work/submission window ends on **April 11, 2026**.
- All internal scheduling and gates must work backward from that public date, not April 12.

---

## Hard Rules

- Raw `train_data/` and `test_data/` split happens before feature engineering.
- `test_data/` is not used for HPO.
- `test_data/` is not used for calibration.
- `models/metrics.json` is the canonical metrics file.
- `src/split.py` owns temporal split logic.
- `src/explain.py:explain_single(...)` returns `factors`, not `top_factors`.
- `requirements.txt` uses `==` exact pins.
- model-quality metrics are against heuristic risk labels unless explicitly marked otherwise
- Human review is limited to two scheduled checkpoints:
  - clean-label review for the calibration subset
  - final proposal editorial pass

---

## Canonical Artifacts

### Data

- `data/raw/ocds_indonesia.jsonl.gz`
- `data/processed/ocds_flat.parquet`
- `data/processed/quality_report.md`
- `data/processed/split_metadata.json`
- `data/processed/clean_labels_protocol.md`
- `data/processed/clean_labels_100.csv`
- `train_data/raw.parquet`
- `train_data/features.parquet`
- `train_data/labels.parquet`
- `test_data/raw.parquet`
- `test_data/features.parquet`
- `test_data/labels.parquet`

### Models

- `models/xgb_model.ubj`
- `models/xgb_model.onnx`
- `models/best_params.json`
- `models/metrics.json`
- `models/calibration.json`
- `models/imputation_values.json`

### Proposal and notebooks

- `proposal/bab1.md`
- `proposal/bab2.md`
- `proposal/bab3.md`
- `proposal/bab4.md`
- `proposal/proposal-final.md`
- `proposal/proposal-final.pdf`
- `proposal/figures/confusion_matrix.png`
- `proposal/figures/calibration_curve.png`
- `proposal/figures/per_class_f1.png`
- `proposal/figures/shap_summary.png`
- `training.ipynb`
- `inference.ipynb`

### Source modules

- `src/data.py`
- `src/split.py`
- `src/features.py`
- `src/labels.py`
- `src/model.py`
- `src/explain.py`
- `src/narrative.py`

### Evidence root

- `.sisyphus/evidence/`

---

## Gate Map

### Gate 0: Foundation

- scaffold exists
- imports work
- `pytest -m p0` passes

### Gate 1: Data freeze and leakage compliance

- flattened canonical dataset exists
- raw external split complete
- split-aware features exist
- leakage guard passes

### Gate 2: Model baseline locked

- HPO completed on internal validation only
- model saved as `.ubj`
- minimum viable quality achieved

### Gate 3: XAI complete

- SHAP works
- ONNX parity works
- narrative works
- counterfactual path works

### Gate 4: Notebook complete

- both notebooks assemble cleanly
- both notebooks execute with `nbconvert`

### Gate 5: Submission ready

- proposal PDF exists
- clean-env reproducibility passes
- final package checklist passes

---

## Risk Controls and Kill Switches

- If bid-level coverage is too sparse for stable bid-derived features, keep those features nullable, document the coverage in `quality_report.md`, and continue without reopening the feature catalog.
- If the custom focal-loss path is unstable or too slow, fall back to class-weighted XGBoost and record that choice in the training notes. Do not reopen earlier tasks.
- If fewer than 80 high-confidence calibration rows are reviewed by the calibration timebox, skip temperature scaling and ship uncalibrated probabilities with the limitation documented in Bab 2 and Bab 4.
- If DiCE exceeds the timebox or breaks notebook latency, disable DiCE and ship the SHAP-based counterfactual path only.
- If proposal text and implementation disagree, implementation artifacts win. Proposal updates must read `models/metrics.json`, the figure files, and the canonical module names.

---

## Task Flow

## Task 1: Project scaffold and exact pins

**Goal**

Create the repo structure and exact-pinned dependency baseline.

**Outputs**

- `.gitignore`
- `src/`
- `tests/`
- `proposal/`
- `requirements.txt`
- `README.md`
- pytest config

**Depends on**

- none

**Acceptance**

- `src` imports cleanly
- `requirements.txt` uses exact `==` pins only
- raw data paths and model artifacts are gitignored

---

## Task 2: Smoke test and pytest markers

**Goal**

Lock the basic project contract before deeper work.

**Outputs**

- `tests/conftest.py`
- `tests/test_smoke.py`
- P0, P1, P2 markers

**Depends on**

- Task 1

**Acceptance**

- `pytest -m p0` passes
- Gate 0 is satisfied

---

## Task 3: Proposal lane kickoff

**Goal**

Open the scoring-critical writing lane immediately.

**Outputs**

- `proposal/bab1.md` initial draft
- `proposal/bab4.md` architecture skeleton with placeholders

**Depends on**

- Task 1

**Acceptance**

- Bab 1 sets framing and scope
- Bab 4 establishes architecture, XAI strategy, and placeholders for real results

---

## Task 4: OCDS download, flatten, and quality report

**Goal**

Create one canonical flattened dataset and measure data viability early.

**Outputs**

- `data/raw/ocds_indonesia.jsonl.gz`
- `data/processed/ocds_flat.parquet`
- `data/processed/quality_report.md`

**Depends on**

- Task 1

**Acceptance**

- `ocds_flat.parquet` is the only canonical flattened source
- quality report documents:
  - row count
  - date range
  - critical fields
  - bid-level coverage
  - major NaN risks
- quality report includes a field-coverage table for every field required by Tasks 7-9 and the weak-label rules
- quality report records a hard decision on `bid-derived features: ON/OFF` with justification
- impossible date values are filtered or quarantined before downstream use
- if bid-level coverage is low, the report explicitly marks bid-derived features as non-blocking but nullable

---

## Task 5: External raw split for C-C4 compliance

**Goal**

Split raw data into `train_data/` and `test_data/` before feature engineering.

**Outputs**

- `train_data/raw.parquet`
- `test_data/raw.parquet`
- `data/processed/split_metadata.json`

**Depends on**

- Task 4

**Acceptance**

- raw train/test files exist
- max train date is earlier than min test date
- split metadata records split date and counts

---

## Task 6: Internal development sub-splits inside train_data

**Goal**

Protect the final test set by introducing train-only development windows.

**Outputs**

- train-only sub-split logic in `src/split.py`
- `data/processed/dev_split_manifest.json`

**Depends on**

- Task 5

**Acceptance**

- `dev_split_manifest.json` records concrete boundaries for:
  - `train_fit`
  - `val_hpo`
  - `val_calibration`
- no downstream task uses `test_data/` for tuning

---

## Task 7: Heuristic labeling on split-aware data

**Goal**

Implement ICW-style weak labels without leakage.

**Outputs**

- labeling functions in `src/labels.py`
- `tests/test_labeling.py`

**Depends on**

- Task 5
- Task 6

**Acceptance**

- labels can be produced for train and test raw splits
- expanding-window rules use past-only history
- label distribution is documented honestly as heuristic
- the labeling notes explicitly state that these are heuristic risk labels, not confirmed fraud outcomes
- any circularity risk between red-flag-style features and red-flag-style labels is documented for Bab 2/Bab 3

---

## Task 8: Tier 1 split-aware features

**Goal**

Build the first 15 feature families directly from split raw data.

**Outputs**

- Tier 1 logic in `src/features.py`
- feature tests

**Depends on**

- Task 5

**Acceptance**

- features are computed from split raw inputs
- all columns exported are numeric-safe for downstream ONNX path

---

## Task 9: Tier 2 split-aware features

**Goal**

Build temporal and aggregated features using past-only windows.

**Outputs**

- Tier 2 logic in `src/features.py`
- combined 30-feature contract

**Depends on**

- Task 8

**Acceptance**

- all 30 feature families exist
- no expanding-window feature looks forward in time
- feature catalog is frozen at the end of this task

---

## Task 10: Materialize feature and label artifacts

**Goal**

Write the frozen train/test artifacts used by the modeling lane.

**Outputs**

- `train_data/features.parquet`
- `train_data/labels.parquet`
- `test_data/features.parquet`
- `test_data/labels.parquet`

**Depends on**

- Task 7
- Task 8
- Task 9

**Acceptance**

- train and test artifacts exist
- feature count in metadata matches the frozen contract

---

## Task 11: Leakage guard and Gate 1 verification

**Goal**

Prove the split and feature pipeline satisfies C-C4.

**Outputs**

- `tests/test_leakage_guard.py`
- evidence in `.sisyphus/evidence/`

**Depends on**

- Task 5
- Task 10

**Acceptance**

- raw split has no overlap
- split-aware features show no look-ahead leakage
- `pytest -m p0` still passes
- Gate 1 is satisfied

---

## Task 12: Bab 2 draft with real data profile

**Goal**

Write methodology only after the data profile is known.

**Outputs**

- `proposal/bab2.md`

**Depends on**

- Task 4
- Task 11

**Acceptance**

- Bab 2 cites actual row counts and quality constraints
- Bab 2 describes weak labeling honestly
- Bab 2 does not claim final results yet

---

## Task 13: XGBoost training and HPO

**Goal**

Train the baseline model using only internal train-only development splits.

**Outputs**

- `models/xgb_model.ubj`
- `models/best_params.json`
- training code in `src/model.py`
- `tests/test_model_training.py`

**Depends on**

- Task 6
- Task 10

**Acceptance**

- HPO reads `train_fit` and `val_hpo`, not `test_data`
- model is saved as `.ubj`
- minimum viable quality threshold is documented and met on validation
- if focal loss is not stable within the training timebox, class-weighted XGBoost becomes the locked fallback

---

## Task 14: Clean-label calibration protocol and sample sheet

**Goal**

Prepare the manual calibration subset without contaminating the final test set.

**Outputs**

- `data/processed/clean_labels_protocol.md`
- sample selection helpers in `src/labels.py`
- verification sheet artifact

**Depends on**

- Task 6
- Task 13

**Acceptance**

- calibration samples come from `val_calibration`
- protocol is explicit about heuristic review
- no sample is drawn from `test_data`
- the protocol defines the minimum reviewed-row threshold needed to enable temperature scaling

---

## Task 15: Manual clean-label review and calibration file

**Goal**

Record the limited human review needed for calibration.

**Outputs**

- `data/processed/clean_labels_100.csv`

**Depends on**

- Task 14

**Acceptance**

- reviewed rows are recorded with `verified_label`
- uncertain rows can be excluded
- this task is explicitly manual and limited in scope
- if fewer than 80 high-confidence rows are reviewed, Task 16 must skip temperature scaling

---

## Task 16: Final evaluation and calibration

**Goal**

Lock final metrics and calibration after the model is chosen.

**Outputs**

- `models/metrics.json`
- `models/calibration.json`
- evaluation figures
- evaluation helpers in `src/model.py`

**Depends on**

- Task 13
- Task 15

**Acceptance**

- temperature scaling uses reviewed calibration labels only
- final reported metrics come from held-out `test_data`
- `models/metrics.json` is the file all later tasks read
- if calibration is skipped, `models/calibration.json` records `"enabled": false`
- `models/metrics.json` explicitly distinguishes heuristic-label evaluation from any clean-label calibration evidence

---

## Task 17: Bab 3 draft from real implementation evidence

**Goal**

Write the compliance chapter after the relevant evidence exists.

**Outputs**

- `proposal/bab3.md`

**Depends on**

- Task 11
- Task 16

**Acceptance**

- Bab 3 addresses C-C1 through C-C5 and G1 through G5
- claims reference actual modules, tests, or artifacts
- CPU-only and offline constraints are grounded in real implementation
- Bab 3 explicitly states that performance is measured against heuristic risk labels unless a clean-label subset is named

---

## Task 18: SHAP explainability pipeline

**Goal**

Implement the local and global explanation stack on the native XGBoost model.

**Outputs**

- SHAP helpers in `src/explain.py`
- `proposal/figures/shap_summary.png`
- `tests/test_explanation.py`

**Depends on**

- Task 13

**Acceptance**

- SHAP uses `models/xgb_model.ubj`
- multi-class handling indexes by predicted class
- `explain_single(...)` returns `predicted_class`, `probability`, and `factors`

---

## Task 19: ONNX export, imputation, and parity

**Goal**

Build the fast offline inference path without breaking parity.

**Outputs**

- `models/xgb_model.onnx`
- `models/imputation_values.json`
- parity logic in `src/model.py`
- `tests/test_onnx_parity.py`

**Depends on**

- Task 13
- Task 10

**Acceptance**

- ONNX conversion works with the frozen numeric feature set
- imputation values are fit from training data only
- parity threshold is met

---

## Task 20: Narrative generator

**Goal**

Turn explanation output into competition-grade Bahasa Indonesia narratives.

**Outputs**

- `src/narrative.py`

**Depends on**

- Task 18
- Task 10

**Acceptance**

- output is Bahasa Indonesia only
- includes disclaimer language
- reads from `factors` contract, not any legacy key

---

## Task 21: Counterfactual stack

**Goal**

Provide a reliable “what should change” path.

**Outputs**

- timeboxed DiCE attempt
- mandatory SHAP-based fallback
- unified counterfactual API in `src/explain.py`

**Depends on**

- Task 18
- Task 20
- Task 10

**Acceptance**

- DiCE is timeboxed and never blocks shipping
- SHAP-based fallback always works
- inference path still works when DiCE times out

---

## Task 22: training.ipynb assembly

**Goal**

Assemble the training notebook around the module contracts.

**Outputs**

- `training.ipynb`

**Depends on**

- Task 16
- Task 18
- Task 19

**Acceptance**

- notebook imports split logic from `src.split`
- notebook reads `models/metrics.json`
- notebook does not rerun expensive HPO unnecessarily

---

## Task 23: inference.ipynb assembly

**Goal**

Assemble the offline inference and XAI showcase.

**Outputs**

- `inference.ipynb`

**Depends on**

- Task 18
- Task 19
- Task 20
- Task 21

**Acceptance**

- notebook runs offline
- notebook loads ONNX for inference and `.ubj` for SHAP
- notebook uses `factors` in the explanation contract

---

## Task 24: Proposal results integration

**Goal**

Replace placeholders with real figures and real metrics.

**Outputs**

- updated `proposal/bab2.md`
- updated `proposal/bab4.md`

**Depends on**

- Task 16
- Task 18
- Task 23

**Acceptance**

- Bab 4 placeholders are gone
- metrics match `models/metrics.json`
- figures reference real files under `proposal/figures/`
- timeline-sensitive text uses the public April 11 / April 30 / May 2 / May 14-16 dates

---

## Task 25: Notebook execution verification

**Goal**

Verify both notebooks from clean kernels before final package assembly.

**Outputs**

- executed-notebook evidence

**Depends on**

- Task 22
- Task 23

**Acceptance**

- both notebooks execute with `nbconvert`
- no hidden state assumptions
- inference path stays under CPU budget
- Gate 4 is satisfied

---

## Task 26: Proposal final revision and PDF

**Goal**

Produce the final proposal package after all technical evidence is frozen.

**Outputs**

- `proposal/proposal-final.md`
- `proposal/proposal-final.pdf`

**Depends on**

- Task 3
- Task 12
- Task 17
- Task 24

**Acceptance**

- all four Bab are present
- no placeholders remain
- all cross-references are internally consistent
- quoted guidebook/constraint language either has page references or is rewritten as team interpretation

---

## Task 27: Clean-environment reproducibility

**Goal**

Prove the package works from a fresh environment.

**Outputs**

- clean-env evidence

**Depends on**

- Task 19
- Task 25
- Task 26

**Acceptance**

- clean virtualenv installs from exact pins
- imports succeed
- tests run
- `inference.ipynb` executes
- no hardcoded absolute paths remain

---

## Task 28: Submission package and final checks

**Goal**

Assemble the final deliverable set and close the loop.

**Outputs**

- final README
- final checklist evidence

**Depends on**

- Task 27

**Acceptance**

- all canonical artifacts exist
- no secrets or `.env` files are included
- no raw data is tracked in git
- Gate 5 is satisfied

---

## Parallel Work Map

### Lane A: Data and modeling

- Tasks 1, 2, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 25, 27, 28

### Lane B: Proposal

- Tasks 3, 12, 17, 24, 26

### Rule

Proposal tasks can start early, but Bab 2, Bab 3, and final Bab 4 updates must wait for the evidence they cite.

---

## Verification Command Set

Use these as the canonical checks.

```bash
# Foundation
pytest -m p0 -v

# Leakage
pytest tests/test_leakage_guard.py -v

# Model
pytest tests/test_model_training.py -v

# SHAP
pytest tests/test_explanation.py -v

# ONNX parity
pytest tests/test_onnx_parity.py -v

# Notebook execution
jupyter nbconvert --to notebook --execute training.ipynb --ExecutePreprocessor.timeout=600
jupyter nbconvert --to notebook --execute inference.ipynb --ExecutePreprocessor.timeout=120

# Clean-env
python3 -m venv /tmp/lpse-x-clean
source /tmp/lpse-x-clean/bin/activate
pip install -r requirements.txt
pytest tests/ -v --tb=short
```

---

## Explicitly Rejected Old Patterns

These belong to the superseded draft and must not return:

- feature engineering before raw split
- HPO on `test_data`
- calibration on `test_data`
- `models/evaluation_report.json` as the proposal source of truth
- notebook imports of `temporal_train_test_split` from `src.model`
- verification scripts expecting `top_factors`
- mixed `>=` and `==` dependency policy
- “zero human intervention” language for the calibration-review step

---

## Relationship to the Skeleton

- [lpse-x-phase2-corrected-skeleton.md](/home/vascosera/LSPE-X/.sisyphus/plans/lpse-x-phase2-corrected-skeleton.md): executive-level corrected structure
