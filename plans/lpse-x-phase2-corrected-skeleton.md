# LPSE-X Phase 2 Corrected Skeleton

> Purpose: define the executive policy that the detailed Phase 2 task plan must follow.
> Strategy: use this file for sequencing and contracts, and `lpse-x-phase2.md` for the detailed task list.
> Deadline: April 12, 2026
> Team: 5 people

---

## Goal

Deliver a competition-ready Phase 2 package for Find IT! 2026 Track C:

- strong proposal in Bahasa Indonesia, especially Bab 4
- offline explainable procurement risk analysis
- XGBoost + SHAP + narrative explanation
- reproducible notebooks and CPU-only inference

---

## What This Skeleton Fixes

This corrected skeleton resolves the major problems in the previous draft:

1. Raw split happens before feature engineering to satisfy C-C4.
2. Final `test_data/` is never used for HPO or calibration.
3. Artifact names are normalized across tasks, notebooks, and proposal.
4. Verification contracts use one API per module instead of conflicting names.
5. Human review is limited to two scheduled checkpoints: clean-label calibration review and final proposal editorial review.

---

## Non-Negotiables

- No cloud inference APIs anywhere in the training or inference path.
- No web app, dashboard, or Phase 3 buildout in Phase 2.
- No SMOTE or synthetic oversampling.
- No XGBoost native categoricals in any path that must export to ONNX.
- Bab 4 is the highest-value writing artifact and must start early.
- C-C4 is hard: raw train/test separation must happen before feature engineering.
- Final `test_data/` is for last-pass reporting only.
- `requirements.txt` uses exact pins with `==`.
- The canonical metrics artifact is `models/metrics.json`.
- The canonical evidence root is `.sisyphus/evidence/`.

---

## Canonical Artifacts

These filenames are the only source of truth for downstream tasks.

### Data

- `data/raw/ocds_indonesia.json.gz`
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

### Models and evaluation

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

---

## Data-Split Policy

This is the most important correction in the whole plan.

### External split for competition compliance

1. Download and flatten OCDS data into `data/processed/ocds_flat.parquet`.
2. Sort by tender date.
3. Create physical raw separation:
   - `train_data/raw.parquet`
   - `test_data/raw.parquet`
4. This raw split happens before feature engineering.

### Internal split for model development

Inside `train_data/raw.parquet`, create temporal sub-splits for development only:

- `train_fit`: model fitting
- `val_hpo`: HPO and early stopping
- `val_calibration`: clean-label verification and temperature scaling

Rules:

- `test_data/` is not used for HPO.
- `test_data/` is not used for temperature scaling.
- `test_data/` is not used for threshold tuning.
- Final reported metrics come from `test_data/` only after the model and calibration are locked.

### Feature-generation rule

Feature engineering must be split-aware:

- train features can only use train history
- validation features can only use prior history available before each validation row
- test features can only use history available before each test row

No feature may look forward in time.

---

## Canonical Module Contracts

These contracts exist to stop downstream mismatch.

### `src/split.py`

- owns temporal splitting
- exports `temporal_train_test_split(...)`
- notebooks import split functions from `src.split`, not `src.model`

### `src/model.py`

- owns training, evaluation, ONNX export, parity, calibration
- writes `models/metrics.json`
- extra diagnostic files are allowed, but no downstream task may depend on them

### `src/explain.py`

- owns SHAP and counterfactual logic
- `explain_single(...)` returns:
  - `predicted_class`
  - `probability`
  - `factors`
- use `factors` consistently everywhere, not `top_factors`

### `src/narrative.py`

- converts explanation output into Bahasa Indonesia narrative
- does not call any cloud model

---

## Phase Skeleton

### Phase 0: Setup and score-first framing

Outputs:

- project scaffold
- `.gitignore`
- exact-pinned `requirements.txt`
- pytest markers and smoke test
- proposal shell
- Bab 1 draft start
- Bab 4 skeleton start

Exit criteria:

- repo imports cleanly
- `pytest -m p0` passes
- proposal lane is open from day 1

### Phase 1: Data acquisition and compliance split

Outputs:

- OCDS download
- canonical flattened dataset at `data/processed/ocds_flat.parquet`
- quality report
- raw `train_data/` and `test_data/` separation
- split metadata

Exit criteria:

- flattened dataset exists and is readable
- quality report documents bid-data coverage
- `train_data/raw.parquet` and `test_data/raw.parquet` exist
- no temporal overlap between raw splits

### Phase 2: Labels and split-aware features

Outputs:

- heuristic labeling
- Tier 1 and Tier 2 features
- `train_data/features.parquet`
- `test_data/features.parquet`
- `train_data/labels.parquet`
- `test_data/labels.parquet`
- leakage guard tests
- frozen 30-feature catalog

Exit criteria:

- all feature generation runs from split raw data
- no future leakage in expanding-window features
- all 30 planned feature families are present

### Phase 3: Modeling and calibration

Outputs:

- tuned XGBoost configuration
- saved `.ubj` model
- internal validation metrics
- clean-label protocol and reviewed calibration set
- temperature scaling parameters
- final `models/metrics.json`

Exit criteria:

- HPO uses only internal train sub-splits
- clean-label review samples come from `val_calibration`, not `test_data`
- final test set remains untouched until final evaluation pass

### Phase 4: Explainability and inference stack

Outputs:

- SHAP explainer
- top-factor extraction
- narrative generator
- ONNX export
- ONNX parity report
- timeboxed DiCE attempt
- SHAP-based counterfactual fallback always available

Exit criteria:

- SHAP uses `.ubj`, never ONNX
- ONNX parity meets threshold
- narrative is fully offline and Bahasa Indonesia
- inference path works even if DiCE fails

### Phase 5: Notebook and proposal integration

Outputs:

- `training.ipynb`
- `inference.ipynb`
- Bab 2 grounded with real data profile
- Bab 3 grounded with real implementation evidence
- Bab 4 updated with real metrics and figures

Exit criteria:

- notebooks import from `src/`
- notebooks run in clean kernels
- proposal references the actual artifact names and module names

### Phase 6: Reproducibility, package, and submission

Outputs:

- clean-environment verification
- final proposal PDF
- final README
- submission checklist
- final tag-ready package

Exit criteria:

- clean install works from `requirements.txt`
- notebooks execute via `nbconvert`
- CPU-only inference stays under budget
- no forbidden dependencies or secrets in repo

---

## Parallel Lanes

Use these lanes, but do not violate dependencies.

### Lane A: Data and model

- setup
- data ingestion
- raw split
- labels and features
- model training
- SHAP and ONNX

### Lane B: Proposal

- Bab 1 and Bab 4 start on day 1
- Bab 2 begins only after the quality report exists
- Bab 3 begins only after compliance evidence exists
- final proposal update happens after metrics and figures are frozen

### Lane C: Verification

- smoke tests from day 1
- leakage tests as soon as split-aware features exist
- parity and notebook execution after model/XAI completion
- clean-env pass before package assembly

---

## Gates

### Gate 0: Foundation

- scaffold exists
- pytest markers work
- smoke test passes

### Gate 1: Data freeze and leakage compliance

- canonical flattened dataset exists
- raw split complete
- leakage guard passes
- feature catalog frozen

### Gate 2: Model baseline locked

- HPO complete on internal validation only
- final model saved
- minimum viable validation quality achieved
- test set still untouched for final reporting

### Gate 3: XAI complete

- SHAP working
- narrative working
- counterfactual path working
- ONNX parity verified

### Gate 4: Notebook-ready

- `training.ipynb` executes
- `inference.ipynb` executes
- offline path demonstrated

### Gate 5: Submission-ready

- proposal PDF generated
- reproducibility passes
- final checklist clean

---

## Manual vs Automated Verification

The previous plan contradicted itself here. The corrected rule is:

### Automated

- unit and integration tests
- leakage tests
- ONNX parity
- notebook execution
- clean-environment install
- CPU latency checks
- no-cloud grep checks
- package completeness checks

### Manual, limited, required

- clean-label verification for the calibration subset only
- final proposal editorial review in Bahasa Indonesia

Everything else should stay automated.

---

## Controlled Risk Outcomes

- Low bid-data coverage does not reopen feature design. Bid-derived features stay nullable and the coverage is documented in `quality_report.md`.
- If focal loss is unstable or too slow, the modeling lane falls back to class-weighted XGBoost without reopening upstream tasks.
- If fewer than 80 high-confidence calibration rows are reviewed by the calibration checkpoint, temperature scaling is disabled and the limitation is documented explicitly.
- If DiCE misses the latency budget, DiCE is disabled and SHAP-based counterfactuals become the only shipping path.
- If proposal text and implementation disagree, canonical implementation artifacts win: `models/metrics.json`, `proposal/figures/*`, and the source module contracts.

---

## Minimal Team Allocation

### Person 1

data ingestion, flattening, quality report, raw split

### Person 2

labels, features, leakage guard

### Person 3

model training, evaluation, calibration, ONNX

### Person 4

SHAP, narrative, counterfactuals, inference notebook

### Person 5

proposal lane, figures, README, package checks

Shared checkpoints:

- end of Phase 1
- end of Phase 3
- end of Phase 5
- final submission review

---

## Recommended Execution Order

1. Setup and smoke test
2. Data download and flatten
3. Raw train/test split
4. Quality report and split metadata
5. Labeling and split-aware features
6. Leakage tests and feature freeze
7. Internal train/validation sub-splits
8. Model training and HPO
9. Clean-label review on calibration subset
10. Final evaluation and `models/metrics.json`
11. SHAP, ONNX, narrative, counterfactuals
12. Notebook assembly
13. Proposal update with real evidence
14. Clean-env and `nbconvert`
15. Final package and submission checks

---

## Relationship to the Main Plan

Use [lpse-x-phase2.md](/home/vascosera/LSPE-X/.sisyphus/plans/lpse-x-phase2.md) as the detailed task-by-task source of truth.

Use this file for:

- executive sequencing
- core policy decisions
- artifact contracts
- gate structure

If the two files ever diverge, update the detailed plan to match this skeleton.
