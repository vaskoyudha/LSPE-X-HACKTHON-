# LPSE-X Hackathon

LPSE-X is an offline, explainable procurement-risk prototype for Find IT! 2026 Track C.

## Current Scope

- real OCDS benchmark slice built from the official Indonesia publication on data.open-contracting.org
- temporal train/test split with split-aware feature engineering
- heuristic risk labeling for Low / Medium / High procurement risk
- XGBoost + SHAP + Bahasa Indonesia narrative explanation
- CPU-safe local inference artifacts (`.ubj` + `.onnx`)
- executable `training.ipynb` and `inference.ipynb`
- proposal bundle (Bab 1-4 + final markdown)
- provenance, robustness, and synthetic-vs-real comparison diagnostics
- larger reviewed-calibration artifacts, operational review metrics, and external-validation artifacts

## Current Data Provenance

The tracked benchmark is now a **real multi-year OCDS slice (2021–2023)**.

Evidence:
- `data/processed/source_manifest.json` records the official source publication and selected year slice
- `data/processed/data_provenance.json` reports `data_kind = "real_or_mixed_ocds"`
- the current benchmark contains **465,184** usable rows after date cleaning, with **618 buyers** and **60,976 suppliers**

Important limitation:
- this is a strong Phase 2 benchmark upgrade, but it still uses the currently selected publication slice and heuristic labels rather than confirmed fraud outcomes

## Review and Validation Upgrades

New validation-layer artifacts now exist:
- `data/processed/calibration_sheet_300.csv`
- `data/processed/clean_labels_300.csv`
- `data/processed/review_benchmark_500.csv`
- `models/operational_metrics.json`
- `models/external_validation.json`
- `models/reviewed_subset_metrics.json`
- `models/explanation_validation.json`

Current status:
- reviewed calibration rows used: **287**
- manual review summary imported for **500 reviewed rows**
- reviewed-subset metrics and explanation-validation metrics are now available
- row-level reviewed-label import path is now available via `scripts/import_reviewed_row_level.py`
- row-level reviewer annotations themselves are still not stored as a full reviewed sheet in the repo

## Synthetic vs Real Benchmark Comparison

Tracked comparison artifact:
- `models/benchmark_comparison.json`
- `proposal/figures/benchmark_comparison.png`

Current comparison:
- synthetic benchmark Macro-F1: **0.9950**
- real 2021–2023 benchmark Macro-F1: **0.9833**
- delta: **-0.0117**

Interpretation: the previous synthetic benchmark still overstates performance, but the hardened 2021–2023 real-data run now transfers much better after replacing dead features, redesigning heuristic labels around real-supported signals, and re-running real calibration.

## Robustness Snapshot

The repository includes a circularity audit at `models/robustness.json` and `proposal/figures/robustness_ablation.png`.

Current real-benchmark findings:
- full model (30 features): Macro-F1 **0.9833**
- core-proxy removed (19 features): Macro-F1 **0.5215**
- broad-proxy removed (13 features): Macro-F1 **0.5204**

Interpretation: the model still relies heavily on features close to the heuristic labeling rules, but the feature-health audit now shows **0 active dead features** in the tracked real benchmark, so the current weakness is circularity rather than stale feature slots.

## Operational Review Metrics

Tracked artifacts:
- `models/operational_metrics.json`
- `proposal/figures/operational_metrics.png`

Current held-out benchmark highlights:
- Precision@50 = **1.00**
- Precision@100 = **1.00**
- Precision@250 = **1.00**
- Precision@500 = **1.00**
- Precision@1000 = **1.00**

Interpretation: under the current benchmark target, the top-ranked High Risk queue is extremely concentrated, which is promising for limited-budget review workflows.

## Manual Review Summary

Tracked artifacts:
- `data/processed/manual_review_summary.csv`
- `models/reviewed_subset_metrics.json`
- `models/explanation_validation.json`
- `models/manual_review_summary.json`
- `proposal/figures/manual_review_summary.png`

Imported 500-row manual review summary highlights:
- overall agreement vs model prediction: **95.8%**
- reviewed-subset Macro-F1: **0.9679**
- reviewed High Risk F1: **0.9603**
- explanation agreement: **95.8%**
- explanation clarity mean: **3.48 / 5**
- explanation actionability mean: **4.03 / 5**

Interpretation: the model aligns strongly with the manual review summary overall, but the remaining errors concentrate at the **Medium ↔ High** boundary, especially in high-uncertainty rows.

## Row-Level Reviewed Label Import Path

If a full reviewed sheet is available later, import it with:

```bash
source .venv/bin/activate
python scripts/import_reviewed_row_level.py /path/to/reviewed_rows.csv
python scripts/run_diagnostics.py
```

This will let the repo prefer row-level reviewed evidence over summary-only imports.

## Canonical Evidence-Label Artifacts

The repo now supports separate stronger evidence-label artifacts beside the baseline heuristic benchmark:

- `data/processed/reviewed_row_labels.parquet` — canonical row-level reviewed labels with provenance
- `data/processed/fraud_outcomes.parquet` — canonical confirmed-outcome labels with provenance
- `models/evidence_label_coverage.json` — current evidence coverage summary across reviewed and confirmed-outcome artifacts
- `models/evaluation_lanes.json` — lane-by-lane summary separating heuristic-risk, reviewed-risk, and confirmed-outcome reporting
- `models/fraud_evidence_metrics.json` — binary fraud-evidence lane metrics using positive-unlabeled style labels
- `models/fraud_evidence_model.ubj` — separate fraud-evidence model artifact

These artifacts do not replace `train_data/labels.parquet` or `test_data/labels.parquet`. They are used to measure how much stronger, row-level evidence exists for future fraud-evidence evaluation.

Separate fraud-evidence lane artifacts:
- `models/fraud_evidence_metrics.json` — binary fraud-evidence lane metrics using positive-unlabeled style labels
- `models/fraud_evidence_model.ubj` — separate fraud-evidence model artifact

If `models/fraud_evidence_metrics.json` is still missing, diagnostics now report the fraud-evidence lane as **pending** instead of crashing, while keeping it separate from the heuristic-risk and reviewed-risk lanes.

## External Validation Snapshot

Tracked artifacts:
- `models/external_validation.json`
- `proposal/figures/external_validation.png`

Current year-holdout summary across **2019–2023**:
- mean Macro-F1: **0.9151**
- min Macro-F1: **0.6956** (2019)
- max Macro-F1: **0.9934** (2023)
- mean High Risk F1: **0.8972**

Interpretation: generalization is strongest on recent years and weakest on the earliest low-history fold, which is useful evidence about temporal robustness.

## Proxy-Reduced Validation Track

Tracked artifacts:
- `models/proxy_reduced_validation.json`
- `proposal/figures/proxy_reduced_validation.png`

Current stricter track:
- selected track: **proxy_core_removed**
- Macro-F1: **0.5215**
- delta vs full model: **-0.4618**

Interpretation: when features nearest to the labeling rules are removed, performance drops sharply. This is the clearest remaining scientific warning sign in the current system.

## Project Structure

```text
src/            core Python modules
tests/          pytest suite
proposal/       proposal drafts and figures
data/           raw and processed data
models/         trained model artifacts and comparison reports
train_data/     raw/features/labels train split
test_data/      raw/features/labels test split
drafts/         research and planning notes
plans/          execution plans
.sisyphus/      lightweight evidence artifacts
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
python -m nbconvert --to notebook --execute training.ipynb --output /tmp/training-executed.ipynb
python -m nbconvert --to notebook --execute inference.ipynb --output /tmp/inference-executed.ipynb
python scripts/run_diagnostics.py
```

## Notes

- Raw download files and model binaries are intentionally ignored by git.
- `models/metrics.json` is the canonical tracked metrics artifact for the current benchmark.
- The current package is strongest as an **explainable risk-screening prototype on a real 2021–2023 OCDS slice**, not as a fully validated production fraud detector.
