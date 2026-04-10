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

## Current Data Provenance

The tracked benchmark is now a **real multi-year OCDS slice (2021–2023)**.

Evidence:
- `data/processed/source_manifest.json` records the official source publication and selected year slice
- `data/processed/data_provenance.json` reports `data_kind = "real_or_mixed_ocds"`
- the current benchmark contains **465,184** usable rows after date cleaning, with **618 buyers** and **60,976 suppliers**

Important limitation:
- this is a strong Phase 2 benchmark upgrade, but it still uses the currently selected publication slice and heuristic labels rather than confirmed fraud outcomes

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
