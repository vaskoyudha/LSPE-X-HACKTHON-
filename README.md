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

The tracked benchmark is now a **real recent-year OCDS slice**.

Evidence:
- `data/processed/source_manifest.json` records the official source publication and selected year slice
- `data/processed/data_provenance.json` reports `data_kind = "real_or_mixed_ocds"`
- the current real benchmark contains **133,774** usable rows after date cleaning, with **583 buyers** and **28,477 suppliers**

Important limitation:
- the benchmark currently uses **2023 only**, so it is far more credible than the previous synthetic benchmark but still narrower than a full historical LPSE migration

## Synthetic vs Real Benchmark Comparison

Tracked comparison artifact:
- `models/benchmark_comparison.json`
- `proposal/figures/benchmark_comparison.png`

Current comparison:
- synthetic benchmark Macro-F1: **0.9950**
- real 2023 benchmark Macro-F1: **0.8309**
- delta: **-0.1641**

Interpretation: the previous synthetic benchmark overstated performance. The real-data run is a better Phase 2 signal because it keeps the full offline/XAI pipeline while forcing the model to operate on noisier, incomplete procurement fields.

## Robustness Snapshot

The repository includes a circularity audit at `models/robustness.json` and `proposal/figures/robustness_ablation.png`.

Current real-benchmark findings:
- full model (30 features): Macro-F1 **0.8299**
- core-proxy removed (21 features): Macro-F1 **0.3466**
- broad-proxy removed (18 features): Macro-F1 **0.3371**

Interpretation: the model still relies heavily on features close to the heuristic labeling rules, but that weakness is now measured on a real data slice instead of only on synthetic data.

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
- The current package is strongest as an **explainable risk-screening prototype on a real recent-year OCDS slice**, not as a fully validated production fraud detector.
