# LPSE-X Hackathon

LPSE-X is an offline, explainable procurement-risk prototype for Find IT! 2026 Track C.

## Current Scope

- temporal train/test split with split-aware feature engineering
- heuristic risk labeling for Low / Medium / High procurement risk
- XGBoost + SHAP + Bahasa Indonesia narrative explanation
- CPU-safe local inference artifacts (`.ubj` + `.onnx`)
- executable `training.ipynb` and `inference.ipynb`
- proposal bundle (Bab 1-4 + final markdown)
- robustness and provenance diagnostics for Phase 2 positioning

## Important Data Provenance Note

The current working dataset is **synthetic**.

Evidence:
- `data/processed/data_provenance.json` marks `dataset_type = "synthetic"`
- OCIDs in the working split use the `ocds-synth-*` prefix
- the current snapshot contains 5,000 rows, 50 buyers, and 200 suppliers

This means the current scores should be interpreted as proof that the pipeline works and that the model can recover the designed heuristic-risk structure. They should **not** be overstated as proof of real-world fraud-detection accuracy on authentic LPSE production data.

## Robustness Snapshot

The repository now includes a circularity audit at `models/robustness.json` and `proposal/figures/robustness_ablation.png`.

Key finding:
- full model (30 features): Macro-F1 **0.9970**
- core-proxy removed (21 features): Macro-F1 **0.3911**
- broad-proxy removed (18 features): Macro-F1 **0.3829**

Interpretation: the current model is highly effective at learning the heuristic risk rules, but much of that strength is concentrated in features that are close to the labeling rubric itself. The audit now quantifies that weakness instead of leaving it implicit.

## Project Structure

```text
src/            core Python modules
tests/          pytest suite
proposal/       proposal drafts and figures
data/           raw and processed data
models/         trained model artifacts and robustness reports
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

- Raw data and model binaries are intentionally ignored by git.
- `models/metrics.json` is the canonical tracked metrics artifact.
- Use the proposal wording carefully: the current package is strongest as an **explainable risk-screening prototype**, not as a proven production fraud detector.
