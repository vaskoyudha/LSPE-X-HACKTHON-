# LPSE-X Hackathon

Initial scaffold for the Find IT! 2026 Track C Phase 2 project.

## Current Scope

- Python package scaffold for the Phase 2 pipeline
- Test harness with smoke test
- Pinned dependency manifest
- Placeholder module layout for data, splitting, features, labels, model, explainability, and narrative generation

## Project Structure

```text
src/            core Python modules
tests/          pytest suite
proposal/       proposal drafts and figures
data/           raw and processed data
models/         trained model artifacts
train_data/     raw/features/labels train split
test_data/      raw/features/labels test split
drafts/         working research and planning notes
plans/          execution plans
```

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-bootstrap.txt
pytest -m p0 -v
```

## Full ML Stack

When you are ready to install the complete Phase 2 environment:

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Notes

- Raw data is intentionally ignored by git.
- Model artifacts are intentionally ignored by git.
- This scaffold is designed to grow into the offline XGBoost + SHAP + ONNX Phase 2 pipeline.
