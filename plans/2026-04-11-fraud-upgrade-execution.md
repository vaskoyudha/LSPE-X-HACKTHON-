# Fraud Upgrade Execution Plan

> For Hermes: implement this plan directly in the LSPE-X-HACKTHON- repo with tests and frequent verification.

Goal: strengthen the current procurement-risk system toward a more fraud-oriented architecture by improving structure preservation, graph-ready features, evidence/label scaffolding, and explainability robustness.

Architecture: keep the current flat risk pipeline working while adding non-breaking foundations for richer fraud detection. Do not replace the current model in one jump. Add compatible data artifacts, fraud-oriented features, and runtime hardening so the repository becomes ready for a stronger reviewed/evidence-backed phase.

Tech Stack: Python, pandas, parquet, XGBoost, pytest.

---

## Task 1: Preserve richer supplier structure
- Add normalized relational extraction for award suppliers and supplier participation metadata.
- Save extra processed artifacts without breaking the current flat parquet pipeline.
- Add tests for multiple-supplier extraction.

## Task 2: Add graph/concentration-ready fraud features
- Add historical buyer/supplier dependency and diversity features.
- Keep the features past-only and history-context aware.
- Add tests covering concentration behavior.

## Task 3: Add evidence/review normalization scaffolding
- Add a normalized evidence/label schema module plus import helpers.
- Keep it source-agnostic and suitable for official-source ingestion later.
- Add tests for schema normalization and validation.

## Task 4: Harden the XAI runtime
- Add a fallback explainer path when SHAP native extensions are unavailable.
- Make explanation tests pass using XGBoost contribution fallback.
- Preserve the current explain_single contract.

## Task 5: Verify and retrain where needed
- Run targeted pytest suites.
- If feature schema changes, refresh model artifacts and evaluate.
- Report completed work, remaining constraints, and next blockers.
