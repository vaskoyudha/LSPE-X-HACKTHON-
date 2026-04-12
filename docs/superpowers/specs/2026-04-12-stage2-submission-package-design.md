# Find IT! 2026 Tahap 2 Submission Package Design

## Status

Approved direction captured on 2026-04-12 for team **BismillahFirstTry-Phase2**.

## Goal

Produce a **judge-safe Tahap 2 submission package** for Find IT! 2026 Track C from the current LPSE-X project, with two deliverables:

1. a clean GitHub/cloud folder for the technical artifacts, and
2. a professional proposal package with stronger structure, graphics, diagrams, and constraint proof.

## Official Constraints and Submission Rules

### Stage 2 upload update

- Submission is centralized through Google Form.
- Proposal is uploaded as a **PDF**.
- Technical materials are submitted as a **cloud folder or GitHub link**.
- Proposal filename must follow:
  - `Proposal_[Nama Tim]_Tahap2_FindIT2026.pdf`
- Technical folder name must follow:
  - `[Nama Tim]_Tahap2_FindIT2026`
- Multiple-model submissions require extra naming and `training_main.ipynb`.

### General technical constraints

The GitHub/cloud submission must clearly contain:

- `proposal.pdf`
- `training.ipynb` with visible logs
- `inference.ipynb` as a clean inference script/notebook
- model file(s)
- `train_data/`
- `test_data/`

### Track C constraints to prove in proposal Bab 3

1. **Explainability wajib**
2. **Human-readable explanation for each prediction**
3. **Anti-black-box**
4. **No data leakage**, with raw `train_data` and `test_data` separation before preprocessing
5. **Offline total**, no cloud inference or cloud explainability

## Chosen Delivery Strategy

### Submission mode

Use **single-model submission**.

### Why single-model is the correct choice

- The project already has one main predictive system: an XGBoost risk model.
- SHAP, calibration, ONNX export, and the evidence lane are supporting mechanisms, not a clean second model lane.
- Single-model packaging minimizes formatting risk close to the deadline.
- Single-model positioning is simpler and more defensible during judging.

### Packaging posture

Use a **minimal judge-safe repo/folder** rather than shipping the entire research repository.

This means the submission repo should contain only the files judges need to inspect and rerun the Tahap 2 solution, while excluding planning notes, redundant diagnostics, and unrelated development history.

## Submission Repo Design

### Canonical folder/repo name

`BismillahFirstTry-Phase2_Tahap2_FindIT2026`

### Canonical proposal filename

`Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf`

### Target top-level contents

```text
BismillahFirstTry-Phase2_Tahap2_FindIT2026/
├── README.md
├── Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf
├── proposal_preview.md
├── requirements.txt
├── training.ipynb
├── inference.ipynb
├── model_risk.ubj
├── model_risk.onnx
├── train_data/
├── test_data/
└── figures/
```

### Inclusion policy

#### Must include

- final proposal PDF
- training notebook
- inference notebook
- main model file(s)
- `train_data/`
- `test_data/`
- a short README explaining how to run the submission locally

#### Should include

- copied proposal figures used inside the PDF
- one markdown preview of the proposal body for fast reading on GitHub
- exact-pinned `requirements.txt`

#### Should exclude

- internal planning docs
- draft articles
- `.sisyphus/`
- unrelated scripts
- redundant diagnostics not needed by judges
- development-only notes and history

## Current-to-Submission Artifact Mapping

| Current artifact | Submission artifact |
| --- | --- |
| `proposal/proposal-final.pdf` | `Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf` |
| `proposal/proposal-final.md` | `proposal_preview.md` |
| `training.ipynb` | `training.ipynb` |
| `inference.ipynb` | `inference.ipynb` |
| `models/xgb_model.ubj` | `model_risk.ubj` |
| `models/xgb_model.onnx` | `model_risk.onnx` |
| `train_data/` | `train_data/` |
| `test_data/` | `test_data/` |
| selected files from `proposal/figures/` | `figures/` |
| `requirements.txt` | `requirements.txt` |

## Proposal Design

### Positioning statement

The proposal should present LPSE-X as an **offline, explainable procurement-risk screening system** for public procurement monitoring in Indonesia, aligned with **Track C: The Explainable Oracle**.

### Narrative stance

The proposal should be ambitious in impact but conservative in scientific claims:

- strong on architecture,
- strong on explainability,
- strong on compliance,
- honest about heuristic labels and remaining circularity risk.

This is the most defensible stance for judges.

## Proposal Structure Upgrades

### Bab 1

Keep the policy/public-service framing, but sharpen the opening around:

- corruption risk in procurement,
- overload of manual auditing,
- need for explainable triage rather than black-box scoring.

### Bab 2

Emphasize:

- split-aware pipeline,
- feature engineering,
- model choice,
- SHAP and narrative generation,
- offline deployment path.

### Bab 3

Make Bab 3 explicitly judge-facing by using a **constraint-by-constraint compliance matrix**:

- C-C1 explainability
- C-C2 readable explanation
- C-C3 anti-black-box
- C-C4 anti-leakage
- C-C5 offline total

Each section should point to concrete artifacts, notebooks, modules, and figures.

### Bab 4

Make Bab 4 the strongest visual and evaluative section:

- headline metrics
- class-level performance
- confusion matrix
- calibration
- SHAP analysis
- robustness warning
- external validation
- operational review metrics
- official evidence showcase / casebook angle

## Visual and Diagram Design

The proposal should include both **performance visuals** and **system visuals**.

### Reuse existing visuals

- confusion matrix
- calibration curve
- per-class F1
- SHAP summary
- robustness ablation
- operational metrics
- manual review summary
- external validation
- benchmark comparison

### New visuals to create

1. **End-to-end pipeline architecture diagram**
   - source data → raw split → feature engineering → labeling → training → explainability → narrative output → audit triage
2. **Constraint compliance diagram/table**
   - map each Track C rule to the exact implementation artifact
3. **Submission package diagram**
   - show what judges receive in the GitHub/cloud folder
4. **Inference flow diagram**
   - input row → model probabilities → SHAP factors → Bahasa Indonesia explanation → risk lane / evidence escalation
5. **Casebook-style decision flow**
   - Aman → Perlu Pantauan → Risiko Tinggi → Risiko Kritis

### Visual ordering recommendation

1. pipeline architecture diagram
2. submission package / system overview
3. headline metrics charts
4. calibration and confusion matrix
5. SHAP and robustness figures
6. external validation and manual review summary
7. casebook / decision-flow visual

## DOCX Production Path

A DOCX export path is desirable for editing and backup, but **PDF remains the submission-critical artifact**.

### Priority order

1. produce the final PDF correctly,
2. optionally generate `.docx` as an editing/export convenience,
3. never let DOCX formatting work block the PDF submission.

### Working rule

The document source of truth should remain markdown + figures until the PDF is correct.

## Verification Requirements

Before declaring the package complete, verify:

1. proposal filename exactly matches the required format
2. technical folder name exactly matches the required format
3. `training.ipynb` opens with visible outputs/logs
4. `inference.ipynb` is cleaner and submission-facing
5. model files are present and reasonably sized
6. `train_data/` and `test_data/` are included
7. Bab 3 explicitly proves all Track C constraints
8. proposal figures render correctly in the final document
9. final PDF opens successfully
10. README explains local execution clearly and briefly

## Risks to Avoid

1. shipping a noisy repo with too many irrelevant files
2. accidentally implying multi-model submission when the package is really single-model
3. weak Bab 3 wording that does not explicitly prove each Track C rule
4. overclaiming fraud detection instead of explainable risk screening
5. spending too much time on DOCX polish while the PDF/package is not locked

## Execution Handoff

The next implementation phase should:

1. create the clean submission repo/folder structure,
2. prepare renamed submission artifacts,
3. upgrade proposal content and visuals,
4. export the final PDF,
5. optionally create a DOCX copy if the path is reliable,
6. verify filenames, structure, and readiness against the Google Form rules.
