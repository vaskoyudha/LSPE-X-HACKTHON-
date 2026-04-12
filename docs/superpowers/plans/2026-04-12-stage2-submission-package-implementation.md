# Stage 2 Submission Package Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a judge-safe Tahap 2 submission package for Find IT! 2026 Track C, including a clean technical repo/folder, upgraded proposal content, and professional diagrams/graphics.

**Architecture:** Keep the current LPSE-X repo as the source of truth, then generate a clean submission bundle with renamed artifacts and curated contents. Strengthen the proposal inside the main repo first so the final PDF and copied submission assets are derived from one consistent markdown-and-figures source.

**Tech Stack:** Markdown, Python 3.12, Jupyter notebooks, existing LPSE-X artifacts, Graphviz/Mermaid-compatible diagram sources where available, git.

---

## File Structure Lock

### Source-of-truth files to modify

- `proposal/bab1.md`
- `proposal/bab2.md`
- `proposal/bab3.md`
- `proposal/bab4.md`
- `proposal/proposal-final.md`
- `proposal/README.md`
- `README.md`

### New files to create

- `proposal/figures/pipeline-architecture.mmd`
- `proposal/figures/pipeline-architecture.svg`
- `proposal/figures/pipeline-architecture.png`
- `proposal/figures/anti-leakage-flow.mmd`
- `proposal/figures/anti-leakage-flow.svg`
- `proposal/figures/anti-leakage-flow.png`
- `proposal/figures/inference-flow.mmd`
- `proposal/figures/inference-flow.svg`
- `proposal/figures/inference-flow.png`
- `proposal/figures/submission-package-map.mmd`
- `proposal/figures/submission-package-map.svg`
- `proposal/figures/submission-package-map.png`
- `proposal/figures/risk-decision-flow.mmd`
- `proposal/figures/risk-decision-flow.svg`
- `proposal/figures/risk-decision-flow.png`
- `scripts/build_submission_bundle.py`
- `submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/README.md`
- `submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/` copied bundle artifacts

### Tests / verification surfaces

- lightweight file/structure verification via script and shell checks
- proposal rendering/file existence checks
- notebook presence and model artifact presence checks

---

### Task 1: Create the bundle builder contract

**Files:**
- Create: `scripts/build_submission_bundle.py`
- Test: shell verification in `submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/`

- [ ] **Step 1: Write the expected bundle mapping into the builder script**

```python
BUNDLE_NAME = "BismillahFirstTry-Phase2_Tahap2_FindIT2026"
PDF_NAME = "Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf"
MODEL_MAP = {
    Path("models/xgb_model.ubj"): Path("models/model_risk.ubj"),
    Path("models/xgb_model.onnx"): Path("models/model_risk.onnx"),
}
DIRECT_COPY_DIRS = [Path("train_data"), Path("test_data"), Path("src")]
```

- [ ] **Step 2: Run a syntax check before using the script**

Run: `python3 -m py_compile scripts/build_submission_bundle.py`
Expected: command exits successfully with no output

- [ ] **Step 3: Implement bundle creation with clean overwrite semantics**

```python
if bundle_root.exists():
    shutil.rmtree(bundle_root)
bundle_root.mkdir(parents=True, exist_ok=True)
for src, dest in MODEL_MAP.items():
    copy_file(src, bundle_root / dest)
```

- [ ] **Step 4: Add copied proposal/figure/requirements/notebook assets**

```python
copy_file(Path("proposal/proposal-final.pdf"), bundle_root / PDF_NAME)
copy_file(Path("training.ipynb"), bundle_root / "training.ipynb")
copy_file(Path("inference.ipynb"), bundle_root / "inference.ipynb")
copy_file(Path("requirements.txt"), bundle_root / "requirements.txt")
copy_tree(Path("proposal/figures"), bundle_root / "proposal/figures")
```

- [ ] **Step 5: Run the builder and inspect the generated tree**

Run: `python3 scripts/build_submission_bundle.py && find submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026 -maxdepth 3 | sort | sed -n '1,120p'`
Expected: the bundle exists with renamed proposal/model files plus train/test/src/proposal figures

- [ ] **Step 6: Commit**

```bash
git add scripts/build_submission_bundle.py submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026
git commit -m "Make the Tahap 2 submission bundle reproducible"
```

### Task 2: Rewrite the submission-facing README content

**Files:**
- Modify: `README.md`
- Create/Modify: `submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/README.md`

- [ ] **Step 1: Draft a short source README section for submission mode**

```markdown
## Stage 2 Submission Bundle

Build the clean submission package with:

```bash
python3 scripts/build_submission_bundle.py
```
```

- [ ] **Step 2: Add a bundle README with judge-facing run instructions**

```markdown
# BismillahFirstTry-Phase2 Tahap 2 Submission

This folder contains the Track C proposal PDF, training notebook, inference notebook,
main model artifacts, and split datasets required by the competition constraints.
```

- [ ] **Step 3: Verify the README texts are concise and non-research-heavy**

Run: `sed -n '1,220p' README.md && printf '\n---\n' && sed -n '1,220p' submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/README.md`
Expected: both files describe only the necessary execution and submission context

- [ ] **Step 4: Commit**

```bash
git add README.md submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/README.md
git commit -m "Clarify the Stage 2 submission instructions"
```

### Task 3: Create the new system and compliance diagrams

**Files:**
- Create: all `proposal/figures/*.mmd`, `*.svg`, `*.png` diagram files listed above
- Modify: `proposal/README.md` if needed to note generated figures

- [ ] **Step 1: Write the pipeline architecture diagram source**

```text
flowchart LR
    A[OCDS source data] --> B[Raw temporal split]
    B --> C[Split-aware feature engineering]
    C --> D[Heuristic risk labeling]
    D --> E[XGBoost training]
    E --> F[SHAP explainability]
    F --> G[Bahasa Indonesia narrative]
    G --> H[Audit triage]
```

- [ ] **Step 2: Write the anti-leakage and inference-flow diagram sources**

```text
flowchart TD
    A[Raw OCDS rows] --> B[train_data/raw.parquet]
    A --> C[test_data/raw.parquet]
    B --> D[train-only fit/hpo/calibration]
    C --> E[final held-out evaluation only]
```

```text
flowchart LR
    A[Input procurement row] --> B[Feature vector]
    B --> C[Risk probabilities]
    C --> D[Top SHAP factors]
    D --> E[Human-readable narrative]
    E --> F[Risk lane and reviewer action]
```

- [ ] **Step 3: Write the submission-package and risk-decision diagram sources**

```text
flowchart TD
    A[Proposal PDF] --> P[Google Form]
    B[training.ipynb] --> R[GitHub/cloud folder]
    C[inference.ipynb] --> R
    D[model files] --> R
    E[train_data] --> R
    F[test_data] --> R
```

```text
flowchart LR
    A[Aman] --> B[Perlu Pantauan]
    B --> C[Risiko Tinggi]
    C --> D[Risiko Kritis]
```

- [ ] **Step 4: Render the diagrams to SVG/PNG**

Run: `python3 scripts/render_diagrams.py` or the available local renderer command chosen during implementation
Expected: each `.mmd` source has matching `.svg` and `.png` outputs in `proposal/figures/`

- [ ] **Step 5: Verify figure outputs exist**

Run: `find proposal/figures -maxdepth 1 \( -name 'pipeline-architecture.*' -o -name 'anti-leakage-flow.*' -o -name 'inference-flow.*' -o -name 'submission-package-map.*' -o -name 'risk-decision-flow.*' \) | sort`
Expected: source and rendered outputs are all present

- [ ] **Step 6: Commit**

```bash
git add proposal/figures proposal/README.md
git commit -m "Add proposal diagrams for system flow and compliance"
```

### Task 4: Strengthen Bab 1-3 for judge-facing clarity

**Files:**
- Modify: `proposal/bab1.md`
- Modify: `proposal/bab2.md`
- Modify: `proposal/bab3.md`
- Modify: `proposal/proposal-final.md`

- [ ] **Step 1: Rewrite Bab 1 opening to sharpen public-service urgency and track fit**

```markdown
LPSE-X diposisikan sebagai sistem skrining risiko pengadaan yang membantu auditor
memprioritaskan kasus berisiko tinggi tanpa menggantikan keputusan manusia.
```

- [ ] **Step 2: Tighten Bab 2 methodology around offline split-aware architecture**

```markdown
Pipeline dibangun dengan pemisahan train/test pada level raw data sebelum preprocessing,
kemudian seluruh feature engineering dilakukan secara split-aware untuk mencegah leakage.
```

- [ ] **Step 3: Rewrite Bab 3 into an explicit Track C compliance matrix**

```markdown
| Constraint | Implementasi LPSE-X | Bukti |
| --- | --- | --- |
| C-C1 Explainability | SHAP + explain_single | `src/explain.py`, `proposal/figures/shap_summary.png` |
| C-C4 Validasi leakage | raw split sebelum preprocessing | `src/split.py`, `train_data/raw.parquet`, `test_data/raw.parquet` |
```

- [ ] **Step 4: Rebuild `proposal/proposal-final.md` so it consistently embeds the new diagrams**

```markdown
![Arsitektur LPSE-X](figures/pipeline-architecture.png)

![Validasi anti-leakage](figures/anti-leakage-flow.png)
```

- [ ] **Step 5: Review the proposal body for overclaiming and fix wording inline**

Run: `rg -n 'fraud detector|terbukti korupsi|guarantee|100%' proposal/*.md || true`
Expected: no overclaiming language remains in the proposal files

- [ ] **Step 6: Commit**

```bash
git add proposal/bab1.md proposal/bab2.md proposal/bab3.md proposal/proposal-final.md
git commit -m "Make the proposal more judge-facing and constraint-explicit"
```

### Task 5: Upgrade Bab 4 visual storytelling and captions

**Files:**
- Modify: `proposal/bab4.md`
- Modify: `proposal/proposal-final.md`

- [ ] **Step 1: Reorder Bab 4 around the strongest evidence flow**

```markdown
1. Ringkasan hasil utama
2. Analisis per kelas
3. Kalibrasi dan confusion matrix
4. Explainability dan robustness
5. External validation dan manual review
6. Evidence lane dan casebook
```

- [ ] **Step 2: Add short judge-friendly captions before or after each major figure**

```markdown
**Gambar 4.x.** Confusion matrix ini menunjukkan trade-off operasional antara false alarm
 dan kasus risiko tinggi yang terdeteksi pada split uji final.
```

- [ ] **Step 3: Remove redundant or weakly justified visuals from the main narrative**

```markdown
Jika dua visual menjelaskan pesan yang sama, pertahankan visual yang paling langsung
terkait keputusan juri dan pindahkan sisanya ke lampiran atau hilangkan.
```

- [ ] **Step 4: Verify the final markdown references only existing figure files**

Run: `python3 - <<'PY'
from pathlib import Path
import re
text = Path('proposal/proposal-final.md').read_text()
for ref in re.findall(r'\((figures/[^)]+)\)', text):
    path = Path('proposal') / ref
    print(ref, 'OK' if path.exists() else 'MISSING')
PY`
Expected: all referenced figure files print `OK`

- [ ] **Step 5: Commit**

```bash
git add proposal/bab4.md proposal/proposal-final.md
git commit -m "Improve proposal visual storytelling for judges"
```

### Task 6: Export and verify the final submission bundle

**Files:**
- Modify/Rebuild: `proposal/proposal-final.pdf`
- Modify/Rebuild: `submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/*`

- [ ] **Step 1: Generate the final proposal PDF from the updated source-of-truth files**

```bash
# Use the locally working PDF export path chosen during implementation
```

- [ ] **Step 2: Rebuild the submission bundle after the PDF is updated**

Run: `python3 scripts/build_submission_bundle.py`
Expected: the bundle is recreated with the final proposal and latest figures

- [ ] **Step 3: Verify the exact required filenames and key artifacts**

Run: `test -f submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/Proposal_BismillahFirstTry-Phase2_Tahap2_FindIT2026.pdf && test -f submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/training.ipynb && test -f submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/inference.ipynb && test -f submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/models/model_risk.ubj && test -f submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026/models/model_risk.onnx`
Expected: command exits successfully

- [ ] **Step 4: Run final lightweight verification**

Run: `python3 -m py_compile scripts/build_submission_bundle.py && git diff --check && find submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026 -maxdepth 2 | sort | sed -n '1,160p'`
Expected: no diff-check errors and the bundle tree looks submission-ready

- [ ] **Step 5: Commit**

```bash
git add proposal/proposal-final.pdf submission/BismillahFirstTry-Phase2_Tahap2_FindIT2026
git commit -m "Prepare the final Tahap 2 submission package"
```
