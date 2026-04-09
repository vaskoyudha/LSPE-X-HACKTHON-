<p align="center">
  <h1 align="center">🔍 LPSE-X</h1>
  <p align="center">
    <strong>Procurement Risk Intelligence Engine</strong>
  </p>
  <p align="center">
    Offline, CPU-only fraud risk detection on Indonesian public procurement data
    <br />
    using XGBoost · SHAP · Bahasa Indonesia Narrative AI
  </p>
  <p align="center">
    <a href="#-quick-start"><img src="https://img.shields.io/badge/python-3.12-blue?style=for-the-badge&logo=python&logoColor=white" alt="Python 3.12"></a>
    <a href="#-tech-stack"><img src="https://img.shields.io/badge/XGBoost-2.1-orange?style=for-the-badge&logo=xgboost&logoColor=white" alt="XGBoost"></a>
    <a href="#-tech-stack"><img src="https://img.shields.io/badge/SHAP-Explainability-green?style=for-the-badge" alt="SHAP"></a>
    <a href="#-test-suite"><img src="https://img.shields.io/badge/tests-88%20passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white" alt="Tests"></a>
    <a href="#-license"><img src="https://img.shields.io/badge/license-MIT-purple?style=for-the-badge" alt="License"></a>
  </p>
</p>

<br />

> **Find IT! 2026** — Track C Phase 2 Submission
>
> _Detecting anomalous procurement patterns in Indonesian OCDS data through interpretable machine learning, with full Bahasa Indonesia narrative explanations._

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🏗️ Architecture](#️-architecture)
- [🛠️ Tech Stack](#️-tech-stack)
- [🚀 Quick Start](#-quick-start)
- [📂 Project Structure](#-project-structure)
- [📊 Pipeline Overview](#-pipeline-overview)
- [🧪 Test Suite](#-test-suite)
- [📓 Notebooks](#-notebooks)
- [📦 Key Artifacts](#-key-artifacts)
- [⚠️ Important Notes](#️-important-notes)
- [👥 Team](#-team)

---

## ✨ Features

<table>
<tr>
<td width="50%">

### 🎯 Risk Detection
- **30-feature engineering** pipeline
- Tier 1: Direct procurement signals
- Tier 2: Temporal buyer-supplier patterns
- **7 heuristic red flags** scoring
- 3-class risk classification (Low / Medium / High)

</td>
<td width="50%">

### 🧠 Explainability (XAI)
- **SHAP TreeExplainer** per-record analysis
- Factor direction indicators
- **Counterfactual recommendations**
- DiCE (timeboxed) + SHAP fallback
- **Bahasa Indonesia** narrative generation

</td>
</tr>
<tr>
<td width="50%">

### 🔒 Anti-Leakage Guarantees
- Zero OCID overlap between splits
- Strict temporal train/test boundary
- Expanding-window features (past-only)
- 15 dedicated leakage guard tests
- HPO uses internal validation only

</td>
<td width="50%">

### ⚡ Production Ready
- ONNX-compatible model export
- Verified export parity (lossless)
- Median imputation for missing values
- CPU-only, fully offline operation
- Temperature-scaled calibrated probabilities

</td>
</tr>
</table>

---

## 🏗️ Architecture

```mermaid
graph LR
    A[📥 OCDS Data] --> B[🔄 Flatten & Clean]
    B --> C[✂️ Temporal Split]
    C --> D[⚙️ Feature Engineering]
    D --> E[🏷️ Heuristic Labels]
    E --> F[🤖 XGBoost HPO]
    F --> G[📊 Evaluation]
    G --> H[🌡️ Calibration]
    H --> I[🧠 SHAP]
    I --> J[📝 Narrative]
    J --> K[🇮🇩 Bahasa Indonesia Report]

    style A fill:#4A90D9,stroke:#333,color:#fff
    style F fill:#E8913A,stroke:#333,color:#fff
    style I fill:#4CAF50,stroke:#333,color:#fff
    style K fill:#DC3545,stroke:#333,color:#fff
```

---

## 🛠️ Tech Stack

| Category | Technology |
|----------|-----------|
| **Language** | Python 3.12 |
| **ML Framework** | XGBoost 2.1 (gradient boosting) |
| **Hyperparameter Tuning** | Optuna (TPE sampler) |
| **Explainability** | SHAP (TreeExplainer) |
| **Data Processing** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Model Export** | ONNX-compatible JSON |
| **Calibration** | Temperature Scaling (scipy) |
| **Testing** | pytest (88 tests, P0/P1/P2 markers) |
| **Notebook Runtime** | Jupyter + nbconvert |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/your-org/LSPE-X-HACKTHON-.git
cd LSPE-X-HACKTHON-

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\Activate.ps1

# Activate (Linux / macOS)
source .venv/bin/activate

# Install dependencies (exact pins)
pip install -r requirements.txt
```

### Run the Pipeline

```bash
# 1️⃣  Materialize data artifacts
python -m scripts.materialize

# 2️⃣  Train model (HPO + final training)
python -c "from src.model import run_training_pipeline; run_training_pipeline()"

# 3️⃣  Evaluate + calibrate
python -c "from src.model import run_evaluation_pipeline; run_evaluation_pipeline()"

# 4️⃣  Generate SHAP summary
python -c "from src.explain import generate_shap_summary; generate_shap_summary()"

# 5️⃣  Run tests
pytest -v
```

---

## 📂 Project Structure

```
LSPE-X-HACKTHON-/
│
├── 📁 src/                          # Core Python modules
│   ├── __init__.py                  # Package init + RANDOM_SEED
│   ├── data.py                      # OCDS download, flatten, quality report
│   ├── split.py                     # Temporal train/test splitting
│   ├── features.py                  # 30-feature engineering (Tier 1 + 2)
│   ├── labels.py                    # Heuristic risk labeling (7 red flags)
│   ├── model.py                     # XGBoost HPO, training, calibration, eval
│   ├── explain.py                   # SHAP + counterfactual explanations
│   └── narrative.py                 # Bahasa Indonesia narrative generator
│
├── 📁 tests/                        # Test suite (88 tests)
│   ├── conftest.py                  # Shared fixtures
│   ├── test_smoke.py                # Import smoke tests
│   ├── test_data.py                 # Data parser tests
│   ├── test_split.py                # Split logic tests
│   ├── test_features.py             # Feature engineering tests
│   ├── test_labeling.py             # Labeling function tests
│   ├── test_leakage_guard.py        # Anti-leakage verification (15 tests)
│   ├── test_model_training.py       # Model training tests
│   ├── test_explanation.py          # SHAP explanation tests
│   └── test_onnx_parity.py          # ONNX export parity tests
│
├── 📁 scripts/                      # Utility scripts
│   ├── materialize.py               # Data materialization pipeline
│   ├── simulate_review.py           # Clean-label review simulation
│   └── generate_calibration_sheet.py # Calibration sheet generator
│
├── 📁 models/                       # Trained model artifacts
│   ├── xgb_model.ubj                # Native XGBoost model
│   ├── best_params.json             # Optuna HPO best parameters
│   ├── metrics.json                 # 📊 Canonical evaluation metrics
│   ├── calibration.json             # Temperature scaling config
│   └── imputation_values.json       # Median imputation for inference
│
├── 📁 proposal/                     # Competition proposal
│   ├── bab1.md                      # Bab 1: Introduction
│   └── figures/                     # Evaluation figures
│       ├── confusion_matrix.png
│       ├── calibration_curve.png
│       ├── per_class_f1.png
│       └── shap_summary.png
│
├── 📁 data/processed/               # Processed data metadata
│   ├── split_metadata.json
│   ├── dev_split_manifest.json
│   ├── feature_manifest.json
│   ├── clean_labels_protocol.md
│   └── clean_labels_100.csv
│
├── 📓 training.ipynb                # Training pipeline notebook
├── 📓 inference.ipynb               # Inference + XAI notebook
├── 📋 requirements.txt              # Exact-pinned dependencies
├── 📋 pytest.ini                    # Test configuration
└── 📄 README.md                     # You are here
```

---

## 📊 Pipeline Overview

### Feature Engineering (30 Features)

<details>
<summary><b>Tier 1 — Direct Signals (15 features)</b></summary>

| # | Feature | Description |
|---|---------|-------------|
| 1 | `f_tender_value_log` | Log-transformed tender value |
| 2 | `f_award_value_log` | Log-transformed award value |
| 3 | `f_price_deviation_ratio` | Award/tender price ratio |
| 4 | `f_tender_duration_days` | Tender period duration |
| 5 | `f_award_duration_days` | Award period duration |
| 6 | `f_num_tenderers` | Number of bidders |
| 7 | `f_single_bidder` | Single-bidder flag (binary) |
| 8 | `f_title_length` | Tender title character count |
| 9 | `f_description_length` | Description character count |
| 10 | `f_procurement_method_enc` | Encoded procurement method |
| 11 | `f_is_q4` | Quarter 4 timing flag |
| 12 | `f_month_sin` | Month (sine component) |
| 13 | `f_month_cos` | Month (cosine component) |
| 14 | `f_day_of_week` | Day of week |
| 15 | `f_has_description` | Description availability flag |

</details>

<details>
<summary><b>Tier 2 — Temporal & Aggregated (15 features)</b></summary>

| # | Feature | Description |
|---|---------|-------------|
| 16 | `f_value_per_char` | Value per title character |
| 17 | `f_tenderer_value_ratio` | Tenderer count / value ratio |
| 18 | `f_buyer_contract_count` | Historical buyer contracts |
| 19 | `f_buyer_total_value` | Buyer cumulative spend |
| 20 | `f_buyer_avg_value` | Buyer average contract value |
| 21 | `f_buyer_supplier_diversity` | Unique suppliers per buyer |
| 22 | `f_buyer_single_bid_ratio` | Buyer's single-bid history |
| 23 | `f_supplier_contract_count` | Supplier contract count |
| 24 | `f_supplier_total_value` | Supplier cumulative revenue |
| 25 | `f_supplier_avg_value` | Supplier average value |
| 26 | `f_supplier_buyer_diversity` | Unique buyers per supplier |
| 27 | `f_supplier_distinct_buyers` | Distinct buyer entities |
| 28 | `f_buyer_method_diversity` | Buyer procurement method mix |
| 29 | `f_buyer_q4_ratio` | Buyer Q4 procurement rate |
| 30 | `f_supplier_q4_ratio` | Supplier Q4 procurement rate |

</details>

### Red Flag Rules (7 Heuristic Labels)

| Flag | Rule | Risk Indication |
|------|------|-----------------|
| 🚩 Single Bidder | `tenderers == 1` | Competition suppression |
| 🚩 Price Deviation | Award/tender ratio ≤ 0.7 or ≥ 1.0 | Price manipulation |
| 🚩 Direct Procurement | Method = direct/limited | Bypassing open competition |
| 🚩 Q4 Timing | Oct–Dec procurement | Budget-spending rush |
| 🚩 Short Title | Title < 20 chars | Obfuscation |
| 🚩 Short Description | Description < 60 chars | Lack of transparency |
| 🚩 High Value | ≥ 90th percentile | Corruption target |

### Risk Classification

| Level | Flag Count | Label |
|-------|-----------|-------|
| 🟢 Low Risk | 0–1 flags | `0` |
| 🟡 Medium Risk | 2–3 flags | `1` |
| 🔴 High Risk | 4+ flags | `2` |

---

## 🧪 Test Suite

```
88 passed ✅  |  0 failed  |  13 warnings (non-blocking)
```

| Test Module | Tests | Coverage |
|------------|-------|----------|
| `test_smoke.py` | 1 | Import verification |
| `test_data.py` | 6 | OCDS parser |
| `test_split.py` | 9 | Temporal splitting |
| `test_features.py` | 9 | Feature engineering |
| `test_labeling.py` | 11 | Heuristic labels |
| `test_leakage_guard.py` | 15 | **Anti-leakage** |
| `test_model_training.py` | 9 | HPO + training |
| `test_explanation.py` | 10 | SHAP explanations |
| `test_onnx_parity.py` | 6 | ONNX export |

```bash
# Run all tests
pytest -v

# Run only critical gate tests
pytest -m p0 -v

# Run specific module
pytest tests/test_leakage_guard.py -v
```

---

## 📓 Notebooks

### `training.ipynb` — Full Training Pipeline

Demonstrates: data loading → model evaluation → SHAP analysis → calibration → ONNX export

### `inference.ipynb` — Offline Inference + XAI

Demonstrates: single-record prediction → SHAP factors → Bahasa Indonesia narrative → counterfactual recommendations → batch inference

```bash
# Register kernel (first time only)
python -m ipykernel install --user --name=lpse-x --display-name "LPSE-X (venv)"

# Execute notebooks
jupyter nbconvert --to notebook --execute training.ipynb --ExecutePreprocessor.kernel_name=lpse-x
jupyter nbconvert --to notebook --execute inference.ipynb --ExecutePreprocessor.kernel_name=lpse-x
```

---

## 📦 Key Artifacts

| Artifact | Path | Description |
|----------|------|-------------|
| 📊 **Metrics** | `models/metrics.json` | Canonical evaluation results (source of truth) |
| 🌡️ **Calibration** | `models/calibration.json` | Temperature scaling parameters |
| 🔧 **Imputation** | `models/imputation_values.json` | Feature medians for ONNX inference |
| ⚙️ **Hyperparams** | `models/best_params.json` | Optuna HPO best configuration |
| ✅ **Clean Labels** | `data/processed/clean_labels_100.csv` | Human-reviewed calibration subset |
| 📈 **SHAP Plot** | `proposal/figures/shap_summary.png` | Feature importance visualization |
| 📉 **Confusion Matrix** | `proposal/figures/confusion_matrix.png` | Classification performance |
| 📊 **Calibration Curve** | `proposal/figures/calibration_curve.png` | Probability reliability |

---

## ⚠️ Important Notes

> [!IMPORTANT]
> All model metrics are measured against **heuristic risk labels**, not confirmed fraud outcomes. These are risk indicators, not forensic evidence.

> [!NOTE]
> The pipeline runs fully **offline** and **CPU-only**. No GPU, API calls, or internet required during inference.

> [!WARNING]
> Raw data (`data/raw/`), model binaries (`models/*.ubj`), and generated splits (`train_data/`, `test_data/`) are **gitignored**. Reproduce them via `scripts/materialize.py`.

---

## 👥 Team

| Member | Role |
|--------|------|
| **Sinholms** | ML Engineering & Development |

---

<p align="center">
  <sub>Built with ❤️ for Find IT! 2026 — Track C Phase 2</sub>
</p>
