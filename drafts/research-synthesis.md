# LPSE-X Research Synthesis & Gap Analysis

> **Generated**: 2026-03-30  
> **Context**: Find IT! 2026 Hackathon — Track C: "The Explainable Oracle"  
> **Phase 2 Deadline**: April 11, 2026 (12 days remaining)  
> **Purpose**: Consolidated research findings + cross-reference analysis against official competition documents

---

## Table of Contents

1. [Competition Requirements Summary](#1-competition-requirements-summary)
2. [Scoring Weight Analysis](#2-scoring-weight-analysis)
3. [Constraint Compliance Map](#3-constraint-compliance-map)
4. [Research Track 1: LPSE/Procurement Data Sources](#4-research-track-1)
5. [Research Track 2: XGBoost + SHAP Best Practices](#5-research-track-2)
6. [Research Track 3: Cardinal Library Reality Check](#6-research-track-3)
7. [Research Track 4: Procurement Fraud Detection Repos & Papers](#7-research-track-4)
8. [Critical Gaps in Current Planning](#8-critical-gaps)
9. [Recommended Corrections](#9-recommended-corrections)

---

## 1. Competition Requirements Summary

### Phase 2 Deliverables (from Guidebook)
- **Proposal** (Bab 1-4): Background, methodology, ethics/constraints, architecture/feasibility
- **training.ipynb**: Full training pipeline with documented preprocessing, feature engineering, model training
- **inference.ipynb**: Inference pipeline with explainability output, offline-capable
- **Trained model file**: Exported model (.json/.ubj/.pkl)
- **requirements.txt**: All dependencies with pinned versions
- **Environment documentation**: Setup instructions, system requirements

### Key Judging Structure
- **3 jury panels** evaluate independently: AI Expert, Software Engineering, Product/System Architect
- Each panel scores different aspects with different weights
- **Final score = weighted combination across all 3 panels**

---

## 2. Scoring Weight Analysis

### Effective Combined Weights (calculated from guidebook)

| Deliverable | AI Expert Weight | SWE Weight | Architect Weight | **Combined Effective** |
|-------------|-----------------|------------|------------------|----------------------|
| Proposal Bab 4 (Architecture + Feasibility) | 10% | 30% | 30% | **23.3%** |
| training.ipynb + model quality | 30% | 15% | 10% | **18.3%** |
| Proposal Bab 3 (Ethics + Constraints) | 15% | 15% | 20% | **16.7%** |
| Proposal Bab 2 (Methodology + Specs) | 20% | 10% | 10% | **13.4%** |
| inference.ipynb | 15% | 10% | 10% | **11.7%** |
| Environment/dependency docs | 5% | 10% | 10% | **8.3%** |
| Code documentation | 5% | 5% | 0% | **3.3%** |
| Proposal Bab 1 (Background) | 0% | 5% | 10% | **5.0%** |

### Key Insight
**Bab 4 (Architecture) is the HIGHEST weighted deliverable at 23.3%** — the current plan under-emphasizes this. It scores heavily with both SWE (30%) and Architect (30%) panels.

### Scoring Optimization Priority
1. **Bab 4** (23.3%) — Architecture diagrams, scalability discussion, deployment feasibility → INCREASE effort
2. **training.ipynb** (18.3%) — Already well-planned
3. **Bab 3** (16.7%) — Restructure as constraint-by-constraint response → RESTRUCTURE
4. **Bab 2** (13.4%) — Methodology is solid but needs "Problem-Model Alignment" framing
5. **inference.ipynb** (11.7%) — Already well-planned
6. **Environment docs** (8.3%) — Easy points, currently under-documented → ADD explicit section

---

## 3. Constraint Compliance Map

### Track C Constraints (all mandatory)

| ID | Constraint | Current Plan Status | Gap? |
|----|-----------|-------------------|------|
| C-C1 | Explainability Wajib (SHAP/LIME/feature importance) | ✅ SHAP TreeExplainer planned | No |
| C-C2 | Output Penjelasan (3+ top variables with direction, human-readable) | ✅ Top-5 features with direction planned | No |
| C-C3 | Anti-Black Box (no opaque models without explainability) | ✅ XGBoost is inherently interpretable + SHAP | No |
| C-C4 | Validasi Data Leakage (folder separation BEFORE preprocessing) | ✅ Explicit leakage protocol in plan | No |
| C-C5 | Offline Total (inference + explainability offline, no cloud API) | ✅ All local, no API dependencies | No |

### General Constraints

| ID | Constraint | Current Plan Status | Gap? |
|----|-----------|-------------------|------|
| G1 | No cloud inference APIs | ✅ All local | No |
| G2 | CPU-only reproducibility (jury validates on CPU) | ⚠️ **NOT DOCUMENTED** | **YES** |
| G3 | Bab 3 = point-by-point constraint compliance | ⚠️ Plan treats Bab 3 as general essay | **YES** |
| G4 | Pinned dependencies in requirements.txt | ✅ Mentioned in plan | No |
| G5 | All code in notebooks (training.ipynb + inference.ipynb) | ✅ Notebook-based pipeline | No |

### Critical Quote from Constraints Document
> "Kegagalan memenuhi seluruh constraint yang telah ditentukan akan berdampak fatal pada penilaian. Constraint bersifat wajib (mandatory), bukan opsional."

> "Model boleh memanfaatkan GPU localhost bawaan laptop peserta saat demo. Namun, model wajib tetap dapat berjalan di lingkungan CPU-only dan memenuhi batas waktu inferensi pada kondisi CPU. Juri memvalidasi constraint kecepatan pada kondisi CPU, bukan GPU."

---

## 4. Research Track 1: LPSE / Procurement Data Sources

### Available Data Sources

| Source | Format | Access | Volume | Relevance |
|--------|--------|--------|--------|-----------|
| **LPSE Portal** (lpse.go.id) | HTML (scraping needed) | Public, no API | Millions of tenders | HIGH — primary Indonesian procurement data |
| **OCDS Indonesia** (ocds-indonesia.org) | JSONL/CSV (OCDS-derived) | Bulk downloads | Large | HIGH — primary structured source, but field coverage must be audited |
| **Opentender.eu** | CSV/JSON (OCDS) | Bulk downloads | EU tenders | MEDIUM — reference format, not Indonesian |
| **Kaggle Dataset** | CSV | Direct download | ~10K tenders | HIGH for prototyping — quick start |
| **SIRUP** | Various | Government portal | Planning data | LOW — complementary budget data |

### Recommended Data Strategy for Hackathon
1. **Start with Kaggle dataset** (~10K tenders CSV) for rapid prototyping and pipeline development
2. **Target OCDS Indonesia** for production data — structured, Cardinal-compatible
3. **LPSE scraping** only if OCDS Indonesia insufficient — high effort, fragile
4. **SIRUP** as supplementary features (budget vs actual spend)

### OCDS Data Structure (key fields for feature engineering)
```
tender.numberOfTenderers          → competition indicator
tender.tenderPeriod.durationInDays → time pressure indicator
tender.value.amount               → contract value
tender.procurementMethod          → procurement method type
awards.value.amount               → award amount
awards.suppliers                  → winner info
bids.details[].value.amount       → individual bid amounts
planning.budget.amount            → planned budget
```

---

## 5. Research Track 2: XGBoost + SHAP Best Practices

### Multi-Class SHAP (4-level risk classification)

**SHAP values shape**: `(n_samples, n_features, n_classes)` for multi-class
- Access class 0 (Low Risk): `shap_values[:, :, 0]`
- Access class 3 (Critical): `shap_values[:, :, 3]`

**Key code pattern:**
```python
import shap
import xgboost as xgb

model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    objective='multi:softprob',
    num_class=4,
    tree_method='hist',  # CPU-optimized
    n_jobs=-1
)

# Class imbalance handling for multi-class
from sklearn.utils.class_weight import compute_sample_weight
sample_weights = compute_sample_weight('balanced', y_train)
model.fit(X_train, y_train, sample_weight=sample_weights)

# SHAP
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)  # shape: (n, features, 4)
```

### Class Imbalance Handling
- **Multi-class**: Use `compute_sample_weight('balanced', y_train)` — NOT `scale_pos_weight` (binary only)
- **Alternative**: Focal loss for extreme imbalance
- **SMOTE**: Possible but be careful with leakage — apply ONLY on training fold

### Performance Optimization
- **FastTreeSHAP** (LinkedIn): speedups are benchmark-dependent; plan around ~1.5-2.5x for XGBoost/single-instance CPU use, not a universal 10-100x
- **Background data**: `shap.kmeans(X_train, k=100)` reduces computation
- **tree_method='hist'**: Fastest CPU training method
- **XGBoost on CPU**: Already fast — no GPU needed for tabular data of this scale

### Human-Readable Explanation Template (Bahasa Indonesia)
```python
EXPLANATION_TEMPLATES = {
    'id': {
        'positive_impact': "{feature} = {value} meningkatkan risiko sebesar {impact:.2%}",
        'negative_impact': "{feature} = {value} menurunkan risiko sebesar {impact:.2%}",
        'summary': "Faktor risiko teratas untuk transaksi ini:"
    }
}
```

### Recommended Hyperparameters for Fraud/Risk

| Parameter | Range | Notes |
|-----------|-------|-------|
| `n_estimators` | 200-500 | More trees = better SHAP stability |
| `max_depth` | 4-8 | Lower prevents overfitting |
| `learning_rate` | 0.05-0.1 | Lower with more trees |
| `min_child_weight` | 1-10 | Higher prevents false positives |
| `subsample` | 0.7-0.9 | Regularization |
| `colsample_bytree` | 0.7-0.9 | Regularization |
| `gamma` | 0.1-1.0 | Min loss reduction |
| `tree_method` | 'hist' | CPU-optimized |

### Evaluation Metrics
- **Primary**: Macro F1 (treats all 4 risk classes equally)
- **Secondary**: Weighted F1 (accounts for class imbalance)
- **Per-class**: Precision/Recall for each risk level (especially High/Critical recall)
- **Report**: `sklearn.metrics.classification_report(y_test, y_pred, target_names=['Low', 'Medium', 'High', 'Critical'])`

### Model Export
- **Recommended**: `.json` format (portable, version-safe, human-readable)
- **Alternative**: `.ubj` (Universal Binary JSON, smaller, XGBoost 2.1+ default)
- **Load pattern**: `model.save_model('model.json')` / `model.load_model('model.json')`

### SHAP Visualization Types
1. **Waterfall plot** — per-prediction explanation (best for single tender analysis)
2. **Force plot** — interactive HTML (good for notebooks)
3. **Summary bar plot** — global feature importance (good for Bab 2 methodology section)
4. **Beeswarm plot** — feature impact distribution (good for proposal)

---

## 6. Research Track 3: Cardinal Library — Reality Check

### ⚠️ CRITICAL FINDING: "73 Cardinal Features" Is Wrong

**Incorrect phrasing**:
> "73 Cardinal features"

**Correct interpretation**:
- **Cardinal** is a specific tool that currently documents **11 implemented red flag indicators**.
- The broader **"73 red flags"** concept belongs to an Open Contracting red-flag mapping framework, not to Cardinal itself.
- Therefore the safe phrasing for this project is:
  - **"11 Cardinal indicators + custom LPSE-X engineered features"**
  - not **"73 Cardinal features"**

**Why this matters**:
- If we say "73 Cardinal features," we overstate what Cardinal provides out of the box.
- That distorts implementation effort, proposal claims, and feature-planning realism.
- It also makes code-to-indicator naming drift more likely, which is dangerous for Bab 2/Bab 4 and for explanation outputs.

**Canonical Cardinal indicator list**:

| ID | Name | Description |
|----|------|-------------|
| R003 | Short submission period | Tender submission window is unusually short |
| R018 | Single bid received | Only one bid is received in an open/selective process |
| R024 | Price close to winner | Price gap between winner and next bid is suspiciously small |
| R025 | Excessive losing ratio | Supplier bidding pattern shows unusual win/loss imbalance |
| R028 | Identical bid prices | Multiple bids share the same amount |
| R030 | Late bid won | Winning bid arrives after the nominal deadline |
| R035 | All others disqualified | Competition collapses after disqualifications |
| R036 | Lowest bid disqualified | Lowest-price bid is disqualified under price-driven award logic |
| R038 | Excessive disqualified ratio | Disqualification ratio is abnormally high |
| R048 | Heterogeneous supplier | Supplier behavior spans unusually heterogeneous categories |
| R058 | Heavily discounted winner | Winner undercuts the next bid by an unusually large margin |

**Practical planning rule**:
- Treat Cardinal as one optional source of **11 documented indicator definitions**.
- Treat everything beyond those 11 as **custom LPSE-X feature engineering**.

### Technical Details
- **Language**: Rust (93.6%), Python bindings (6.4%)
- **Input format**: OCDS JSONL (JSON Lines)
- **License**: MIT
- **Stars**: 13
- **Maintainer**: Open Contracting Partnership
- **CLI tool** — not importable as Python library directly
- **Docs**: https://cardinal.readthedocs.io/

### Impact on LPSE-X Plan
1. **Feature count must be corrected**: Cardinal contributes 11 documented indicators, not 73.
2. **Terminology must be corrected**: Use "73 red flags mapping" only when referring to the broader OCP framework, never to Cardinal.
3. **Integration approach**: Either call Cardinal CLI from Python or reimplement the 11 indicator definitions directly in Python.
4. **Feature gap**: LPSE-X still needs a large custom feature layer beyond Cardinal's 11.
5. **Positive**: Cardinal remains useful as a rigor anchor for a subset of procurement red flags.

### Recommended Additional Features (beyond Cardinal's 11)

**Price features** (~10):
- Bid-to-budget ratio, coefficient of variation of bids, price clustering score, historical price deviation, bid amount round-number frequency

**Competition features** (~8):
- Number of bidders, bidder concentration (HHI), new bidder rate, bidder overlap across tenders, geographic diversity of bidders

**Timing features** (~8):
- Submission time clustering (do bids arrive suspiciously together?), time between announcement and deadline, weekend/holiday publishing, evaluation duration

**Bidder behavior features** (~10):
- Historical win rate, market share, tender participation frequency, geographic scope, specialization score, subcontracting patterns

**Procurement process features** (~8):
- Number of amendments, complaint history, extension count, method type risk score, evaluation method, qualification criteria count

**Network/Association features** (~6):
- Shared address/phone, shared directors, bidder co-occurrence frequency, geographic clustering

**Total estimated features**: 11 (Cardinal-aligned indicators) + ~50 (custom LPSE-X features) = ~61 features

---

## 7. Research Track 4: Procurement Fraud Detection Repos & Papers

### Top GitHub Repositories

| Repository | Stars | Focus | ML Approach | Relevance |
|-----------|-------|-------|-------------|-----------|
| **open-tender-watch** | 101 | Full procurement monitoring platform | Rules + analytics | HIGH — architecture reference |
| **Cardinal-rs** | 13 | OCDS red flags calculation | Rule-based (11 indicators) | HIGH — direct red flag calculation |
| **buenosaires_ocds_redflags** | — | Buenos Aires OCDS red flags | Statistical indicators | HIGH — OCDS implementation pattern |
| **OCDS_RedFlags** (Project PODER) | — | Mexican procurement (Groucho system) | Red flags + visualization | MEDIUM — reporting reference |
| **BidRiggingDetection** | 3 | Social network analysis for bid rigging | SNA (network graph) | MEDIUM — network feature ideas |
| **Anomaly-Detection-in-Procurement** | 0 | Unsupervised procurement anomaly detection | Isolation Forest, LOF | MEDIUM — unsupervised baseline |
| **P2Predict** | — | Procurement price prediction | ML price estimation | LOW — supplementary |
| **FraudDetection-SupplyChain** | — | Supply chain fraud (DL) | Deep learning, PCA, SMOTE | LOW — different domain |

### Key Academic Papers

| Paper | Year | Method | Key Contribution |
|-------|------|--------|-----------------|
| **Deep Learning for Bid Rigging** (arXiv:2104.11142) | 2021 | CNN | Converts bid sequences to feature matrices for CNN classification |
| **GAT for Bid-rigging** (arXiv:2507.12369) | 2025 | Graph Attention Network | Dynamic bidder relationship graphs with attention |
| **ML for Incomplete Cartels** (arXiv:2004.05629) | 2020 | Random Forest, GBT | Detects partial collusion; excellent feature engineering catalog |
| **GNN for Collusion Detection** (arXiv:2410.07091) | 2024 | GCN, GAT, GraphSAGE | Systematic GNN comparison; GAT best in most scenarios |
| **SPSE Data Mining Fraud Detection** (IEEE) | — | Data mining | Directly on Indonesian SPSE system |
| **Collusion Fraud Risk in E-Tendering** (BPKP) | — | Data analytics | Indonesian context, BPKP auditor perspective |

### Feature Engineering Catalog (from papers)

**From "ML for Incomplete Cartels" (best feature catalog found):**
- Bid price statistics: mean, std, CV, skewness, kurtosis
- Bid spacing: gaps between consecutive bids, price clustering
- Cover bidding indicators: bids that seem designed to lose
- Rotation patterns: do the same firms take turns winning?
- Market allocation: geographic/sector concentration of wins

**From "Deep Learning for Bid Rigging":**
- Temporal bid patterns (submission timing relative to deadline)
- Bid revision frequency and magnitude
- Historical pairwise co-occurrence of bidders

### Key Insight
**No high-star Python repo exists for procurement fraud + XGBoost + SHAP specifically.** LPSE-X would be relatively novel in combining:
1. Indonesian procurement data (LPSE/OCDS)
2. XGBoost gradient boosting
3. SHAP TreeExplainer for explainability
4. 4-level risk classification (not just binary fraud/not-fraud)

This is a **strength** — the judges will see original work, not a clone.

---

## 8. Critical Gaps in Current Planning

### 🔴 Must Fix (affects scoring or disqualification risk)

| # | Gap | Impact | Source |
|---|-----|--------|--------|
| 1 | **"73 Cardinal features" is wrong** — only 11 indicators | Feature engineering scope is miscalculated; need ~50 additional custom features | Cardinal docs |
| 2 | **CPU-only requirement not documented** | Jury validates on CPU; must prove XGBoost+SHAP runs within time limit on CPU | Constraints doc G2 |
| 3 | **Bab 3 is not structured as constraint-by-constraint** | Constraints doc MANDATES point-by-point compliance for each C-C1 through C-C5 | Constraints doc G3 |
| 4 | **Bab 4 is under-prioritized** | Highest weighted deliverable (23.3%) but plan allocates less effort than training notebook | Scoring analysis |
| 5 | **Conflicting plans** — `lpse-x-build-track-c-revised.md` includes Phase 3 scope | Risk of team members working on out-of-scope items (frontend, backend, app) | Plan review |

### 🟡 Should Fix (improves scoring)

| # | Gap | Impact | Source |
|---|-----|--------|--------|
| 6 | **No "Problem-Model Alignment" framing** in Bab 2 | AI Expert panel looks for explicit alignment between problem characteristics and model choice | Guidebook |
| 7 | **No bias mitigation discussion** | Bab 3 (Ethics) should address demographic bias in procurement data | Guidebook |
| 8 | **No data augmentation reasoning** | Should document why augmentation is/isn't needed for this dataset | Guidebook |
| 9 | **No resource efficiency metrics** | CPU inference time, memory footprint, model size should be documented | Guidebook |
| 10 | **No explicit model export format documentation** | Should state .json format choice with reasoning | Guidebook |

### 🟢 Nice to Have

| # | Gap | Impact |
|---|-----|--------|
| 11 | Bahasa Indonesia explanation templates | Demonstrates localization awareness |
| 12 | SHAP visualization variety (waterfall + beeswarm + bar) | Shows depth of explainability |
| 13 | Comparison table (XGBoost vs alternatives) in Bab 2 | Shows rigorous method selection |

---

## 9. Recommended Corrections

### Immediate Actions (Days 1-2, March 30-31)

1. **Correct Cardinal feature count**: Replace "73 Cardinal features" with "11 Cardinal indicators + ~50 custom-engineered features = ~61 total features"
2. **Add CPU benchmark section**: Include `%%timeit` cell in inference.ipynb; target <2s per prediction on CPU
3. **Restructure Bab 3 outline**: Create 5 subsections (C-C1 through C-C5), each with: constraint text → implementation → evidence → verification
4. **Archive `lpse-x-build-track-c-revised.md`**: Rename to `phase-3-vision.md` or add clear "PHASE 3 ONLY" header
5. **Elevate Bab 4 priority**: Allocate dedicated day for architecture diagrams, scalability discussion, deployment plan

### Feature Engineering Plan (corrected)

```
Layer 1: Cardinal Indicators (11 features)
├── R003: submission_period_short
├── R018: single_bid_received
├── R024: price_close_to_winner
├── R025: excessive_losing_ratio
├── R028: identical_bid_prices
├── R030: late_bid_won
├── R035: all_others_disqualified
├── R036: lowest_bid_disqualified
├── R038: excessive_disqualified_ratio
├── R048: heterogeneous_supplier
└── R058: heavily_discounted_winner

Layer 2: Custom Price Features (~10)
├── bid_to_budget_ratio
├── bid_coefficient_of_variation
├── price_clustering_score
├── round_number_frequency
├── historical_price_deviation
└── ...

Layer 3: Custom Competition Features (~8)
├── number_of_bidders
├── bidder_hhi (Herfindahl-Hirschman Index)
├── new_bidder_rate
├── bidder_overlap_score
└── ...

Layer 4: Custom Timing Features (~8)
├── submission_time_clustering
├── announcement_to_deadline_days
├── weekend_holiday_publishing
├── evaluation_duration_zscore
└── ...

Layer 5: Custom Behavior Features (~10)
├── historical_win_rate
├── market_share
├── specialization_score
├── geographic_scope
└── ...

Layer 6: Custom Process Features (~8)
├── amendment_count
├── complaint_history
├── method_type_risk_score
├── qualification_criteria_count
└── ...

TOTAL: ~55-65 features
```

### Reference Repositories to Study

| Priority | Repository | What to Learn |
|----------|-----------|---------------|
| 1 | Cardinal-rs docs | Exact red flag calculation logic for reimplementation |
| 2 | buenosaires_ocds_redflags | OCDS data processing pipeline pattern |
| 3 | ML for Incomplete Cartels paper | Feature engineering catalog |
| 4 | open-tender-watch | System architecture for Phase 3 |
| 5 | FastTreeSHAP | SHAP performance optimization |

---

## External References

### Data Sources
- LPSE Portal: https://lpse.go.id/
- OCDS Indonesia: https://ocds-indonesia.org/
- Cardinal docs: https://cardinal.readthedocs.io/
- Cardinal repo: https://github.com/open-contracting/cardinal-rs
- Open Contracting red flags guide: https://www.open-contracting.org/resources/red-flags-in-public-procurement-a-guide-to-using-data-to-detect-and-mitigate-risks/
- Red Flags to OCDS Mapping: https://docs.google.com/spreadsheets/d/12PFkUlQH09jQvcnORjcbh9-8d-NnIuk4mAQwdGiXeSM/edit
- Opentender.eu: https://opentender.eu/
- OCDS standard: https://standard.open-contracting.org/

### Libraries
- XGBoost: https://xgboost.readthedocs.io/
- SHAP: https://shap.readthedocs.io/
- FastTreeSHAP: https://github.com/linkedin/FastTreeSHAP

### Papers
- XGBoost: https://arxiv.org/abs/1603.02754
- SHAP: https://arxiv.org/abs/1705.07874
- Deep Learning for Bid Rigging: https://arxiv.org/abs/2104.11142
- GAT for Bid-rigging: https://arxiv.org/abs/2507.12369
- ML for Incomplete Cartels: https://arxiv.org/abs/2004.05629
- GNN for Collusion: https://arxiv.org/abs/2410.07091

### Competition Documents
- Guidebook: `/home/vascosera/Downloads/Hackathon-Guidebook-2026 (1).pdf`
- Constraints: `/home/vascosera/Downloads/Constraints Hackathon 2026 (2).pdf`
- Extracted text: `/tmp/guidebook.txt`, `/tmp/constraints.txt`

---

# Round 2 Research (2026-03-30)

## 10. LPSE Data Availability — Confirmed Sources

### Source-by-Source Assessment

| Source | URL | Format | Volume | Access | Currency |
|--------|-----|--------|--------|--------|----------|
| **OCDS Indonesia** | ocds-indonesia.org / OCP Registry | JSONL (OCDS), CSV | Hundreds of thousands | Bulk download (free account for some) | Historical (2013–2023, lag in OCDS conversion) |
| **LPSE Portal** | lpse.go.id + regional portals | HTML (scraping) | Real-time, all active tenders | Scraping via `pyspse` library | **Current/Real-time** |
| **SIRUP API** | api.lpse.or.id (SMEP 3 SIRUP API) | JSON via API, XLSX via web | Budget/planning data 2019–2027 | **Public API** | Current |
| **LKPP Open Data** | data.lkpp.go.id | XLSX, CSV, JSON | Large (aggregate + detailed) | Direct download | Updated periodically (2025/2026 data) |
| **data.go.id** | data.go.id | JSON, CSV | Mirrors LKPP data | Direct download / API | Slightly delayed |
| **Kaggle** | Various datasets | CSV | 10K–50K rows (regional) | Direct download | **Outdated** (2018–2022 snapshots) |

### Key Findings
- **SIRUP has a public API** (api.lpse.or.id) — this is huge for comparing planned vs actual procurement
- **`pyspse` Python library** exists for scraping LPSE portals
- **LKPP publishes open data** at data.lkpp.go.id including UMKM participation, SIRUP summaries
- **For hackathon**: Kaggle for quick prototyping → OCDS/LKPP for production model
- **Critical caution**: OCP Registry publication 101 reports zero tenderers and zero contracts/documents/milestones/amendments in aggregate counts, so bid/tenderer-dependent features must be treated as unverified until the coverage audit runs

### Recommended Data Strategy

| Component | Source | Method | Purpose |
|-----------|--------|--------|---------|
| Historical Baseline | OCDS / Kaggle | Bulk Download | Model training, anomaly baseline |
| Real-time Detection | LPSE Portals | Scraper (pyspse) | Current tender data, bid prices, winners |
| Integrity Check | SIRUP API | API Pull | "Ghost Projects" — tenders without budget plans |
| Vendor Profiling | LKPP SIKaP / data.go.id | API / Scrape | Vendor eligibility, blacklists |

---

## 11. Indonesian Procurement Fraud Patterns

### Bid-Rigging Techniques (Persekongkolan Tender)

| Pattern | Indonesian Term | Description | Feature Idea |
|---------|----------------|-------------|--------------|
| **Phantom Bidders** | Perusahaan Boneka | Single mastermind uses multiple shell companies | Overlapping ownership, shared addresses/IP |
| **Rotation Schemes** | Gantian Pemenang | Cartel members take turns winning | Win-loss pattern analysis across company groups |
| **Split Procurement** | Pemecahan Paket | Split large projects below IDR 200M threshold | Ratio of similar projects vs threshold in same agency |
| **Complementary Bidding** | Pendamping | "Losers" submit intentionally flawed/high bids | Bid price distribution analysis, flawed document patterns |
| **Spec Locking** | Pengaturan Spesifikasi | Specs written to match only one vendor | Spec-to-vendor brochure similarity (advanced NLP) |
| **Electronic Sabotage** | — | Prevent competitors from uploading documents | Upload failure rate near deadline |

### KPK Case Patterns
- **"Fee" system**: 5–15% kickback promised before tender begins
- **Technical specification locking**: Only one vendor can meet hyper-specific requirements
- **Electronic sabotage**: Server manipulation or bots during upload windows (LPSE Lampung 2026 case)

### BPKP Audit Red Flags (Directly Measurable)

| Red Flag | SPSE Signal | Feature Engineering |
|----------|-------------|---------------------|
| **HPS Proximity** | Bids within 1-2% of HPS | `bid_price / hps_price` — flag if >0.98 |
| **Late Addendums** | Document changes near deadline | `addendum_count` + `days_before_deadline` |
| **Participation Drop-off** | Many download, few submit | `downloaders / submitters` ratio |
| **Document Fingerprinting** | Identical metadata across "competing" bids | PDF Author field, Creation Date, identical typos |
| **Submission Synchronicity** | Multiple bids uploaded within minutes | `max_time_gap_between_submissions` |

### Top Features for LPSE-X Classifier

1. **HPS Proximity**: `bid_price / hps_price` — values >0.985 are high-risk
2. **Submission Synchronicity**: Time delta between bid uploads — flag if <5 minutes
3. **Vendor Network Centrality**: Graph feature — how often companies A, B, C appear together
4. **Tender Addendum Frequency**: Number of spec changes during announcement phase
5. **Single Bidder Frequency**: Win rate with single bidder per agency
6. **Winner Rotation Score**: Entropy of winner distribution within a bidder group
7. **Downloaders-to-Submitters Ratio**: High download, low submission = intimidation signal

---

## 12. Winning Hackathon Proposal Strategies

### Find IT! / Gemastik Winning Patterns

1. **Structure**: Follow Bab I–IV format strictly as per guidebook
2. **"So What?" Factor**: Solve a SPECIFIC Indonesian problem, not generic ML
3. **Visual Dominance**: Architecture diagram is the "hero" of the proposal — use professional tools
4. **Indonesian Context**: Winning teams contextualize everything to Indonesia

### The XAI Pyramid (Differentiator for Track C)

Most teams stop at level 2. Level 3 wins:

```
Level 3: ACTIONABLE INSIGHTS (Counterfactual Explanations)
  "If this tender had 3+ bidders, risk would drop from 85% to 23%"
  → This is the winning differentiator

Level 2: LOCAL INTERPRETABILITY (SHAP/LIME per prediction)
  "This tender is high-risk because HPS proximity = 99.5%"
  → Most teams will do this

Level 1: GLOBAL INTERPRETABILITY (Feature Importance)
  "Top features: HPS proximity, single bidder rate, vendor win rate"
  → Table stakes, everyone does this
```

### Architecture Diagram Framework (Highest Scoring Weight)

Must include these 4 layers:
1. **Data Ingestion Layer** — Raw data → preprocessing → feature engineering
2. **Predictive Engine** — The "Oracle" (XGBoost + hyperparameter tuning)
3. **Explainability Wrapper** — Dedicated layer showing SHAP/LIME integration
4. **UI/UX Delivery** — How explanations reach non-technical users (dashboard / NL summary)

### Methodology Framing

Use CRISP-DM + Explainability Audit:
- Phase 1: Data Understanding & Preprocessing (handle imbalanced classes)
- Phase 2: Model Development & Hyperparameter Tuning
- Phase 3: Interpretability Analysis (XAI techniques)
- Phase 4: Human-in-the-loop Validation

### Reference Repos
- **Find IT! 2022 DAC**: `evanekawijaya/DAC-FIND-IT-UGM-2022` — EDA + metrics focus
- For 2026: Must extend with dedicated "Explainability" module + counterfactuals

---

## 13. Indonesian Academic Papers on SPSE/LPSE

### Paper 1: ITB — Data Mining on SPSE (2016)

| Field | Value |
|-------|-------|
| **Title** | Fraud detection based-on data mining on Indonesian E-Procurement System (SPSE) |
| **Authors** | Saptawati, G. A. P., et al. (ITB) |
| **Venue** | 2016 International Conference on Data and Software Engineering (ICoDSE), IEEE |
| **Dataset** | SPSE tender data — company profiles, bidding prices, tender schedules |
| **Method** | Decision Trees (C4.5), Naive Bayes, Neural Networks, K-Means clustering |
| **Features** | Bid-to-HPS ratio, submission time relative to deadline, company co-occurrence, winning frequency |
| **Results** | Decision Trees best: **85-92% accuracy** depending on fraud type |
| **Key Finding** | Bid rigging and collusive tendering are most detectable patterns |
| **Limitations** | Rare labeled fraud instances; focuses only on tendering stage |
| **Code** | Not available |

### Paper 2: BPKP — Collusion Fraud Risk Mitigation (2022)

| Field | Value |
|-------|-------|
| **Title** | Collusion Fraud Risk Mitigation with Integration of Data Analytics in E-Tendering |
| **Author** | Mustofa Kamal (Pusdiklatwas BPKP) |
| **Venue** | Asia Pacific Fraud Journal, Vol 7 No 2 |
| **Dataset** | SPSE E-Tendering metadata from local government LPSE |
| **Method** | Benford's Law + Social Network Analysis (SNA) |
| **Features** | Price clustering, IP/MAC address matching, document metadata fingerprinting, bidder network frequency |
| **Key Finding** | **Metadata analysis (IP/MAC/PDF properties) is most effective "smoking gun"** for collusion |
| **Limitations** | BPKP data is siloed; needs cross-agency data |
| **Code** | Not available |

### Paper 3: UGM — Fraudulent Behaviors in SPSE (2017)

| Field | Value |
|-------|-------|
| **Title** | Potential fraudulent behaviors in e-procurement implementation in Indonesia |
| **Authors** | Huda et al. (UGM) |
| **Focus** | Categorizing fraud types in SPSE |
| **Patterns** | Tailor-made specifications, restrictive requirements, tender splitting |

### Paper 4: ITS — Process Mining on Procurement Logs (2017)

| Field | Value |
|-------|-------|
| **Title** | Fraud detection on event logs using Heuristics Miner |
| **Method** | Process Mining |
| **Key Finding** | Analyzing event logs to find deviations from standard LKPP workflow (skipped steps, unusually fast transitions) |
| **Tool** | `pm4py` recommended for implementation |

### Benchmark Summary

| Paper | Best Method | Accuracy | Dataset Size |
|-------|------------|----------|-------------|
| ITB 2016 | Decision Tree C4.5 | 85-92% | SPSE (unspecified) |
| BPKP 2022 | Benford's Law + SNA | N/A (qualitative) | Local gov LPSE |

**Gap**: No prior work uses XGBoost + SHAP on Indonesian procurement data. LPSE-X would be the first.

---

## 14. OCDS Schema for Feature Engineering

### Top-Level OCDS Structure

| Section | Field Path | Description |
|---------|-----------|-------------|
| **Parties** | `parties[]` | All organizations (buyer, supplier, tenderer) |
| **Planning** | `planning` | Budget, project ID, rationale |
| **Tender** | `tender` | Procurement process details |
| **Awards** | `awards[]` | Award decisions and winners |
| **Contracts** | `contracts[]` | Signed agreements |
| **Bids*** | `bids.details[]` | Individual bid data (via Bids Extension) |

### Key Fields for Risk Classification

| Field Path | Type | Description | Risk Use |
|-----------|------|-------------|----------|
| `tender.numberOfTenderers` | integer | Count of unique bidders | Single bidder detection |
| `tender.tenderPeriod.startDate` | dateTime | Submission window start | Short notice calculation |
| `tender.tenderPeriod.endDate` | dateTime | Submission deadline | Duration calculation |
| `tender.procurementMethod` | string | `open`, `selective`, `limited` | Method risk scoring |
| `tender.procurementMethodDetails` | string | Indonesian: Tender Umum, Seleksi, Pengadaan Langsung | Local method mapping |
| `tender.amendments[]` | array | Changes to tender | Late modification count |
| `planning.budget.amount.value` | number | Budgeted amount | Budget utilization ratio |
| `awards[].value.amount` | number | Award price | Price anomaly detection |
| `awards[].suppliers[].id` | string | Winner ID | Repeat winner analysis |
| `awards[].date` | dateTime | Award date | Award delay calculation |
| `bids.details[].value.amount` | number | Individual bid price | Bid spread analysis |
| `bids.details[].tenderers[].id` | string | Bidder ID | Network analysis |
| `bids.details[].status` | string | `valid`, `disqualified`, `withdrawn` | Disqualification ratio |
| `parties[].identifier.id` | string | NPWP (tax ID) | Entity resolution |

### Indonesia-Specific OCDS Implementation
- **Extensions used**: Bids Extension, Lots Extension
- **Localization**: `title_id`, `description_id` for Bahasa Indonesia
- **tender.id** maps to LPSE tender ID
- **procurementMethodDetails** contains Indonesian names (Tender Umum, Seleksi, Pengadaan Langsung)
- **parties[].identifier.id** uses NPWP format (inconsistent formatting is a known issue)

### Data Quality Issues (Critical for Feature Engineering)
1. **Missing bids array**: Many LPSE instances don't sync detailed bid data
2. **Inconsistent NPWP**: Tax ID formatting varies — need normalization
3. **Missing planning section**: Budget data often in SIRUP, not in OCDS release
4. **Lag in OCDS conversion**: Real-time LPSE data may not appear in OCDS for months

### OCDS → Flat Feature Table Mapping

```python
# Temporal Features
tender_duration_days = (tender.tenderPeriod.endDate - tender.tenderPeriod.startDate).days
award_delay_days = (awards[0].date - tender.tenderPeriod.endDate).days

# Monetary Features  
budget_utilization = awards[0].value.amount / planning.budget.amount.value
relative_bid_spread = (max(bids) - min(bids)) / mean(bids)
hps_proximity = awards[0].value.amount / planning.budget.amount.value

# Competition Features
n_bidders = tender.numberOfTenderers
disqualification_ratio = count(bids where status='disqualified') / count(bids)

# Categorical (One-Hot)
procurement_method = tender.procurementMethod  # open, selective, limited
main_category = tender.mainProcurementCategory  # goods, works, services

# Network/History Features (require aggregation across releases)
supplier_win_rate = historical_wins(supplier.id) / historical_bids(supplier.id)
buyer_concentration = HHI(awards grouped by buyer.id)
```

### Sample OCDS JSON (Indonesian Context)
```json
{
  "ocid": "ocds-87SD23-12345",
  "id": "release-1",
  "date": "2026-03-30T10:00:00Z",
  "tag": ["tender"],
  "initiationType": "tender",
  "tender": {
    "id": "12345",
    "title": "Pengadaan Jasa Konsultansi IT",
    "status": "active",
    "procurementMethod": "open",
    "procurementMethodDetails": "Tender Umum",
    "tenderPeriod": {
      "startDate": "2026-03-01T08:00:00Z",
      "endDate": "2026-03-10T16:00:00Z"
    },
    "numberOfTenderers": 1
  },
  "parties": [
    {
      "id": "ID-NPWP-01.234.567.8-999.000",
      "name": "PT. Solusi Teknologi",
      "roles": ["tenderer"]
    }
  ],
  "bids": {
    "details": [
      {
        "id": "bid-1",
        "status": "valid",
        "value": { "amount": 500000000, "currency": "IDR" },
        "tenderers": [{ "id": "ID-NPWP-01.234.567.8-999.000" }]
      }
    ]
  }
}
```

---

## 15. Complete Feature Engineering Specification (Round 3)

### 15.1 Cardinal Red Flags — All 11 Indicators (Documented)

| ID | Feature Name | Category | Formula / Logic | OCDS Fields | Risk Direction |
|:---|:---|:---|:---|:---|:---|
| R003 | `submission_period_short` | Timing | `(tenderPeriod/endDate - tenderPeriod/startDate) < 15 days` | `tender/tenderPeriod` | Binary (1=Risk) |
| R018 | `single_bid_received` | Competition | `numberOfTenderers == 1` AND method is 'open'/'selective' | `tender/numberOfTenderers`, `tender/procurementMethod` | Binary (1=Risk) |
| R024 | `price_close_to_winner` | Price | `(2ndLowestBid - WinningBid) / WinningBid <= LowerFence` | `bids/details`, `awards/value` | Continuous (Ratio) |
| R025 | `excessive_losing_ratio` | Behavioral | `Count(Wins) / Count(ValidBids)` — low outlier for top bidders | `bids/details`, `awards/suppliers` | Continuous (Ratio) |
| R028 | `identical_bid_prices` | Price | `Count(UniqueTenderers per (Amount, Currency)) > 1` | `bids/details` | Binary (1=Risk) |
| R030 | `late_bid_won` | Timing | `bids/date > tender/tenderPeriod/endDate` AND `bidder == winner` | `bids/details/date`, `tender/tenderPeriod/endDate` | Binary (1=Risk) |
| R035 | `all_others_disqualified` | Competition | `Count(DisqualifiedBidders) > 0` AND `Count(ValidBidders) == 1` | `bids/details/status`, `awards/suppliers` | Binary (1=Risk) |
| R036 | `lowest_bid_disqualified` | Competition | `LowestBid/status == 'disqualified'` AND `awardCriteria == 'price'` | `bids/details`, `tender/awardCriteria` | Binary (1=Risk) |
| R038 | `excessive_disqualified_ratio` | Process | `Count(DisqualifiedBids) / Count(SubmittedBids)` — high outlier | `bids/details/status` | Continuous (Ratio) |
| R048 | `heterogeneous_supplier` | Behavioral | `Count(Distinct(item/classification/id[:2]))` — high outlier | `awards/items/classification/id` | Continuous (Count) |
| R058 | `heavily_discounted_winner` | Price | `(2ndLowestBid - WinningBid) / WinningBid >= UpperFence` | `bids/details`, `awards/value` | Continuous (Ratio) |

### 15.2 Academic Features (Wallimann & Imhof / Huber & Imhof)

Sources: arXiv:2004.05629 (ML incomplete cartels), arXiv:2104.11142 (CNN bid-rigging)

| Feature Name | Category | Formula | OCDS Fields | Risk Direction |
|:---|:---|:---|:---|:---|
| `rel_distance_mean` | Price | `(Bid - Mean(Bids)) / Mean(Bids)` | `bids/details/value` | Low Dist = Rigging |
| `rel_distance_median` | Price | `(Bid - Median(Bids)) / Median(Bids)` | `bids/details/value` | Low Dist = Rigging |
| `cv_bids` | Price | `StdDev(Bids) / Mean(Bids)` | `bids/details/value` | Low CV = Suspicious |
| `kurtosis_bids` | Price | `Kurtosis of bid amounts in a tender` | `bids/details/value` | High = Cluster |
| `skewness_bids` | Price | `Skewness of bid amounts in a tender` | `bids/details/value` | Deviations from Normal |
| `benford_d_stat` | Price | `KS D-stat vs Benford's 1st Digit Law` | `bids/details/value` | Higher = Fraud |
| `herd_count` | Network | `Count of bidders submitting together in last N months` | `bids/details/tenderers` | Higher = Collusion |
| `winner_reoccurrence` | Network | `Frequency of same (Winner, Buyer) pair in N months` | `buyer/id`, `awards/suppliers` | Higher = Favoritism |
| `losing_bidder_reoccurrence` | Network | `Frequency of same (Loser, Winner) co-appearance` | `bids/details/tenderers` | Higher = Cover Bid |
| `bid_rank_stability` | Behavioral | `Stability of bidder ranks across sequential tenders` | `bids/details` | High stability = Rigged |
| `price_gap_ratio` | Price | `(Bid_i+1 - Bid_i) / Mean(Bids)` | `bids/details/value` | Small Gaps = Rigging |

### 15.3 OECD / World Bank Indicators

| Feature Name | Category | Formula | OCDS Fields | Risk Direction |
|:---|:---|:---|:---|:---|
| `limited_procurement_usage` | Process | `Count(LimitedProcurement) / Total(Tenders) per Buyer` | `tender/procurementMethod` | Higher = Risk |
| `award_delay` | Process | `award/date - tender/tenderPeriod/endDate` | `award/date`, `tenderPeriod` | High delay = Payoff |
| `contract_extension_ratio` | Process | `Count(Extensions) / Total(Contracts)` | `contracts/statusDetails` | Higher = Negotiation |
| `sole_source_justification` | Process | `Binary(HasJustificationField) if method == 'limited'` | `tender/procurementMethod` | No justification = Risk |
| `high_discrepancy_value` | Price | `(contract/value - award/value) / award/value` | `contracts/value`, `awards/value` | High = Post-award fraud |
| `change_in_specs_after_notice` | Process | `Count(tender/amendments)` | `tender/amendments` | High = Directed |

### 15.4 Indonesian-Specific Features (BPKP / Audit Methodology)

| Feature Name | Category | Formula | OCDS Fields | Risk Direction |
|:---|:---|:---|:---|:---|
| `hps_proximity_winning` | Price | `(tender/value - award/value) / tender/value` | `tender/value`, `awards/value` | < 1% = Suspicious |
| `hps_proximity_bids` | Price | `(tender/value - bids/value) / tender/value` | `tender/value`, `bids/details` | Cluster at < 5% HPS |
| `submission_time_cluster` | Timing | `StdDev(bids/details/date) within a tender` | `bids/details/date` | Low StdDev = Automated |
| `submission_last_hour` | Timing | `Binary(bid/date within 1 hour of deadline)` | `bids/details/date` | High volume = Last-min |
| `metadata_id_match` | Network | `Check for identical Browser Fingerprints/IPs` | `bids/details/custom/ip` | Binary (1=Matched) |
| `npwp_verification_fail` | Behavioral | `Binary(Tenderer/id not in National Tax Database)` | `tenderers/id` (NPWP) | Binary (1=Risk) |
| `director_overlap` | Network | `Count overlapping directors in tenderer pool` | `tenderers/parties` | Higher = Collusion |
| `address_similarity` | Network | `LevenshteinDistance(Address_A, Address_B)` | `tenderers/address` | High similarity = Risk |
| `domicile_risk` | Network | `Binary(Winner address is residential/virtual office)` | `tenderers/address` | Binary (1=Risk) |
| `tender_splitting` | Process | `Count(Tenders) same CPV code, same location, same week` | `tender/items/classification` | Higher = Splitting |

### 15.5 Feature Summary by Category

| Category | Count | Key Signals |
|:---------|:------|:------------|
| Price | 15 | HPS proximity, CV, Benford's Law, Gap ratios, Bid distance |
| Competition | 12 | Single bidder, Disqualification patterns, Rank stability |
| Timing | 8 | Submission duration, Clustering, Late bids, Award delays |
| Process | 10 | Method usage, Extensions, Amendments, Discrepancies, Splitting |
| Network/Behavioral | 10 | Director overlap, Address similarity, Co-occurrence, CPV diversity |
| Indonesian/Audit | 10 | NPWP verification, Metadata matching, HPS clustering |
| **TOTAL** | **65** | |

### 15.6 Recommended Feature Tiers for Implementation

**Tier 1 — Core (must have, available from OCDS):** ~25 features
- All Cardinal indicators (11)
- `cv_bids`, `num_bidders`, `tender_duration`, `hps_proximity_winning`
- `limited_procurement_usage`, `award_delay`
- `submission_time_cluster`, `submission_last_hour`

**Tier 2 — Derived (requires aggregation across tenders):** ~20 features
- `winner_reoccurrence`, `herd_count`, `losing_bidder_reoccurrence`
- `bid_rank_stability`, `tender_splitting`
- Rolling vendor stats, HHI market concentration
- `benford_d_stat`, `kurtosis_bids`, `skewness_bids`

**Tier 3 — External/Advanced (requires external data or custom fields):** ~20 features
- `director_overlap`, `address_similarity`, `npwp_verification_fail`
- `metadata_id_match`, `domicile_risk`
- Network graph centrality features
- Document metadata fingerprinting

### 15.7 XGBoost Training Recommendations

- **Scale ratio features**: Normalize `hps_proximity` and `cv_bids`
- **Target encoding**: Use for `buyer/id` and `tenderers/id` on large datasets
- **SHAP explainer**: Use `TreeExplainer` on all 65 features — verify which red flags drive "Critical" classification
- **FastTreeSHAP**: Use LinkedIn's FastTreeSHAP for CPU-efficient SHAP on large datasets
- **Feature selection**: Start with Tier 1, add Tier 2, benchmark before adding Tier 3

---

## 16. OCDS Feature Engineering Code Patterns (Round 3)

### 16.1 Flattening OCDS JSON → Tabular

**Primary Tool: `flatten-tool`** by Open Data Services
- Repo: https://github.com/OpenDataServices/flatten-tool
- Handles roll-up of nested arrays (tenderers, awards) into a single row per `ocid`

```python
from flattentool import flatten

flatten(
    'input_ocds_release.json',
    output_name='flattened_data',
    output_format='csv',
    main_sheet_name='main',
    root_list_path='releases',
    rollup=True,               # Flattens nested arrays into the main row
    root_id='ocid'
)

import pandas as pd
df = pd.read_csv('flattened_data/main.csv')
```

### 16.2 Core Feature Extraction Pattern

```python
def extract_ocds_features(df):
    # Time-based features
    df['tender_start'] = pd.to_datetime(df['tender/tenderPeriod/startDate'])
    df['tender_end'] = pd.to_datetime(df['tender/tenderPeriod/endDate'])
    df['tender_duration'] = (df['tender_end'] - df['tender_start']).dt.days

    # Competition features
    df['num_bidders'] = df['tender/numberOfTenderers'].fillna(0)

    # Financial features
    df['relative_award_value'] = df['awards/0/value/amount'] / df['tender/value/amount']

    # Categorical encoding (Standard OCDS Codelists)
    df['method'] = df['tender/procurementMethod'].astype('category')
    return df
```

### 16.3 HHI Market Concentration

```python
def calculate_hhi(df, market_col='tender/items/0/classification/id'):
    market_totals = df.groupby(market_col)['awards/0/value/amount'].sum()
    supplier_totals = df.groupby([market_col, 'awards/0/suppliers/0/id'])['awards/0/value/amount'].sum()
    shares = (supplier_totals / market_totals.reindex(supplier_totals.index, level=0)) * 100
    hhi = (shares**2).groupby(level=0).sum()
    return hhi
```

### 16.4 Temporal / Rolling Window Features

```python
def get_rolling_vendor_stats(df):
    df = df.sort_values('tender/tenderPeriod/startDate')
    df['vendor_rolling_win_rate'] = df.groupby('awards/0/suppliers/0/id')['is_winner']\
                                      .transform(lambda x: x.rolling(window=10, min_periods=1).mean())
    return df
```

### 16.5 OCDS Data Quality Handling

- **Merge releases**: Use latest release per `ocid`: `df.sort_values('date').groupby('ocid').tail(1)`
- **Standardize IDs**: Combine `scheme` + `id` (e.g., `ID-NPWP-01.234.567.8-999.000`)
- **Missing arrays**: `flatten-tool` leaves columns null for missing arrays
- **Validation**: Use `lib-cove-ocds` (https://github.com/open-contracting/lib-cove-ocds)

### 16.6 Key Libraries & Repos

| Library | Purpose | URL |
|---------|---------|-----|
| flatten-tool | OCDS JSON → CSV/tabular | https://github.com/OpenDataServices/flatten-tool |
| notebooks-ocds | Red flag & indicator logic | https://github.com/open-contracting/notebooks-ocds |
| kingfisher-process | Large-scale OCDS collection/cleaning | https://github.com/open-contracting/kingfisher-process |
| lib-cove-ocds | Data quality validation | https://github.com/open-contracting/lib-cove-ocds |

---

## 17. Previous Attempt Analysis — vaskoyudha/LPSE-X (Round 4)

Source: https://github.com/vaskoyudha/LPSE-X

### 17.1 What Was Built (Surprisingly Advanced)

The previous attempt was a **full-stack production application**, not a simple notebook:

```
lpse-x/
├── backend/
│   ├── api/routes/       # FastAPI endpoints (tenders, oracle, graph, config)
│   ├── config/           # Runtime config & Dynamic Injection logic
│   ├── data/             # Ingestion (opentender.net OCDS) & SQLite storage
│   ├── features/         # 82-feature extractor based on ICW methodology
│   ├── graph/            # Leiden community detection & cartel scoring
│   ├── ml/               # XGBoost + Isolation Forest ensemble
│   ├── reports/          # IIA 2025 pre-investigation reports (Jinja2)
│   ├── schemas/          # Pydantic models for API validation
│   └── xai/              # Oracle Sandwich: SHAP, Anchors, DiCE, Benford
├── frontend/             # React 18 + TS + Vite + Tailwind CSS v4
├── data/                 # lpse_x.db (SQLite)
├── models/               # Pre-trained XGBoost (.ubj) and IForest (.pkl)
└── scripts/              # Batch processing & seeding scripts
```

### 17.2 Architecture — "Oracle Sandwich" (5-Layer XAI)

| Layer | Technique | Purpose | Quality |
|:------|:----------|:--------|:--------|
| L1 | **SHAP** (TreeSHAP) | Global/local feature importance | ✅ High |
| L2 | **DiCE** | Counterfactual explanations | ⚠️ Medium (cached/async) |
| L3 | **Anchors** | Rule-based If-Then explanations | ⚠️ Medium (slow, <5s SLA) |
| L4 | **Leiden Graph** | Cartel detection via co-bidding | ✅ High |
| L5 | **Benford's Law** | HPS digit distribution analysis | ✅ High |

Fault tolerant: each layer runs independently, one failure doesn't crash others.

### 17.3 Model Approach

- **Ensemble**: XGBoost (supervised, ICW weak labels) + Isolation Forest (unsupervised anomaly)
- **4-class output**: Aman, Perlu Pantauan, Risiko Tinggi, Risiko Kritis
- **82 features** per tender (ICW methodology-based)
- **Temporal split**: Train ≤2021, Val 2022, Test ≥2023 (leakage-safe ✅)
- **Optuna** hyperparameter tuning, optimizing Macro F1
- **SMOTE** for class imbalance handling
- **TimeSeriesSplit** for cross-validation

### 17.4 Cartel Detection (4-Signal Scoring)

| Signal | Weight | Method |
|:-------|:-------|:-------|
| Intra-bid Frequency | 30% | Co-bidding rate |
| Win Rotation | 30% | Shannon entropy of winner distribution |
| Price Similarity | 20% | Bid amount clustering |
| Geographic Overlap | 20% | Buyer/institution concentration |

### 17.5 Other Notable Features

- **Data source**: opentender.net OCDS data
- **Privacy**: SHA-256(NPWP) + last 4 digits — raw NPWPs never stored
- **Storage**: aiosqlite for async local SQLite (100% offline)
- **Dynamic Injection**: Runtime config changes via PUT /api/config/inject without restart
- **Audit Trail**: All config injections tracked for transparency
- **ONNX export**: onnx_export.py exists for lightweight inference

### 17.6 STRENGTHS (Reuse in New Plan)

1. ✅ **Temporal split** — Correct leakage prevention approach
2. ✅ **5-layer XAI** — Far beyond competition requirement (SHAP/LIME minimum)
3. ✅ **Cartel detection** — Graph-based community detection is differentiating
4. ✅ **82 features** — Comprehensive feature set based on ICW methodology
5. ✅ **4-class classification** — Matches Track C requirement exactly
6. ✅ **Optuna tuning** — Automated hyperparameter optimization
7. ✅ **SMOTE** — Class imbalance handling
8. ✅ **NPWP hashing** — Privacy-aware design
9. ✅ **Dynamic Injection** — Runtime reconfigurability
10. ✅ **Fault-tolerant XAI** — Each layer independent

### 17.7 FLAWS & WEAKNESSES (Fix in New Plan)

#### CRITICAL — Would Cause Score Loss

1. ❌ **Over-scoped for Phase 2**: Full backend/frontend/API is Phase 3 scope. Phase 2 requires ONLY training.ipynb + inference.ipynb + proposal. All the FastAPI/React work is WASTED EFFORT for Phase 2 submission.

2. ❌ **Weak ground truth (ICW weak labels)**: XGBoost trained on ICW-generated weak labels. If ICW scores are biased or incomplete, model mirrors those biases. Jury will question label quality.

3. ❌ **Scraping fragility (pyproc/opentender.net)**: LPSE sites have Cloudflare protection. Scraping is unreliable. Fallback exists but data acquisition is the #1 risk.

4. ❌ **No dedicated proposal writing**: The competition is 60%+ scored on the PROPOSAL (Bab 1-4), not the code. The previous attempt prioritized code over proposal.

5. ❌ **Anchors layer too slow**: <5s SLA for Anchors is risky. CPU-only inference constraint (G2) makes this worse. Could breach time limits during jury evaluation.

#### MAJOR — Would Reduce Competitiveness

6. ⚠️ **Sequential XAI computation**: Oracle Sandwich layers compute mostly sequentially. Parallelizing would cut inference time significantly.

7. ⚠️ **XAI outputs too technical**: SHAP values and anchors rules are math, not stories. Jury wants "Bahasa Indonesia executive summaries" not raw numbers.

8. ⚠️ **No Bab 3 constraint-by-constraint mapping**: Track C requires Bab 3 to address EACH constraint (C-C1 through C-C5) individually. Previous approach likely treated this as a general essay.

9. ⚠️ **Bab 4 under-prioritized**: Architecture section = 23.3% of total score (highest weighted deliverable). Previous attempt's architecture is good but needs explicit jury-facing documentation.

10. ⚠️ **Static graph analysis**: Leiden runs on a static co-bidding graph. Temporal graph evolution would detect changing cartel behavior.

#### MINOR — Nice to Fix

11. 💡 **No ONNX inference in main pipeline**: onnx_export.py exists but isn't the default inference path. ONNX would be faster for CPU-only demos.
12. 💡 **No Benford's Law feature in XGBoost**: Benford analysis exists as separate XAI layer but not as a training feature.
13. 💡 **Reports are static Jinja2**: Not interactive. Fine for Phase 2 but weak for Phase 3 demo.

### 17.8 Key Decision for New Plan: Phase 2 Focus

**The #1 lesson from the previous attempt: SCOPE DISCIPLINE.**

Phase 2 deliverables (ALL that matters before April 11 public submission cutoff):
1. `training.ipynb` — Data loading, preprocessing, feature engineering, model training, evaluation
2. `inference.ipynb` — Load model, predict on new data, generate explanations
3. `Proposal (Bab 1-4)` — Written document (60%+ of score)
4. `requirements.txt` + environment docs
5. Optional: `data/` folder with sample data

**NOT Phase 2** (cut from plan): FastAPI backend, React frontend, Docker, database, live scraping, report generation, dynamic injection. ALL of this is Phase 3.

### 17.9 What to Carry Forward vs. Rebuild

| Component | Decision | Rationale |
|:----------|:---------|:----------|
| XGBoost + 4-class | **KEEP** | Correct approach, matches Track C |
| Temporal split | **KEEP** | Leakage-safe, correct |
| 82 ICW features | **ADAPT** → 65 features (our spec) | Better sourced, OCDS-mapped |
| SHAP (TreeSHAP) | **KEEP** | Required by C-C1 |
| DiCE counterfactuals | **KEEP** (simplified) | Level 3 XAI differentiator |
| Anchors | **CUT** for Phase 2 | Too slow for CPU, add in Phase 3 |
| Leiden cartel detection | **SIMPLIFY** → co-bidding features | Graph features yes, full Leiden in Phase 3 |
| Benford's Law | **KEEP** as feature + visualization | Strong differentiator |
| Optuna tuning | **KEEP** | Automated, jury-impressive |
| SMOTE | **KEEP** | Class imbalance handling |
| FastAPI/React/SQLite | **CUT** entirely | Phase 3 scope |
| NPWP hashing | **KEEP** concept | Privacy-aware design for proposal |
| Dynamic Injection | **CUT** | Phase 3 scope |
| ONNX export | **ADD** to inference.ipynb | CPU-speed advantage |

---

## 18. Key Decisions (Confirmed by User)

### 18.1 Ground Truth Strategy
- **Decision**: ICW Weak Labels (same as previous attempt)
- **Rationale**: ICW methodology is well-documented and defensible. Acknowledge limitation in Bab 3 (ethics section) and add confidence calibration.
- **Mitigation**: Explicitly document label quality limitations. Add confidence scores. Discuss in Bab 3 as ethical consideration.

### 18.2 Test Strategy
- **Decision**: Full Test Suite (pytest)
- **Infrastructure**: Need to set up pytest from scratch (no existing test infra)
- **Scope**: Unit tests for feature engineering, model training, inference pipeline
- **Impact**: Adds ~1 day of setup + testing tasks to the plan, but strengthens code quality deliverable (3.3% of score) and catches bugs early
- **Agent QA**: Still mandatory for ALL tasks regardless of test suite

### 18.3 Phase 2 Scope Lock
- **IN SCOPE**: training.ipynb, inference.ipynb, proposal (Bab 1-4), requirements.txt, pytest test suite, sample data
- **OUT OF SCOPE**: FastAPI backend, React frontend, Docker, SQLite, live scraping, dynamic injection, report generation — all Phase 3

---

## 19. Strategic Brainstorm Session (Pre-Plan Rethink)

### 19.1 Context Update
- **Team**: 5 people (new — previous attempt was solo practice)
- **Previous attempt**: Solo practice run, never submitted. GitHub repo is reference, not submission.
- **Top worries**: (1) Technical differentiation, (2) Proposal writing quality, (5) Model performance
- **Mode**: Full comprehensive rethink — strategy, architecture, proposal, technical fixes, phase continuity

### 19.2 Team & Role Context
- **Team size**: 5 people (mostly technical)
- **Proposal writing**: AI-assisted (Prometheus helps draft, humans review/polish)
- **Previous attempt**: Solo practice run, never submitted. Code is reference only.
- **Data status**: Sources identified, not yet downloaded

### 19.3 Differentiation Strategy — CONFIRMED
- **Primary**: Narrative XAI (Bahasa Indonesia human-readable explanations) + Indonesian Fraud Specialization
- **Secondary**: Cartel co-bidding patterns as XGBoost features (full graph viz deferred to Phase 3)
- **NOT in Phase 2**: Full Oracle Sandwich, Anchors, Leiden visualization, graph UI

### 19.4 Headline Pitch
"LPSE-X doesn't just detect fraud — it explains fraud in language an auditor understands, trained on Indonesian-specific corruption patterns that foreign models miss."

### 19.5 Refined Architecture

Pipeline: OCDS JSON → flatten-tool → pandas → temporal split (BEFORE preprocessing) → 65 features (Tier 1+2) → XGBoost multi:softprob → SHAP + DiCE → Narrative Generator → Structured JSON Output

Phase 2 deliverables: training.ipynb + inference.ipynb + proposal (Bab 1-4) + pytest tests + requirements.txt

XAI Pyramid (3 levels):
- Level 1 (APA): SHAP — which features drive this prediction?
- Level 2 (BAGAIMANA JIKA): DiCE counterfactuals — what could change the risk?
- Level 3 (JADI APA): Narrative Generator — Bahasa Indonesia explanation for auditors

### 19.6 Work Streams (for 5-person team)
- WS1: Data Pipeline (download, flatten, clean, split) → Day 1 start
- WS2: Feature Engineering (65 features, 3 tiers) → depends on WS1
- WS3: Model Training & Eval (XGBoost, Optuna, SMOTE, ONNX) → depends on WS2
- WS4: XAI & Inference (SHAP, DiCE, Narrative Gen) → depends on WS3
- WS5: Proposal Writing (Bab 1-4, diagrams) → Day 1 parallel
- WS6: Testing (pytest, 40 tests) → depends on WS2-WS4
- WS7: Environment & Docs → final integration

### 19.7 Feature Feasibility
- Tier 1 (25 features): Definitely computable from OCDS ✅
- Tier 2 (20 features): Computable WITH bid-level data ⚠️ (key risk)
- Tier 3 (20 features): Requires external data ❌ (Phase 3 future work)
- Realistic Phase 2 target: 25-35 features

### 19.8 Technical Fix List (13 fixes from previous attempt flaws)
See Section 17.7 for full list. Key fixes:
1. Scope to Phase 2 only (no backend/frontend)
2. Confidence calibration for ICW labels
3. Pre-downloaded data (no live scraping)
4. Cut Anchors, add Narrative Generator
5. ONNX as default inference
6. Bab 3 point-by-point constraint mapping

### 19.9 Testing Strategy
- Framework: pytest
- Structure: 6 test files (~40 tests)
- Priority: P0 (features, temporal split, e2e) → P1 (SHAP format, narrative) → P2 (counterfactuals, edge cases)
- Estimated effort: 1-1.5 days

### 19.10 Data Acquisition Plan (Verified March 30, 2026)

**Primary source (confirmed accessible):**
- OCDS Indonesia (ICW) via OCP Data Registry: data.open-contracting.org/en/publication/101
- Format: JSON (OCDS Release Package), annual bulk files
- Records: 500K+ across 2018-2025
- Bid-level data: YES (awards + bids objects)
- HPS/tender value: YES
- Auth: None required
- Download time: ~30-60 minutes

**Secondary source (gap filler for 2025-2026):**
- pyspse library (pip install pyspse, v0.1.0, Dec 2025)
- Scrapes SPSE v4.5 interfaces
- Bid-level data: YES
- Use: fill gaps in most recent data

**Verification/cross-reference:**
- opentender.net: ICW's gold standard, includes "Potensi Fraud" scores
- Export limited to 5-10K rows, redirects to OCDS registry for bulk

**NOT useful for Phase 2:**
- LKPP portal: aggregated only, no bid-level data
- Kaggle: <10K rows, no bid-level for 2018-2024
- SIRUP API: public endpoint(s) appear to exist, but it remains planning-oriented and not sufficient for bid-level feature extraction

**Data strategy:**
1. Day 1: Download OCDS Indonesia bulk from OCP Registry (primary)
2. Day 1: pip install pyspse, scrape recent 2025 tenders if needed
3. Day 2: Flatten via flatten-tool, temporal split, verify bid-level fields
4. Day 2: Check which of 65 features are actually computable → finalize feature list
5. Fallback: if OCDS data quality is poor, generate synthetic data based on distributions

---

## 20. Model Performance Risk Analysis (Deep Dive)

> **Source**: Dedicated librarian research — brutally honest assessment of our XGBoost + weak labels approach.

### 20.1 The Weak Label Trap: Noise and SHAP Reliability

**Core problem**: ICW heuristic labels are a PROXY for fraud, not ground truth.

- **Accuracy degradation**: With ~20% heuristic error rate, model's theoretical upper bound is significantly capped. In multi-class settings, noise "blurs" boundaries between adjacent classes (Pantauan vs Tinggi especially).
- **SHAP circular logic (CRITICAL RISK)**: SHAP values explain why the model predicted the *heuristic label*, NOT why the transaction is actually fraudulent.
  - If heuristic says "Single Bidder = Fraud," SHAP will assign high importance to `single_bidder` even if heuristic is logically flawed
  - Creates "circular logic loop" — model learns to mimic heuristics, SHAP confirms it did so, zero new insight
- **Mitigation strategy**: Frame this as "risk profiling based on internationally validated procurement red flags" NOT "fraud detection." We are flagging anomalous patterns, not making legal determinations.

**Literature reference**: Snorkel framework (Ratner et al.) — use heuristics as "labeling functions" to generate probabilistic labels, train on probabilities as weights.

### 20.2 Class Imbalance: 4-Class Realities

**Expected distribution**: ~60/25/10/5 (Aman/Pantauan/Tinggi/Kritis)

- **DO NOT use SMOTE** for 10K-50K samples with tree models — creates synthetic points that overfit on noisy boundaries
- **Recommendation**: Use **Focal Loss** (custom XGBoost objective) or **class weights** (sample_weight parameter)
- **Focal Loss**: `-(1-pt)^gamma * log(pt)` — forces model to focus on hard-to-classify Kritis samples
- **Realistic ceiling**: Macro F1 of 0.70-0.75 before label noise makes metric meaningless

### 20.3 Feature Engineering Risks

**Multicollinearity**: XGBoost handles correlated features better than linear models, BUT randomly splits importance between them. If `hps_proximity` and `cv_bids` are 90% correlated, SHAP dilutes both — neither looks critical.
- **Fix**: Correlation audit → drop or combine features with Pearson > 0.85

**Data Leakage (HIGH RISK)**: Features like `winner_reoccurrence` encode the label if calculated over entire dataset. Must use **expanding window** (look-back only, no future data).
- **Audit protocol**: Every aggregated feature must use `txn_date <= current_row_date` filter

**Feature power ranking** (from literature):
1. `hps_proximity` — strongest signal for bid-rigging (Kritis)
2. `single_bidder` — often legal edge case (Pantauan), not necessarily fraud
3. `winner_reoccurrence` — strong if properly windowed
4. Benford deviation — weak individually, strong in ensemble

### 20.4 Calibration: The Silent Killer

**Problem**: Platt scaling and Isotonic regression rely on clean ground truth. With noisy heuristic labels, calibration simply makes model "confident about its mistakes."

**Better alternative**: **Temperature Scaling** on a small clean hold-out set.
- Even 100 human-verified samples (50 Kritis, 50 Aman) would dramatically improve calibration
- Without clean labels, "calibration" is a mathematical illusion

**Our defense for jury**: 
- Acknowledge limitation transparently
- Show calibration curves on validation set
- Frame confidence as "model certainty about risk indicators" not "probability of fraud"

### 20.5 ONNX Conversion Risks

**Prediction drift**: XGBoost → ONNX is stable for `tree_method='hist'`, BUT categorical features (XGBoost 1.5+) often fail during conversion.
- **Fix**: Use `OneHotEncoder` explicitly in sklearn pipeline, not XGBoost native categoricals

**SHAP compatibility (CRITICAL)**:
- ⚠️ **TreeSHAP does NOT work on ONNX models** — ONNX loses tree structure needed for O(TLD²) algorithm
- Would require KernelSHAP (100x slower, approximation only)
- **Strategy**: Use ONNX only for fast inference in inference.ipynb. Keep original XGBoost model for SHAP explanations. Load both.

**Speed**: ONNX Runtime (CPU) is 2-4x faster than native XGBoost for single-row inference, slower for large batch.

### 20.6 Brutal Recommendations (Incorporated into Strategy)

| # | Recommendation | Our Response |
|---|---------------|-------------|
| 1 | Stop treating heuristics as ground truth | ✅ Frame as "risk profiling," use probabilistic labels (Snorkel-style weights) |
| 2 | SHAP is a mirror, not a microscope | ✅ Acknowledge in Bab 3; frame as "auditing risk indicators" |
| 3 | Ditch SMOTE | ✅ Use Focal Loss or class weights instead |
| 4 | Audit for leakage | ✅ Expanding window for all temporal features, strict temporal split |
| 5 | Clean 100 samples | ⚠️ Time-constrained — attempt 50 Kritis + 50 Aman manual review if time allows |

### 20.7 Realistic Performance Expectations

| Metric | Optimistic | Realistic | Pessimistic |
|--------|-----------|-----------|-------------|
| Macro F1 | 0.75 | 0.65-0.70 | 0.50-0.55 |
| Kritis Recall | 0.80 | 0.60-0.70 | 0.40-0.50 |
| Aman Precision | 0.90 | 0.80-0.85 | 0.70 |
| Calibration (ECE) | 0.05 | 0.10-0.15 | 0.20+ |

**Jury framing**: "Our model achieves Macro F1 of X on temporally-split validation data using internationally validated procurement risk indicators. We acknowledge label noise limitations and present calibration analysis in Section 4.3."

---

## 21. Strategic Brainstorming: Unresolved Deep Questions

> These are the hard questions that require "1000x deeper" thinking before committing to a plan.

### 21.1 The Label Honesty Dilemma

**Question**: How aggressively should we acknowledge weak label limitations in the proposal?

**Option A — Full Transparency**: Dedicate a subsection in Bab 3 to "Label Quality & Limitations." Show we understand the problem deeply. Risk: jury thinks our model is unreliable.

**Option B — Confident Framing**: Present ICW methodology as "internationally validated risk scoring" (it IS used by Transparency International). Mention calibration as a strength. Risk: jury who understands ML will see through it.

**Option C — Hybrid (RECOMMENDED)**: Acknowledge the challenge in 1-2 sentences, then immediately pivot to our mitigations (Platt scaling, temporal validation, Snorkel-inspired probabilistic weighting). Show MATURITY not WEAKNESS.

### 21.2 The ONNX Dual-Model Question

**New insight from research**: We CANNOT use TreeSHAP on ONNX. This means:
- `inference.ipynb` loads ONNX for fast prediction
- `inference.ipynb` ALSO loads original XGBoost .ubj for SHAP explanations
- Two model files in submission

**Is this a constraint violation?** Need to check: does the guidebook specify "one model file"?
**Is this an advantage?** Could frame as "production architecture pattern — fast inference + detailed explanations"

### 21.3 The Feature Count Sweet Spot

**Previous attempt**: 82 features (over-engineered, many were noise)
**Research says**: 25-35 is sweet spot for interpretability + performance

**Deeper question**: Should we optimize for:
- **Model performance** (more features, higher F1, harder to explain)
- **Explainability quality** (fewer features, cleaner SHAP, better narratives)
- **Jury impression** (show we COULD do 65 but CHOSE 30 for interpretability — demonstrates maturity)

**Recommendation**: 25-30 features, with explicit "Feature Selection Rationale" section in Bab 2 showing we evaluated 65 and selected based on: availability, correlation analysis, and interpretability value.

### 21.4 The Narrative XAI Depth vs Polish Trade-off

**Current design**: Template-based Bahasa Indonesia narratives
**Question**: How much time should we invest in narrative quality vs model quality?

Given scoring weights:
- 58.4% is WRITING (proposal)
- 33.3% is CODE (notebooks + model)
- Narrative XAI is the INTERSECTION — it appears in BOTH

**Hypothesis**: Narrative XAI is our highest-ROI investment because:
1. It differentiates us from every team doing raw SHAP plots
2. It scores in BOTH writing (Bab 4 architecture) and code (inference.ipynb)
3. It directly addresses "Explainable" in the track title
4. It's memorable — jury will remember "the team that explains in Indonesian"

### 21.5 The Phase 2→3 Bridge Architecture

**Risk**: If Phase 2 notebooks are too "notebook-y" (messy, inline), Phase 3 migration to web app is painful.
**Counter-risk**: If we over-modularize notebooks, they look like a web app framework crammed into Jupyter — jury might question why we didn't just build a web app.

**Sweet spot**: Functions are modular and importable, but notebook FLOW is linear and readable. Each function has a clear docstring. No class hierarchies in notebooks.

### 21.6 What Could Make Us LOSE

**Disqualification risks:**
1. ❌ Using cloud API → DQ (we don't, but must be explicit in Bab 3)
2. ❌ Inference timeout on CPU → hard constraint violation
3. ❌ Missing required deliverable file

**Competitive loss risks:**
1. 🔴 Another team has REAL labeled data (not heuristic)
2. 🔴 Another team has a more impressive XAI visualization
3. 🔴 Another team writes a better proposal (58% of score!)
4. 🔴 Our model F1 is below 0.60 (jury perceives as not working)
5. 🟡 Jury doesn't understand Bahasa Indonesia narratives (unlikely — Indonesian judges)
6. 🟡 Our feature count seems low compared to teams claiming 100+ features

### 21.7 What Could Make Us WIN

**Differentiation vectors:**
1. 🟢 Only team with human-readable Bahasa Indonesia explanations
2. 🟢 Only team that acknowledges AND mitigates weak label limitations
3. 🟢 Production-ready architecture thinking (ONNX + modular + importable)
4. 🟢 Domain expertise in Indonesian procurement (LKPP, SPSE, OCDS)
5. 🟢 Rigorous validation (temporal split, calibration curves, leakage audit)
6. 🟢 Proposal quality (AI co-written, structured, addresses every scoring rubric point)

---

## 22. Competitive Intelligence & Differentiation Analysis

> **Source**: Librarian research on past Find IT winners, narrative XAI in literature, and XAI judging patterns.

### 22.1 Past Find IT Winners (2022-2024) — Gap Analysis

| Year | Winner | Approach | XAI Level |
|------|--------|----------|-----------|
| 2022 | Sipending Team | XGBoost + CatBoost + LGBM VotingRegressor (Salary Prediction) | Global Feature Importance bar charts only |
| 2023 | Oh Data Euy | CatBoost + Optuna tuning | ZERO XAI implementation |

**Critical insight**: Past winners focused on **accuracy** with basic EDA. None implemented:
- Local explanations (SHAP per-instance)
- Counterfactual "what-if" scenarios
- Natural language explanations
- Bahasa Indonesia output

**Our opportunity**: We are likely the ONLY team providing narrative XAI in Bahasa Indonesia. This is a massive differentiation gap.

### 22.2 Academic Backing for Narrative XAI

**Must-cite papers for proposal:**
1. **Martens et al. (2025)** — "Tell Me a Story! Narrative-Driven XAI with Large Language Models" (Decision Support Systems) — argues SHAP values too complex for non-technical stakeholders, proposes narrative XAI
2. **Zytek et al. (2024)** — "Explingo" — maps SHAP/LIME importance to natural language templates, increases user trust and "mental model" alignment
3. **DiCE counterfactuals** — considered "Gold Standard" for human-centric AI (provides roadmap for change)

**Template-based approach justification**: We use templates NOT LLMs, which:
- Satisfies CPU-only constraint
- Ensures deterministic, reproducible explanations
- Avoids hallucination risk of LLM-based narratives
- Can cite Explingo as academic precedent

### 22.3 XAI Hackathon Judging Patterns (2026)

From IJCNN 2026 XAI Challenge and similar competitions, judges value:
1. **Fidelity**: Does narrative accurately reflect model's internal logic?
2. **Human-in-the-loop utility**: Can a non-data-scientist USE the explanations?
3. **Cultural context**: Bahasa Indonesia is a NECESSITY, not a feature, for Indonesian procurement context
4. **Actionability**: "What should the auditor DO with this information?"

### 22.4 Positioning Strategy

| Feature | Competitors (Past Winners) | **LPSE-X** |
|---------|--------------------------|-----------|
| Model | Ensemble (XGB+LGBM+RF) | XGBoost (focused, optimized) |
| Explanation | Global Feature Importance bar chart | **Local SHAP + DiCE counterfactual** |
| Output | Probability score (0.82) | **Narrative story in Bahasa Indonesia** |
| Language | English/Technical | **Bahasa Indonesia (auditor-friendly)** |
| Impact | High accuracy | **High auditability & trust** |
| Framing | "What" (prediction) | **"Why" (narrative) + "How to fix" (counterfactual)** |

**Pitch line for proposal**: "Sementara solusi sebelumnya mengoptimalkan 'apa' (prediksi), LPSE-X memberikan 'mengapa' (narasi) dan 'bagaimana memperbaiki' (kontrafaktual), menjadikannya satu-satunya solusi yang siap untuk deployment nyata di pemerintahan Indonesia."

### 22.5 Procurement Fraud Detection Industry Context

- Most existing fraud detection in Indonesia (BPK/KPK) is **rule-based** with high false positive rates
- Winning ML approaches focus on **relational features**: tender splitting, bid rigging (circular patterns), vendor networks
- Position as **Decision Support System**, NOT a verdict system
- "Model ini bukan pengganti auditor, melainkan 'Lensa Pembesar' yang mendeteksi anomali"

---

## 23. Proposal Writing Strategy (Deep Analysis)

> **Source**: Librarian research on winning hackathon proposal patterns for mixed AI/SWE/Architect jury panels.
> **Key insight**: 58.4% of score is writing. Treat proposal as "System Design & Product Manifesto," not a technical report.

### 23.1 The Mixed Jury Strategy

Three distinct "appetites" to satisfy simultaneously:

| Jury Panel | What They Want | Our Strategy |
|-----------|---------------|-------------|
| **AI Expert** | Validation rigor, methodology depth | Weak label handling, SHAP fidelity, temporal validation |
| **Software Engineer** | Feasibility, integration, clean code | System architecture diagram, SPSE workflow integration, modular design |
| **Product/System Architect** | Actionability, real-world impact | Narrative XAI demo, auditor workflow, deployment path |

### 23.2 Writing Time Allocation (Score-Weighted)

| Section | Weight | Time % | Strategy | Key "Winning" Device |
|---------|--------|--------|----------|---------------------|
| **Bab 4** (Implementation) | 23.3% | **35%** | Most critical. Center the Narrative Engine. | Mock audit report showing XAI output |
| **Bab 3** (Solution/Ethics) | 16.7% | **25%** | Point-by-point constraint compliance | Full-stack architecture diagram |
| **Bab 2** (Methodology) | 13.4% | **25%** | Feature engineering depth | Data Flow Diagram + SHAP summary plot |
| **Bab 1** (Background) | 5.0% | **15%** | Sharp and emotional | "Fraud Gap" statistic in Indonesia |

**Rule**: Draft Bab 4 FIRST. If the output isn't compelling, the model doesn't matter.

### 23.3 The Three-Layer Explanation Pattern

For EVERY technical claim in the proposal, layer three levels:
1. **Technical Fact**: "Kami menggunakan XGBoost dengan monotone constraints."
2. **Product Value**: "Ini memastikan bahwa semakin tinggi 'Variance Harga', skor risiko tidak pernah menurun."
3. **Human Impact**: "Ini membangun kepercayaan auditor karena logika model konsisten dengan hukum pengadaan."

### 23.4 Required Visual Aids

| Chapter | Visual | Purpose |
|---------|--------|---------|
| Bab 1 | Infographic: fraud loss statistics in Indonesia | Emotional hook |
| Bab 2 | SHAP Summary Plot (Bahasa Indonesia labels) | Show feature engineering depth |
| Bab 2 | Data Flow Diagram (OCDS → features → model) | Show methodology rigor |
| Bab 3 | System Architecture Diagram | Show "Predictive Engine" vs "Explanation Wrapper" |
| Bab 3 | Constraint compliance table (point-by-point) | Show thoroughness |
| Bab 4 | **XAI UI Mockup** (what auditor sees) | KILLER visual — shows real-world impact |
| Bab 4 | Calibration curve | Show confidence is trustworthy |
| Bab 4 | Narrative XAI example output (full Bahasa Indonesia) | Demonstrate the differentiator |

### 23.5 Weak Label Framing (The Persuasive Pivot)

**DO NOT hide this. Use it as strategic strength:**

> "Model ini memanfaatkan pendekatan Semi-Supervised Heuristic Labeling. Kami tidak mengklaim memprediksi 'Kesalahan,' melainkan 'Risiko Anomali.' Melalui XAI, kami memberdayakan auditor untuk memverifikasi heuristik ini, menjadikan model sebagai Decision Support Tool, bukan hakim akhir."

**Metaphor**: "Lensa Pembesar" (Magnifying Glass) — model detects anomalies based on historical patterns, narrative XAI serves as initial justification for further investigation.

### 23.6 Common Proposal Mistakes to AVOID

| Mistake | Why It Kills | Our Counter |
|---------|-------------|-------------|
| Buzzword salad ("Deep Learning", "LLM") | Judges see through it | Justify XGBoost explicitly (Grinsztajn 2022: trees > DL on tabular) |
| Claiming 99% accuracy | Incredible with noisy data | Focus on Recall + calibration curves |
| XAI as afterthought | Track is literally called "Explainable Oracle" | XAI is CENTER of Bab 4 |
| Missing feasibility | Architects want real-world viability | CPU latency budget, ONNX deployment path |
| Over-long Bab 1 | 5% weight, don't over-invest | 1-2 pages max, punchy statistics |

### 23.7 The "Auditor Test"

**Internal quality gate before submission**: Have a non-technical teammate read Bab 4. If they don't understand WHY a project is flagged as fraud, rewrite the narrative XAI section.

---

## 24. DiCE Counterfactual Analysis (Deep Dive)

> **Source**: Two librarian research passes — compatibility, speed, alternatives, pre-computation, and fallback strategies.

### 24.1 DiCE + XGBoost Compatibility

- ✅ DiCE-ML officially supports multi-class classification via `sklearn` backend
- ✅ XGBoost's `XGBClassifier` (uses `softprob` by default for multiclass) is compatible
- ⚠️ **Feature ordering friction**: DiCE's `dice_ml.Data` object must match exact feature order expected by XGBoost booster
- ⚠️ If using raw `Booster` instead of `XGBClassifier`, must wrap in custom class with `predict_proba`
- **Backend**: Use `genetic` or `random` search method for tree models. `KDTree` (model-agnostic) is most stable for non-differentiable models.

### 24.2 DiCE Speed Benchmarks (CPU)

For single instance, 25-35 features, modern laptop CPU:

| Method | Latency | Notes |
|--------|---------|-------|
| `genetic` | 1.5-4.0s (up to 8s worst case) | More robust, slower. total_CFs matters hugely. |
| `random` | 1.5-3.0s | Less diverse but faster |
| `KDTree` (model-agnostic) | 0.5-1.5s | Most stable for tree models |

**Key optimization levers:**
1. **Limit `total_CFs`**: Request 2 counterfactuals, not 5 (default)
2. **`features_to_vary`**: Restrict to top 10 actionable features (not all 30)
3. **`permitted_range`**: Hard-code ranges to reduce search space
4. **`proximity_weight`**: Lower for faster convergence (less "realistic")

### 24.3 DiCE Alternatives (Speed-Ranked)

| Library | Latency | Quality | CPU-Friendly? |
|---------|---------|---------|--------------|
| **NICE** (Nearest Instance CF) | **< 200ms** | Good (search-based on training data) | ✅ Best |
| Pre-computed + NN lookup | **< 50ms** | Moderate (may feel generic) | ✅ Fastest |
| **DiCE** (genetic) | 2-4s | High (diverse, optimal) | ⚠️ Acceptable |
| **Alibi** (CEM/Proto) | 2-5s | High (prototypical) | ⚠️ Similar to DiCE |
| SHAP-based hybrid | ~10ms (post-SHAP) | Moderate (top-feature perturbation) | ✅ Trivial |

### 24.4 SHAP-Based Counterfactual Fallback (CRITICAL INSIGHT)

**Can derive counterfactuals from SHAP without DiCE:**
1. Identify top 2-3 features with negative SHAP values (pushing prediction away from target)
2. Perturb those features toward mean/median of the target class
3. Check if prediction flips
4. **Latency**: ~10ms post-SHAP computation
5. Not "optimal" or "diverse" like DiCE, but GUARANTEED fast

**JP Morgan's `cf-shap`**: Uses SHAP importance to weight counterfactual generation — hybrid approach.

### 24.5 Strategic Decision: DiCE vs Alternatives

**RECOMMENDED APPROACH — Tiered Counterfactual Strategy:**

```
Primary: DiCE (genetic, total_CFs=2, features_to_vary=top_10)
  → Expected: 2-3 seconds
  → Timeout: 5 seconds

Fallback: SHAP-based perturbation (if DiCE times out)
  → Expected: 10ms
  → Always succeeds
  → Less diverse but still meaningful
```

**Narrative template handles BOTH**: The Bahasa Indonesia narrative template doesn't care HOW the counterfactual was generated — it just needs "if feature X changed from A to B, risk drops from C to D."

### 24.6 Code Pattern

```python
import dice_ml

# Initialize
d = dice_ml.Data(dataframe=train_df, continuous_features=continuous_cols, outcome_name='risk_class')
m = dice_ml.Model(model=xgb_classifier, backend='sklearn')
exp = dice_ml.Dice(d, m, method='random')

# Generate with timeout
cf = exp.generate_counterfactuals(
    query_instance, 
    total_CFs=2, 
    desired_class="opposite",
    features_to_vary=top_10_features  # Restrict search space
)
```

---

## 25. CPU Inference Latency Budget (Deep Dive)

> **Source**: Librarian research on TreeSHAP, FastTreeSHAP, DiCE, and ONNX benchmarks for CPU-only hackathon demo.

### 25.1 Full Pipeline Latency Budget

For **300 trees, 30 features, 4-class, modern laptop CPU** (i7/M2):

| Stage | Tool | Latency (ms) | Notes |
|-------|------|-------------|-------|
| 1. Inference | ONNX Runtime | 5-15 | Extremely fast |
| 2. SHAP Explanation | TreeSHAP | 50-250 | Multi-class adds N_classes multiplier |
| 3. Counterfactual | DiCE (genetic) | 2,500-8,000 | **THE BOTTLENECK** |
| 4. Narrative Generation | Python templates | 1-5 | Negligible |
| **TOTAL** | | **~3-8 seconds** | **Fits within 10s demo budget** |

### 25.2 TreeSHAP Deep Analysis

- **Single instance**: ~100ms for 300 trees (C++ optimized)
- **Multi-class impact**: XGBoost implements multi-class as N separate trees per iteration. 4 classes → 4x multiplier → ~400ms
- **FastTreeSHAP** (LinkedIn): 1.5-2x speedup for single instance (~200ms for 4-class)
  - Supports `.ubj` and `.json` models
  - Modest single-instance gain (30-50% reduction); excels at batch (up to 100x)
- **Best setting**: `feature_perturbation="tree_path_dependent"` — no background dataset needed, significantly faster for single instances

### 25.3 Optimization Strategies

| Strategy | Speed Gain | Accuracy Cost | Recommendation |
|----------|-----------|---------------|---------------|
| Reduce trees: 500→150 | 3.3x faster SHAP | < 2% accuracy loss | ✅ DO THIS |
| FastTreeSHAP | 1.5-2x faster | None | ✅ DO THIS |
| `tree_path_dependent` | ~2x faster than interventional | Slightly different values | ✅ DO THIS |
| Parallel SHAP + DiCE | Minor (DiCE dominates) | None | ⚠️ Optional |
| Limit DiCE to 2 CFs | 2-3x faster | Fewer diverse options | ✅ DO THIS |
| SHAP fallback for CFs | 100x faster than DiCE | Less diverse | ✅ ALWAYS HAVE |

### 25.4 Optimized Target Configuration

```
Model: 150-200 trees, max_depth=6
Explainer: FastTreeSHAP, tree_path_dependent
DiCE: genetic, total_CFs=2, restricted features
Narrative: Python templates

Expected total: < 4 seconds on standard CPU
```

### 25.5 Worst-Case Fallback Chain

```
IF DiCE completes in < 5s:
  → Full pipeline: ONNX + SHAP + DiCE + Narrative (~3-4s)

IF DiCE times out at 5s:
  → Fallback: ONNX + SHAP + SHAP-based CF + Narrative (~0.5s)
  → Still produces: risk class, confidence, top-5 features, 2 counterfactuals, narrative

IF even SHAP is slow (shouldn't happen):
  → Emergency: ONNX + top feature weights from model + narrative (~0.1s)
  → Degraded but functional
```

### 25.6 Jury Demo Strategy

- **Normal demo**: Show full pipeline with DiCE (~4s) — impressive explanation quality
- **Speed demo**: Show SHAP-only fallback (~0.5s) — proves CPU constraint compliance
- **Stress test**: Run 10 sequential predictions — shows reliability and consistent timing
- **Frame**: "Our architecture gracefully degrades — always produces explanations, never fails to respond"

---

## 26. Strategic Brainstorming: Updated Decision Matrix

> After all deep dives, here's the consolidated decision state.

### 26.1 Architecture Decisions (CONFIRMED)

| Component | Decision | Confidence | Risk |
|-----------|---------|------------|------|
| Model | XGBoost multi:softprob, 4-class | HIGH | Low — proven for tabular |
| Trees | 150-200 (not 500) | HIGH | Low — speed/accuracy sweet spot |
| XAI Layer 1 | FastTreeSHAP (tree_path_dependent) | HIGH | Low — well-benchmarked |
| XAI Layer 2 | DiCE (genetic, 2 CFs) + SHAP fallback | MEDIUM | Medium — DiCE timing varies |
| XAI Layer 3 | Template-based Bahasa Indonesia narrative | HIGH | Low — deterministic, fast |
| Inference | ONNX Runtime (prediction) + XGBoost .ubj (SHAP) | HIGH | Low — dual-model is standard |
| Labels | ICW heuristics → 4-class + Platt scaling | MEDIUM | Medium — weak labels acknowledged |
| Features | 25-30 from Tier 1+2 of 65-feature catalog | HIGH | Low — realistic, interpretable |
| Imbalance | Focal Loss or class weights (NOT SMOTE) | HIGH | Low — research-backed |
| Data | OCDS Indonesia (OCP Registry) primary | HIGH | Low — verified accessible |

### 26.2 Final Decisions (User Confirmed — March 30, 2026)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| **Counterfactual strategy** | Tiered: DiCE (primary) + SHAP-based fallback | Best quality with guaranteed response. DiCE timeout → instant SHAP CFs. |
| **Clean label verification** | YES — 100 samples (50 Kritis + 50 Aman) | Dramatically strengthens Bab 3 defense and calibration curves. ~2-3 hours. |
| **Feature count** | 30 features (Tier 1 + some Tier 2) | Balanced — good F1, manageable narratives. |
| **Phase 3 in proposal** | Dedicated subsection in Bab 4 | 1-page "Roadmap & Scalability" — shows Architect jury we think beyond notebooks. |
| **Brainstorming status** | COMPLETE — ready for plan generation | All research tracks done, all decisions made. |

---

## 27. Research Complete

All 20+ research tracks complete. All strategic decisions confirmed. Ready for execution plan generation.
