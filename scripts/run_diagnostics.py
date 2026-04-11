"""Generate provenance and robustness diagnostics for the current Phase 2 bundle."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import xgboost as xgb

from src.data import PROCESSED_DIR
from src.diagnostics import (
    build_explanation_validation_from_summary,
    build_reviewed_subset_metrics_from_summary,
    compute_operational_review_metrics,
    load_canonical_reviewed_labels,
    load_confirmed_outcome_labels,
    load_metrics_artifact,
    load_reviewed_labels,
    load_manual_review_summary,
    load_row_level_reviewed_benchmark,
    run_circularity_ablation,
    select_reviewed_rows,
    summarize_explanation_validation,
    summarize_data_provenance,
    summarize_evidence_label_coverage,
    summarize_confirmed_outcome_alignment,
    summarize_evaluation_lanes,
    summarize_fraud_evidence_lane,
    summarize_feature_health,
    summarize_feature_health_overview,
)
from src.model import (
    apply_temperature,
    evaluate,
    load_decision_thresholds,
    load_model,
    predict_with_thresholds,
)

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "proposal" / "figures"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"
EVALUATION_LANES_PATH = MODELS_DIR / "evaluation_lanes.json"


def load_optional_json(path: Path) -> dict[str, object]:
    """Load an optional JSON artifact without crashing when it is absent."""
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    train_raw = pd.read_parquet(ROOT / "train_data" / "raw.parquet")
    test_raw = pd.read_parquet(ROOT / "test_data" / "raw.parquet")
    train_X = pd.read_parquet(ROOT / "train_data" / "features.parquet")
    train_y = pd.read_parquet(ROOT / "train_data" / "labels.parquet")["risk_label"]
    test_X = pd.read_parquet(ROOT / "test_data" / "features.parquet")
    test_y = pd.read_parquet(ROOT / "test_data" / "labels.parquet")["risk_label"]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    reviewed_evidence = load_canonical_reviewed_labels()
    confirmed_outcomes = load_confirmed_outcome_labels()
    evidence_coverage = summarize_evidence_label_coverage(
        reviewed_evidence, confirmed_outcomes
    )
    (MODELS_DIR / "evidence_label_coverage.json").write_text(
        json.dumps(evidence_coverage, indent=2),
        encoding="utf-8",
    )

    provenance = summarize_data_provenance(train_raw, test_raw)
    (PROCESSED_DIR / "data_provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")

    feature_health = summarize_feature_health(train_X)
    feature_health_overview = summarize_feature_health_overview(feature_health)
    (MODELS_DIR / "feature_health.json").write_text(
        json.dumps(feature_health, indent=2),
        encoding="utf-8",
    )

    robustness = run_circularity_ablation(train_X, train_y, test_X, test_y)
    (MODELS_DIR / "robustness.json").write_text(json.dumps(robustness, indent=2), encoding="utf-8")
    (EVIDENCE_DIR / "weakness-diagnostics.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "evidence_label_coverage": evidence_coverage,
                "feature_health_overview": feature_health_overview,
                "robustness": robustness,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    labels = ["Full", "Core proxies removed", "Broad proxies removed"]
    macro_f1 = [
        robustness["full"]["macro_f1"],
        robustness["proxy_core_removed"]["macro_f1"],
        robustness["proxy_broad_removed"]["macro_f1"],
    ]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, macro_f1, color=["#1f77b4", "#ff7f0e", "#d62728"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro-F1")
    ax.set_title("Robustness Check: Impact of Removing Heuristic Proxy Features")
    for bar, value in zip(bars, macro_f1):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.4f}", ha="center", va="bottom")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "robustness_ablation.png", dpi=150)
    plt.close(fig)

    missing_pairs = sorted(
        (
            (feature, stats["missing_pct"])
            for feature, stats in feature_health.items()
        ),
        key=lambda item: (-item[1], item[0]),
    )
    top_missing = missing_pairs[:10]
    fig, ax = plt.subplots(figsize=(9, 5))
    labels_missing = [name for name, _ in top_missing]
    values_missing = [value for _, value in top_missing]
    colors = [
        "#d62728"
        if feature in feature_health_overview["active_dead_features"]
        else "#1f77b4"
        for feature in labels_missing
    ]
    ax.barh(labels_missing, values_missing, color=colors)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Missing percentage")
    ax.set_title("Feature Health: Highest Missingness in Active Real-Benchmark Features")
    ax.invert_yaxis()
    for idx, value in enumerate(values_missing):
        ax.text(value + 1, idx, f"{value:.2f}%", va="center", fontsize=9)
    summary_text = (
        f"Active dead features: {feature_health_overview['active_dead_feature_count']} | "
        f"Retired slots removed: {len(feature_health_overview['retired_dead_features_removed'])}/"
        f"{len(feature_health_overview['retired_dead_features_removed']) + len(feature_health_overview['retired_dead_features_present'])}"
    )
    fig.text(0.02, 0.01, summary_text, fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "feature_health.png", dpi=150)
    plt.close(fig)

    model = load_model()
    thresholds = load_decision_thresholds()
    calibration_path = MODELS_DIR / "calibration.json"
    calibration = None
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text())

    dtest = xgb.DMatrix(test_X)
    probs = model.predict(dtest)
    if calibration and calibration.get("enabled"):
        probs = apply_temperature(probs, calibration["temperature"])
    preds = predict_with_thresholds(probs, thresholds)

    operational = compute_operational_review_metrics(probs, test_y)
    (MODELS_DIR / "operational_metrics.json").write_text(
        json.dumps(operational, indent=2),
        encoding="utf-8",
    )

    budgets = operational["budgets"]
    fig, ax = plt.subplots(figsize=(8, 5))
    x_labels = list(budgets.keys())
    precision_values = [budgets[key]["precision_at_k"] for key in x_labels]
    recall_values = [budgets[key]["recall_at_k"] for key in x_labels]
    ax.plot(x_labels, precision_values, marker="o", label="Precision@k")
    ax.plot(x_labels, recall_values, marker="o", label="Recall@k")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Review budget (k rows)")
    ax.set_ylabel("Score")
    ax.set_title("Operational Review Metrics on the Current Test Benchmark")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "operational_metrics.png", dpi=150)
    plt.close(fig)

    manual_summary = load_manual_review_summary()
    reviewed_metrics: dict[str, object]
    explanation_validation: dict[str, object]
    row_level_reviewed = load_row_level_reviewed_benchmark()
    if len(row_level_reviewed) > 0:
        explanation_validation = summarize_explanation_validation(row_level_reviewed)
        review_raw, review_X, review_y = select_reviewed_rows(row_level_reviewed, test_raw, test_X)
        if len(review_X) == 0:
            reviewed_metrics = {
                "status": "unmatched_review_rows",
                "message": "Row-level reviewed rows could not be aligned back to the test benchmark.",
            }
        else:
            reviewed_metrics = evaluate(
                model,
                review_X,
                review_y,
                partition_name="reviewed_subset_row_level",
                thresholds=thresholds,
                label_type="reviewed_risk_labels",
            )
            reviewed_metrics["status"] = "available"
            reviewed_metrics["matched_rows"] = int(len(review_X))
            reviewed_metrics["source"] = "row_level_reviewed_benchmark"
    elif not manual_summary.empty:
        reviewed_metrics = build_reviewed_subset_metrics_from_summary(manual_summary)
        explanation_validation = build_explanation_validation_from_summary(manual_summary)
        (MODELS_DIR / "manual_review_summary.json").write_text(
            manual_summary.to_json(orient="records", indent=2),
            encoding="utf-8",
        )
    else:
        reviewed = load_reviewed_labels()
        explanation_validation = summarize_explanation_validation(reviewed)
        if reviewed.empty:
            reviewed_metrics = {
                "status": "pending_human_review",
                "message": "No reviewed benchmark rows with reviewed_label available yet.",
            }
        else:
            review_raw, review_X, review_y = select_reviewed_rows(reviewed, test_raw, test_X)
            if len(review_X) == 0:
                reviewed_metrics = {
                    "status": "unmatched_review_rows",
                    "message": "Reviewed rows could not be aligned back to the test benchmark.",
                }
            else:
                reviewed_metrics = evaluate(
                    model,
                    review_X,
                    review_y,
                    partition_name="reviewed_subset",
                    thresholds=thresholds,
                    label_type="reviewed_risk_labels",
                )
                reviewed_metrics["status"] = "available"
                reviewed_metrics["matched_rows"] = int(len(review_X))

    (MODELS_DIR / "reviewed_subset_metrics.json").write_text(
        json.dumps(reviewed_metrics, indent=2),
        encoding="utf-8",
    )
    (MODELS_DIR / "explanation_validation.json").write_text(
        json.dumps(explanation_validation, indent=2),
        encoding="utf-8",
    )
    heuristic_metrics = load_metrics_artifact()
    fraud_evidence_metrics = load_optional_json(
        MODELS_DIR / "fraud_evidence_metrics.json"
    )
    fraud_evidence_lane = summarize_fraud_evidence_lane(fraud_evidence_metrics)
    confirmed_outcome_alignment = summarize_confirmed_outcome_alignment(
        confirmed_outcomes,
        test_raw,
        probs,
        preds,
    )
    evaluation_lanes = summarize_evaluation_lanes(
        heuristic_metrics,
        reviewed_metrics,
        explanation_validation,
        evidence_coverage,
        confirmed_outcome_alignment,
        fraud_evidence_lane,
    )
    EVALUATION_LANES_PATH.write_text(
        json.dumps(evaluation_lanes, indent=2),
        encoding="utf-8",
    )

    (EVIDENCE_DIR / "weakness-diagnostics.json").write_text(
        json.dumps(
            {
                "provenance": provenance,
                "evidence_label_coverage": evidence_coverage,
                "evaluation_lanes": evaluation_lanes,
                "fraud_evidence_lane": fraud_evidence_lane,
                "feature_health_overview": feature_health_overview,
                "robustness": robustness,
                "operational_metrics": operational,
                "reviewed_subset_metrics": reviewed_metrics,
                "explanation_validation": explanation_validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
