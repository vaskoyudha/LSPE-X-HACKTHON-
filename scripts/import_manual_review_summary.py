"""Import a manual benchmark review summary CSV into tracked repo artifacts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import PROCESSED_DIR
from src.diagnostics import (
    build_explanation_validation_from_summary,
    build_reviewed_subset_metrics_from_summary,
)

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "proposal" / "figures"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"


def _load_summary(source: Path) -> pd.DataFrame:
    if not source.exists():
        raise FileNotFoundError(f"Manual review summary not found: {source}")
    return pd.read_csv(source)


def _plot_manual_review_summary(
    reviewed_metrics: dict[str, object],
    explanation_validation: dict[str, object],
) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    cm = np.array(reviewed_metrics["confusion_matrix"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    im = ax.imshow(cm, cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    labels = ["Low", "Medium", "High"]
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Reviewed")
    ax.set_title("Manual Review Confusion Matrix")
    for i in range(3):
        for j in range(3):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center")
    fig.colorbar(im, ax=ax, shrink=0.8)

    ax = axes[1]
    score_names = ["Agreement", "Clarity", "Actionable"]
    score_values = [
        float(reviewed_metrics["overall_agreement"]),
        float(explanation_validation["clarity_mean"]) / 5.0,
        float(explanation_validation["actionable_mean"]) / 5.0,
    ]
    bars = ax.bar(score_names, score_values, color=["#1f77b4", "#ff7f0e", "#2ca02c"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Normalized score")
    ax.set_title("Manual Review Summary")
    for bar, value in zip(bars, score_values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.02, f"{value:.2f}", ha="center")

    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "manual_review_summary.png", dpi=150)
    plt.close(fig)


def main(source: Path) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    summary_df = _load_summary(source)
    dest_csv = PROCESSED_DIR / "manual_review_summary.csv"
    summary_df.to_csv(dest_csv, index=False)

    reviewed_metrics = build_reviewed_subset_metrics_from_summary(summary_df)
    explanation_validation = build_explanation_validation_from_summary(summary_df)

    (MODELS_DIR / "reviewed_subset_metrics.json").write_text(
        json.dumps(reviewed_metrics, indent=2),
        encoding="utf-8",
    )
    (MODELS_DIR / "explanation_validation.json").write_text(
        json.dumps(explanation_validation, indent=2),
        encoding="utf-8",
    )
    (MODELS_DIR / "manual_review_summary.json").write_text(
        summary_df.to_json(orient="records", indent=2),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "manual-review-summary.json").write_text(
        json.dumps(
            {
                "reviewed_subset_metrics": reviewed_metrics,
                "explanation_validation": explanation_validation,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _plot_manual_review_summary(reviewed_metrics, explanation_validation)
    print("Imported manual review summary into repo artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "source",
        nargs="?",
        default="/home/vascosera/Downloads/benchmark_analysis.csv",
    )
    args = parser.parse_args()
    main(Path(args.source))
