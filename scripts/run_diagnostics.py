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

from src.data import PROCESSED_DIR
from src.diagnostics import (
    run_circularity_ablation,
    summarize_data_provenance,
    summarize_feature_health,
    summarize_feature_health_overview,
)

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "proposal" / "figures"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"


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


if __name__ == "__main__":
    main()
