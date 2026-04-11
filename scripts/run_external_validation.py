"""Run year-holdout external validation across a broader real-data window."""

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
import pandas as pd

from src.data import (
    AVAILABLE_YEARS,
    RAW_DIR,
    clean_dates,
    download_all,
    flatten_jsonl_gz,
)
from src.features import compute_all_features
from src.labels import compute_risk_labels
from src.model import BEST_PARAMS_PATH, evaluate, train_final_model
from src.split import internal_dev_splits

MODELS_DIR = ROOT / "models"
FIGURES_DIR = ROOT / "proposal" / "figures"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"


def _discover_local_years() -> list[int]:
    years: list[int] = []
    for path in RAW_DIR.glob("*.jsonl.gz"):
        if path.name[:4].isdigit():
            years.append(int(path.name[:4]))
    return sorted(set(years))


def _load_year_window(years: list[int], download_missing: bool) -> pd.DataFrame:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    if download_missing:
        download_all(years)

    paths = [RAW_DIR / f"{year}.jsonl.gz" for year in years if (RAW_DIR / f"{year}.jsonl.gz").exists()]
    if not paths:
        raise FileNotFoundError("No raw year files available for external validation.")

    df = flatten_jsonl_gz(paths)
    return clean_dates(df)


def _build_labels(raw_df: pd.DataFrame, feature_df: pd.DataFrame) -> pd.Series:
    merged = pd.concat([raw_df.reset_index(drop=True), feature_df.reset_index(drop=True)], axis=1)
    return compute_risk_labels(merged)["risk_label"]


def run_external_validation(years: list[int], download_missing: bool = False) -> dict[str, object]:
    df = _load_year_window(years, download_missing=download_missing)
    df = df[df["tender_datePublished"].notna()].copy()
    df["year"] = df["tender_datePublished"].dt.year

    best_params = json.loads(BEST_PARAMS_PATH.read_text())
    folds: list[dict[str, object]] = []

    for holdout_year in sorted(df["year"].dropna().astype(int).unique()):
        train_raw = df[df["year"] < holdout_year].copy().sort_values("tender_datePublished")
        test_raw = df[df["year"] == holdout_year].copy().sort_values("tender_datePublished")
        if len(train_raw) < 1000 or len(test_raw) < 1000:
            continue
        print(
            f"[external-validation] fold holdout={holdout_year} "
            f"train_rows={len(train_raw)} test_rows={len(test_raw)}",
            flush=True,
        )

        dev = internal_dev_splits(train_raw)
        train_fit_raw = dev["train_fit"].reset_index(drop=True)
        val_hpo_raw = dev["val_hpo"].reset_index(drop=True)
        test_raw = test_raw.reset_index(drop=True)

        X_fit = compute_all_features(train_fit_raw)
        y_fit = _build_labels(train_fit_raw, X_fit)
        X_hpo = compute_all_features(val_hpo_raw, history_df=train_fit_raw)
        y_hpo = _build_labels(val_hpo_raw, X_hpo)
        X_test = compute_all_features(test_raw, history_df=train_raw)
        y_test = _build_labels(test_raw, X_test)

        model = train_final_model(X_fit, y_fit, X_hpo, y_hpo, best_params.copy())
        metrics = evaluate(
            model,
            X_test,
            y_test,
            partition_name=f"external_holdout_{holdout_year}",
        )
        print(
            f"[external-validation] completed holdout={holdout_year} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"high_risk_f1={metrics['per_class_f1']['High Risk']:.4f}",
            flush=True,
        )
        folds.append(
            {
                "holdout_year": int(holdout_year),
                "train_rows": int(len(train_raw)),
                "test_rows": int(len(test_raw)),
                "metrics": metrics,
            }
        )

    if not folds:
        raise RuntimeError("No usable external-validation folds were produced.")

    macro_scores = [fold["metrics"]["macro_f1"] for fold in folds]
    high_scores = [fold["metrics"]["per_class_f1"]["High Risk"] for fold in folds]

    result = {
        "years_covered": years,
        "fold_count": len(folds),
        "folds": folds,
        "summary": {
            "macro_f1_mean": round(float(pd.Series(macro_scores).mean()), 4),
            "macro_f1_min": round(float(pd.Series(macro_scores).min()), 4),
            "macro_f1_max": round(float(pd.Series(macro_scores).max()), 4),
            "high_risk_f1_mean": round(float(pd.Series(high_scores).mean()), 4),
            "high_risk_f1_min": round(float(pd.Series(high_scores).min()), 4),
            "high_risk_f1_max": round(float(pd.Series(high_scores).max()), 4),
        },
        "note": (
            "External validation trains on prior years and evaluates on a held-out future year. "
            "It broadens coverage without replacing the tracked benchmark bundle."
        ),
    }
    return result


def save_artifacts(result: dict[str, object]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    (MODELS_DIR / "external_validation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )
    (EVIDENCE_DIR / "external-validation.json").write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    years = [fold["holdout_year"] for fold in result["folds"]]
    macro_scores = [fold["metrics"]["macro_f1"] for fold in result["folds"]]
    high_scores = [fold["metrics"]["per_class_f1"]["High Risk"] for fold in result["folds"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(years, macro_scores, marker="o", label="Macro-F1")
    ax.plot(years, high_scores, marker="o", label="High Risk F1")
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("Holdout year")
    ax.set_ylabel("Score")
    ax.set_title("External Validation Across Holdout Years")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "external_validation.png", dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="*", type=int, default=None)
    parser.add_argument("--download-missing", action="store_true")
    args = parser.parse_args()

    years = args.years or _discover_local_years() or AVAILABLE_YEARS
    result = run_external_validation(years, download_missing=args.download_missing)
    save_artifacts(result)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
