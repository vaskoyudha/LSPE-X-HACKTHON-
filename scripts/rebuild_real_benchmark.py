"""Rebuild the Phase 2 benchmark on a real OCDS year slice and compare to the prior synthetic bundle."""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import xgboost as xgb
from onnxmltools import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as rt

from scripts.materialize import materialize
from src.data import RAW_DIR, PROCESSED_DIR, run_pipeline
from src.model import compute_sample_weights, run_evaluation_pipeline, run_training_pipeline

MODELS_DIR = ROOT / "models"
EVIDENCE_DIR = ROOT / ".sisyphus" / "evidence"
PUBLICATION_URL = "https://data.open-contracting.org/en/publication/101"
SELECTED_YEARS = [2021, 2022, 2023]
HPO_TRIALS = 3
HPO_TIMEOUT = 60


def _read_json(path: Path) -> dict | None:
    if path.exists():
        return json.loads(path.read_text())
    return None



def _load_historical_synthetic_snapshot() -> dict | None:
    """Recover the last tracked synthetic benchmark snapshot from git history."""
    try:
        revs = subprocess.check_output(
            ["git", "log", "--format=%H", "--", "data/processed/data_provenance.json"],
            cwd=ROOT,
            text=True,
        ).splitlines()
    except subprocess.CalledProcessError:
        return None

    for rev in revs:
        try:
            provenance_text = subprocess.check_output(
                ["git", "show", f"{rev}:data/processed/data_provenance.json"], cwd=ROOT, text=True
            )
            provenance = json.loads(provenance_text)
        except subprocess.CalledProcessError:
            continue
        if provenance.get("data_kind") != "synthetic_structured_benchmark":
            continue
        metrics = json.loads(subprocess.check_output(["git", "show", f"{rev}:models/metrics.json"], cwd=ROOT, text=True))
        robustness = json.loads(subprocess.check_output(["git", "show", f"{rev}:models/robustness.json"], cwd=ROOT, text=True))
        return {"metrics": metrics, "provenance": provenance, "robustness": robustness}
    return None

def _save_source_manifest() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "real_recent_multi_year_slice",
        "publication_url": PUBLICATION_URL,
        "selected_years": SELECTED_YEARS,
        "downloaded_files": [str((RAW_DIR / f"{year}.jsonl.gz").relative_to(ROOT)) for year in SELECTED_YEARS],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "note": (
            "This benchmark intentionally uses a recent multi-year real data slice to improve external validity while keeping the repo runnable. "
            "It should be compared against the earlier synthetic benchmark in models/benchmark_comparison.json."
        ),
    }
    (PROCESSED_DIR / "source_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _build_local_onnx_artifacts() -> dict:
    train_X = pd.read_parquet(ROOT / "train_data" / "features.parquet")
    train_y = pd.read_parquet(ROOT / "train_data" / "labels.parquet")["risk_label"]
    params_path = MODELS_DIR / "best_params.json"
    params = json.loads(params_path.read_text())
    n_rounds = int(params.pop("n_rounds", 449))
    weights = compute_sample_weights(train_y)

    # Save .ubj using real feature names for native XGBoost use.
    ubj_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        seed=42,
        n_estimators=n_rounds,
        n_jobs=-1,
        **params,
    )
    ubj_model.fit(train_X, train_y, sample_weight=weights)
    ubj_path = MODELS_DIR / "xgb_model.ubj"
    ubj_model.save_model(str(ubj_path))

    # Train a renamed-feature clone for ONNX export compatibility.
    renamed_cols = [f"f{i}" for i in range(train_X.shape[1])]
    train_X_onnx = train_X.copy()
    train_X_onnx.columns = renamed_cols
    onnx_model = xgb.XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        eval_metric="mlogloss",
        tree_method="hist",
        seed=42,
        n_estimators=n_rounds,
        n_jobs=-1,
        **params,
    )
    onnx_model.fit(train_X_onnx, train_y, sample_weight=weights)
    onnx_proto = convert_xgboost(
        onnx_model,
        initial_types=[("float_input", FloatTensorType([None, train_X_onnx.shape[1]]))],
    )
    onnx_path = MODELS_DIR / "xgb_model.onnx"
    onnx_path.write_bytes(onnx_proto.SerializeToString())

    session = rt.InferenceSession(str(onnx_path))
    raw = session.run(None, {session.get_inputs()[0].name: train_X_onnx.head(64).to_numpy(dtype=np.float32)})[1]
    onnx_probs = np.array([[row[i] for i in range(3)] for row in raw]) if isinstance(raw[0], dict) else np.array(raw)
    native_probs = onnx_model.predict_proba(train_X_onnx.head(64))
    mean_abs_diff = float(np.mean(np.abs(native_probs - onnx_probs)))
    return {
        "ubj_path": str(ubj_path.relative_to(ROOT)),
        "onnx_path": str(onnx_path.relative_to(ROOT)),
        "onnx_mean_abs_diff_head64": mean_abs_diff,
    }


def main() -> None:
    EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)

    synthetic_snapshot = _load_historical_synthetic_snapshot() or {
        "metrics": _read_json(MODELS_DIR / "metrics.json"),
        "provenance": _read_json(PROCESSED_DIR / "data_provenance.json"),
        "robustness": _read_json(MODELS_DIR / "robustness.json"),
    }

    # Step 1: fetch real raw data + flatten canonical real parquet.
    run_pipeline(years=SELECTED_YEARS, skip_download=False)
    _save_source_manifest()

    # Step 2: split / features / labels using the real flat parquet.
    materialize(use_synthetic=False)

    # Step 3: retrain and evaluate on the real benchmark.
    run_training_pipeline(n_trials=HPO_TRIALS, hpo_timeout=HPO_TIMEOUT)
    real_metrics = run_evaluation_pipeline()

    # Step 4: rebuild local model artifacts and diagnostics.
    onnx_info = _build_local_onnx_artifacts()

    # Diagnostics script uses the current train/test artifacts.
    import subprocess
    subprocess.run([str(ROOT / '.venv' / 'bin' / 'python'), str(ROOT / 'scripts' / 'run_diagnostics.py')], check=True)

    real_snapshot = {
        "metrics": _read_json(MODELS_DIR / "metrics.json"),
        "provenance": _read_json(PROCESSED_DIR / "data_provenance.json"),
        "robustness": _read_json(MODELS_DIR / "robustness.json"),
        "source_manifest": _read_json(PROCESSED_DIR / "source_manifest.json"),
        "onnx": onnx_info,
    }

    synthetic_macro = (synthetic_snapshot.get("metrics") or {}).get("final_test", {}).get("macro_f1")
    real_macro = (real_snapshot.get("metrics") or {}).get("final_test", {}).get("macro_f1")
    comparison = {
        "synthetic_before": synthetic_snapshot,
        "real_after": real_snapshot,
        "summary": {
            "synthetic_macro_f1": synthetic_macro,
            "real_macro_f1": real_macro,
            "macro_f1_delta_real_minus_synthetic": None if synthetic_macro is None or real_macro is None else round(real_macro - synthetic_macro, 4),
            "synthetic_data_kind": (synthetic_snapshot.get("provenance") or {}).get("data_kind"),
            "real_data_kind": (real_snapshot.get("provenance") or {}).get("data_kind"),
        },
    }
    (MODELS_DIR / "benchmark_comparison.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    (EVIDENCE_DIR / "real-benchmark-migration.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    print(json.dumps(comparison["summary"], indent=2))


if __name__ == "__main__":
    main()
