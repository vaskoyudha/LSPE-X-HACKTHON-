"""Materialization pipeline — Task 10.

Orchestrates the full data → split → features → labels pipeline.
Produces all canonical train/test artifacts needed by the modeling lane.

Can operate in two modes:
  1. Real: downloads and processes actual OCDS data
  2. Synthetic: generates realistic synthetic data for dev/testing
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Import all pipeline modules
from src.data import (
    PROJECT_ROOT,
    PROCESSED_DIR,
    FLAT_PARQUET,
    run_pipeline as data_pipeline,
    load_flat,
    clean_dates,
)
from src.split import (
    TRAIN_DIR,
    TEST_DIR,
    external_raw_split,
    save_raw_splits,
    internal_dev_splits,
    save_dev_split_manifest,
)
from src.features import compute_all_features, save_features, FEATURE_CATALOG
from src.labels import compute_risk_labels, save_labels

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

MODELS_DIR = PROJECT_ROOT / "models"
FEATURE_MANIFEST = PROCESSED_DIR / "feature_manifest.json"


# ---------------------------------------------------------------------------
# Synthetic data generator (for development without download)
# ---------------------------------------------------------------------------


def generate_synthetic_ocds(n: int = 5000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic synthetic procurement data matching OCDS schema.

    Produces data with realistic Indonesian procurement characteristics:
    - Value distributions matching LPSE patterns
    - Temporal spread across 2014-2023
    - Realistic NaN patterns
    - Multiple buyers and suppliers
    """
    rng = np.random.default_rng(seed)

    # Temporal spread: 2014-2023
    start_ts = pd.Timestamp("2014-01-01", tz="UTC")
    end_ts = pd.Timestamp("2023-12-31", tz="UTC")
    total_days = (end_ts - start_ts).days
    random_days = rng.integers(0, total_days, size=n)
    dates = pd.to_datetime(
        [start_ts + pd.Timedelta(days=int(d)) for d in sorted(random_days)]
    )

    # Buyer and supplier pools
    n_buyers = 50
    n_suppliers = 200
    buyers = [f"buyer-{i:03d}" for i in range(n_buyers)]
    suppliers = [f"supplier-{i:04d}" for i in range(n_suppliers)]
    buyer_names = [f"Dinas {chr(65 + i % 26)} Kab {i // 26 + 1}" for i in range(n_buyers)]
    supplier_names = [f"PT {chr(65 + i % 26)}{chr(97 + (i//26) % 26)} Sejahtera {i}" for i in range(n_suppliers)]

    # Value distributions (log-normal, in IDR)
    tender_values = rng.lognormal(mean=20, sigma=1.5, size=n)  # ~50M - 5B IDR
    tender_values = np.clip(tender_values, 1e7, 1e11)

    # Award values typically 70-100% of tender
    award_ratios = rng.uniform(0.6, 1.0, size=n)
    award_values = tender_values * award_ratios

    # Contract values close to award
    contract_ratios = rng.uniform(0.95, 1.05, size=n)
    contract_values = award_values * contract_ratios

    # Number of tenderers (with realistic skew)
    tenderer_counts = rng.choice(
        [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, np.nan],
        size=n,
        p=[0.15, 0.10, 0.20, 0.15, 0.15, 0.08, 0.05, 0.04, 0.03, 0.02, 0.03],
    )

    # Procurement methods
    methods = rng.choice(
        ["open", "direct", "limited", "selective"],
        size=n,
        p=[0.55, 0.20, 0.15, 0.10],
    )

    # Title and description generation
    prefixes = [
        "Pengadaan", "Pembangunan", "Rehabilitasi", "Peningkatan",
        "Pemeliharaan", "Konsultansi", "Jasa", "Pekerjaan",
    ]
    objects = [
        "Alat Kesehatan", "Peralatan IT", "Jalan Nasional",
        "Gedung Kantor", "Sarana Pendidikan", "Infrastruktur Air",
        "Kendaraan Dinas", "Obat-obatan", "Bahan Bangunan",
        "Sistem Informasi", "Peralatan Laboratorium",
    ]

    titles = []
    descriptions = []
    for i in range(n):
        prefix = rng.choice(prefixes)
        obj = rng.choice(objects)
        year = dates[i].year
        titles.append(f"{prefix} {obj} Tahun {year}")
        descriptions.append(
            f"{prefix} {obj} untuk mendukung program pembangunan "
            f"di wilayah kerja {buyer_names[rng.integers(0, n_buyers)]} "
            f"tahun anggaran {year}. Pekerjaan meliputi pengadaan barang "
            f"dan jasa sesuai dengan spesifikasi teknis yang telah ditentukan."
        )

    # Build DataFrame
    buyer_idx = rng.integers(0, n_buyers, size=n)
    supplier_idx = rng.integers(0, n_suppliers, size=n)

    df = pd.DataFrame(
        {
            "ocid": [f"ocds-synth-{i:06d}" for i in range(n)],
            "tender_id": [f"T-{i:06d}" for i in range(n)],
            "tender_datePublished": dates,
            "tender_title": titles,
            "tender_description": descriptions,
            "tender_status": rng.choice(
                ["complete", "active", "cancelled"], size=n, p=[0.75, 0.15, 0.10]
            ),
            "tender_procurementMethod": methods,
            "tender_value_amount": tender_values,
            "tender_value_currency": "IDR",
            "tender_tenderPeriod_startDate": dates - pd.Timedelta(days=7),
            "tender_tenderPeriod_endDate": dates + pd.Timedelta(days=int(rng.integers(7, 60))),
            "tender_numberOfTenderers": tenderer_counts,
            "buyer_id": [buyers[i] for i in buyer_idx],
            "buyer_name": [buyer_names[i] for i in buyer_idx],
            "award_id": [f"A-{i:06d}" for i in range(n)],
            "award_status": rng.choice(
                ["active", "unsuccessful"], size=n, p=[0.85, 0.15]
            ),
            "award_date": dates + pd.Timedelta(days=int(rng.integers(14, 90))),
            "award_value_amount": award_values,
            "award_value_currency": "IDR",
            "supplier_id": [suppliers[i] for i in supplier_idx],
            "supplier_name": [supplier_names[i] for i in supplier_idx],
            "contract_id": [f"C-{i:06d}" for i in range(n)],
            "contract_value_amount": contract_values,
            "contract_dateSigned": dates + pd.Timedelta(days=int(rng.integers(30, 120))),
        }
    )

    # Introduce realistic NaN patterns
    nan_mask = rng.random(size=n)
    df.loc[nan_mask < 0.05, "tender_description"] = None
    df.loc[nan_mask < 0.08, "contract_value_amount"] = None
    df.loc[nan_mask < 0.03, "award_value_amount"] = None

    logger.info("Generated %d synthetic OCDS records", len(df))
    return df


# ---------------------------------------------------------------------------
# Materialization pipeline
# ---------------------------------------------------------------------------


def materialize(use_synthetic: bool = False, n_synthetic: int = 5000) -> dict:
    """Run the full materialization pipeline.

    Steps:
      1. Load or generate flat data
      2. External raw split (temporal 80/20)
      3. Internal dev sub-splits
      4. Compute features per partition
      5. Compute labels per partition
      6. Save all artifacts
      7. Record feature manifest

    Returns dict with summary statistics.
    """
    logger.info("=" * 60)
    logger.info("MATERIALIZATION PIPELINE START")
    logger.info("=" * 60)

    # Step 1: Get flat data
    if use_synthetic:
        logger.info("Using SYNTHETIC data (n=%d)", n_synthetic)
        df = generate_synthetic_ocds(n=n_synthetic)
    else:
        if FLAT_PARQUET.exists():
            logger.info("Loading existing flat parquet: %s", FLAT_PARQUET)
            df = load_flat()
        else:
            logger.info("Running full data pipeline (download + flatten)...")
            df = data_pipeline()

    df = clean_dates(df)
    logger.info("Loaded %d rows, %d columns", len(df), len(df.columns))

    # Step 2: External raw split
    logger.info("--- External raw split ---")
    train_raw, test_raw = external_raw_split(df, test_ratio=0.2)
    save_raw_splits(train_raw, test_raw)
    logger.info("Train: %d rows, Test: %d rows", len(train_raw), len(test_raw))

    # Step 3: Internal dev sub-splits (train only)
    logger.info("--- Internal dev sub-splits ---")
    dev_splits = internal_dev_splits(train_raw)
    save_dev_split_manifest(dev_splits)
    for name, split_df in dev_splits.items():
        logger.info("  %s: %d rows", name, len(split_df))

    # Step 4: Compute features per partition
    logger.info("--- Computing features ---")
    train_features = compute_all_features(train_raw)
    test_features = compute_all_features(test_raw)
    save_features(train_features, "train")
    save_features(test_features, "test")
    logger.info(
        "Features: %d columns (train=%d rows, test=%d rows)",
        len(train_features.columns),
        len(train_features),
        len(test_features),
    )

    # Step 5: Compute labels per partition
    logger.info("--- Computing labels ---")
    train_labels = compute_risk_labels(train_raw)
    test_labels = compute_risk_labels(test_raw)
    save_labels(train_labels, "train")
    save_labels(test_labels, "test")

    # Step 6: Record feature manifest
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    manifest = {
        "feature_count": len(train_features.columns),
        "feature_names": list(train_features.columns),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "label_distribution": {
            "train": {
                "low_risk": int((train_labels["risk_label"] == 0).sum()),
                "medium_risk": int((train_labels["risk_label"] == 1).sum()),
                "high_risk": int((train_labels["risk_label"] == 2).sum()),
            },
            "test": {
                "low_risk": int((test_labels["risk_label"] == 0).sum()),
                "medium_risk": int((test_labels["risk_label"] == 1).sum()),
                "high_risk": int((test_labels["risk_label"] == 2).sum()),
            },
        },
    }
    FEATURE_MANIFEST.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )

    logger.info("=" * 60)
    logger.info("MATERIALIZATION COMPLETE")
    logger.info("Feature manifest: %s", FEATURE_MANIFEST)
    logger.info("=" * 60)

    return manifest


if __name__ == "__main__":
    # Default: try real data, fall back to synthetic
    use_syn = "--synthetic" in sys.argv
    materialize(use_synthetic=use_syn)
