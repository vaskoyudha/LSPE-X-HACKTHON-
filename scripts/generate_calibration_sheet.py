"""Generate the calibration sample sheet from val_calibration.

This script:
  1. Loads train_data/raw.parquet
  2. Applies internal dev splits to extract val_calibration
  3. Computes features for val_calibration using only prior train_fit + val_hpo
     rows as history context, then merges raw + feature inputs
  4. Computes heuristic labels for the merged val_calibration frame
  5. Selects 100 stratified samples
  6. Saves as data/processed/calibration_sheet_100.csv

SAFETY: Only pre-val_calibration train rows are used as history context.
test_data is never touched.
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import PROCESSED_DIR
from src.split import internal_dev_splits, load_raw_split
from src.features import compute_all_features
from src.labels import compute_risk_labels, select_calibration_samples, save_calibration_sheet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main(n_samples: int = 100) -> None:
    # Step 1: Load training raw data
    logger.info("Loading train raw data...")
    train_raw = load_raw_split("train")
    logger.info("Train raw: %d rows", len(train_raw))

    # Step 2: Extract val_calibration via dev splits
    logger.info("Applying internal dev splits...")
    dev_splits = internal_dev_splits(train_raw)
    train_fit = dev_splits["train_fit"].reset_index(drop=True)
    val_hpo = dev_splits["val_hpo"].reset_index(drop=True)
    val_cal = dev_splits["val_calibration"].reset_index(drop=True)
    logger.info("val_calibration: %d rows", len(val_cal))

    # Step 3: Compute features and merged label inputs for val_calibration
    logger.info("Computing features for val_calibration...")
    prior_history = pd.concat([train_fit, val_hpo], axis=0, ignore_index=True)
    val_cal_features = compute_all_features(val_cal, history_df=prior_history)
    val_cal_label_inputs = pd.concat(
        [val_cal.reset_index(drop=True), val_cal_features.reset_index(drop=True)],
        axis=1,
    )

    # Step 4: Compute heuristic labels for merged val_calibration inputs
    logger.info("Computing heuristic labels for merged val_calibration inputs...")
    cal_labels = compute_risk_labels(val_cal_label_inputs)
    logger.info(
        "Label distribution in val_calibration: Low=%d, Medium=%d, High=%d",
        (cal_labels["risk_label"] == 0).sum(),
        (cal_labels["risk_label"] == 1).sum(),
        (cal_labels["risk_label"] == 2).sum(),
    )

    # Step 5: Select stratified samples
    logger.info("Selecting %d stratified calibration samples...", n_samples)
    samples = select_calibration_samples(cal_labels, val_cal, n_samples=n_samples)

    # Step 6: Save
    sheet_path = save_calibration_sheet(
        samples,
        path=PROCESSED_DIR / f"calibration_sheet_{n_samples}.csv",
    )
    logger.info("Calibration sheet saved to: %s", sheet_path)
    logger.info("Columns: %s", list(samples.columns))
    logger.info("Total samples: %d", len(samples))
    logger.info(
        "Sample class distribution: Low=%d, Medium=%d, High=%d",
        (samples["risk_label"] == 0).sum(),
        (samples["risk_label"] == 1).sum(),
        (samples["risk_label"] == 2).sum(),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int, default=100)
    args = parser.parse_args()
    main(n_samples=args.n_samples)
