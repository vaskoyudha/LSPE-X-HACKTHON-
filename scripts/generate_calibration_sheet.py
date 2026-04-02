"""Generate the calibration sample sheet from val_calibration.

This script:
  1. Loads train_data/raw.parquet
  2. Applies internal dev splits to extract val_calibration
  3. Computes heuristic labels for val_calibration
  4. Selects 100 stratified samples
  5. Saves as data/processed/calibration_sheet_100.csv

SAFETY: Only val_calibration data is used. test_data is never touched.
"""

import logging
import sys

from src.data import PROCESSED_DIR
from src.split import internal_dev_splits, load_raw_split
from src.labels import compute_risk_labels, select_calibration_samples, save_calibration_sheet

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def main() -> None:
    # Step 1: Load training raw data
    logger.info("Loading train raw data...")
    train_raw = load_raw_split("train")
    logger.info("Train raw: %d rows", len(train_raw))

    # Step 2: Extract val_calibration via dev splits
    logger.info("Applying internal dev splits...")
    dev_splits = internal_dev_splits(train_raw)
    val_cal = dev_splits["val_calibration"]
    logger.info("val_calibration: %d rows", len(val_cal))

    # Step 3: Compute heuristic labels for val_calibration only
    logger.info("Computing heuristic labels for val_calibration...")
    cal_labels = compute_risk_labels(val_cal)
    logger.info(
        "Label distribution in val_calibration: Low=%d, Medium=%d, High=%d",
        (cal_labels["risk_label"] == 0).sum(),
        (cal_labels["risk_label"] == 1).sum(),
        (cal_labels["risk_label"] == 2).sum(),
    )

    # Step 4: Select 100 stratified samples
    logger.info("Selecting 100 stratified calibration samples...")
    samples = select_calibration_samples(cal_labels, val_cal, n_samples=100)

    # Step 5: Save
    sheet_path = save_calibration_sheet(samples)
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
    main()
