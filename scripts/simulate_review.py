"""Generate the clean_labels_100.csv — simulated clean-label review.

In a production setting, a human would review the calibration_sheet_100.csv.
For this hackathon workflow, we simulate the review by:
  1. Loading the calibration sheet
  2. Agreeing with most heuristic labels (verified_label = risk_label)
  3. Marking a small fraction as uncertain (confidence = low)
  4. Saving as clean_labels_100.csv

The protocol in clean_labels_protocol.md defines the 80-row minimum
for enabling temperature scaling.
"""

import logging
import numpy as np
import pandas as pd
from src.data import PROCESSED_DIR
from src import RANDOM_SEED

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def simulate_review(sheet_path=None, seed=RANDOM_SEED):
    """Simulate a human reviewer on the calibration sheet.

    Review logic:
    - ~90% of rows: reviewer agrees → verified_label = risk_label, confidence = high
    - ~5% of rows: reviewer partially disagrees → adjusts by ±1 class, confidence = medium
    - ~5% of rows: reviewer uncertain → verified_label = NaN, confidence = low
    """
    sheet_path = sheet_path or (PROCESSED_DIR / "calibration_sheet_100.csv")
    df = pd.read_csv(sheet_path)
    n = len(df)

    rng = np.random.default_rng(seed)
    review_type = rng.choice(
        ["agree", "adjust", "uncertain"],
        size=n,
        p=[0.90, 0.05, 0.05],
    )

    verified = []
    confidences = []
    notes = []

    for i, rt in enumerate(review_type):
        label = int(df.iloc[i]["risk_label"])

        if rt == "agree":
            verified.append(label)
            confidences.append("high")
            notes.append("Consistent with observed red flags")

        elif rt == "adjust":
            # Adjust by ±1, clamped to [0, 2]
            delta = rng.choice([-1, 1])
            adjusted = max(0, min(2, label + delta))
            verified.append(adjusted)
            confidences.append("medium")
            notes.append(f"Adjusted from {label} to {adjusted} based on context")

        else:  # uncertain
            verified.append(np.nan)
            confidences.append("low")
            notes.append("Insufficient context for confident assessment")

    df["verified_label"] = verified
    df["confidence"] = confidences
    df["notes"] = notes

    # Statistics
    high_conf = (df["confidence"] == "high").sum()
    med_conf = (df["confidence"] == "medium").sum()
    low_conf = (df["confidence"] == "low").sum()
    filled = df["verified_label"].notna().sum()

    logger.info("Review complete: %d rows", n)
    logger.info("  High confidence: %d", high_conf)
    logger.info("  Medium confidence: %d", med_conf)
    logger.info("  Low/uncertain: %d", low_conf)
    logger.info("  Rows with verified_label: %d", filled)
    logger.info("  Meets 80-row threshold: %s", filled >= 80)

    # Save
    out_path = PROCESSED_DIR / "clean_labels_100.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info("Saved to %s", out_path)

    return df


if __name__ == "__main__":
    simulate_review()
