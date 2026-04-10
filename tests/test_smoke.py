import src
from pathlib import Path

import pytest


@pytest.mark.p0
def test_src_importable():
    assert src.__version__ == "0.1.0"
    assert src.RANDOM_SEED == 42


@pytest.mark.p1
def test_readme_mentions_evidence_label_coverage_artifact():
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "models/evidence_label_coverage.json" in readme
    assert "reviewed_row_labels.parquet" in readme
    assert "fraud_outcomes.parquet" in readme
