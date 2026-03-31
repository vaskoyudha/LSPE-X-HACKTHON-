"""Shared pytest fixtures for the LPSE-X project."""

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import RANDOM_SEED


@pytest.fixture
def random_seed() -> int:
    """Expose the canonical project random seed to tests."""
    return RANDOM_SEED
