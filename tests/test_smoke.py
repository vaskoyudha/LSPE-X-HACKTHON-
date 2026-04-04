import src

import pytest


@pytest.mark.p0
def test_src_importable():
    assert src.__version__ == "0.1.0"
    assert src.RANDOM_SEED == 42
