from __future__ import annotations

import numpy as np

from scripts.generate_review_benchmark import (
    _nearest_threshold_margin,
    _review_priority_score,
    _select_review_rows,
)


def test_nearest_threshold_margin_uses_closest_low_or_high_boundary() -> None:
    probs = np.array(
        [
            [0.48, 0.42, 0.10],
            [0.15, 0.30, 0.55],
            [0.05, 0.35, 0.60],
        ],
        dtype=float,
    )
    thresholds = {"high_risk": 0.60, "low_risk": 0.50}

    margins = _nearest_threshold_margin(probs, thresholds)

    np.testing.assert_allclose(margins, np.array([0.02, 0.05, 0.0]), atol=1e-9)



def test_review_priority_score_prefers_disagreement_uncertainty_and_high_risk() -> None:
    probs = np.array(
        [
            [0.05, 0.15, 0.80],
            [0.34, 0.33, 0.33],
            [0.88, 0.08, 0.04],
        ],
        dtype=float,
    )
    preds = np.array([2, 0, 0], dtype=int)
    heuristic = np.array([2, 2, 0], dtype=int)
    thresholds = {"high_risk": 0.60, "low_risk": 0.50}

    scores = _review_priority_score(probs, preds, heuristic, thresholds)

    assert scores[0] > scores[2]
    assert scores[1] > scores[2]



def test_select_review_rows_includes_disagreement_and_boundary_cases() -> None:
    probs = np.array(
        [
            [0.05, 0.15, 0.80],
            [0.52, 0.36, 0.12],
            [0.10, 0.35, 0.55],
            [0.36, 0.32, 0.32],
            [0.20, 0.50, 0.30],
            [0.12, 0.18, 0.70],
        ],
        dtype=float,
    )
    preds = np.array([2, 0, 1, 0, 1, 2], dtype=int)
    heuristic = np.array([2, 1, 2, 0, 1, 0], dtype=int)
    thresholds = {"high_risk": 0.60, "low_risk": 0.50}

    selected, reasons = _select_review_rows(probs, preds, heuristic, thresholds, n_rows=6)

    assert len(selected) == 6
    assert len(reasons) == 6
    assert set(selected.tolist()) == {0, 1, 2, 3, 4, 5}
    assert any(reason == "model_heuristic_disagreement" for reason in reasons)
    assert any(reason == "near_decision_threshold" for reason in reasons)
    assert any(reason == "priority_mix" for reason in reasons)
