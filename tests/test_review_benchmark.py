from __future__ import annotations

import numpy as np
import pandas as pd

from scripts.generate_review_benchmark import (
    _load_linked_evidence_by_ocid,
    _nearest_threshold_margin,
    _prioritize_evidence_rows,
    _review_priority_score,
    _select_review_rows,
    _summarize_evidence_records,
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



def test_summarize_evidence_records_compacts_families_and_sources() -> None:
    families, sources, has_official_evidence = _summarize_evidence_records(
        [
            {
                "label_family": "confirmed_fraud",
                "source_name": "kpk_procurement_case",
                "reviewer_needed": False,
            },
            {
                "label_family": "sanctioned_supplier",
                "source_name": "lkpp_inaproc_blacklist",
                "reviewer_needed": True,
            },
        ]
    )

    assert families == "confirmed_fraud|sanctioned_supplier"
    assert sources == "kpk_procurement_case|lkpp_inaproc_blacklist"
    assert has_official_evidence is True



def test_prioritize_evidence_rows_places_official_cases_first() -> None:
    raw = pd.DataFrame(
        {
            "ocid": ["ocds-a", "ocds-b", "ocds-c", "ocds-d"],
        }
    )
    evidence_by_ocid = {
        "ocds-b": [
            {
                "label_family": "confirmed_fraud",
                "source_name": "kpk_procurement_case",
                "reviewer_needed": False,
            }
        ],
        "ocds-c": [
            {
                "label_family": "sanctioned_supplier",
                "source_name": "lkpp_inaproc_blacklist",
                "reviewer_needed": True,
            }
        ],
        "ocds-d": [
            {
                "label_family": "confirmed_irregularity",
                "source_name": "bpk",
                "reviewer_needed": False,
            }
        ],
    }

    selected, reasons = _prioritize_evidence_rows(raw, evidence_by_ocid, limit=3)

    assert selected == [1, 3, 2]
    assert reasons == [
        "official_evidence_linked",
        "official_evidence_linked",
        "evidence_needs_review",
    ]



def test_load_linked_evidence_by_ocid_groups_rows_from_parquet(tmp_path) -> None:
    evidence_path = tmp_path / "linked_label_records.parquet"
    pd.DataFrame(
        [
            {
                "ocid": "ocds-1",
                "label_family": "confirmed_fraud",
                "source_name": "kpk_procurement_case",
                "reviewer_needed": False,
            },
            {
                "ocid": "ocds-1",
                "label_family": "sanctioned_supplier",
                "source_name": "lkpp_inaproc_blacklist",
                "reviewer_needed": True,
            },
            {
                "ocid": "ocds-2",
                "label_family": "reviewed_risk",
                "source_name": "manual_review",
                "reviewer_needed": True,
            },
        ]
    ).to_parquet(evidence_path, index=False)

    grouped = _load_linked_evidence_by_ocid(evidence_path)

    assert sorted(grouped.keys()) == ["ocds-1", "ocds-2"]
    assert len(grouped["ocds-1"]) == 2
    assert grouped["ocds-2"][0]["label_family"] == "reviewed_risk"
