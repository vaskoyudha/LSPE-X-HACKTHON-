# Leakage-Safe Evaluation Lane Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Report heuristic-risk, reviewed-risk, and confirmed-outcome evidence families as separate evaluation lanes without contaminating the baseline benchmark.

**Architecture:** Extend diagnostics with lane-specific summaries rather than merging label families. The heuristic lane keeps the current canonical `models/metrics.json` output, the reviewed lane reuses existing reviewed metrics and explanation validation, and the confirmed-outcome lane stays descriptive-only until a dedicated fraud-evidence model exists.

**Tech Stack:** Python 3.12, pandas, pytest.

---

## File Map

- Modify: `src/diagnostics.py`
- Modify: `scripts/run_diagnostics.py`
- Modify: `tests/test_diagnostics.py`
- Modify: `tests/test_smoke.py`
- Modify: `README.md`

## Requirements Summary

1. Heuristic, reviewed, and confirmed-outcome evidence must be reported in distinct lanes.
2. Confirmed outcomes must not be evaluated as if they were the same target as heuristic three-class labels.
3. Diagnostics must always emit a stable lane summary artifact, even when some lanes have no data.

## Acceptance Criteria

- `models/evaluation_lanes.json` is written by `scripts/run_diagnostics.py`
- lane summaries clearly distinguish label family and source
- confirmed-outcome lane is descriptive-only unless a dedicated model exists
- affected tests pass
