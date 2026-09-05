from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]


def _load():
    tools = str(ROOT / "tools/v3")
    if tools not in sys.path:
        sys.path.insert(0, tools)
    spec = importlib.util.spec_from_file_location(
        "run_frontier_attempt_outcome_audit_v1",
        ROOT / "tools/v3/run_frontier_attempt_outcome_audit_v1.py",
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _case(outcomes, events, *, lookahead=4.0, event_travel=8.0):
    nonmatching = sum(
        count for outcome, count in outcomes.items()
        if outcome != "matched_verified_departure"
    )
    return {
        "waypoint_lookahead_m": lookahead,
        "minimum_event_travel_m": event_travel,
        "frontier_attempt": {
            "outcome_counts": outcomes,
            "nonmatching_attempt_event_count": nonmatching,
            "events": events,
        },
    }


def _event(outcome, travel):
    return {
        "outcome": outcome,
        "target_run_route_arc_m_before_event": travel,
    }


def test_aggregate_uses_only_frozen_existing_distance_scales():
    module = _load()
    first = _case(
        {"matched_verified_departure": 1, "divergent_verified_departure": 2},
        [_event("matched_verified_departure", 9.0), _event("divergent_verified_departure", 5.0), _event("divergent_verified_departure", 9.0)],
    )
    remaining = [_case({}, []) for _ in range(29)]
    value = module.summarize_frontier_attempt_cases([first, *remaining])
    assert value["case_count"] == 30
    assert value["case_count_with_nonmatching_attempt"] == 1
    assert value["nonmatching_attempt_event_count"] == 2
    assert value["nonmatching_at_least_frozen_waypoint_lookahead_count"] == 2
    assert value["nonmatching_at_least_frozen_minimum_event_travel_count"] == 1
    assert value["uses_new_tuned_threshold"] is False
    assert value["uses_evaluator_gt"] is False


def test_aggregate_rejects_incomplete_case_distribution():
    module = _load()
    with pytest.raises(RuntimeError, match="exactly 30"):
        module.summarize_frontier_attempt_cases([])
