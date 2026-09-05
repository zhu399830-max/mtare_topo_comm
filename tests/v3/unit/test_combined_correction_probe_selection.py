from __future__ import annotations

import copy

import pytest

from mtare_topo.evaluation.combined_correction_probe_selection import (
    select_combined_correction_probe_case,
)


def _cases():
    cases = []
    for index in range(30):
        cases.append({
            "case_id": f"case_{index:02d}",
            "world": "tunnel" if index % 2 == 0 else "garage",
            "environment_seed": index,
            "checkpoint_seed": index % 3,
            "final_explored_volume_m3": float(10000 - index),
            "traveling_distance_m": float(index),
            "reanchor_evidence": {"first_proxy_frame": None},
            "frontier_attempt": {"events": []},
        })
    cases[7]["reanchor_evidence"]["first_proxy_frame"] = 80
    cases[7]["frontier_attempt"]["events"] = [
        {"frame_index": 120, "outcome": "divergent_verified_departure"}
    ]
    cases[9]["reanchor_evidence"]["first_proxy_frame"] = 100
    cases[9]["frontier_attempt"]["events"] = [
        {"frame_index": 110, "outcome": "same_node_loop_merge"}
    ]
    return cases


def test_selects_earliest_frame_where_both_mechanisms_have_appeared():
    selected = select_combined_correction_probe_case(_cases())
    assert selected["selected_case"]["case_id"] == "case_09"
    assert selected["selected_case"]["both_mechanisms_observed_by_frame"] == 110
    assert selected["eligible_case_count"] == 2
    assert selected["performance_outcomes_used_for_selection"] == []


def test_selection_is_invariant_to_coverage_and_travel_outcomes():
    first = _cases()
    second = copy.deepcopy(first)
    for index, case in enumerate(second):
        case["final_explored_volume_m3"] = float(index * 1_000_000)
        case["traveling_distance_m"] = float(1_000_000 - index)
    assert (
        select_combined_correction_probe_case(first)["selected_case"]
        == select_combined_correction_probe_case(second)["selected_case"]
    )


def test_rejects_incomplete_or_malformed_mechanism_evidence():
    with pytest.raises(ValueError, match="exactly 30"):
        select_combined_correction_probe_case(_cases()[:-1])
    malformed = _cases()
    malformed[0]["frontier_attempt"]["events"] = None
    with pytest.raises(ValueError, match="events are missing"):
        select_combined_correction_probe_case(malformed)
