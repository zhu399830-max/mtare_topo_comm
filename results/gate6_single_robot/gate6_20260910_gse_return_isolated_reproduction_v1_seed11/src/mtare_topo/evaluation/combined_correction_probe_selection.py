"""Deterministic, outcome-blind selection for the combined-correction probe."""

from __future__ import annotations

from typing import Any, Mapping, Sequence


NONMATCHING_OUTCOMES = {
    "divergent_verified_departure",
    "same_node_loop_merge",
}


def select_combined_correction_probe_case(
    cases: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the case where both diagnosed mechanisms appear earliest.

    The rule intentionally ignores coverage, travel, latency and every other
    performance outcome.  It uses only causal mechanism-event frame indices.
    """

    if len(cases) != 30:
        raise ValueError("combined-correction probe selection requires exactly 30 V9 cases")
    case_ids = [str(case.get("case_id", "")) for case in cases]
    if any(not case_id for case_id in case_ids) or len(set(case_ids)) != len(case_ids):
        raise ValueError("combined-correction probe case identities must be unique and nonempty")

    candidates = []
    for case in cases:
        case_id = str(case["case_id"])
        reanchor = case.get("reanchor_evidence")
        attempt = case.get("frontier_attempt")
        if not isinstance(reanchor, Mapping) or not isinstance(attempt, Mapping):
            raise ValueError(f"combined-correction mechanism evidence is missing: {case_id}")
        first_proxy_raw = reanchor.get("first_proxy_frame")
        events = attempt.get("events")
        if not isinstance(events, list):
            raise ValueError(f"combined-correction frontier events are missing: {case_id}")
        nonmatching_frames = []
        for event in events:
            if not isinstance(event, Mapping):
                raise ValueError(f"combined-correction frontier event is malformed: {case_id}")
            if event.get("outcome") in NONMATCHING_OUTCOMES:
                frame = event.get("frame_index")
                if type(frame) is not int or frame < 0:
                    raise ValueError(f"combined-correction frontier frame is invalid: {case_id}")
                nonmatching_frames.append(frame)
        if first_proxy_raw is None or not nonmatching_frames:
            continue
        if type(first_proxy_raw) is not int or first_proxy_raw < 0:
            raise ValueError(f"combined-correction proxy frame is invalid: {case_id}")
        first_nonmatching = min(nonmatching_frames)
        candidates.append({
            "case_id": case_id,
            "world": str(case["world"]),
            "environment_seed": int(case["environment_seed"]),
            "checkpoint_seed": int(case["checkpoint_seed"]),
            "first_reanchor_proxy_frame": first_proxy_raw,
            "first_nonmatching_frontier_event_frame": first_nonmatching,
            "both_mechanisms_observed_by_frame": max(first_proxy_raw, first_nonmatching),
        })
    if not candidates:
        raise ValueError("no V9 case contains both diagnosed correction mechanisms")
    candidates.sort(key=lambda row: (row["both_mechanisms_observed_by_frame"], row["case_id"]))
    return {
        "schema_version": "combined_correction_probe_selection_v1",
        "selection_rule": "minimize_max_first_mechanism_frame_then_case_id",
        "performance_outcomes_used_for_selection": [],
        "eligible_case_count": len(candidates),
        "selected_case": candidates[0],
        "ranked_candidates": candidates,
    }


__all__ = ["NONMATCHING_OUTCOMES", "select_combined_correction_probe_case"]
