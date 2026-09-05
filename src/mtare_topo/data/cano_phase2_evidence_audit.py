"""Pure helpers for the Phase-2 sealed-dataset corrective evidence audit."""

from __future__ import annotations

from collections import Counter
from typing import Any, Mapping, Sequence

import numpy as np


def compare_scan_replay(
    stored_range: np.ndarray,
    stored_valid: np.ndarray,
    audit_range_a: np.ndarray,
    audit_valid_a: np.ndarray,
    audit_range_b: np.ndarray,
    audit_valid_b: np.ndarray,
    threshold_m: float = 1e-6,
) -> dict[str, Any]:
    """Compare a sealed scan against two independently constructed scenes."""

    arrays = [stored_range, stored_valid, audit_range_a, audit_valid_a, audit_range_b, audit_valid_b]
    if any(np.asarray(value).shape != (16, 720) for value in arrays):
        raise ValueError("all scan arrays must have shape (16, 720)")
    stored_valid_bool = np.asarray(stored_valid, dtype=bool)
    valid_a_bool = np.asarray(audit_valid_a, dtype=bool)
    valid_b_bool = np.asarray(audit_valid_b, dtype=bool)

    def maximum(left: np.ndarray, right: np.ndarray, mask: np.ndarray) -> float:
        if not np.any(mask):
            return 0.0
        return float(np.max(np.abs(np.asarray(left)[mask] - np.asarray(right)[mask])))

    stored_a_mask = stored_valid_bool & valid_a_bool
    a_b_mask = valid_a_bool & valid_b_bool
    stored_a_max = maximum(stored_range, audit_range_a, stored_a_mask)
    a_b_max = maximum(audit_range_a, audit_range_b, a_b_mask)
    stored_valid_equal = bool(np.array_equal(stored_valid_bool, valid_a_bool))
    independent_valid_equal = bool(np.array_equal(valid_a_bool, valid_b_bool))
    passed = bool(
        stored_valid_equal
        and independent_valid_equal
        and stored_a_max <= threshold_m
        and a_b_max <= threshold_m
    )
    return {
        "passed": passed,
        "stored_valid_equal": stored_valid_equal,
        "independent_valid_equal": independent_valid_equal,
        "stored_to_a_max_difference_m": stored_a_max,
        "scene_a_to_b_max_difference_m": a_b_max,
        "valid_ratio": float(stored_valid_bool.mean()),
        "threshold_m": float(threshold_m),
    }


def frame_failure_reasons(
    evaluation: Mapping[str, Any], replay: Mapping[str, Any]
) -> list[str]:
    """Return explicit frozen-contract failure reasons for one candidate frame."""

    reasons: list[str] = []
    branch_count = int(evaluation["branch_count"])
    if not 1 <= branch_count <= 8:
        reasons.append("branch_count_outside_1_8")
    if float(evaluation["minimum_horizontal_clearance_m"]) < 0.8:
        reasons.append("minimum_horizontal_clearance_below_0p8m")
    los = list(evaluation.get("branch_los", []))
    if not los:
        reasons.append("no_representative_branch_los")
    elif not all(bool(value) for value in los):
        reasons.append("representative_branch_los_failed")
    if replay.get("passed") is not True:
        reasons.append("independent_scene_replay_failed")
    return reasons


def selection_identity_audit(
    recreated: Sequence[Mapping[str, Any]], sealed: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    recreated_ids = [str(item["cluster_id"]) for item in recreated]
    sealed_ids = [str(item["cluster_id"]) for item in sealed]
    return {
        "passed": recreated_ids == sealed_ids,
        "recreated_count": len(recreated_ids),
        "sealed_count": len(sealed_ids),
        "first_mismatch_index": next(
            (index for index, pair in enumerate(zip(recreated_ids, sealed_ids)) if pair[0] != pair[1]),
            None if len(recreated_ids) == len(sealed_ids) else min(len(recreated_ids), len(sealed_ids)),
        ),
        "recreated_role_counts": dict(Counter(str(item["primary_role"]) for item in recreated)),
        "sealed_role_counts": dict(Counter(str(item["primary_role"]) for item in sealed)),
    }


def event_coverage(clusters: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    junctions = {str(value) for item in clusters for value in item.get("junction_event_ids", []) if item["primary_role"] == "junction"}
    terminals = {str(value) for item in clusters for value in item.get("terminal_event_ids", []) if item["primary_role"] == "terminal"}
    return {
        "junction_event_count": len(junctions),
        "terminal_event_count": len(terminals),
        "junction_event_ids": sorted(junctions),
        "terminal_event_ids": sorted(terminals),
    }
