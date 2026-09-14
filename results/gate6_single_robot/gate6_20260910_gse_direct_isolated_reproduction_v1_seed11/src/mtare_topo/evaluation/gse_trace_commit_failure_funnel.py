"""Read-only attribution helpers for trace-commit graph failures."""

from __future__ import annotations

from collections import defaultdict
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np


def cross_trace_pair_state(
    rows: Sequence[int],
    traversal_id: Sequence[str],
    xyz_m: np.ndarray,
    association_valid: np.ndarray,
    accepted_pairs: Mapping[tuple[int, int], bool],
    *,
    distance_cap_m: float = 4.0,
) -> dict[str, object]:
    """Summarize whether one identity can receive independent association support."""

    selected = sorted({int(value) for value in rows})
    traversal = np.asarray(traversal_id, dtype=str)
    xyz = np.asarray(xyz_m, dtype=np.float64)
    valid = np.asarray(association_valid, dtype=np.bool_)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or len(traversal) != len(xyz) or valid.shape != (len(xyz),):
        raise ValueError("failure-funnel association population drift")
    cross = []
    for left, right in combinations(selected, 2):
        if traversal[left] == traversal[right]:
            continue
        distance = float(np.linalg.norm(xyz[left] - xyz[right]))
        key = (left, right)
        cross.append({
            "left": left,
            "right": right,
            "distance_m": distance,
            "both_association_valid": bool(valid[left] and valid[right]),
            "within_radius": bool(valid[left] and valid[right] and distance <= distance_cap_m + 1e-12),
            "accepted": bool(accepted_pairs.get(key, False)),
        })
    return {
        "distinct_traversals": len({str(traversal[value]) for value in selected}),
        "cross_trace_pairs": len(cross),
        "minimum_distance_m": min((value["distance_m"] for value in cross), default=None),
        "has_within_radius_pair": any(value["within_radius"] for value in cross),
        "has_accepted_pair": any(value["accepted"] for value in cross),
        "accepted_pair_count": sum(value["accepted"] for value in cross),
    }


def classify_identity_failure(
    proposal_rows: Sequence[int],
    pair_state: Mapping[str, object],
    committed_hypotheses: Sequence[int],
) -> str:
    """Assign one mutually exclusive causal stage to a true node identity."""

    if committed_hypotheses:
        return "committed_duplicate" if len(set(committed_hypotheses)) > 1 else "committed_unique"
    if not proposal_rows:
        return "proposal_missing"
    if int(pair_state["distinct_traversals"]) < 2:
        return "insufficient_traversal_support"
    if not bool(pair_state["has_within_radius_pair"]):
        return "center_outside_4m"
    if not bool(pair_state["has_accepted_pair"]):
        return "association_verifier_reject"
    return "ambiguity_or_commit_logic"


def true_relations(
    rows: Iterable[int],
    traversal_id: Sequence[str],
    sequence_index: Sequence[int],
    identity: Sequence[str | None],
    event: Sequence[str],
) -> dict[tuple[str, str], set[str]]:
    """Return observed undirected semantic relations and supporting traces."""

    by_trace: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        by_trace[str(traversal_id[row])].append(int(row))
    relations: dict[tuple[str, str], set[str]] = defaultdict(set)
    for trace, selected in by_trace.items():
        selected.sort(key=lambda value: (int(sequence_index[value]), value))
        ordered = []
        for row in selected:
            name = identity[row]
            if event[row] not in ("junction", "terminal") or name is None:
                continue
            if not ordered or ordered[-1] != name:
                ordered.append(name)
        for left, right in zip(ordered, ordered[1:]):
            if left != right:
                relations[tuple(sorted((str(left), str(right))))].add(trace)
    return dict(relations)


def classify_relation_failure(
    relation: tuple[str, str],
    support_traces: set[str],
    proposal_rows_by_identity: Mapping[str, Sequence[int]],
    committed_hypotheses_by_identity: Mapping[str, Sequence[int]],
    committed_rows_by_identity_trace: Mapping[tuple[str, str], Sequence[int]],
    predicted_relations: set[tuple[str, str]],
) -> str:
    """Assign one mutually exclusive stage to an observed true edge relation."""

    left, right = relation
    if not proposal_rows_by_identity.get(left) or not proposal_rows_by_identity.get(right):
        return "endpoint_proposal_missing"
    if not committed_hypotheses_by_identity.get(left) or not committed_hypotheses_by_identity.get(right):
        return "endpoint_not_committed"
    common = any(
        committed_rows_by_identity_trace.get((left, trace))
        and committed_rows_by_identity_trace.get((right, trace))
        for trace in support_traces
    )
    if not common:
        return "no_common_committed_trace"
    if relation in predicted_relations:
        return "recovered"
    return "edge_assembly_miss"


__all__ = [
    "classify_identity_failure", "classify_relation_failure",
    "cross_trace_pair_state", "true_relations",
]
