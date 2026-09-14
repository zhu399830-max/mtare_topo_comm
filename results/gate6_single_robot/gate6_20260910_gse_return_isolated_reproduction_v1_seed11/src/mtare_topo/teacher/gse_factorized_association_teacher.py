"""Identity-balanced Teacher for factorized decision-node association."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from mtare_topo.evaluation.gse_factorized_association_inventory import (
    DECISION_EVENTS,
    identity_node_id,
)


PROFILE_OBSERVATIONS = 5
PROFILE_SCALES = np.asarray((30.0, 30.0, 45.0, 0.1), dtype=np.float64)


def causal_history_row_references(
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    *,
    maximum: int = PROFILE_OBSERVATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize left-padded observation references within one traversal."""

    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    if traversal.ndim != 1 or sequence.shape != traversal.shape or maximum <= 0:
        raise ValueError("history reference input contract drift")
    references = np.full((len(traversal), maximum), -1, dtype=np.int64)
    mask = np.zeros((len(traversal), maximum), dtype=np.bool_)
    seen: set[str] = set()
    start = 0
    while start < len(traversal):
        current = str(traversal[start])
        if current in seen:
            raise ValueError("one traversal appears in multiple row blocks")
        seen.add(current)
        end = start + 1
        while end < len(traversal) and traversal[end] == current:
            end += 1
        local_sequence = sequence[start:end]
        if not np.array_equal(local_sequence, np.arange(local_sequence[0], local_sequence[0] + len(local_sequence))):
            raise ValueError("sequence indices are not consecutive within traversal")
        for row in range(start, end):
            first = max(start, row - maximum + 1)
            values = np.arange(first, row + 1, dtype=np.int64)
            references[row, -len(values):] = values
            mask[row, -len(values):] = True
        start = end
    return references, mask


def causal_history_row_references_unordered(
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    *,
    maximum: int = PROFILE_OBSERVATIONS,
) -> tuple[np.ndarray, np.ndarray]:
    """Materialize causal references when frozen inference rows are shuffled.

    The output retains the caller's row order.  History is joined only by the
    typed ``(traversal_id, sequence_index)`` key and can never cross a
    traversal reset.
    """

    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    if traversal.ndim != 1 or sequence.shape != traversal.shape or maximum <= 0:
        raise ValueError("unordered history reference input contract drift")
    lookup: dict[tuple[str, int], int] = {}
    by_traversal: dict[str, list[int]] = {}
    for row, (name, current) in enumerate(zip(traversal, sequence, strict=True)):
        key = (str(name), int(current))
        if int(current) < 0 or key in lookup:
            raise ValueError("unordered history contains negative or duplicate identity")
        lookup[key] = row
        by_traversal.setdefault(str(name), []).append(int(current))
    for values in by_traversal.values():
        ordered = np.sort(np.asarray(values, dtype=np.int64))
        if not np.array_equal(ordered, np.arange(ordered[0], ordered[0] + len(ordered))):
            raise ValueError("unordered history sequence indices are not consecutive")
    references = np.full((len(traversal), maximum), -1, dtype=np.int64)
    mask = np.zeros((len(traversal), maximum), dtype=np.bool_)
    for row, (name, current) in enumerate(zip(traversal, sequence, strict=True)):
        first = max(min(by_traversal[str(name)]), int(current) - maximum + 1)
        values = [lookup[(str(name), value)] for value in range(first, int(current) + 1)]
        references[row, -len(values) :] = values
        mask[row, -len(values) :] = True
    return references, mask


def objective_geometry_profile(
    rows: Sequence[Mapping[str, object]],
    history_references: np.ndarray,
    history_mask: np.ndarray,
    row_index: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return fixed-scale mean/std profile features and explicit validity."""

    if (
        history_references.shape != history_mask.shape
        or history_references.ndim != 2
        or history_references.shape[0] != len(rows)
        or not 0 <= row_index < len(rows)
    ):
        raise ValueError("objective profile reference contract drift")
    indices = history_references[row_index, history_mask[row_index]]
    if len(indices) == 0 or np.any(indices < 0):
        raise ValueError("objective profile has no causal observation")
    geometry_valid = np.asarray([bool(rows[index]["geometry_valid"]) for index in indices])
    values = np.asarray([
        (
            float(rows[index]["width_m"]) if geometry_valid[offset] else 0.0,
            float(rows[index]["height_m"]) if geometry_valid[offset] else 0.0,
            float(rows[index]["slope_deg"]), float(rows[index]["curvature_per_m"]),
        )
        for offset, index in enumerate(indices)
    ], dtype=np.float64) / PROFILE_SCALES
    if not np.all(np.isfinite(values)):
        raise ValueError("objective profile geometry is nonfinite")
    dimension_valid = np.ones((len(indices), 4), dtype=np.bool_)
    dimension_valid[:, :2] = geometry_valid[:, None]
    mean = np.zeros(4, dtype=np.float64)
    std = np.zeros(4, dtype=np.float64)
    valid = np.zeros(4, dtype=np.bool_)
    for dimension in range(4):
        selected = values[dimension_valid[:, dimension], dimension]
        if len(selected):
            mean[dimension] = float(np.mean(selected))
            std[dimension] = float(np.std(selected))
            valid[dimension] = True
    return np.concatenate((mean, std)), np.concatenate((valid, valid))


def masked_profile_distance(
    left: np.ndarray,
    left_mask: np.ndarray,
    right: np.ndarray,
    right_mask: np.ndarray,
) -> float:
    """RMS distance over profile dimensions valid for both identities."""

    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    left_mask = np.asarray(left_mask, dtype=np.bool_)
    right_mask = np.asarray(right_mask, dtype=np.bool_)
    if left.shape != (8,) or right.shape != (8,) or left_mask.shape != (8,) or right_mask.shape != (8,):
        raise ValueError("masked profile distance contract drift")
    common = left_mask & right_mask
    # Slope and curvature mean/std are always objective-valid, so four common
    # dimensions are the minimum scientifically meaningful comparison.
    if int(np.sum(common)) < 4:
        raise ValueError("profiles lack four common objective dimensions")
    return float(np.sqrt(np.mean(np.square(left[common] - right[common]))))


def choose_positive_rows(
    rows: Sequence[Mapping[str, object]],
    indices: Sequence[int],
    history_mask: np.ndarray,
    association_valid: np.ndarray,
) -> tuple[int, int, str]:
    """Choose one deterministic cross-view positive pair for one identity."""

    candidates = [int(index) for index in indices if bool(association_valid[int(index)])]
    if not candidates:
        raise ValueError("decision identity lacks an association-valid observation")
    node = identity_node_id(str(rows[candidates[0]]["identity"]))

    def rank(index: int) -> tuple[int, int, int]:
        return (-int(np.sum(history_mask[index])), -int(bool(rows[index]["geometry_valid"])), int(rows[index]["global_sequence_index"]))

    inbound = [index for index in candidates if str(rows[index]["to_node_id"]) == node]
    if not inbound:
        raise ValueError("decision identity has no causal inbound observation")
    query = min(inbound, key=rank)
    if len(candidates) == 1:
        # Five objectively short terminal approaches in C01-C08 contain only
        # one 1 m-sampled terminal observation. Retain those identities with
        # an explicit circular-shift augmentation contract rather than
        # dropping them or calling a repeated tensor a physical revisit.
        if str(rows[query]["event"]) != "terminal" or int(np.sum(history_mask[query])) < PROFILE_OBSERVATIONS:
            raise ValueError("singleton decision identity is not a full-history terminal")
        return query, query, "singleton_circular_shift_augmentation"
    other_edge = [index for index in inbound if str(rows[index]["edge_id"]) != str(rows[query]["edge_id"])]
    if other_edge:
        positive = min(other_edge, key=rank)
        kind = "different_physical_edge"
    else:
        outbound = [
            index for index in candidates
            if index != query and str(rows[index]["from_node_id"]) == node
            and str(rows[index]["traversal_id"]) != str(rows[query]["traversal_id"])
        ]
        if outbound:
            positive = min(outbound, key=rank)
            kind = "same_edge_reverse_view"
        else:
            remaining = [index for index in candidates if index != query]
            positive = min(remaining, key=rank)
            kind = "same_identity_distinct_observation"
    return query, positive, kind


def choose_hard_negative_identity(
    query_identity: str,
    metadata: Mapping[str, Mapping[str, object]],
) -> tuple[str, float]:
    """Choose nearest different identity with identical event and degree."""

    query = metadata[query_identity]
    candidates: list[tuple[float, str]] = []
    for identity, candidate in metadata.items():
        if (
            identity == query_identity
            or candidate["partition"] != query["partition"]
            or candidate["event"] != query["event"]
            or int(candidate["degree"]) != int(query["degree"])
        ):
            continue
        distance = masked_profile_distance(
            np.asarray(query["profile"]), np.asarray(query["profile_mask"]),
            np.asarray(candidate["profile"]), np.asarray(candidate["profile_mask"]),
        )
        candidates.append((distance, identity))
    if not candidates:
        raise ValueError("decision identity has no same-event/degree hard-negative candidate")
    distance, identity = min(candidates, key=lambda value: (value[0], value[1]))
    return identity, float(distance)


__all__ = [
    "PROFILE_OBSERVATIONS",
    "PROFILE_SCALES",
    "causal_history_row_references",
    "causal_history_row_references_unordered",
    "choose_hard_negative_identity",
    "choose_positive_rows",
    "masked_profile_distance",
    "objective_geometry_profile",
]
