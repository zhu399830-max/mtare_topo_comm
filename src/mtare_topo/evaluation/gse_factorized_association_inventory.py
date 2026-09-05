"""Inventory helpers for factorized decision-node association."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


DECISION_EVENTS = ("junction", "terminal")


def past_observation_lengths(
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    *,
    maximum: int = 5,
) -> np.ndarray:
    """Return available causal observation count without crossing traversal."""

    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    if traversal.shape != sequence.shape or traversal.ndim != 1 or maximum <= 0:
        raise ValueError("causal profile length input contract drift")
    result = np.empty(len(traversal), dtype=np.int16)
    seen: set[str] = set()
    start = 0
    while start < len(traversal):
        current = str(traversal[start])
        if current in seen:
            raise ValueError("one traversal appears in multiple blocks")
        seen.add(current)
        end = start + 1
        while end < len(traversal) and traversal[end] == current:
            end += 1
        local = sequence[start:end]
        if not np.array_equal(local, np.arange(local[0], local[0] + len(local))):
            raise ValueError("traversal sequence is not consecutive")
        result[start:end] = np.minimum(np.arange(1, len(local) + 1), maximum)
        start = end
    return result


def identity_node_id(identity: str) -> str:
    parts = str(identity).split(":")
    if len(parts) < 3 or parts[-2] != "node" or not parts[-1]:
        raise ValueError("decision identity is not a node identity")
    return parts[-1]


def inbound_profile_identity_counts(
    rows: Sequence[Mapping[str, object]],
    profile_length: np.ndarray,
    partition: np.ndarray,
    *,
    required_length: int = 5,
) -> dict[str, int | float]:
    """Count decision identities with a causal approach profile on any edge."""

    selected = np.asarray(partition, dtype=np.bool_)
    length = np.asarray(profile_length, dtype=np.int64)
    if selected.shape != (len(rows),) or length.shape != (len(rows),):
        raise ValueError("inbound profile inventory alignment drift")
    identities: set[str] = set()
    covered: set[str] = set()
    covered_edges: set[tuple[str, str]] = set()
    for index, row in enumerate(rows):
        if not selected[index] or str(row["event"]) not in DECISION_EVENTS:
            continue
        identity = str(row["identity"])
        node = identity_node_id(identity)
        identities.add(identity)
        # Only a row approaching its to-node has a physically past profile on
        # this directed traversal.  Reverse traversals provide the opposite
        # edge direction without borrowing future observations.
        if str(row["to_node_id"]) == node and int(length[index]) >= required_length:
            covered.add(identity)
            covered_edges.add((identity, str(row["edge_id"])))
    return {
        "decision_identities": len(identities),
        "identities_with_inbound_profile": len(covered),
        "identity_coverage": len(covered) / len(identities) if identities else 0.0,
        "covered_identity_edge_pairs": len(covered_edges),
    }


def decision_pair_inventory(
    left: np.ndarray,
    right: np.ndarray,
    label: np.ndarray,
    distance_m: np.ndarray,
    family: Sequence[str],
    event: Sequence[str],
    identity: Sequence[str],
    edge_id: Sequence[str],
    *,
    maximum_distance_m: float = 16.0,
) -> dict:
    """Describe online-eligible decision-node positive and hard-negative pairs."""

    left = np.asarray(left, dtype=np.int64)
    right = np.asarray(right, dtype=np.int64)
    label = np.asarray(label, dtype=np.uint8)
    distance = np.asarray(distance_m, dtype=np.float64)
    families = np.asarray(family, dtype=str)
    events = np.asarray(event, dtype=str)
    identities = np.asarray(identity, dtype=str)
    edges = np.asarray(edge_id, dtype=str)
    pair_count = len(left)
    if (
        any(value.shape != (pair_count,) for value in (right, label, distance, families))
        or events.ndim != 1
        or identities.shape != events.shape
        or edges.shape != events.shape
        or np.any(left < 0)
        or np.any(right < 0)
        or np.any(left >= len(events))
        or np.any(right >= len(events))
        or np.any((label != 0) & (label != 1))
        or not np.all(np.isfinite(distance))
    ):
        raise ValueError("decision pair inventory contract drift")
    decision = np.isin(events, DECISION_EVENTS)
    selected = (distance <= maximum_distance_m) & decision[left] & decision[right]
    positive = selected & (label == 1)
    negative = selected & (label == 0)
    if np.any(positive & (identities[left] != identities[right])):
        raise RuntimeError("positive decision pair identity drift")
    per_family = {}
    for name in sorted(np.unique(families)):
        rows = selected & (families == name)
        per_family[str(name)] = {
            "pairs": int(np.sum(rows)),
            "positive": int(np.sum(rows & (label == 1))),
            "negative": int(np.sum(rows & (label == 0))),
        }
    return {
        "online_eligible_decision_pairs": int(np.sum(selected)),
        "positive_pairs": int(np.sum(positive)),
        "negative_pairs": int(np.sum(negative)),
        "positive_same_physical_edge": int(np.sum(positive & (edges[left] == edges[right]))),
        "positive_different_physical_edge": int(np.sum(positive & (edges[left] != edges[right]))),
        "negative_same_event_type": int(np.sum(negative & (events[left] == events[right]))),
        "per_family": per_family,
    }


__all__ = [
    "DECISION_EVENTS",
    "decision_pair_inventory",
    "identity_node_id",
    "inbound_profile_identity_counts",
    "past_observation_lengths",
]
