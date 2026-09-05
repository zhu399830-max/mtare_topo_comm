"""Exchangeable local composition-slot Teacher utilities.

The representation assigns every endpoint participating in an observable
physical composition to one exchangeable slot.  Endpoints without a complete
observable connection use the dustbin label ``-1``.  Pair attachments are
decoded only as equality of non-dustbin slot labels.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np


ENDPOINTS = 64
CAPACITY_CANDIDATES = (8, 16, 32)
CAPACITY_MARGIN = 1.25


@dataclass(frozen=True)
class CompositionSlotDecomposition:
    labels: np.ndarray
    cluster_sizes: tuple[int, ...]
    observed_attachment: np.ndarray
    reconstructed_attachment: np.ndarray
    is_clique_partition: bool
    overlap_violations: int

    @property
    def cluster_count(self) -> int:
        return len(self.cluster_sizes)


def decode_slot_attachment(labels: np.ndarray) -> np.ndarray:
    value = np.asarray(labels, dtype=np.int16)
    if value.shape != (ENDPOINTS,) or np.any(value < -1):
        raise ValueError("composition-slot labels must be int [64] with dustbin=-1")
    decoded = (value[:, None] >= 0) & (value[:, None] == value[None, :])
    np.fill_diagonal(decoded, False)
    return decoded


def decompose_observable_attachment(
    attachment: np.ndarray,
    endpoint_observed: np.ndarray,
    disconnected_overlap: np.ndarray,
) -> CompositionSlotDecomposition:
    relation = np.asarray(attachment, dtype=np.bool_)
    observed = np.asarray(endpoint_observed, dtype=np.bool_)
    overlap = np.asarray(disconnected_overlap, dtype=np.bool_)
    if relation.shape != (ENDPOINTS, ENDPOINTS):
        raise ValueError("endpoint attachment must be [64,64]")
    if observed.shape != (ENDPOINTS,) or overlap.shape != (ENDPOINTS, ENDPOINTS):
        raise ValueError("endpoint observability/overlap shape drift")
    if not np.array_equal(relation, relation.T) or np.any(np.diag(relation)):
        raise ValueError("endpoint attachment must be symmetric without self loops")
    if not np.array_equal(overlap, overlap.T):
        raise ValueError("disconnected overlap must be symmetric")
    qualified = relation & observed[:, None] & observed[None, :]
    labels = np.full(ENDPOINTS, -1, dtype=np.int16)
    visited = np.zeros(ENDPOINTS, dtype=np.bool_)
    cluster_sizes: list[int] = []
    is_clique = True
    for seed in np.flatnonzero(qualified.any(axis=1)):
        if visited[seed]:
            continue
        stack = [int(seed)]; component: list[int] = []
        visited[seed] = True
        while stack:
            current = stack.pop(); component.append(current)
            for neighbor in np.flatnonzero(qualified[current]):
                neighbor = int(neighbor)
                if not visited[neighbor]:
                    visited[neighbor] = True; stack.append(neighbor)
        component.sort()
        index = len(cluster_sizes)
        labels[component] = index
        cluster_sizes.append(len(component))
        block = qualified[np.ix_(component, component)].copy()
        np.fill_diagonal(block, True)
        is_clique = is_clique and bool(block.all())
    reconstructed = decode_slot_attachment(labels)
    violations = int(np.count_nonzero(np.triu(reconstructed & overlap, k=1)))
    return CompositionSlotDecomposition(
        labels=labels,
        cluster_sizes=tuple(cluster_sizes),
        observed_attachment=qualified,
        reconstructed_attachment=reconstructed,
        is_clique_partition=is_clique,
        overlap_violations=violations,
    )


def select_slot_capacity(maximum_fit_clusters: int) -> dict:
    if maximum_fit_clusters < 0:
        raise ValueError("maximum fit cluster count must be nonnegative")
    required_with_margin = int(math.ceil(maximum_fit_clusters * CAPACITY_MARGIN))
    selected = next(
        (candidate for candidate in CAPACITY_CANDIDATES if candidate >= required_with_margin),
        None,
    )
    return {
        "maximum_fit_clusters": int(maximum_fit_clusters),
        "margin": CAPACITY_MARGIN,
        "required_with_margin": required_with_margin,
        "candidates": list(CAPACITY_CANDIDATES),
        "selected_capacity": selected,
        "available": selected is not None,
    }


__all__ = [
    "CAPACITY_CANDIDATES", "CAPACITY_MARGIN", "CompositionSlotDecomposition",
    "decode_slot_attachment", "decompose_observable_attachment", "select_slot_capacity",
]
