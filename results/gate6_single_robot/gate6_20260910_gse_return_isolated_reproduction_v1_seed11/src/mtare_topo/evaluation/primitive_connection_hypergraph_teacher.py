"""Teacher feasibility checks for endpoint-to-connection-cluster decoding.

The P1b Teacher stores, for every visible primitive endpoint, at most three
other endpoints incident on the same procedural graph node.  A connection
cluster is learnable without construction identity only when this directed
neighbor storage is a disjoint union of complete, symmetric endpoint graphs.
This module validates that contract and summarizes the cluster population
without reading sensor input or running a model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


MAXIMUM_PRIMITIVES = 32
ENDPOINTS_PER_PRIMITIVE = 2
MAXIMUM_ENDPOINTS = MAXIMUM_PRIMITIVES * ENDPOINTS_PER_PRIMITIVE
MAXIMUM_NEIGHBORS = 3
CAPACITY_CANDIDATES = (4, 8, 16, 32)
CAPACITY_MARGIN = 1.25
_SENTINEL = MAXIMUM_ENDPOINTS


@dataclass(frozen=True)
class ConnectionClusterBatchSummary:
    """Additive counts from one row batch.

    Histograms use fixed bins so summaries from different shards can be
    merged without retaining row-level Teacher data.
    """

    rows: int
    active_primitives: int
    active_endpoints: int
    observed_endpoints: int
    directed_attachment_labels: int
    undirected_attachment_labels: int
    observable_undirected_attachment_labels: int
    physical_clusters: int
    observable_multimember_clusters: int
    observed_singletons_from_physical_multimember: int
    fully_hidden_physical_clusters: int
    isolated_active_endpoints: int
    isolated_observed_endpoints: int
    pair_candidates: int
    dense_assignment_candidates_by_capacity: tuple[int, ...]
    supervised_pair_positive_labels: int
    supervised_cluster_memberships: int
    observable_cluster_memberships: int
    cluster_count_histogram: tuple[int, ...]
    cluster_size_histogram: tuple[int, ...]
    observed_member_count_histogram: tuple[int, ...]
    maximum_clusters_per_row: int
    maximum_cluster_size: int
    canonical_padding: bool
    symmetric: bool
    clique_consistent: bool
    active_destination_only: bool
    cross_primitive_only: bool

    @property
    def passed(self) -> bool:
        return all((
            self.canonical_padding,
            self.symmetric,
            self.clique_consistent,
            self.active_destination_only,
            self.cross_primitive_only,
        ))

    def to_dict(self) -> dict[str, object]:
        return {
            "rows": self.rows,
            "active_primitives": self.active_primitives,
            "active_endpoints": self.active_endpoints,
            "observed_endpoints": self.observed_endpoints,
            "directed_attachment_labels": self.directed_attachment_labels,
            "undirected_attachment_labels": self.undirected_attachment_labels,
            "observable_undirected_attachment_labels": (
                self.observable_undirected_attachment_labels
            ),
            "physical_clusters": self.physical_clusters,
            "observable_multimember_clusters": self.observable_multimember_clusters,
            "observed_singletons_from_physical_multimember": (
                self.observed_singletons_from_physical_multimember
            ),
            "fully_hidden_physical_clusters": self.fully_hidden_physical_clusters,
            "isolated_active_endpoints": self.isolated_active_endpoints,
            "isolated_observed_endpoints": self.isolated_observed_endpoints,
            "pair_candidates": self.pair_candidates,
            "dense_assignment_candidates_by_capacity": {
                str(capacity): int(value)
                for capacity, value in zip(
                    CAPACITY_CANDIDATES,
                    self.dense_assignment_candidates_by_capacity,
                    strict=True,
                )
            },
            "supervised_pair_positive_labels": self.supervised_pair_positive_labels,
            "supervised_cluster_memberships": self.supervised_cluster_memberships,
            "observable_cluster_memberships": self.observable_cluster_memberships,
            "cluster_count_histogram": list(self.cluster_count_histogram),
            "cluster_size_histogram": list(self.cluster_size_histogram),
            "observed_member_count_histogram": list(self.observed_member_count_histogram),
            "maximum_clusters_per_row": self.maximum_clusters_per_row,
            "maximum_cluster_size": self.maximum_cluster_size,
            "checks": {
                "canonical_padding": self.canonical_padding,
                "symmetric": self.symmetric,
                "clique_consistent": self.clique_consistent,
                "active_destination_only": self.active_destination_only,
                "cross_primitive_only": self.cross_primitive_only,
            },
            "passed": self.passed,
        }


def _binary(name: str, value: np.ndarray, shape: tuple[int, ...]) -> np.ndarray:
    result = np.asarray(value, dtype=np.uint8)
    if result.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.any((result != 0) & (result != 1)):
        raise ValueError(f"{name} must be binary")
    return result


def _histogram(values: np.ndarray, length: int) -> tuple[int, ...]:
    return tuple(
        int(value) for value in np.bincount(
            np.asarray(values, dtype=np.int64), minlength=length,
        )[:length]
    )


def audit_connection_cluster_batch(
    primitive_mask: np.ndarray,
    endpoint_neighbor: np.ndarray,
    endpoint_observed: np.ndarray,
) -> ConnectionClusterBatchSummary:
    """Validate and summarize one batch of compact P1b endpoint relations.

    ``endpoint_observed`` is the frozen dual-endpoint evidence sidecar.  Full
    physical clusters are audited first; observable clusters are then the
    induced subgraphs on directly evidenced endpoints.  No hidden endpoint is
    converted into a negative target.
    """

    primitive = np.asarray(primitive_mask, dtype=np.uint8)
    if primitive.ndim != 2 or primitive.shape[1:] != (MAXIMUM_PRIMITIVES,):
        raise ValueError("primitive_mask must have shape [batch,32]")
    batch = len(primitive)
    primitive = _binary("primitive_mask", primitive, (batch, MAXIMUM_PRIMITIVES))
    observed = _binary(
        "endpoint_observed",
        endpoint_observed,
        (batch, MAXIMUM_PRIMITIVES, ENDPOINTS_PER_PRIMITIVE),
    ).reshape(batch, MAXIMUM_ENDPOINTS).astype(bool)
    active = np.repeat(primitive.astype(bool), ENDPOINTS_PER_PRIMITIVE, axis=1)
    if np.any(observed & ~active):
        raise ValueError("inactive primitive endpoint cannot be observed")

    neighbor = np.asarray(endpoint_neighbor, dtype=np.int16)
    expected = (batch, MAXIMUM_PRIMITIVES, ENDPOINTS_PER_PRIMITIVE, MAXIMUM_NEIGHBORS)
    if neighbor.shape != expected:
        raise ValueError(f"endpoint_neighbor must have shape {expected}")
    flat = neighbor.reshape(batch, MAXIMUM_ENDPOINTS, MAXIMUM_NEIGHBORS)
    valid = flat >= 0
    in_range = (~valid) | (flat < MAXIMUM_ENDPOINTS)
    if not np.all(in_range):
        raise ValueError("endpoint neighbor lies outside [0,63]")

    # The materializer writes ascending unique destinations followed by -1.
    canonical_values = np.where(valid, flat, _SENTINEL)
    canonical_padding = bool(
        not np.any((~valid[:, :, :-1]) & valid[:, :, 1:])
        and np.all(
            (~(valid[:, :, :-1] & valid[:, :, 1:]))
            | (np.diff(canonical_values, axis=2) > 0)
        )
    )
    sources = np.arange(MAXIMUM_ENDPOINTS, dtype=np.int16)[None, :, None]
    no_self = bool(np.all((~valid) | (flat != sources)))
    cross_primitive_only = no_self and bool(np.all(
        (~valid) | ((flat // ENDPOINTS_PER_PRIMITIVE) != (sources // ENDPOINTS_PER_PRIMITIVE))
    ))
    source_active = active[:, :, None]
    safe_destination = np.where(valid, flat, 0)
    row_index = np.arange(batch, dtype=np.int64)[:, None, None]
    destination_active = active[row_index, safe_destination]
    active_destination_only = bool(np.all(
        (~valid) | (source_active & destination_active)
    ))

    # A graph is a disjoint union of cliques iff every adjacent pair has the
    # same closed neighborhood.  This avoids materializing a 64x64 matrix per
    # row and is exact for the frozen degree<=4 construction.
    closed = np.concatenate((sources + np.zeros((batch, 1, 1), dtype=np.int16), flat), axis=2)
    closed = np.where(np.concatenate((active[:, :, None], valid), axis=2), closed, _SENTINEL)
    closed.sort(axis=2)
    gathered_closed = closed[row_index, safe_destination]
    source_closed = closed[:, :, None, :]
    edge_closed_equal = np.all(gathered_closed == source_closed, axis=3)
    clique_consistent = bool(np.all((~valid) | edge_closed_equal))
    symmetric = clique_consistent

    degree = np.sum(valid, axis=2, dtype=np.int16)
    representative = closed[:, :, 0]
    endpoint_index = np.arange(MAXIMUM_ENDPOINTS, dtype=np.int16)[None, :]
    representative_mask = active & (degree > 0) & (representative == endpoint_index)
    cluster_count = np.sum(representative_mask, axis=1, dtype=np.int16)
    cluster_size = degree + 1
    physical_sizes = cluster_size[representative_mask]

    safe_closed = np.where(closed < MAXIMUM_ENDPOINTS, closed, 0)
    member_observed = observed[
        np.arange(batch, dtype=np.int64)[:, None, None], safe_closed,
    ] & (closed < MAXIMUM_ENDPOINTS)
    observed_members = np.sum(member_observed, axis=2, dtype=np.int16)
    physical_observed_members = observed_members[representative_mask]

    active_primitives_per_row = np.sum(primitive, axis=1, dtype=np.int64)
    pair_candidates = 2 * active_primitives_per_row * (active_primitives_per_row - 1)
    active_endpoints_per_row = 2 * active_primitives_per_row
    assignment = tuple(
        int(np.sum(active_endpoints_per_row * capacity))
        for capacity in CAPACITY_CANDIDATES
    )
    directed_labels = int(np.sum(valid))
    supervised_memberships = int(np.sum(physical_sizes))
    observable_pairs = int(np.sum(
        physical_observed_members * (physical_observed_members - 1) // 2
    ))

    return ConnectionClusterBatchSummary(
        rows=batch,
        active_primitives=int(np.sum(active_primitives_per_row)),
        active_endpoints=int(np.sum(active_endpoints_per_row)),
        observed_endpoints=int(np.sum(observed)),
        directed_attachment_labels=directed_labels,
        undirected_attachment_labels=directed_labels // 2,
        observable_undirected_attachment_labels=observable_pairs,
        physical_clusters=int(np.sum(representative_mask)),
        observable_multimember_clusters=int(np.sum(physical_observed_members >= 2)),
        observed_singletons_from_physical_multimember=int(np.sum(physical_observed_members == 1)),
        fully_hidden_physical_clusters=int(np.sum(physical_observed_members == 0)),
        isolated_active_endpoints=int(np.sum(active & (degree == 0))),
        isolated_observed_endpoints=int(np.sum(observed & (degree == 0))),
        pair_candidates=int(np.sum(pair_candidates)),
        dense_assignment_candidates_by_capacity=assignment,
        supervised_pair_positive_labels=directed_labels // 2,
        supervised_cluster_memberships=supervised_memberships,
        observable_cluster_memberships=int(np.sum(physical_observed_members)),
        cluster_count_histogram=_histogram(cluster_count, MAXIMUM_ENDPOINTS + 1),
        cluster_size_histogram=_histogram(physical_sizes, MAXIMUM_NEIGHBORS + 2),
        observed_member_count_histogram=_histogram(
            physical_observed_members, MAXIMUM_NEIGHBORS + 2,
        ),
        maximum_clusters_per_row=int(np.max(cluster_count, initial=0)),
        maximum_cluster_size=int(np.max(physical_sizes, initial=0)),
        canonical_padding=canonical_padding,
        symmetric=symmetric,
        clique_consistent=clique_consistent,
        active_destination_only=active_destination_only,
        cross_primitive_only=cross_primitive_only,
    )


def merge_connection_cluster_summaries(
    summaries: Iterable[ConnectionClusterBatchSummary],
) -> ConnectionClusterBatchSummary:
    values = tuple(summaries)
    if not values:
        raise ValueError("at least one cluster summary is required")

    def total(name: str) -> int:
        return sum(int(getattr(value, name)) for value in values)

    def vector(name: str) -> tuple[int, ...]:
        rows = tuple(getattr(value, name) for value in values)
        return tuple(sum(int(row[index]) for row in rows) for index in range(len(rows[0])))

    return ConnectionClusterBatchSummary(
        rows=total("rows"),
        active_primitives=total("active_primitives"),
        active_endpoints=total("active_endpoints"),
        observed_endpoints=total("observed_endpoints"),
        directed_attachment_labels=total("directed_attachment_labels"),
        undirected_attachment_labels=total("undirected_attachment_labels"),
        observable_undirected_attachment_labels=total(
            "observable_undirected_attachment_labels"
        ),
        physical_clusters=total("physical_clusters"),
        observable_multimember_clusters=total("observable_multimember_clusters"),
        observed_singletons_from_physical_multimember=total(
            "observed_singletons_from_physical_multimember"
        ),
        fully_hidden_physical_clusters=total("fully_hidden_physical_clusters"),
        isolated_active_endpoints=total("isolated_active_endpoints"),
        isolated_observed_endpoints=total("isolated_observed_endpoints"),
        pair_candidates=total("pair_candidates"),
        dense_assignment_candidates_by_capacity=vector(
            "dense_assignment_candidates_by_capacity"
        ),
        supervised_pair_positive_labels=total("supervised_pair_positive_labels"),
        supervised_cluster_memberships=total("supervised_cluster_memberships"),
        observable_cluster_memberships=total("observable_cluster_memberships"),
        cluster_count_histogram=vector("cluster_count_histogram"),
        cluster_size_histogram=vector("cluster_size_histogram"),
        observed_member_count_histogram=vector("observed_member_count_histogram"),
        maximum_clusters_per_row=max(value.maximum_clusters_per_row for value in values),
        maximum_cluster_size=max(value.maximum_cluster_size for value in values),
        canonical_padding=all(value.canonical_padding for value in values),
        symmetric=all(value.symmetric for value in values),
        clique_consistent=all(value.clique_consistent for value in values),
        active_destination_only=all(value.active_destination_only for value in values),
        cross_primitive_only=all(value.cross_primitive_only for value in values),
    )


def select_fit_only_cluster_capacity(
    maximum_fit_clusters_per_row: int,
    *,
    margin: float = CAPACITY_MARGIN,
    candidates: tuple[int, ...] = CAPACITY_CANDIDATES,
) -> dict[str, int | float | tuple[int, ...]]:
    """Freeze the smallest registered capacity covering a fit-only margin."""

    maximum = int(maximum_fit_clusters_per_row)
    if maximum < 1 or margin < 1.0 or not candidates:
        raise ValueError("cluster capacity inputs are invalid")
    ordered = tuple(int(value) for value in candidates)
    if any(value < 1 for value in ordered) or any(
        second <= first for first, second in zip(ordered, ordered[1:])
    ):
        raise ValueError("cluster capacity candidates must be positive and increasing")
    required = int(np.ceil(maximum * float(margin)))
    selected = next((value for value in ordered if value >= required), None)
    if selected is None:
        raise OverflowError("fit cluster population exceeds registered decoder capacities")
    return {
        "maximum_fit_clusters_per_row": maximum,
        "margin": float(margin),
        "required_capacity": required,
        "selected_capacity": selected,
        "candidates": ordered,
    }


__all__ = [
    "CAPACITY_CANDIDATES",
    "CAPACITY_MARGIN",
    "ConnectionClusterBatchSummary",
    "audit_connection_cluster_batch",
    "merge_connection_cluster_summaries",
    "select_fit_only_cluster_capacity",
]
