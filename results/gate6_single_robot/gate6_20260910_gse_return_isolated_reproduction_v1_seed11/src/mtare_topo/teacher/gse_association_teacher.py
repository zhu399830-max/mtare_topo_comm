"""Deterministic cross-direction identities for learned GSE association."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

import numpy as np

from mtare_topo.semantics.geometric_semantics import StructuralEvent


@dataclass(frozen=True)
class EdgeEventSample:
    traversal_id: str
    frame_index: int
    canonical_edge_arc_m: float

    def __post_init__(self) -> None:
        if not self.traversal_id:
            raise ValueError("traversal_id must be nonempty")
        if self.frame_index < 0:
            raise ValueError("frame_index must be nonnegative")
        if not math.isfinite(self.canonical_edge_arc_m) or self.canonical_edge_arc_m < 0.0:
            raise ValueError("canonical edge arc must be finite and nonnegative")

    @property
    def key(self) -> tuple[str, int]:
        return self.traversal_id, self.frame_index


@dataclass(frozen=True)
class EdgeEventIdentity:
    identity: str
    event: str
    canonical_start_arc_m: float
    canonical_end_arc_m: float
    canonical_center_arc_m: float
    sample_count: int
    traversal_count: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AssociationTeacherExample:
    observation_id: str
    parent_id: str
    traversal_id: str
    identity: str
    event: str
    geometry_signature: tuple[float, ...]

    def __post_init__(self) -> None:
        if not all((self.observation_id, self.parent_id, self.traversal_id, self.identity, self.event)):
            raise ValueError("association example identities must be nonempty")
        signature = np.asarray(self.geometry_signature, dtype=np.float64)
        if signature.ndim != 1 or len(signature) == 0 or not np.all(np.isfinite(signature)):
            raise ValueError("geometry_signature must be a nonempty finite vector")


@dataclass(frozen=True)
class AssociationTeacherPair:
    anchor_observation_id: str
    paired_observation_id: str
    same_identity: bool
    pair_kind: str
    geometry_distance: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def cluster_edge_event_identities(
    *,
    parent_id: str,
    edge_id: str,
    event: StructuralEvent,
    samples: Iterable[EdgeEventSample],
    maximum_gap_m: float = 2.0,
) -> tuple[dict[tuple[str, int], str], tuple[EdgeEventIdentity, ...]]:
    """Group the same physical turn/transition observed from both directions.

    Canonical arc is always measured from the edge's declared first node, so
    reverse traversals map back into the same one-dimensional coordinate.
    """

    if event not in {StructuralEvent.TURN, StructuralEvent.GEOMETRY_TRANSITION}:
        raise ValueError("edge event clustering is only for turn or geometry-transition")
    if not parent_id or not edge_id:
        raise ValueError("parent_id and edge_id must be nonempty")
    if not math.isfinite(float(maximum_gap_m)) or maximum_gap_m <= 0.0:
        raise ValueError("maximum_gap_m must be positive and finite")
    ordered = sorted(
        tuple(samples),
        key=lambda item: (item.canonical_edge_arc_m, item.traversal_id, item.frame_index),
    )
    if len({sample.key for sample in ordered}) != len(ordered):
        raise ValueError("edge event sample keys must be unique")
    if not ordered:
        return {}, ()
    groups: list[list[EdgeEventSample]] = [[ordered[0]]]
    for sample in ordered[1:]:
        if sample.canonical_edge_arc_m - groups[-1][-1].canonical_edge_arc_m <= maximum_gap_m:
            groups[-1].append(sample)
        else:
            groups.append([sample])

    mapping: dict[tuple[str, int], str] = {}
    identities: list[EdgeEventIdentity] = []
    for cluster_index, group in enumerate(groups):
        identity = f"{parent_id}:{edge_id}:{event.value}:{cluster_index:03d}"
        arcs = [sample.canonical_edge_arc_m for sample in group]
        for sample in group:
            mapping[sample.key] = identity
        identities.append(
            EdgeEventIdentity(
                identity=identity,
                event=event.value,
                canonical_start_arc_m=float(min(arcs)),
                canonical_end_arc_m=float(max(arcs)),
                canonical_center_arc_m=float((min(arcs) + max(arcs)) / 2.0),
                sample_count=len(group),
                traversal_count=len({sample.traversal_id for sample in group}),
            )
        )
    return mapping, tuple(identities)


def node_event_identity(parent_id: str, node_id: str) -> str:
    if not parent_id or not node_id:
        raise ValueError("parent_id and node_id must be nonempty")
    return f"{parent_id}:node:{node_id}"


def deterministic_association_pairs(
    examples: Iterable[AssociationTeacherExample],
    *,
    hard_negatives_per_anchor: int = 1,
) -> tuple[AssociationTeacherPair, ...]:
    """Build cross-traversal positives and same-event geometry-hard negatives."""

    if hard_negatives_per_anchor < 1:
        raise ValueError("hard_negatives_per_anchor must be positive")
    records = sorted(tuple(examples), key=lambda item: item.observation_id)
    if len({record.observation_id for record in records}) != len(records):
        raise ValueError("association observation IDs must be unique")
    dimensions = {len(record.geometry_signature) for record in records}
    if len(dimensions) > 1:
        raise ValueError("association geometry signatures must share one dimension")
    by_identity: dict[str, list[AssociationTeacherExample]] = {}
    for record in records:
        by_identity.setdefault(record.identity, []).append(record)
    pairs: list[AssociationTeacherPair] = []
    for anchor in records:
        positives = [
            candidate
            for candidate in by_identity[anchor.identity]
            if candidate.observation_id != anchor.observation_id
        ]
        positives.sort(
            key=lambda candidate: (
                candidate.traversal_id == anchor.traversal_id,
                candidate.observation_id,
            )
        )
        if positives:
            positive = positives[0]
            pairs.append(
                AssociationTeacherPair(
                    anchor_observation_id=anchor.observation_id,
                    paired_observation_id=positive.observation_id,
                    same_identity=True,
                    pair_kind=(
                        "cross_traversal_positive"
                        if positive.traversal_id != anchor.traversal_id
                        else "same_traversal_revisit_positive"
                    ),
                    geometry_distance=float(
                        np.linalg.norm(
                            np.asarray(anchor.geometry_signature)
                            - np.asarray(positive.geometry_signature)
                        )
                    ),
                )
            )
        negatives = [
            candidate
            for candidate in records
            if candidate.parent_id == anchor.parent_id
            and candidate.event == anchor.event
            and candidate.identity != anchor.identity
        ]
        negatives.sort(
            key=lambda candidate: (
                float(
                    np.linalg.norm(
                        np.asarray(anchor.geometry_signature)
                        - np.asarray(candidate.geometry_signature)
                    )
                ),
                candidate.observation_id,
            )
        )
        for negative in negatives[:hard_negatives_per_anchor]:
            pairs.append(
                AssociationTeacherPair(
                    anchor_observation_id=anchor.observation_id,
                    paired_observation_id=negative.observation_id,
                    same_identity=False,
                    pair_kind="same_event_geometry_hard_negative",
                    geometry_distance=float(
                        np.linalg.norm(
                            np.asarray(anchor.geometry_signature)
                            - np.asarray(negative.geometry_signature)
                        )
                    ),
                )
            )
    return tuple(pairs)


__all__ = [
    "EdgeEventIdentity",
    "EdgeEventSample",
    "AssociationTeacherExample",
    "AssociationTeacherPair",
    "cluster_edge_event_identities",
    "deterministic_association_pairs",
    "node_event_identity",
]
