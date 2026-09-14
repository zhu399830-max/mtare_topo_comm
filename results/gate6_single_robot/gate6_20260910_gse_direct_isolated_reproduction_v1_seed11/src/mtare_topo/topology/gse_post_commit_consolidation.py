"""Fail-closed consolidation of execution-committed GSE graph nodes.

The online 4 m association remains unchanged.  This module acts only after a
physical traversal has been completed: two committed hypotheses may represent
the same endpoint when their accumulated execution traces name the same side
of the same traversed physical edge.  Edge endpoint tokens use only route arc,
traversal direction and completed-edge length; objective node identity is
never an input.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class TraversedEdgeEndpoint:
    physical_edge_id: str
    endpoint_side: int

    def __post_init__(self) -> None:
        if not self.physical_edge_id or self.endpoint_side not in (0, 1):
            raise ValueError("traversed-edge endpoint contract drift")


@dataclass(frozen=True)
class CommittedEndpointSignature:
    hypothesis_id: int
    world: str
    event: str
    endpoints: tuple[TraversedEdgeEndpoint, ...]

    def __post_init__(self) -> None:
        if (
            self.hypothesis_id < 0 or not self.world
            or self.event not in ("junction", "terminal")
            or not self.endpoints or len(self.endpoints) != len(set(self.endpoints))
            or any(not value.physical_edge_id.startswith(f"{self.world}:") for value in self.endpoints)
        ):
            raise ValueError("committed endpoint signature contract drift")


@dataclass(frozen=True)
class TraversedEndpointGeometry:
    endpoint: TraversedEdgeEndpoint
    xyz_m: tuple[float, float, float]
    outward_unit_xyz: tuple[float, float, float]

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz_m, dtype=np.float64)
        outward = np.asarray(self.outward_unit_xyz, dtype=np.float64)
        if (
            xyz.shape != (3,) or outward.shape != (3,)
            or not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(outward))
            or not np.isclose(np.linalg.norm(outward), 1.0, atol=2e-6)
        ):
            raise ValueError("traversed endpoint geometry contract drift")


def traversed_edge_endpoint(
    traversal_id: str,
    traversal_arc_m: float,
    traversal_length_m: float,
) -> TraversedEdgeEndpoint:
    """Return an orientation-invariant physical-edge endpoint token.

    Side 0 is the d0 traversal start and side 1 is its end.  Reversing the
    traversal swaps start/end but leaves the physical endpoint token intact.
    A midpoint tie is assigned to the traversal start deterministically.
    """

    if ":d" not in traversal_id:
        raise ValueError("traversal identity has no directed suffix")
    physical, suffix = traversal_id.rsplit(":d", 1)
    if suffix not in ("0", "1") or not physical:
        raise ValueError("traversal direction contract drift")
    arc = float(traversal_arc_m)
    length = float(traversal_length_m)
    if (
        not math.isfinite(arc) or not math.isfinite(length) or length <= 0.0
        or arc < -1e-9 or arc > length + 1e-9
    ):
        raise ValueError("traversal arc/length contract drift")
    direction = int(suffix)
    near_start = arc <= length / 2.0
    side = direction if near_start else 1 - direction
    return TraversedEdgeEndpoint(physical, side)


def endpoint_consolidation_mapping(
    signatures: Sequence[CommittedEndpointSignature],
) -> tuple[dict[int, int], tuple[tuple[int, ...], ...]]:
    """Union only same-world/same-event hypotheses sharing an exact token."""

    identifiers = [value.hypothesis_id for value in signatures]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("committed endpoint hypothesis IDs must be unique")
    parent = {value: value for value in identifiers}

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: int, right: int) -> None:
        first, second = find(left), find(right)
        if first == second:
            return
        representative, child = min(first, second), max(first, second)
        parent[child] = representative

    owners: dict[tuple[str, str, TraversedEdgeEndpoint], int] = {}
    for signature in sorted(signatures, key=lambda value: value.hypothesis_id):
        for endpoint in signature.endpoints:
            key = (signature.world, signature.event, endpoint)
            prior = owners.get(key)
            if prior is None:
                owners[key] = signature.hypothesis_id
            else:
                union(prior, signature.hypothesis_id)

    mapping = {value: find(value) for value in identifiers}
    groups: dict[int, list[int]] = {}
    for value, representative in mapping.items():
        groups.setdefault(representative, []).append(value)
    components = tuple(
        tuple(sorted(group)) for _, group in sorted(groups.items()) if len(group) > 1
    )
    return mapping, components


def endpoint_anchor_consensus(
    geometries: Sequence[TraversedEndpointGeometry],
    *,
    route_sample_spacing_m: float = 1.0,
) -> tuple[bool, tuple[float, float, float], float]:
    """Check whether executed edge ends meet at one topometric node.

    Each stored route endpoint can be at most one sample spacing from the
    physical endpoint, so two incident traces may differ by at most twice that
    spacing.  Both sides of one physical edge are always contradictory.
    """

    if not geometries or not math.isfinite(route_sample_spacing_m) or route_sample_spacing_m <= 0.0:
        raise ValueError("endpoint anchor consensus contract drift")
    endpoints = [value.endpoint for value in geometries]
    if len(endpoints) != len(set(endpoints)):
        raise ValueError("endpoint geometries must be unique")
    physical_sides: dict[str, set[int]] = {}
    for endpoint in endpoints:
        physical_sides.setdefault(endpoint.physical_edge_id, set()).add(endpoint.endpoint_side)
    contradictory = any(len(value) > 1 for value in physical_sides.values())
    xyz = np.asarray([value.xyz_m for value in geometries], dtype=np.float64)
    maximum = max(
        (float(np.linalg.norm(xyz[left] - xyz[right]))
         for left in range(len(xyz)) for right in range(left + 1, len(xyz))),
        default=0.0,
    )
    center = tuple(float(value) for value in np.mean(xyz, axis=0))
    allowed = 2.0 * route_sample_spacing_m + 1e-9
    return bool(not contradictory and maximum <= allowed), center, maximum


def executed_branch_witness(
    geometries: Sequence[TraversedEndpointGeometry],
) -> bool:
    """Require executed evidence that cannot be a straight two-edge corridor.

    Three distinct physical incidences directly witness a junction.  With
    exactly two, both outward directions must lie in the same open half-space
    (positive dot product, an acute opening).  This zero-threshold sign test is
    deterministic and introduces no angle grid or selection-set tuning.
    """

    by_physical: dict[str, TraversedEndpointGeometry] = {}
    for value in geometries:
        prior = by_physical.get(value.endpoint.physical_edge_id)
        if prior is not None and prior.endpoint != value.endpoint:
            return False
        by_physical[value.endpoint.physical_edge_id] = value
    if len(by_physical) >= 3:
        return True
    if len(by_physical) != 2:
        return False
    first, second = by_physical.values()
    return float(np.dot(first.outward_unit_xyz, second.outward_unit_xyz)) > 0.0


def remap_verified_edges(
    edges: Iterable[Mapping[str, object]],
    mapping: Mapping[int, int],
) -> tuple[list[dict[str, object]], int, int]:
    """Remap verified edges, removing merge-created self loops and duplicates."""

    result: list[dict[str, object]] = []
    keys: set[tuple[int, int]] = set()
    self_loops = 0
    duplicates = 0
    for edge in edges:
        left_raw = int(edge["from_hypothesis"])
        right_raw = int(edge["to_hypothesis"])
        if left_raw not in mapping or right_raw not in mapping:
            continue
        left, right = mapping[left_raw], mapping[right_raw]
        if left == right:
            self_loops += 1
            continue
        key = tuple(sorted((left, right)))
        if key in keys:
            duplicates += 1
            continue
        keys.add(key)
        record = dict(edge)
        record["id"] = len(result)
        record["from_hypothesis"], record["to_hypothesis"] = key
        record["post_commit_consolidation"] = "shared_traversed_edge_endpoint_v1"
        result.append(record)
    return result, self_loops, duplicates


__all__ = [
    "CommittedEndpointSignature", "TraversedEdgeEndpoint", "TraversedEndpointGeometry",
    "endpoint_anchor_consensus", "endpoint_consolidation_mapping", "executed_branch_witness",
    "remap_verified_edges", "traversed_edge_endpoint",
]
