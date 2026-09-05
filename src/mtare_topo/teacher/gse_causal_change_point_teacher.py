"""Persistent bidirectional causal geometry change-point Teacher.

This module turns local width/height profiles into sparse structural events.
Unlike the historical frame mask, an event exists only when the same physical
change is sustained over the frozen comparison span and is independently
supported by both directions of one physical edge.  The online detector uses
past samples only and back-projects the delayed evidence to the change boundary.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.teacher.gse_geometry_teacher import GSETeacherConfig
from mtare_topo.teacher.gse_mesh_geometry_teacher import MeshGeometryTeacherConfig


@dataclass(frozen=True)
class CausalChangePointTeacherConfig:
    """Reuse the frozen geometry and node-distance scales without new tuning."""

    spacing_m: float = 1.0
    comparison_span_m: float = MeshGeometryTeacherConfig().transition_comparison_span_m
    width_change_m: float = MeshGeometryTeacherConfig().transition_width_change_m
    height_change_m: float = MeshGeometryTeacherConfig().transition_height_change_m
    endpoint_merge_radius_m: float = GSETeacherConfig().node_event_radius_m

    def __post_init__(self) -> None:
        values = (
            self.spacing_m,
            self.comparison_span_m,
            self.width_change_m,
            self.height_change_m,
            self.endpoint_merge_radius_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("change-point scales must be positive and finite")
        ratio = self.comparison_span_m / self.spacing_m
        if not math.isclose(ratio, round(ratio), rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("comparison span must be an integer number of samples")

    @property
    def span_samples(self) -> int:
        return int(round(self.comparison_span_m / self.spacing_m))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DirectionalGeometryProfile:
    """One physical edge observed in one traversal direction."""

    traversal_id: str
    edge_id: str
    direction_index: int
    edge_length_m: float
    traversal_arc_m: tuple[float, ...]
    width_m: tuple[float, ...]
    height_m: tuple[float, ...]
    valid_mask: tuple[bool, ...]

    def __post_init__(self) -> None:
        if not self.traversal_id or not self.edge_id:
            raise ValueError("profile identities must be nonempty")
        if self.direction_index not in (0, 1):
            raise ValueError("direction_index must be 0 or 1")
        if not math.isfinite(self.edge_length_m) or self.edge_length_m <= 0.0:
            raise ValueError("edge length must be positive and finite")
        count = len(self.traversal_arc_m)
        if count == 0 or any(len(values) != count for values in (self.width_m, self.height_m, self.valid_mask)):
            raise ValueError("profile vectors must be nonempty and aligned")
        arcs = np.asarray(self.traversal_arc_m, dtype=np.float64)
        if not np.all(np.isfinite(arcs)) or np.any(np.diff(arcs) <= 0.0):
            raise ValueError("traversal arcs must be finite and strictly increasing")
        if arcs[0] < 0.0 or arcs[-1] > self.edge_length_m + 1e-8:
            raise ValueError("profile arcs lie outside the physical edge")

    def canonical_arc(self, traversal_arc_m: float) -> float:
        return (
            float(traversal_arc_m)
            if self.direction_index == 0
            else self.edge_length_m - float(traversal_arc_m)
        )


@dataclass(frozen=True)
class DirectionalChangeEpisode:
    traversal_id: str
    edge_id: str
    direction_index: int
    traversal_start_arc_m: float
    traversal_end_arc_m: float
    canonical_start_arc_m: float
    canonical_end_arc_m: float
    canonical_center_arc_m: float
    canonical_width_delta_m: float
    canonical_height_delta_m: float
    active_dimensions: tuple[str, ...]
    sample_count: int

    def __post_init__(self) -> None:
        if self.direction_index not in (0, 1) or self.sample_count < 1:
            raise ValueError("invalid directional episode")
        if self.canonical_start_arc_m > self.canonical_end_arc_m:
            raise ValueError("canonical episode interval is reversed")
        if not self.active_dimensions or not set(self.active_dimensions) <= {"width", "height"}:
            raise ValueError("episode must have an active geometry dimension")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CausalChangeEpisode:
    traversal_id: str
    edge_id: str
    direction_index: int
    first_detection_arc_m: float
    emission_arc_m: float
    canonical_boundary_start_arc_m: float
    canonical_boundary_end_arc_m: float
    canonical_boundary_center_arc_m: float
    canonical_width_delta_m: float
    canonical_height_delta_m: float
    active_dimensions: tuple[str, ...]
    detection_count: int
    causal_delay_m: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class BidirectionalChangePoint:
    edge_id: str
    canonical_start_arc_m: float
    canonical_end_arc_m: float
    canonical_center_arc_m: float
    active_dimensions: tuple[str, ...]
    direction0: DirectionalChangeEpisode
    direction1: DirectionalChangeEpisode

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ChangePointPairingResult:
    accepted: tuple[BidirectionalChangePoint, ...]
    unilateral: tuple[DirectionalChangeEpisode, ...]
    ambiguous: tuple[DirectionalChangeEpisode, ...]


@dataclass(frozen=True)
class ChangePointIdentityAssignment:
    identity: str | None
    identity_kind: str
    endpoint_node_id: str | None
    suppressing_node_id: str | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _profile_arrays(
    profile: DirectionalGeometryProfile,
    config: CausalChangePointTeacherConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    arcs = np.asarray(profile.traversal_arc_m, dtype=np.float64)
    width = np.asarray(profile.width_m, dtype=np.float64)
    height = np.asarray(profile.height_m, dtype=np.float64)
    valid = np.asarray(profile.valid_mask, dtype=bool)
    if len(arcs) > 1 and not np.allclose(
        np.diff(arcs), config.spacing_m, rtol=0.0, atol=1e-8
    ):
        raise ValueError("profile spacing differs from the frozen sampling interval")
    valid &= np.isfinite(width) & np.isfinite(height) & (width > 0.0) & (height > 0.0)
    return arcs, width, height, valid


def _runs(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return ()
    starts = np.r_[indices[0], indices[1:][np.diff(indices) > 1]]
    ends = np.r_[indices[:-1][np.diff(indices) > 1], indices[-1]]
    return tuple((int(start), int(end)) for start, end in zip(starts, ends, strict=True))


def persistent_two_sided_change_episodes(
    profile: DirectionalGeometryProfile,
    *,
    config: CausalChangePointTeacherConfig | None = None,
) -> tuple[DirectionalChangeEpisode, ...]:
    """Return only two-sided changes whose evidence spans the frozen 5 m scale."""

    cfg = config or CausalChangePointTeacherConfig()
    arcs, width, height, valid = _profile_arrays(profile, cfg)
    span = cfg.span_samples
    mask = np.zeros(len(arcs), dtype=bool)
    width_delta = np.zeros(len(arcs), dtype=np.float64)
    height_delta = np.zeros(len(arcs), dtype=np.float64)
    for index in range(span, len(arcs) - span):
        before = slice(index - span, index)
        after = slice(index + 1, index + 1 + span)
        if not np.all(valid[before]) or not np.all(valid[after]):
            continue
        width_delta[index] = float(np.median(width[after]) - np.median(width[before]))
        height_delta[index] = float(np.median(height[after]) - np.median(height[before]))
        mask[index] = bool(
            abs(width_delta[index]) >= cfg.width_change_m
            or abs(height_delta[index]) >= cfg.height_change_m
        )

    episodes: list[DirectionalChangeEpisode] = []
    canonical_sign = 1.0 if profile.direction_index == 0 else -1.0
    for start, end in _runs(mask):
        # This is the explicit persistence contract.  An ideal stable step under
        # the historical 5 m/5 m median comparison produces exactly a 5 m run.
        if arcs[end] - arcs[start] + 1e-9 < cfg.comparison_span_m:
            continue
        directional_arcs = arcs[start : end + 1]
        canonical_arcs = np.asarray(
            [profile.canonical_arc(value) for value in directional_arcs], dtype=np.float64
        )
        dw = canonical_sign * float(np.median(width_delta[start : end + 1]))
        dh = canonical_sign * float(np.median(height_delta[start : end + 1]))
        active = tuple(
            name
            for name, delta, threshold in (
                ("width", dw, cfg.width_change_m),
                ("height", dh, cfg.height_change_m),
            )
            if abs(delta) >= threshold
        )
        if not active:
            # Alternating dimensions inside one run are not one stable change.
            continue
        low = float(np.min(canonical_arcs))
        high = float(np.max(canonical_arcs))
        episodes.append(
            DirectionalChangeEpisode(
                traversal_id=profile.traversal_id,
                edge_id=profile.edge_id,
                direction_index=profile.direction_index,
                traversal_start_arc_m=float(arcs[start]),
                traversal_end_arc_m=float(arcs[end]),
                canonical_start_arc_m=low,
                canonical_end_arc_m=high,
                canonical_center_arc_m=(low + high) / 2.0,
                canonical_width_delta_m=dw,
                canonical_height_delta_m=dh,
                active_dimensions=active,
                sample_count=end - start + 1,
            )
        )
    return tuple(episodes)


def past_only_causal_change_episodes(
    profile: DirectionalGeometryProfile,
    *,
    config: CausalChangePointTeacherConfig | None = None,
) -> tuple[CausalChangeEpisode, ...]:
    """Detect changes from two completed past-only windows and back-project them."""

    cfg = config or CausalChangePointTeacherConfig()
    arcs, width, height, valid = _profile_arrays(profile, cfg)
    span = cfg.span_samples
    mask = np.zeros(len(arcs), dtype=bool)
    width_delta = np.zeros(len(arcs), dtype=np.float64)
    height_delta = np.zeros(len(arcs), dtype=np.float64)
    boundary = np.full(len(arcs), np.nan, dtype=np.float64)
    for index in range(2 * span - 1, len(arcs)):
        old = slice(index - 2 * span + 1, index - span + 1)
        recent = slice(index - span + 1, index + 1)
        if not np.all(valid[old]) or not np.all(valid[recent]):
            continue
        width_delta[index] = float(np.median(width[recent]) - np.median(width[old]))
        height_delta[index] = float(np.median(height[recent]) - np.median(height[old]))
        mask[index] = bool(
            abs(width_delta[index]) >= cfg.width_change_m
            or abs(height_delta[index]) >= cfg.height_change_m
        )
        boundary[index] = (arcs[index - span] + arcs[index - span + 1]) / 2.0

    episodes: list[CausalChangeEpisode] = []
    canonical_sign = 1.0 if profile.direction_index == 0 else -1.0
    for start, end in _runs(mask):
        # A run can only be closed after the next (non-change) sample arrives.
        # Returning a run that touches the current prefix boundary would use an
        # implicit future decision about whether the episode continues.
        if end == len(arcs) - 1:
            continue
        boundaries = boundary[start : end + 1]
        if not np.all(np.isfinite(boundaries)):
            raise RuntimeError("causal change episode has an undefined boundary")
        canonical_boundaries = np.asarray(
            [profile.canonical_arc(value) for value in boundaries], dtype=np.float64
        )
        dw = canonical_sign * float(np.median(width_delta[start : end + 1]))
        dh = canonical_sign * float(np.median(height_delta[start : end + 1]))
        active = tuple(
            name
            for name, delta, threshold in (
                ("width", dw, cfg.width_change_m),
                ("height", dh, cfg.height_change_m),
            )
            if abs(delta) >= threshold
        )
        if not active:
            continue
        low = float(np.min(canonical_boundaries))
        high = float(np.max(canonical_boundaries))
        center = (low + high) / 2.0
        emission = float(arcs[end + 1])
        boundary_traversal = center if profile.direction_index == 0 else profile.edge_length_m - center
        episodes.append(
            CausalChangeEpisode(
                traversal_id=profile.traversal_id,
                edge_id=profile.edge_id,
                direction_index=profile.direction_index,
                first_detection_arc_m=float(arcs[start]),
                emission_arc_m=emission,
                canonical_boundary_start_arc_m=low,
                canonical_boundary_end_arc_m=high,
                canonical_boundary_center_arc_m=center,
                canonical_width_delta_m=dw,
                canonical_height_delta_m=dh,
                active_dimensions=active,
                detection_count=end - start + 1,
                causal_delay_m=emission - boundary_traversal,
            )
        )
    return tuple(episodes)


def _same_signed_dimension(
    left: DirectionalChangeEpisode,
    right: DirectionalChangeEpisode,
) -> tuple[str, ...]:
    shared = set(left.active_dimensions) & set(right.active_dimensions)
    agreed: list[str] = []
    for name in ("width", "height"):
        if name not in shared:
            continue
        left_value = getattr(left, f"canonical_{name}_delta_m")
        right_value = getattr(right, f"canonical_{name}_delta_m")
        if left_value * right_value > 0.0:
            agreed.append(name)
    return tuple(agreed)


def pair_bidirectional_change_points(
    direction0: Sequence[DirectionalChangeEpisode],
    direction1: Sequence[DirectionalChangeEpisode],
) -> ChangePointPairingResult:
    """Keep only unique interval-overlapping, same-signed two-direction pairs."""

    left = tuple(direction0)
    right = tuple(direction1)
    if any(item.direction_index != 0 for item in left) or any(item.direction_index != 1 for item in right):
        raise ValueError("change episodes were supplied to the wrong direction")
    edge_ids = {item.edge_id for item in left + right}
    if len(edge_ids) > 1:
        raise ValueError("bidirectional pairing must stay within one physical edge")
    matches: dict[tuple[int, int], tuple[str, ...]] = {}
    for left_index, left_item in enumerate(left):
        for right_index, right_item in enumerate(right):
            overlap_start = max(left_item.canonical_start_arc_m, right_item.canonical_start_arc_m)
            overlap_end = min(left_item.canonical_end_arc_m, right_item.canonical_end_arc_m)
            dimensions = _same_signed_dimension(left_item, right_item)
            if overlap_start <= overlap_end + 1e-9 and dimensions:
                matches[(left_index, right_index)] = dimensions

    left_degree = {index: 0 for index in range(len(left))}
    right_degree = {index: 0 for index in range(len(right))}
    for left_index, right_index in matches:
        left_degree[left_index] += 1
        right_degree[right_index] += 1

    accepted: list[BidirectionalChangePoint] = []
    ambiguous_indices_left: set[int] = set()
    ambiguous_indices_right: set[int] = set()
    for left_index, right_index in matches:
        if left_degree[left_index] > 1 or right_degree[right_index] > 1:
            ambiguous_indices_left.add(left_index)
            ambiguous_indices_right.add(right_index)
    for (left_index, right_index), dimensions in sorted(matches.items()):
        if left_degree[left_index] != 1 or right_degree[right_index] != 1:
            continue
        left_item = left[left_index]
        right_item = right[right_index]
        low = max(left_item.canonical_start_arc_m, right_item.canonical_start_arc_m)
        high = min(left_item.canonical_end_arc_m, right_item.canonical_end_arc_m)
        accepted.append(
            BidirectionalChangePoint(
                edge_id=left_item.edge_id,
                canonical_start_arc_m=float(low),
                canonical_end_arc_m=float(high),
                canonical_center_arc_m=float(
                    (left_item.canonical_center_arc_m + right_item.canonical_center_arc_m) / 2.0
                ),
                active_dimensions=dimensions,
                direction0=left_item,
                direction1=right_item,
            )
        )
    unilateral = tuple(
        [item for index, item in enumerate(left) if left_degree[index] == 0]
        + [item for index, item in enumerate(right) if right_degree[index] == 0]
    )
    ambiguous = tuple(
        [left[index] for index in sorted(ambiguous_indices_left)]
        + [right[index] for index in sorted(ambiguous_indices_right)]
    )
    return ChangePointPairingResult(tuple(accepted), unilateral, ambiguous)


def assign_change_point_identity(
    *,
    parent_id: str,
    point: BidirectionalChangePoint,
    edge_length_m: float,
    from_node_id: str,
    to_node_id: str,
    node_degree_by_id: Mapping[str, int],
    point_index: int,
    config: CausalChangePointTeacherConfig | None = None,
) -> ChangePointIdentityAssignment:
    """Merge endpoint-near changes only into unique degree-two TNG nodes."""

    cfg = config or CausalChangePointTeacherConfig()
    if not parent_id or point_index < 0:
        raise ValueError("parent identity and point index must be valid")
    if not math.isfinite(edge_length_m) or edge_length_m <= 0.0:
        raise ValueError("edge length must be positive and finite")
    if not 0.0 <= point.canonical_center_arc_m <= edge_length_m:
        raise ValueError("change point lies outside the physical edge")
    candidates = []
    for distance, node_id in (
        (point.canonical_center_arc_m, from_node_id),
        (edge_length_m - point.canonical_center_arc_m, to_node_id),
    ):
        if distance <= cfg.endpoint_merge_radius_m:
            if node_id not in node_degree_by_id:
                raise ValueError("endpoint node degree is missing")
            candidates.append((float(distance), str(node_id), int(node_degree_by_id[node_id])))
    candidates.sort(key=lambda item: (item[0], item[1]))
    protected = [item for item in candidates if item[2] == 1 or item[2] >= 3]
    if protected:
        return ChangePointIdentityAssignment(
            identity=None,
            identity_kind="suppressed_by_terminal_or_junction",
            endpoint_node_id=None,
            suppressing_node_id=protected[0][1],
        )
    degree_two = [item for item in candidates if item[2] == 2]
    if len(degree_two) > 1:
        return ChangePointIdentityAssignment(
            identity=None,
            identity_kind="ambiguous_degree_two_endpoints",
            endpoint_node_id=None,
            suppressing_node_id=None,
        )
    if len(degree_two) == 1:
        node_id = degree_two[0][1]
        return ChangePointIdentityAssignment(
            identity=f"{parent_id}:node:{node_id}",
            identity_kind="degree_two_endpoint",
            endpoint_node_id=node_id,
            suppressing_node_id=None,
        )
    return ChangePointIdentityAssignment(
        identity=f"{parent_id}:{point.edge_id}:geometry_change_point:{point_index:03d}",
        identity_kind="interior_edge",
        endpoint_node_id=None,
        suppressing_node_id=None,
    )


def causal_episode_matches_point(
    causal: CausalChangeEpisode,
    point: BidirectionalChangePoint,
) -> bool:
    """Match without a tuned radius: boundary intervals must overlap and agree."""

    if causal.edge_id != point.edge_id:
        return False
    overlap = max(causal.canonical_boundary_start_arc_m, point.canonical_start_arc_m) <= min(
        causal.canonical_boundary_end_arc_m, point.canonical_end_arc_m
    ) + 1e-9
    if not overlap:
        return False
    point_episode = point.direction0 if causal.direction_index == 0 else point.direction1
    shared = set(causal.active_dimensions) & set(point.active_dimensions)
    return any(
        getattr(causal, f"canonical_{name}_delta_m")
        * getattr(point_episode, f"canonical_{name}_delta_m")
        > 0.0
        for name in shared
    )


__all__ = [
    "BidirectionalChangePoint",
    "CausalChangeEpisode",
    "CausalChangePointTeacherConfig",
    "ChangePointIdentityAssignment",
    "ChangePointPairingResult",
    "DirectionalChangeEpisode",
    "DirectionalGeometryProfile",
    "assign_change_point_identity",
    "causal_episode_matches_point",
    "pair_bidirectional_change_points",
    "past_only_causal_change_episodes",
    "persistent_two_sided_change_episodes",
]
