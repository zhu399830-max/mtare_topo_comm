"""Objective spline/TNG geometry teacher primitives for GSE-Graph."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.semantics.geometric_semantics import StructuralEvent


@dataclass(frozen=True)
class GSETeacherConfig:
    node_event_radius_m: float = 10.0
    local_geometry_span_m: float = 5.0
    turn_heading_change_deg: float = 15.0

    def __post_init__(self) -> None:
        if self.node_event_radius_m <= 0.0 or self.local_geometry_span_m <= 0.0:
            raise ValueError("teacher distance scales must be positive")
        if not 0.0 < self.turn_heading_change_deg < 180.0:
            raise ValueError("turn_heading_change_deg must be in (0,180)")
    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class LocalGeometryTarget:
    axis: tuple[float, float, float]
    width_m: float
    height_m: float
    slope_deg: float
    curvature_per_m: float
    heading_change_deg: float


class PolylineGeometrySampler:
    """Cache one oriented traversal polyline for many one-metre anchors."""

    def __init__(self, points: np.ndarray) -> None:
        self.points = np.asarray(points, dtype=np.float64)
        self.cumulative = _polyline_cumulative(self.points)
        self.length_m = float(self.cumulative[-1])

    def interpolate(self, arc_m: float) -> np.ndarray:
        arc = float(np.clip(float(arc_m), 0.0, self.length_m))
        index = min(int(np.searchsorted(self.cumulative, arc, side="right") - 1), len(self.points) - 2)
        segment_length = self.cumulative[index + 1] - self.cumulative[index]
        ratio = (arc - self.cumulative[index]) / segment_length
        return self.points[index] + ratio * (self.points[index + 1] - self.points[index])

    def target(self, arc_m: float, *, tunnel_radius_m: float, span_m: float = 5.0) -> LocalGeometryTarget:
        return _local_geometry_from_sampler(self, arc_m, tunnel_radius_m=tunnel_radius_m, span_m=span_m)


def _polyline_cumulative(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or len(values) < 2:
        raise ValueError("polyline must have shape [N>=2,3]")
    lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("polyline contains degenerate segments")
    return np.concatenate(([0.0], np.cumsum(lengths)))


def local_geometry_target(
    points: np.ndarray,
    arc_m: float,
    *,
    tunnel_radius_m: float,
    span_m: float = 5.0,
) -> LocalGeometryTarget:
    """Compute direction-invariant local geometry from a sealed spline."""

    sampler = PolylineGeometrySampler(points)
    return _local_geometry_from_sampler(sampler, arc_m, tunnel_radius_m=tunnel_radius_m, span_m=span_m)


def _local_geometry_from_sampler(
    sampler: PolylineGeometrySampler,
    arc_m: float,
    *,
    tunnel_radius_m: float,
    span_m: float,
) -> LocalGeometryTarget:
    length = sampler.length_m
    arc = float(np.clip(float(arc_m), 0.0, length))
    half_span = min(float(span_m), max(length / 2.0, 1e-6))
    low = max(0.0, arc - half_span)
    high = min(length, arc + half_span)
    if high - low <= 1e-8:
        raise ValueError("local geometry interval is degenerate")
    low_point = sampler.interpolate(low)
    high_point = sampler.interpolate(high)
    axis = high_point - low_point
    norm = float(np.linalg.norm(axis))
    if norm <= 1e-8:
        raise ValueError("local spline axis is degenerate")
    axis /= norm
    slope = math.degrees(math.asin(float(np.clip(axis[2], -1.0, 1.0))))

    center = sampler.interpolate(arc)
    incoming = center - low_point
    outgoing = high_point - center
    incoming_norm = float(np.linalg.norm(incoming))
    outgoing_norm = float(np.linalg.norm(outgoing))
    if incoming_norm <= 1e-8 or outgoing_norm <= 1e-8:
        heading_change = 0.0
    else:
        cosine = float(np.clip((incoming / incoming_norm) @ (outgoing / outgoing_norm), -1.0, 1.0))
        heading_change = math.degrees(math.acos(cosine))
    curvature = math.radians(heading_change) / (high - low)
    radius = float(tunnel_radius_m)
    if not math.isfinite(radius) or radius <= 0.0:
        raise ValueError("tunnel_radius_m must be positive and finite")
    return LocalGeometryTarget(
        axis=tuple(float(value) for value in axis),
        width_m=2.0 * radius,
        height_m=2.0 * radius,
        slope_deg=float(slope),
        curvature_per_m=float(curvature),
        heading_change_deg=float(heading_change),
    )


def node_has_geometry_transition(
    node: Mapping[str, Any],
    tunnel_radius_by_id: Mapping[str, float],
    *,
    minimum_ratio: float,
) -> bool:
    radii = [float(tunnel_radius_by_id[str(value)]) for value in node.get("incident_tunnel_ids", ())]
    if len(radii) < 2 or any(not math.isfinite(value) or value <= 0.0 for value in radii):
        return False
    return max(radii) / min(radii) >= float(minimum_ratio)


def classify_structural_event(
    *,
    traversal_arc_m: float,
    traversal_length_m: float,
    from_node: Mapping[str, Any],
    to_node: Mapping[str, Any],
    local_geometry: LocalGeometryTarget,
    tunnel_radius_by_id: Mapping[str, float] | None = None,
    geometry_transition: bool = False,
    geometry_transition_identity: str | None = None,
    config: GSETeacherConfig | None = None,
) -> tuple[StructuralEvent, str | None]:
    """Assign one objective event with a deterministic priority contract."""

    cfg = config or GSETeacherConfig()
    arc = float(traversal_arc_m)
    length = float(traversal_length_m)
    if not 0.0 <= arc <= length:
        raise ValueError("traversal arc lies outside traversal")
    endpoint_candidates = []
    for distance, node in ((arc, from_node), (length - arc, to_node)):
        if distance <= cfg.node_event_radius_m:
            endpoint_candidates.append((float(distance), str(node["id"]), node))
    endpoint_candidates.sort(key=lambda item: (item[0], item[1]))
    for _, node_id, node in endpoint_candidates:
        if int(node["degree"]) >= 3:
            return StructuralEvent.JUNCTION, node_id
    for _, node_id, node in endpoint_candidates:
        if int(node["degree"]) == 1:
            return StructuralEvent.TERMINAL, node_id
    # Geometry-transition is intentionally not inferred from the generator's
    # per-tunnel radius.  The main teacher supplies it from a sustained native-
    # mesh width/height profile.  The radius map remains an accepted argument
    # only so historical analytic inventories can be audited unchanged.
    del tunnel_radius_by_id
    if geometry_transition:
        return StructuralEvent.GEOMETRY_TRANSITION, geometry_transition_identity
    if local_geometry.heading_change_deg >= cfg.turn_heading_change_deg:
        return StructuralEvent.TURN, None
    return StructuralEvent.CORRIDOR, None


__all__ = [
    "GSETeacherConfig",
    "LocalGeometryTarget",
    "PolylineGeometrySampler",
    "classify_structural_event",
    "local_geometry_target",
    "node_has_geometry_transition",
]
