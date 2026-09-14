"""Topology-isolated local implicit-union window contracts for Gate 4."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
from scipy.spatial import cKDTree


RAY_EXIT_BISECTION_ITERATIONS = 32


@dataclass(frozen=True)
class LocalUnionWindow:
    node_id: str
    center_xyz_m: tuple[float, float, float]
    incident_tunnel_ids: tuple[int, ...]
    radius_m: float


@dataclass(frozen=True)
class ArcIncidence:
    edge_id: str
    node_id: str
    endpoint: str
    tunnel_id: int
    arc_m: float
    connector_error_m: float
    projection_xyz_m: tuple[float, float, float]
    away_tangent: tuple[float, float, float]


@dataclass(frozen=True)
class RouteSupportQuery:
    """Identity required to select one route-conditioned support surface."""

    tunnel_id: int
    edge_id: str
    window_node_id: str | None = None


@dataclass(frozen=True)
class IncidentSupportArc:
    """One edge-endpoint arc retained independently inside a junction window."""

    edge_id: str
    node_id: str
    tunnel_id: int
    endpoint: str
    points_xyz_m: np.ndarray


def _project_xy(point_xy: np.ndarray, polyline: np.ndarray) -> tuple[float, float]:
    """Return projected z and XY distance on a non-degenerate 3-D polyline."""
    segments = np.diff(polyline, axis=0)
    lengths_xy_sq = np.sum(segments[:, :2] * segments[:, :2], axis=1)
    usable = lengths_xy_sq > 1e-18
    if not usable.any():
        raise ValueError("support spline has no horizontally projectable segment")
    ratios = np.zeros(len(segments), dtype=np.float64)
    ratios[usable] = np.clip(
        np.sum((point_xy - polyline[:-1, :2]) * segments[:, :2], axis=1)[usable]
        / lengths_xy_sq[usable],
        0.0,
        1.0,
    )
    candidates = polyline[:-1] + ratios[:, None] * segments
    distances = np.linalg.norm(candidates[:, :2] - point_xy, axis=1)
    distances[~usable] = np.inf
    index = int(np.argmin(distances))
    return float(candidates[index, 2]), float(distances[index])


def _slice_polyline_by_arc(polyline: np.ndarray, start_arc: float, end_arc: float) -> np.ndarray:
    """Extract an oriented exact arc interval including interpolated endpoints."""
    reverse = end_arc < start_arc
    low, high = sorted((float(start_arc), float(end_arc)))
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    if low < -1e-9 or high > cumulative[-1] + 1e-9 or high - low <= 1e-12:
        raise ValueError("invalid support arc interval")

    def interpolate(arc: float) -> np.ndarray:
        index = min(int(np.searchsorted(cumulative, arc, side="right") - 1), len(lengths) - 1)
        ratio = (arc - cumulative[index]) / lengths[index]
        return polyline[index] + ratio * (polyline[index + 1] - polyline[index])

    interior = polyline[(cumulative > low + 1e-12) & (cumulative < high - 1e-12)]
    result = np.vstack((interpolate(low), interior, interpolate(high)))
    return result[::-1].copy() if reverse else result


def _clip_away_arc_to_sphere(
    polyline: np.ndarray, start_arc: float, direction: float, center_xyz_m: np.ndarray, radius_m: float
) -> np.ndarray:
    """Clip one oriented spline arc at its first exit from a spherical window."""
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    total = float(lengths.sum())
    end_arc = total if direction > 0.0 else 0.0
    oriented = _slice_polyline_by_arc(polyline, start_arc, end_arc)
    relative = oriented - center_xyz_m
    if np.linalg.norm(relative[0]) > radius_m + 1e-9:
        raise ValueError("incident arc projection starts outside its window")
    result = [oriented[0]]
    for first, second in zip(oriented[:-1], oriented[1:]):
        if np.linalg.norm(second - center_xyz_m) <= radius_m + 1e-12:
            result.append(second)
            continue
        delta = second - first
        offset = first - center_xyz_m
        coefficients = (
            float(np.dot(delta, delta)),
            2.0 * float(np.dot(offset, delta)),
            float(np.dot(offset, offset) - radius_m * radius_m),
        )
        roots = np.roots(coefficients[:2] + (coefficients[2],))
        valid = sorted(float(root.real) for root in roots if abs(root.imag) <= 1e-10 and -1e-10 <= root.real <= 1.0 + 1e-10)
        if not valid:
            raise ValueError("incident arc crosses window boundary without a finite intersection")
        result.append(first + np.clip(valid[-1], 0.0, 1.0) * delta)
        return np.asarray(result, dtype=np.float64)
    raise ValueError("incident arc never exits its window")


class RouteConditionedSupportField:
    """Analytic route-conditioned floor surface, independent of collision SDFs.

    Outside a frozen junction membership the selected physical spline is the
    only support source.  Inside it, independently retained incident edge arcs
    are combined by nearest horizontal projection, then C1-blended back to the
    selected spline at the spherical window boundary.
    """

    AMBIGUITY_DISTANCE_TOLERANCE_M = 1e-9
    AMBIGUITY_HEIGHT_TOLERANCE_M = 1e-9

    def __init__(
        self,
        splines_by_tunnel: Mapping[int, np.ndarray],
        fta_distance_m: float,
        windows: tuple[LocalUnionWindow, ...],
        incident_arcs: tuple[IncidentSupportArc, ...],
    ):
        self.splines_by_tunnel = {
            int(key): np.asarray(value, dtype=np.float64) for key, value in splines_by_tunnel.items()
        }
        self.fta_distance_m = float(fta_distance_m)
        self.windows = {window.node_id: window for window in windows}
        self.incident_arcs_by_node: dict[str, tuple[IncidentSupportArc, ...]] = {}
        for node_id in self.windows:
            arcs = tuple(item for item in incident_arcs if item.node_id == node_id)
            if not arcs:
                raise ValueError(f"junction {node_id} has no incident support arcs")
            identities = [(item.edge_id, item.endpoint) for item in arcs]
            if len(identities) != len(set(identities)):
                raise ValueError(f"junction {node_id} has duplicate incident support arcs")
            self.incident_arcs_by_node[node_id] = arcs

    @classmethod
    def from_documents(
        cls,
        graph: Mapping[str, Any],
        splines: Mapping[str, Any],
        geometry: Mapping[str, Any],
        windows: tuple[LocalUnionWindow, ...],
    ) -> "RouteConditionedSupportField":
        curves = {
            int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
            for item in splines["tunnels"]
        }
        records = edge_arc_incidence(graph, splines)
        window_by_node = {item.node_id: item for item in windows}
        incident_arcs: list[IncidentSupportArc] = []
        for record in records:
            if record.node_id not in window_by_node:
                continue
            curve = curves[record.tunnel_id]
            tangent = np.asarray(record.away_tangent, dtype=np.float64)
            _, _, _, local_tangent = _project(np.asarray(record.projection_xyz_m), curve)
            direction = 1.0 if float(np.dot(tangent, local_tangent)) >= 0.0 else -1.0
            window = window_by_node[record.node_id]
            points = _clip_away_arc_to_sphere(
                curve,
                record.arc_m,
                direction,
                np.asarray(window.center_xyz_m, dtype=np.float64),
                window.radius_m,
            )
            incident_arcs.append(IncidentSupportArc(
                record.edge_id, record.node_id, record.tunnel_id, record.endpoint, points
            ))
        return cls(curves, float(geometry["fta_distance_m"]), windows, tuple(incident_arcs))

    @staticmethod
    def incident_weight(normalized_radius: np.ndarray) -> np.ndarray:
        return ContinuousLayeredHybridField.incident_weight(normalized_radius)

    @staticmethod
    def incident_weight_derivative(normalized_radius: np.ndarray) -> np.ndarray:
        return ContinuousLayeredHybridField.incident_weight_derivative(normalized_radius)

    def _selected_height(self, point: np.ndarray, tunnel_id: int) -> float:
        if tunnel_id not in self.splines_by_tunnel:
            raise ValueError(f"unknown support tunnel identity {tunnel_id}")
        z, _ = _project_xy(point[:2], self.splines_by_tunnel[tunnel_id])
        return z + self.fta_distance_m

    def _incident_height(self, point: np.ndarray, query: RouteSupportQuery) -> float:
        assert query.window_node_id is not None
        window = self.windows.get(query.window_node_id)
        if window is None:
            raise ValueError(f"unknown support window {query.window_node_id}")
        if query.tunnel_id not in window.incident_tunnel_ids:
            raise ValueError(f"non-incident traversal {query.edge_id} entered support window {window.node_id}")
        candidates = []
        for arc in self.incident_arcs_by_node[window.node_id]:
            z, distance = _project_xy(point[:2], arc.points_xyz_m)
            candidates.append((distance, z + self.fta_distance_m, arc.edge_id, arc.endpoint))
        minimum = min(item[0] for item in candidates)
        nearest = [
            item for item in candidates
            if abs(item[0] - minimum) <= self.AMBIGUITY_DISTANCE_TOLERANCE_M
        ]
        heights = [item[1] for item in nearest]
        if max(heights) - min(heights) > self.AMBIGUITY_HEIGHT_TOLERANCE_M:
            identities = [(item[2], item[3]) for item in nearest]
            raise ValueError(f"ambiguous incident support at {window.node_id}: {identities}")
        return float(sum(heights) / len(heights))

    def support_heights(self, graph_axis_xyz_m: np.ndarray, queries: tuple[RouteSupportQuery, ...]) -> np.ndarray:
        points = np.asarray(graph_axis_xyz_m, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3 or len(points) != len(queries):
            raise ValueError("support queries require matching Nx3 graph-axis points")
        result = np.empty(len(points), dtype=np.float64)
        for index, (point, query) in enumerate(zip(points, queries)):
            selected = self._selected_height(point, query.tunnel_id)
            if query.window_node_id is None:
                result[index] = selected
                continue
            window = self.windows[query.window_node_id]
            radius = float(np.linalg.norm(point - np.asarray(window.center_xyz_m)))
            if radius > window.radius_m + 1e-9:
                raise ValueError(f"frozen support membership escaped window {window.node_id}")
            incident = self._incident_height(point, query)
            weight = float(self.incident_weight(np.asarray([radius / window.radius_m]))[0])
            result[index] = selected + weight * (incident - selected)
        if not np.isfinite(result).all():
            raise ValueError("support field produced a non-finite height")
        return result

    def downward_distances(
        self,
        sensor_xyz_m: np.ndarray,
        graph_axis_xyz_m: np.ndarray,
        queries: tuple[RouteSupportQuery, ...],
    ) -> np.ndarray:
        sensor = np.asarray(sensor_xyz_m, dtype=np.float64)
        if sensor.shape != np.asarray(graph_axis_xyz_m).shape:
            raise ValueError("sensor and graph-axis arrays must have identical shapes")
        return sensor[:, 2] - self.support_heights(graph_axis_xyz_m, queries)


class SampledSweptTubeField:
    """Deterministic continuous tube-union field sampled along sealed splines."""

    def __init__(self, samples_by_tunnel: Mapping[int, np.ndarray], radii_by_tunnel: Mapping[int, float], spacing_m: float):
        if spacing_m <= 0 or set(samples_by_tunnel) != set(radii_by_tunnel):
            raise ValueError("sample/radius tunnel identities differ")
        self.spacing_m = float(spacing_m)
        self.samples_by_tunnel = {int(key): np.asarray(value, dtype=np.float64) for key, value in samples_by_tunnel.items()}
        self.radii_by_tunnel = {int(key): float(value) for key, value in radii_by_tunnel.items()}
        self._trees = {key: cKDTree(value) for key, value in self.samples_by_tunnel.items()}

    @classmethod
    def from_documents(cls, splines: Mapping[str, Any], geometry: Mapping[str, Any], spacing_m: float = 0.025) -> "SampledSweptTubeField":
        radii = {int(item["tunnel_id"]): float(item["radius_m"]) for item in geometry["tunnels"]}
        samples: dict[int, np.ndarray] = {}
        for item in splines["tunnels"]:
            tunnel_id = int(item["tunnel_id"])
            points = np.asarray(item["points"], dtype=np.float64)
            lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
            cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
            arcs = np.arange(0.0, cumulative[-1], spacing_m, dtype=np.float64)
            arcs = np.append(arcs, cumulative[-1])
            indices = np.minimum(np.searchsorted(cumulative, arcs, side="right") - 1, len(lengths) - 1)
            ratios = (arcs - cumulative[indices]) / lengths[indices]
            samples[tunnel_id] = points[indices] + ratios[:, None] * (points[indices + 1] - points[indices])
        return cls(samples, radii, spacing_m)

    def signed_distance(self, points: np.ndarray, tunnel_ids: tuple[int, ...] | None = None) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        flat = query.reshape(-1, 3)
        identities = tuple(sorted(self._trees)) if tunnel_ids is None else tuple(sorted(set(tunnel_ids)))
        values = []
        for tunnel_id in identities:
            distance, _ = self._trees[tunnel_id].query(flat, k=1, workers=1)
            values.append(distance - self.radii_by_tunnel[tunnel_id])
        return np.min(np.stack(values, axis=0), axis=0).reshape(query.shape[:-1])

    def ray_exit_distances(self, origins: np.ndarray, directions: np.ndarray, maximum_m: float = 20.0, iterations: int = 128) -> np.ndarray:
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        directions = directions / np.linalg.norm(directions, axis=1, keepdims=True)
        travel = np.zeros(len(origins), dtype=np.float64)
        last_inside = np.zeros(len(origins), dtype=np.float64)
        active = np.ones(len(origins), dtype=bool)
        hits = np.full(len(origins), np.inf, dtype=np.float64)
        for _ in range(iterations):
            if not active.any():
                break
            indices = np.flatnonzero(active)
            sdf = self.signed_distance(origins[indices] + directions[indices] * travel[indices, None])
            escaped = sdf >= 0.0
            if escaped.any():
                hit_indices = indices[escaped]
                low = last_inside[hit_indices].copy()
                high = travel[hit_indices].copy()
                refinable = high > low
                for _ in range(RAY_EXIT_BISECTION_ITERATIONS):
                    if not refinable.any():
                        break
                    midpoint = 0.5 * (low[refinable] + high[refinable])
                    refine_indices = hit_indices[refinable]
                    midpoint_sdf = self.signed_distance(
                        origins[refine_indices] + directions[refine_indices] * midpoint[:, None]
                    )
                    outside = midpoint_sdf >= 0.0
                    refined_low = low[refinable]
                    refined_high = high[refinable]
                    refined_high[outside] = midpoint[outside]
                    refined_low[~outside] = midpoint[~outside]
                    low[refinable] = refined_low
                    high[refinable] = refined_high
                hits[hit_indices] = high
                active[hit_indices] = False
            remaining = indices[~escaped]
            if len(remaining):
                last_inside[remaining] = travel[remaining]
                step = np.maximum(np.abs(sdf[~escaped]) * 0.8, self.spacing_m * 0.5)
                travel[remaining] += step
                active[remaining[travel[remaining] > maximum_m]] = False
        return hits


class LayeredHybridField:
    """Traversal-conditioned tube field with incident unions only in junction windows.

    Outside a compatible junction window each query is evaluated against its
    sealed physical-tunnel layer.  Inside a window whose incident set contains
    that layer, the field switches to the union of that window's incident
    tunnels.  A spatially overlapping non-incident tunnel can therefore never
    steal collision or support from the selected traversal.
    """

    def __init__(self, base: SampledSweptTubeField, windows: tuple[LocalUnionWindow, ...]):
        incompatible = [item for item in overlapping_window_pairs(windows) if not item["incident_sets_identical"]]
        if incompatible:
            raise ValueError(f"incompatible layered windows overlap: {incompatible[:3]}")
        self.base = base
        self.windows = tuple(windows)
        self.spacing_m = base.spacing_m

    def signed_distance(self, points: np.ndarray, layer_tunnel_ids: np.ndarray) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        if query.shape[-1] != 3:
            raise ValueError("points must end in xyz")
        flat = query.reshape(-1, 3)
        layers = np.asarray(layer_tunnel_ids, dtype=np.int64).reshape(-1)
        if len(layers) != len(flat):
            raise ValueError("one layer tunnel identity is required per query point")
        unknown = sorted(set(layers.tolist()) - set(self.base.samples_by_tunnel))
        if unknown:
            raise ValueError(f"unknown layer tunnel identities: {unknown[:3]}")

        values = np.empty(len(flat), dtype=np.float64)
        for tunnel_id in np.unique(layers):
            selected = layers == tunnel_id
            values[selected] = self.base.signed_distance(flat[selected], (int(tunnel_id),))
        for window in self.windows:
            compatible = np.isin(layers, np.asarray(window.incident_tunnel_ids, dtype=np.int64))
            inside = np.linalg.norm(flat - np.asarray(window.center_xyz_m), axis=1) <= window.radius_m
            selected = compatible & inside
            if selected.any():
                values[selected] = self.base.signed_distance(flat[selected], window.incident_tunnel_ids)
        return values.reshape(query.shape[:-1])

    def ray_exit_distances(
        self,
        origins: np.ndarray,
        directions: np.ndarray,
        layer_tunnel_ids: np.ndarray,
        maximum_m: float = 20.0,
        iterations: int = 128,
    ) -> np.ndarray:
        origins = np.asarray(origins, dtype=np.float64)
        directions = np.asarray(directions, dtype=np.float64)
        layers = np.asarray(layer_tunnel_ids, dtype=np.int64).reshape(-1)
        if origins.shape != directions.shape or origins.ndim != 2 or origins.shape[1] != 3:
            raise ValueError("origins and directions must be matching Nx3 arrays")
        if len(layers) != len(origins):
            raise ValueError("one layer tunnel identity is required per ray")
        norms = np.linalg.norm(directions, axis=1, keepdims=True)
        if np.any(norms <= 0.0):
            raise ValueError("ray directions must be nonzero")
        directions = directions / norms
        travel = np.zeros(len(origins), dtype=np.float64)
        last_inside = np.zeros(len(origins), dtype=np.float64)
        active = np.ones(len(origins), dtype=bool)
        hits = np.full(len(origins), np.inf, dtype=np.float64)
        for _ in range(iterations):
            if not active.any():
                break
            indices = np.flatnonzero(active)
            sdf = self.signed_distance(
                origins[indices] + directions[indices] * travel[indices, None], layers[indices]
            )
            escaped = sdf >= 0.0
            if escaped.any():
                hit_indices = indices[escaped]
                low = last_inside[hit_indices].copy()
                high = travel[hit_indices].copy()
                refinable = high > low
                for _ in range(RAY_EXIT_BISECTION_ITERATIONS):
                    if not refinable.any():
                        break
                    midpoint = 0.5 * (low[refinable] + high[refinable])
                    refine_indices = hit_indices[refinable]
                    midpoint_sdf = self.signed_distance(
                        origins[refine_indices] + directions[refine_indices] * midpoint[:, None],
                        layers[refine_indices],
                    )
                    outside = midpoint_sdf >= 0.0
                    refined_low = low[refinable]
                    refined_high = high[refinable]
                    refined_high[outside] = midpoint[outside]
                    refined_low[~outside] = midpoint[~outside]
                    low[refinable] = refined_low
                    high[refinable] = refined_high
                hits[hit_indices] = high
                active[hit_indices] = False
            remaining = indices[~escaped]
            if len(remaining):
                last_inside[remaining] = travel[remaining]
                step = np.maximum(np.abs(sdf[~escaped]) * 0.8, self.spacing_m * 0.5)
                travel[remaining] += step
                active[remaining[travel[remaining] > maximum_m]] = False
        return hits


class ContinuousLayeredHybridField(LayeredHybridField):
    """C1 radial transition from a selected layer to a junction incident union."""

    @staticmethod
    def incident_weight(normalized_radius: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(normalized_radius, dtype=np.float64), 0.0, 1.0)
        return 1.0 - (3.0 * u * u - 2.0 * u * u * u)

    @staticmethod
    def incident_weight_derivative(normalized_radius: np.ndarray) -> np.ndarray:
        u = np.clip(np.asarray(normalized_radius, dtype=np.float64), 0.0, 1.0)
        derivative = -6.0 * u + 6.0 * u * u
        outside = (np.asarray(normalized_radius) < 0.0) | (np.asarray(normalized_radius) > 1.0)
        return np.where(outside, 0.0, derivative)

    def signed_distance(self, points: np.ndarray, layer_tunnel_ids: np.ndarray) -> np.ndarray:
        query = np.asarray(points, dtype=np.float64)
        if query.shape[-1] != 3:
            raise ValueError("points must end in xyz")
        flat = query.reshape(-1, 3)
        layers = np.asarray(layer_tunnel_ids, dtype=np.int64).reshape(-1)
        if len(layers) != len(flat):
            raise ValueError("one layer tunnel identity is required per query point")
        unknown = sorted(set(layers.tolist()) - set(self.base.samples_by_tunnel))
        if unknown:
            raise ValueError(f"unknown layer tunnel identities: {unknown[:3]}")

        values = np.empty(len(flat), dtype=np.float64)
        for tunnel_id in np.unique(layers):
            selected = layers == tunnel_id
            values[selected] = self.base.signed_distance(flat[selected], (int(tunnel_id),))
        for window in self.windows:
            compatible = np.isin(layers, np.asarray(window.incident_tunnel_ids, dtype=np.int64))
            radius = np.linalg.norm(flat - np.asarray(window.center_xyz_m), axis=1)
            selected = compatible & (radius <= window.radius_m)
            if selected.any():
                union = self.base.signed_distance(flat[selected], window.incident_tunnel_ids)
                weight = self.incident_weight(radius[selected] / window.radius_m)
                values[selected] += weight * (union - values[selected])
        return values.reshape(query.shape[:-1])


def build_local_union_windows(
    graph: Mapping[str, Any], geometry_parameters: Mapping[str, Any], *, scale: float = 1.10
) -> tuple[LocalUnionWindow, ...]:
    """Return deterministic degree>=3 windows from topology/radii only."""
    if scale <= 1.0:
        raise ValueError("window scale must exceed 1 to cover the incident tube boundary")
    radii = {int(item["tunnel_id"]): float(item["radius_m"]) for item in geometry_parameters["tunnels"]}
    result: list[LocalUnionWindow] = []
    for node in sorted(graph["nodes"], key=lambda item: str(item["id"])):
        if int(node["degree"]) < 3:
            continue
        incident = tuple(sorted(int(value) for value in node["incident_tunnel_ids"]))
        if not incident or any(value not in radii for value in incident):
            raise ValueError(f"node {node['id']} has invalid incident tunnel identity")
        center = tuple(float(value) for value in node["xyz"])
        if len(center) != 3 or not np.isfinite(center).all():
            raise ValueError(f"node {node['id']} has invalid xyz")
        result.append(LocalUnionWindow(str(node["id"]), center, incident, scale * max(radii[value] for value in incident)))
    return tuple(result)


def overlapping_window_pairs(windows: tuple[LocalUnionWindow, ...]) -> list[dict[str, Any]]:
    """Report geometric overlaps; only identical incident sets are contract-compatible."""
    result: list[dict[str, Any]] = []
    for index, first in enumerate(windows):
        for second in windows[index + 1:]:
            distance = float(np.linalg.norm(np.asarray(first.center_xyz_m) - np.asarray(second.center_xyz_m)))
            overlap = first.radius_m + second.radius_m - distance
            if overlap > 1e-9:
                result.append({
                    "first_node_id": first.node_id,
                    "second_node_id": second.node_id,
                    "center_distance_m": distance,
                    "overlap_m": overlap,
                    "incident_sets_identical": first.incident_tunnel_ids == second.incident_tunnel_ids,
                })
    return result


def _project(point: np.ndarray, polyline: np.ndarray) -> tuple[float, float, np.ndarray, np.ndarray]:
    segments = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if len(segments) == 0 or np.any(lengths <= 1e-12):
        raise ValueError("spline has a degenerate segment")
    ratios = np.clip(np.sum((point - polyline[:-1]) * segments, axis=1) / (lengths * lengths), 0.0, 1.0)
    candidates = polyline[:-1] + ratios[:, None] * segments
    index = int(np.argmin(np.linalg.norm(candidates - point, axis=1)))
    arc = float(np.sum(lengths[:index]) + ratios[index] * lengths[index])
    return arc, float(np.linalg.norm(candidates[index] - point)), candidates[index], segments[index] / lengths[index]


def edge_arc_incidence(graph: Mapping[str, Any], splines: Mapping[str, Any]) -> tuple[ArcIncidence, ...]:
    """Project every graph-edge endpoint to its one physical tunnel spline."""
    nodes = {str(item["id"]): np.asarray(item["xyz"], dtype=np.float64) for item in graph["nodes"]}
    curves = {int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64) for item in splines["tunnels"]}
    records: list[ArcIncidence] = []
    for edge in sorted(graph["edges"], key=lambda item: str(item["id"])):
        tunnel_ids = tuple(int(value) for value in edge["tunnel_ids"])
        if len(tunnel_ids) != 1 or tunnel_ids[0] not in curves:
            raise ValueError(f"edge {edge['id']} has no unique existing tunnel")
        first, second = (str(value) for value in edge["node_ids"])
        if first not in nodes or second not in nodes:
            raise ValueError(f"edge {edge['id']} references an absent node")
        first_arc, first_error, first_xyz, first_tangent = _project(nodes[first], curves[tunnel_ids[0]])
        second_arc, second_error, second_xyz, second_tangent = _project(nodes[second], curves[tunnel_ids[0]])
        if abs(first_arc - second_arc) <= 1e-9:
            raise ValueError(f"edge {edge['id']} endpoints project to one spline arc")
        sign = 1.0 if second_arc > first_arc else -1.0
        for endpoint, node_id, arc, error, xyz, tangent, direction in (
            ("first", first, first_arc, first_error, first_xyz, first_tangent, sign * first_tangent),
            ("second", second, second_arc, second_error, second_xyz, second_tangent, -sign * second_tangent),
        ):
            records.append(ArcIncidence(str(edge["id"]), node_id, endpoint, tunnel_ids[0], arc, error, tuple(xyz.astype(float)), tuple(direction.astype(float))))
    return tuple(records)


def window_boundary_continuity(field: SampledSweptTubeField, window: LocalUnionWindow, sample_count: int = 2048) -> dict[str, float | bool]:
    """Compare global and incident-only fields on a deterministic spherical seam."""
    index = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / sample_count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    unit = np.column_stack((radius_xy * np.cos(angle), radius_xy * np.sin(angle), z))
    points = np.asarray(window.center_xyz_m) + window.radius_m * unit
    global_sdf = field.signed_distance(points)
    local_sdf = field.signed_distance(points, window.incident_tunnel_ids)
    error = np.abs(global_sdf - local_sdf)
    return {"sample_count": sample_count, "maximum_field_error_m": float(error.max()), "mean_field_error_m": float(error.mean()), "passed": bool(error.max() <= field.spacing_m)}


def window_volume_isolation(field: SampledSweptTubeField, window: LocalUnionWindow, sample_count: int = 4096) -> dict[str, float | bool]:
    """Prove that non-incident tubes do not alter a deterministic window sample set."""
    index = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / sample_count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    # Cubic-root radii give a deterministic approximately uniform volume audit.
    radial = ((index * 0.7548776662466927) % 1.0) ** (1.0 / 3.0)
    unit = radial[:, None] * np.column_stack((radius_xy * np.cos(angle), radius_xy * np.sin(angle), z))
    points = np.asarray(window.center_xyz_m) + window.radius_m * unit
    global_sdf = field.signed_distance(points)
    local_sdf = field.signed_distance(points, window.incident_tunnel_ids)
    error = np.abs(global_sdf - local_sdf)
    return {
        "sample_count": sample_count,
        "maximum_field_error_m": float(error.max()),
        "mean_field_error_m": float(error.mean()),
        "passed": bool(error.max() <= field.spacing_m),
    }


def layered_window_seam_continuity(
    field: SampledSweptTubeField, window: LocalUnionWindow, sample_count: int = 2048
) -> dict[str, Any]:
    """Audit every incident layer against the incident union on its window seam.

    Only seam samples within one sampling interval of an incident layer's tube
    are relevant: these are the openings through which a traversal can cross
    the spherical field boundary.
    """
    index = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / sample_count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    unit = np.column_stack((radius_xy * np.cos(angle), radius_xy * np.sin(angle), z))
    points = np.asarray(window.center_xyz_m) + window.radius_m * unit
    union_sdf = field.signed_distance(points, window.incident_tunnel_ids)
    per_layer = []
    all_errors = []
    for tunnel_id in window.incident_tunnel_ids:
        selected_sdf = field.signed_distance(points, (tunnel_id,))
        relevant = selected_sdf <= field.spacing_m
        errors = np.abs(union_sdf[relevant] - selected_sdf[relevant])
        maximum = float(errors.max()) if len(errors) else 0.0
        per_layer.append({
            "tunnel_id": tunnel_id,
            "opening_sample_count": int(relevant.sum()),
            "maximum_field_error_m": maximum,
            "passed": bool(len(errors) and maximum <= field.spacing_m),
        })
        if len(errors):
            all_errors.append(errors)
    maximum = float(max(item.max() for item in all_errors)) if all_errors else float("inf")
    return {
        "sample_count": sample_count,
        "opening_sample_count": int(sum(item["opening_sample_count"] for item in per_layer)),
        "maximum_field_error_m": maximum,
        "per_layer": per_layer,
        "passed": bool(all(item["passed"] for item in per_layer)),
    }


def layered_window_semantics_audit(
    field: LayeredHybridField, window: LocalUnionWindow, sample_count: int = 4096
) -> dict[str, float | int | bool]:
    """Verify incident-union and non-incident-exclusion dispatch inside a window."""
    index = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / sample_count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    radial = ((index * 0.7548776662466927) % 1.0) ** (1.0 / 3.0)
    unit = radial[:, None] * np.column_stack((radius_xy * np.cos(angle), radius_xy * np.sin(angle), z))
    points = np.asarray(window.center_xyz_m) + window.radius_m * unit

    incident = np.asarray(window.incident_tunnel_ids, dtype=np.int64)
    incident_layers = np.resize(incident, sample_count)
    incident_actual = field.signed_distance(points, incident_layers)
    incident_expected = field.base.signed_distance(points, window.incident_tunnel_ids)
    incident_error = np.abs(incident_actual - incident_expected)

    nonincident = np.asarray(sorted(set(field.base.samples_by_tunnel) - set(window.incident_tunnel_ids)), dtype=np.int64)
    if len(nonincident):
        nonincident_layers = np.resize(nonincident, sample_count)
        nonincident_actual = field.signed_distance(points, nonincident_layers)
        nonincident_expected = np.empty(sample_count, dtype=np.float64)
        for tunnel_id in np.unique(nonincident_layers):
            selected = nonincident_layers == tunnel_id
            nonincident_expected[selected] = field.base.signed_distance(points[selected], (int(tunnel_id),))
        nonincident_error = np.abs(nonincident_actual - nonincident_expected)
    else:
        nonincident_error = np.zeros(1, dtype=np.float64)
    maximum = float(max(incident_error.max(), nonincident_error.max()))
    return {
        "sample_count": sample_count,
        "incident_layer_count": int(len(incident)),
        "nonincident_layer_count": int(len(nonincident)),
        "maximum_incident_dispatch_error_m": float(incident_error.max()),
        "maximum_nonincident_dispatch_error_m": float(nonincident_error.max()),
        "passed": bool(maximum <= 1e-12),
    }


def continuous_transition_seam_audit(
    field: ContinuousLayeredHybridField, window: LocalUnionWindow, sample_count: int = 2048
) -> dict[str, Any]:
    """Prove exact value and analytic first-order agreement on a window sphere."""
    index = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / sample_count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    unit = np.column_stack((radius_xy * np.cos(angle), radius_xy * np.sin(angle), z))
    points = np.asarray(window.center_xyz_m) + window.radius_m * unit
    per_layer = []
    for tunnel_id in window.incident_tunnel_ids:
        layers = np.full(sample_count, tunnel_id, dtype=np.int64)
        actual = field.signed_distance(points, layers)
        selected = field.base.signed_distance(points, (tunnel_id,))
        error = np.abs(actual - selected)
        per_layer.append({
            "tunnel_id": tunnel_id,
            "maximum_boundary_value_error_m": float(error.max()),
            "passed": bool(error.max() <= 1e-12),
        })
    boundary_weight = float(field.incident_weight(np.asarray([1.0]))[0])
    boundary_derivative = float(field.incident_weight_derivative(np.asarray([1.0]))[0])
    return {
        "sample_count": sample_count,
        "maximum_boundary_value_error_m": max(item["maximum_boundary_value_error_m"] for item in per_layer),
        "boundary_incident_weight": boundary_weight,
        "boundary_normalized_weight_derivative": boundary_derivative,
        "per_layer": per_layer,
        "passed": bool(
            all(item["passed"] for item in per_layer)
            and abs(boundary_weight) <= 1e-15
            and abs(boundary_derivative) <= 1e-15
        ),
    }


def continuous_transition_semantics_audit(
    field: ContinuousLayeredHybridField, window: LocalUnionWindow, sample_count: int = 4096
) -> dict[str, float | int | bool]:
    """Verify the radial blend formula and strict exclusion of non-incident layers."""
    index = np.arange(sample_count, dtype=np.float64) + 0.5
    z = 1.0 - 2.0 * index / sample_count
    angle = np.pi * (1.0 + np.sqrt(5.0)) * index
    radius_xy = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    radial = ((index * 0.7548776662466927) % 1.0) ** (1.0 / 3.0)
    unit = radial[:, None] * np.column_stack((radius_xy * np.cos(angle), radius_xy * np.sin(angle), z))
    points = np.asarray(window.center_xyz_m) + window.radius_m * unit

    incident = np.asarray(window.incident_tunnel_ids, dtype=np.int64)
    incident_layers = np.resize(incident, sample_count)
    actual = field.signed_distance(points, incident_layers)
    selected = np.empty(sample_count, dtype=np.float64)
    for tunnel_id in np.unique(incident_layers):
        mask = incident_layers == tunnel_id
        selected[mask] = field.base.signed_distance(points[mask], (int(tunnel_id),))
    union = field.base.signed_distance(points, window.incident_tunnel_ids)
    weight = field.incident_weight(radial)
    expected = selected + weight * (union - selected)
    incident_error = np.abs(actual - expected)

    nonincident = np.asarray(sorted(set(field.base.samples_by_tunnel) - set(window.incident_tunnel_ids)), dtype=np.int64)
    if len(nonincident):
        layers = np.resize(nonincident, sample_count)
        actual_nonincident = field.signed_distance(points, layers)
        expected_nonincident = np.empty(sample_count, dtype=np.float64)
        for tunnel_id in np.unique(layers):
            mask = layers == tunnel_id
            expected_nonincident[mask] = field.base.signed_distance(points[mask], (int(tunnel_id),))
        nonincident_error = np.abs(actual_nonincident - expected_nonincident)
    else:
        nonincident_error = np.zeros(1, dtype=np.float64)
    return {
        "sample_count": sample_count,
        "incident_layer_count": int(len(incident)),
        "nonincident_layer_count": int(len(nonincident)),
        "maximum_transition_formula_error_m": float(incident_error.max()),
        "maximum_nonincident_dispatch_error_m": float(nonincident_error.max()),
        "passed": bool(max(incident_error.max(), nonincident_error.max()) <= 1e-12),
    }
