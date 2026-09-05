"""Objective metric and identity targets for variable GSE exit tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.teacher.gse_geometry_teacher import PolylineGeometrySampler
from mtare_topo.teacher.gse_mesh_geometry_teacher import (
    BatchRaycastFunction,
    MeshGeometryMeasurement,
    MeshGeometryTeacherConfig,
    measure_mesh_geometry_batch,
    sensor_origin_from_axis,
)
from mtare_topo.topology.continuous_trajectory import project_to_polyline


@dataclass(frozen=True)
class ObjectiveExitGeometryTarget:
    identity: str
    heading_robot_deg: float
    opening_width_m: float
    vertical_profile_m: tuple[float, float, float, float]
    source_tunnel_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.identity:
            raise ValueError("exit identity must be nonempty")
        if not math.isfinite(self.heading_robot_deg):
            raise ValueError("exit heading must be finite")
        if not math.isfinite(self.opening_width_m) or self.opening_width_m <= 0.0:
            raise ValueError("exit opening width must be positive and finite")
        if len(self.vertical_profile_m) != 4 or not all(math.isfinite(value) for value in self.vertical_profile_m):
            raise ValueError("exit vertical profile must contain four finite values")
        if not self.source_tunnel_ids or not all(self.source_tunnel_ids):
            raise ValueError("exit target must preserve at least one source tunnel identity")
        object.__setattr__(self, "heading_robot_deg", float(self.heading_robot_deg) % 360.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class IncidentExitCandidate:
    identity: str
    edge_id: str
    from_node_id: str
    to_node_id: str
    source_tunnel_ids: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExitRouteTarget:
    candidate: IncidentExitCandidate
    heading_robot_deg: float
    representative_axis_xyz_m: tuple[float, float, float]
    representative_sensor_xyz_m: tuple[float, float, float]
    away_tangent_xyz: tuple[float, float, float]
    vertical_profile_m: tuple[float, float, float, float]


@dataclass(frozen=True)
class AuditedExitGeometryToken:
    identity: str
    heading_robot_deg: float
    opening_width_m: float | None
    opening_width_valid: bool
    vertical_profile_m: tuple[float, float, float, float]
    source_tunnel_ids: tuple[str, ...]
    visible: bool
    line_of_sight_distance_m: float
    first_hit_distance_m: float | None

    def __post_init__(self) -> None:
        if not self.identity or not math.isfinite(self.heading_robot_deg):
            raise ValueError("audited exit identity and heading must be valid")
        if self.opening_width_valid != (self.opening_width_m is not None):
            raise ValueError("opening width mask and value disagree")
        if self.opening_width_m is not None and (
            not math.isfinite(self.opening_width_m) or self.opening_width_m <= 0.0
        ):
            raise ValueError("opening width must be positive and finite")
        if len(self.vertical_profile_m) != 4 or not all(math.isfinite(value) for value in self.vertical_profile_m):
            raise ValueError("vertical profile must contain four finite values")
        if not math.isfinite(self.line_of_sight_distance_m) or self.line_of_sight_distance_m < 0.0:
            raise ValueError("LOS distance must be finite and nonnegative")
        if self.first_hit_distance_m is not None and (
            not math.isfinite(self.first_hit_distance_m) or self.first_hit_distance_m < 0.0
        ):
            raise ValueError("first-hit distance must be finite and nonnegative")
        object.__setattr__(self, "heading_robot_deg", float(self.heading_robot_deg) % 360.0)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ExitRouteGeometryContext:
    """Cache one world's TNG/splines for objective incident-exit geometry."""

    def __init__(
        self,
        *,
        parent_id: str,
        graph: Mapping[str, Any],
        spline_document: Mapping[str, Any],
        geometry_parameters: Mapping[str, Any],
    ) -> None:
        if not isinstance(parent_id, str) or not parent_id.strip():
            raise ValueError("exit route parent identity must be a nonempty string")
        self.parent_id = parent_id
        self.graph = graph
        self.nodes = {}
        for node in graph["nodes"]:
            node_id = str(node["id"])
            xyz = np.asarray(node["xyz"], dtype=np.float64)
            if not node_id or xyz.shape != (3,) or not np.all(np.isfinite(xyz)):
                raise ValueError("exit route graph contains an invalid node")
            self.nodes[node_id] = xyz
        self.edges = {str(edge["id"]): edge for edge in graph["edges"]}
        self.splines = {}
        for item in spline_document["tunnels"]:
            tunnel_id = str(item["tunnel_id"])
            points = np.asarray(item["points"], dtype=np.float64)
            if (
                not tunnel_id
                or points.ndim != 2
                or points.shape[1:] != (3,)
                or len(points) < 2
                or not np.all(np.isfinite(points))
            ):
                raise ValueError("exit route spline document contains an invalid tunnel")
            self.splines[tunnel_id] = points
        self.fta_distance_m = float(geometry_parameters["fta_distance_m"])
        if not math.isfinite(self.fta_distance_m):
            raise ValueError("exit route context identity/FTA is invalid")

    def route_target(
        self,
        *,
        candidate: IncidentExitCandidate,
        current_axis_xyz_m: Sequence[float],
        robot_yaw_deg: float,
        selected_node_id: str | None,
        representative_offset_m: float = 2.5,
    ) -> ExitRouteTarget:
        current_axis = np.asarray(current_axis_xyz_m, dtype=np.float64)
        if (
            candidate.edge_id not in self.edges
            or current_axis.shape != (3,)
            or not np.all(np.isfinite(current_axis))
            or not math.isfinite(float(robot_yaw_deg))
            or not math.isfinite(float(representative_offset_m))
            or representative_offset_m <= 0.0
        ):
            raise ValueError("exit candidate edge or representative offset is invalid")
        endpoints = tuple(str(value) for value in self.edges[candidate.edge_id].get("node_ids", ()))
        if len(endpoints) != 2 or any(node_id not in self.nodes for node_id in endpoints):
            raise ValueError("exit candidate edge has invalid endpoints")
        # Kept local to avoid a package-initialization cycle when data modules
        # import the teacher package from a cold Python process.
        from mtare_topo.data.gse_teacher_inventory import oriented_edge_polyline

        points, _, from_node, to_node = oriented_edge_polyline(
            edge=self.edges[candidate.edge_id],
            nodes=self.nodes,
            splines=self.splines,
            reverse=(str(self.edges[candidate.edge_id]["node_ids"][0]) != candidate.from_node_id),
        )
        if (from_node, to_node) != (candidate.from_node_id, candidate.to_node_id):
            raise RuntimeError("candidate orientation does not match physical edge")
        sampler = PolylineGeometrySampler(points)
        if selected_node_id is None:
            base_arc = float(project_to_polyline(current_axis, points).arc_m)
        else:
            if selected_node_id != candidate.from_node_id:
                raise ValueError("node-event candidate must point away from selected node")
            base_arc = 0.0
        representative_arc = min(base_arc + float(representative_offset_m), sampler.length_m)
        axis = sampler.interpolate(representative_arc)
        tangent = np.asarray(
            sampler.target(representative_arc, tunnel_radius_m=1.0).axis,
            dtype=np.float64,
        )
        heading_world = math.degrees(math.atan2(float(tangent[1]), float(tangent[0])))
        return ExitRouteTarget(
            candidate=candidate,
            heading_robot_deg=float(heading_world - float(robot_yaw_deg)) % 360.0,
            representative_axis_xyz_m=tuple(float(value) for value in axis),
            representative_sensor_xyz_m=tuple(
                float(value)
                for value in sensor_origin_from_axis(axis, fta_distance_m=self.fta_distance_m)
            ),
            away_tangent_xyz=tuple(float(value) for value in tangent),
            vertical_profile_m=spline_vertical_profile(
                points,
                current_arc_m=base_arc,
                direction_sign=1,
            ),
        )


def audit_exit_route_targets(
    targets: Sequence[ExitRouteTarget],
    *,
    current_sensor_xyz_m: np.ndarray,
    cast_distances: BatchRaycastFunction,
    mesh_config: MeshGeometryTeacherConfig | None = None,
    los_margin_m: float = 0.25,
) -> tuple[AuditedExitGeometryToken, ...]:
    """Batch native-mesh opening-width and first-return LOS qualification."""

    cfg = mesh_config or MeshGeometryTeacherConfig()
    current = np.asarray(current_sensor_xyz_m, dtype=np.float64)
    if (
        not math.isfinite(float(los_margin_m))
        or los_margin_m < 0.0
        or current.shape != (len(targets), 3)
        or not np.all(np.isfinite(current))
    ):
        raise ValueError("exit target LOS inputs are invalid")
    if not targets:
        return ()
    representative_sensors = np.asarray(
        [target.representative_sensor_xyz_m for target in targets],
        dtype=np.float64,
    )
    tangents = np.asarray([target.away_tangent_xyz for target in targets], dtype=np.float64)
    measurements, _ = measure_mesh_geometry_batch(
        origins_xyz_m=representative_sensors,
        tangents_xyz=tangents,
        cast_distances=cast_distances,
        config=cfg,
    )
    vectors = representative_sensors - current
    distances = np.linalg.norm(vectors, axis=1)
    nonzero = distances > 1e-8
    directions = np.zeros((len(targets), 1, 3), dtype=np.float64)
    directions[nonzero, 0] = vectors[nonzero] / distances[nonzero, None]
    directions[~nonzero, 0, 0] = 1.0
    raw_hits = np.asarray(cast_distances(current, directions), dtype=np.float64)
    if raw_hits.shape != (len(targets), 1):
        raise ValueError("batch raycaster returned the wrong LOS distance shape")
    hits = raw_hits[:, 0]
    if np.any(np.isnan(hits)) or np.any(hits < 0.0):
        raise ValueError("batch raycaster returned invalid LOS distances")
    result = []
    for index, target in enumerate(targets):
        hit = float(hits[index]) if math.isfinite(float(hits[index])) else None
        visible = bool(
            not nonzero[index]
            or hit is None
            or hit >= float(distances[index]) - float(los_margin_m)
        )
        measurement = measurements[index]
        result.append(
            AuditedExitGeometryToken(
                identity=target.candidate.identity,
                heading_robot_deg=target.heading_robot_deg,
                opening_width_m=measurement.width_m if measurement is not None else None,
                opening_width_valid=measurement is not None,
                vertical_profile_m=target.vertical_profile_m,
                source_tunnel_ids=target.candidate.source_tunnel_ids,
                visible=visible,
                line_of_sight_distance_m=float(distances[index]),
                first_hit_distance_m=hit,
            )
        )
    return tuple(result)


def directed_exit_identity(
    *,
    parent_id: str,
    edge_id: str,
    from_node_id: str,
    to_node_id: str,
) -> str:
    if not all((parent_id, edge_id, from_node_id, to_node_id)) or from_node_id == to_node_id:
        raise ValueError("directed exit identity requires nonempty distinct endpoints")
    return f"{parent_id}:{edge_id}:{from_node_id}->{to_node_id}"


def incident_exit_candidates(
    *,
    parent_id: str,
    graph: dict[str, Any],
    current_edge_id: str,
    selected_node_id: str | None,
) -> tuple[IncidentExitCandidate, ...]:
    """Enumerate physical directed exits without collapsing tunnel IDs."""

    edges = {str(edge["id"]): edge for edge in graph["edges"]}
    if current_edge_id not in edges:
        raise ValueError("current physical edge is absent from graph")
    selected: list[tuple[str, str, str, tuple[str, ...]]] = []
    if selected_node_id is None:
        edge = edges[current_edge_id]
        endpoints = tuple(str(value) for value in edge["node_ids"])
        if len(endpoints) != 2 or endpoints[0] == endpoints[1]:
            raise ValueError("physical edge must have two distinct endpoints")
        tunnels = tuple(sorted(str(value) for value in edge.get("tunnel_ids", ())))
        selected.extend(((current_edge_id, endpoints[0], endpoints[1], tunnels), (current_edge_id, endpoints[1], endpoints[0], tunnels)))
    else:
        if selected_node_id not in {str(node["id"]) for node in graph["nodes"]}:
            raise ValueError("selected structural node is absent from graph")
        for edge_id, edge in edges.items():
            endpoints = tuple(str(value) for value in edge["node_ids"])
            if len(endpoints) != 2 or endpoints[0] == endpoints[1]:
                raise ValueError("physical edge must have two distinct endpoints")
            if selected_node_id not in endpoints:
                continue
            other = endpoints[1] if endpoints[0] == selected_node_id else endpoints[0]
            tunnels = tuple(sorted(str(value) for value in edge.get("tunnel_ids", ())))
            selected.append((edge_id, selected_node_id, other, tunnels))
    if not selected:
        raise ValueError("exit candidate set is empty")
    candidates = tuple(
        IncidentExitCandidate(
            identity=directed_exit_identity(parent_id=parent_id, edge_id=edge_id, from_node_id=from_node, to_node_id=to_node),
            edge_id=edge_id,
            from_node_id=from_node,
            to_node_id=to_node,
            source_tunnel_ids=tunnels,
        )
        for edge_id, from_node, to_node, tunnels in sorted(selected)
    )
    if len({candidate.identity for candidate in candidates}) != len(candidates):
        raise RuntimeError("directed exit candidate identities are not unique")
    return candidates


def spline_vertical_profile(
    points: np.ndarray,
    *,
    current_arc_m: float,
    direction_sign: int,
    lookahead_offsets_m: Sequence[float] = (2.5, 5.0, 7.5, 10.0),
) -> tuple[float, float, float, float]:
    """Return metric centerline height change along one visible exit."""

    if direction_sign not in {-1, 1}:
        raise ValueError("direction_sign must be -1 or 1")
    offsets = tuple(float(value) for value in lookahead_offsets_m)
    if len(offsets) != 4 or tuple(sorted(offsets)) != offsets or any(value <= 0.0 for value in offsets):
        raise ValueError("lookahead offsets must be four sorted positive distances")
    sampler = PolylineGeometrySampler(np.asarray(points, dtype=np.float64))
    current_arc = float(np.clip(float(current_arc_m), 0.0, sampler.length_m))
    current = sampler.interpolate(current_arc)
    profile = []
    for offset in offsets:
        arc = float(np.clip(current_arc + direction_sign * offset, 0.0, sampler.length_m))
        profile.append(float(sampler.interpolate(arc)[2] - current[2]))
    return tuple(profile)  # type: ignore[return-value]


def make_objective_exit_target(
    *,
    identity: str,
    heading_world_deg: float,
    robot_yaw_deg: float,
    cross_section: MeshGeometryMeasurement,
    vertical_profile_m: Sequence[float],
    source_tunnel_ids: Sequence[str | int],
) -> ObjectiveExitGeometryTarget:
    profile = tuple(float(value) for value in vertical_profile_m)
    tunnels = tuple(sorted({str(value) for value in source_tunnel_ids}))
    return ObjectiveExitGeometryTarget(
        identity=identity,
        heading_robot_deg=float(heading_world_deg) - float(robot_yaw_deg),
        opening_width_m=cross_section.width_m,
        vertical_profile_m=profile,  # type: ignore[arg-type]
        source_tunnel_ids=tunnels,
    )


__all__ = [
    "AuditedExitGeometryToken",
    "ExitRouteGeometryContext",
    "ExitRouteTarget",
    "IncidentExitCandidate",
    "ObjectiveExitGeometryTarget",
    "audit_exit_route_targets",
    "directed_exit_identity",
    "incident_exit_candidates",
    "make_objective_exit_target",
    "spline_vertical_profile",
]
