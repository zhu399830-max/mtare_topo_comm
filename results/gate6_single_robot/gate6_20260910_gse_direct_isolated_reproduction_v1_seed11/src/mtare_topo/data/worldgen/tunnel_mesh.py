"""Deterministic voxel-union meshing for TNG-first underground worlds.

The occupied voxels represent tunnel free space.  Their closed boundary is
exported as the tunnel wall/floor/ceiling collision surface.  This first smoke
implementation keeps the original straight V0 path reproducible and adds an
optional navigation-grade curved-centerline path with explicit robot probes.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import random
import struct
from typing import Iterable, Mapping

from .tng import SeedBundle, TunnelNetworkGraph, Vec3


class MeshGenerationError(RuntimeError):
    """Raised when geometry would violate the topology/mesh contract."""


@dataclass(frozen=True)
class TunnelMeshParameters:
    geometry_variant_id: str = "g000"
    voxel_size_m: float = 0.5
    lateral_half_width_m: float = 2.5
    lateral_half_width_noise_m: float = 0.2
    vertical_radius_m: float = 2.5
    vertical_radius_noise_m: float = 0.15
    floor_depth_m: float = 1.5
    geometry_guard_m: float = 1.0
    minimum_robot_radius_m: float = 0.6
    centerline_mode: str = "straight"
    centerline_sample_spacing_m: float = 0.5
    curve_tangent_scale: float = 0.8
    curve_wander_m: float = 0.0
    floor_profile_mode: str = "segment_normal"
    junction_chamber_scale: float = 1.0
    navigation_audit_enabled: bool = False
    robot_height_m: float = 1.5
    robot_safety_margin_m: float = 0.2
    robot_ground_clearance_m: float = 0.25
    maximum_path_pitch_rad: float = 0.22
    minimum_path_turn_radius_m: float = 2.5
    surface_noise_enabled: bool = False
    clutter_enabled: bool = False
    frame: str = "right_handed_z_up"
    unit: str = "meter"

    @classmethod
    def from_dict(cls, payload: Mapping[str, object]) -> "TunnelMeshParameters":
        if payload.get("schema_version") not in {
            "tunnel_mesh_config_v1",
            "tunnel_mesh_config_v2",
        }:
            raise ValueError("unsupported tunnel mesh config schema")
        fields = {
            key: value
            for key, value in payload.items()
            if key != "schema_version"
        }
        result = cls(**fields)
        result.validate()
        return result

    def validate(self) -> None:
        positive = {
            "voxel_size_m": self.voxel_size_m,
            "lateral_half_width_m": self.lateral_half_width_m,
            "vertical_radius_m": self.vertical_radius_m,
            "floor_depth_m": self.floor_depth_m,
            "geometry_guard_m": self.geometry_guard_m,
            "minimum_robot_radius_m": self.minimum_robot_radius_m,
            "centerline_sample_spacing_m": self.centerline_sample_spacing_m,
            "robot_height_m": self.robot_height_m,
            "robot_ground_clearance_m": self.robot_ground_clearance_m,
            "maximum_path_pitch_rad": self.maximum_path_pitch_rad,
            "minimum_path_turn_radius_m": self.minimum_path_turn_radius_m,
        }
        for name, value in positive.items():
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        if self.lateral_half_width_noise_m < 0 or self.vertical_radius_noise_m < 0:
            raise ValueError("geometry noise magnitudes cannot be negative")
        if self.curve_wander_m < 0:
            raise ValueError("curve_wander_m cannot be negative")
        if not 0 < self.curve_tangent_scale <= 1.5:
            raise ValueError("curve_tangent_scale must be in (0, 1.5]")
        if self.robot_safety_margin_m < 0:
            raise ValueError("robot_safety_margin_m cannot be negative")
        if self.junction_chamber_scale < 1.0:
            raise ValueError("junction_chamber_scale cannot be below 1")
        if self.floor_depth_m >= self.vertical_radius_m - self.vertical_radius_noise_m:
            raise ValueError("floor_depth_m must be below every vertical radius")
        if self.minimum_robot_radius_m >= self.lateral_half_width_m - self.lateral_half_width_noise_m:
            raise ValueError("minimum robot radius does not fit the narrowest tunnel")
        if self.centerline_mode not in {"straight", "cubic_hermite"}:
            raise ValueError("unsupported centerline_mode")
        if self.floor_profile_mode not in {"segment_normal", "global_z"}:
            raise ValueError("unsupported floor_profile_mode")
        if self.navigation_audit_enabled:
            minimum_half_width = (
                self.lateral_half_width_m - self.lateral_half_width_noise_m
            )
            minimum_headroom = (
                self.vertical_radius_m
                - self.vertical_radius_noise_m
                + self.floor_depth_m
            )
            if (
                self.minimum_robot_radius_m + self.robot_safety_margin_m
                >= minimum_half_width
            ):
                raise ValueError("robot radius plus safety margin does not fit")
            if (
                self.robot_height_m
                + self.robot_ground_clearance_m
                + self.robot_safety_margin_m
                >= minimum_headroom
            ):
                raise ValueError("robot height plus vertical margins does not fit")
            if self.centerline_mode != "cubic_hermite":
                raise ValueError("navigation audit requires cubic_hermite centerlines")
            if self.floor_profile_mode != "global_z":
                raise ValueError("navigation audit requires a global_z floor profile")
        if self.surface_noise_enabled or self.clutter_enabled:
            raise ValueError("surface noise and clutter are intentionally disabled in this smoke")
        if self.frame != "right_handed_z_up" or self.unit != "meter":
            raise ValueError("only right-handed Z-up metres are supported")


@dataclass(frozen=True)
class TriangleMesh:
    vertices: tuple[Vec3, ...]
    faces: tuple[tuple[int, int, int], ...]
    voxel_origin: Vec3
    voxel_size_m: float
    occupied_voxel_count: int
    geometry_variant_id: str
    topology_parent_id: str
    geometry_seed: int
    navigation_paths: tuple["NavigationPath", ...]
    navigation_audit: Mapping[str, object]

    def canonical_hash(self) -> str:
        digest = hashlib.sha256()
        digest.update(b"mtare-tunnel-mesh-v1\0")
        digest.update(self.topology_parent_id.encode("utf-8"))
        digest.update(self.geometry_variant_id.encode("utf-8"))
        digest.update(struct.pack(">Qd", self.geometry_seed, self.voxel_size_m))
        for vertex in self.vertices:
            digest.update(struct.pack(">ddd", vertex.x, vertex.y, vertex.z))
        for face in self.faces:
            digest.update(struct.pack(">QQQ", *face))
        return digest.hexdigest()


@dataclass(frozen=True)
class _SegmentGeometry:
    edge_id: str
    start: Vec3
    end: Vec3
    tangent: Vec3
    lateral: Vec3
    normal: Vec3
    length: float
    lateral_half_width: float
    vertical_radius: float


@dataclass(frozen=True)
class NavigationPath:
    edge_id: str
    u: str
    v: str
    points: tuple[Vec3, ...]
    lateral_half_width: float
    vertical_radius: float


def _dot(first: Vec3, second: Vec3) -> float:
    return first.x * second.x + first.y * second.y + first.z * second.z


def _sub(first: Vec3, second: Vec3) -> Vec3:
    return Vec3(first.x - second.x, first.y - second.y, first.z - second.z)


def _cross(first: Vec3, second: Vec3) -> Vec3:
    return Vec3(
        first.y * second.z - first.z * second.y,
        first.z * second.x - first.x * second.z,
        first.x * second.y - first.y * second.x,
    )


def _scale(vector: Vec3, scalar: float) -> Vec3:
    return Vec3(vector.x * scalar, vector.y * scalar, vector.z * scalar)


def _add(first: Vec3, second: Vec3) -> Vec3:
    return Vec3(first.x + second.x, first.y + second.y, first.z + second.z)


def _norm(vector: Vec3) -> float:
    return math.sqrt(_dot(vector, vector))


def _normalize(vector: Vec3) -> Vec3:
    length = _norm(vector)
    if length <= 1e-12:
        raise MeshGenerationError("cannot normalize a zero vector")
    return _scale(vector, 1.0 / length)


def _lerp(first: Vec3, second: Vec3, ratio: float) -> Vec3:
    return _add(_scale(first, 1.0 - ratio), _scale(second, ratio))


def _node_continuation_axes(graph: TunnelNetworkGraph) -> dict[str, Vec3]:
    node_map = graph.node_map()
    adjacency = graph.adjacency()
    axes: dict[str, Vec3] = {}
    for node_id, neighbours in adjacency.items():
        if len(neighbours) != 2:
            continue
        first_id, second_id = sorted(neighbours)
        axis = _sub(node_map[second_id].position, node_map[first_id].position)
        if _norm(axis) > 1e-9:
            axes[node_id] = _normalize(axis)
    return axes


def _oriented_node_direction(
    node_id: str,
    other_id: str,
    graph: TunnelNetworkGraph,
    continuation_axes: Mapping[str, Vec3],
) -> Vec3:
    node_map = graph.node_map()
    direct = _normalize(_sub(node_map[other_id].position, node_map[node_id].position))
    axis = continuation_axes.get(node_id)
    if axis is None:
        return direct
    if _dot(axis, direct) < 0:
        axis = _scale(axis, -1.0)
    # Keep the graph edge direction dominant so smoothing cannot fold backward.
    return _normalize(_add(_scale(direct, 0.45), _scale(axis, 0.55)))


def _cubic_hermite_point(
    start: Vec3,
    end: Vec3,
    start_tangent: Vec3,
    end_tangent: Vec3,
    ratio: float,
) -> Vec3:
    t2 = ratio * ratio
    t3 = t2 * ratio
    return _add(
        _add(_scale(start, 2 * t3 - 3 * t2 + 1), _scale(start_tangent, t3 - 2 * t2 + ratio)),
        _add(_scale(end, -2 * t3 + 3 * t2), _scale(end_tangent, t3 - t2)),
    )


def _navigation_paths(
    graph: TunnelNetworkGraph,
    geometry_seed: int,
    parameters: TunnelMeshParameters,
) -> tuple[NavigationPath, ...]:
    rng = random.Random(geometry_seed)
    node_map = graph.node_map()
    continuation_axes = _node_continuation_axes(graph)
    result: list[NavigationPath] = []
    up = Vec3(0.0, 0.0, 1.0)
    for edge in sorted(graph.edges, key=lambda item: item.id):
        start = node_map[edge.u].position
        end = node_map[edge.v].position
        delta = _sub(end, start)
        length = _norm(delta)
        direct = _scale(delta, 1.0 / length)
        lateral_half_width = parameters.lateral_half_width_m + rng.uniform(
            -parameters.lateral_half_width_noise_m,
            parameters.lateral_half_width_noise_m,
        )
        vertical_radius = parameters.vertical_radius_m + rng.uniform(
            -parameters.vertical_radius_noise_m,
            parameters.vertical_radius_noise_m,
        )
        if parameters.centerline_mode == "straight":
            points = (start, end)
        else:
            start_direction = _oriented_node_direction(
                edge.u, edge.v, graph, continuation_axes
            )
            end_outward = _oriented_node_direction(
                edge.v, edge.u, graph, continuation_axes
            )
            tangent_length = length * parameters.curve_tangent_scale
            start_tangent = _scale(start_direction, tangent_length)
            end_tangent = _scale(end_outward, -tangent_length)
            lateral_raw = _cross(up, direct)
            if _norm(lateral_raw) <= 1e-8:
                lateral_raw = Vec3(1.0, 0.0, 0.0)
            lateral = _normalize(lateral_raw)
            signed_wander = rng.uniform(
                -parameters.curve_wander_m, parameters.curve_wander_m
            )
            sample_count = max(
                2, math.ceil(length / parameters.centerline_sample_spacing_m) + 1
            )
            generated: list[Vec3] = []
            for sample_index in range(sample_count):
                ratio = sample_index / (sample_count - 1)
                point = _cubic_hermite_point(
                    start, end, start_tangent, end_tangent, ratio
                )
                # sin(pi*t)^2 keeps endpoint positions and tangents unchanged.
                wander = signed_wander * math.sin(math.pi * ratio) ** 2
                curved = _add(point, _scale(lateral, wander))
                # Horizontal curvature must not invent a steeper vertical profile.
                generated.append(
                    Vec3(curved.x, curved.y, _lerp(start, end, ratio).z)
                )
            horizontal_steps = [
                math.hypot(second.x - first.x, second.y - first.y)
                for first, second in zip(generated, generated[1:])
            ]
            horizontal_length = sum(horizontal_steps)
            if horizontal_length <= 1e-9:
                raise MeshGenerationError(
                    f"edge {edge.id} has no horizontal navigation extent"
                )
            accumulated = 0.0
            elevation_adjusted: list[Vec3] = []
            for point_index, point in enumerate(generated):
                if point_index > 0:
                    accumulated += horizontal_steps[point_index - 1]
                elevation_ratio = accumulated / horizontal_length
                elevation_adjusted.append(
                    Vec3(
                        point.x,
                        point.y,
                        start.z + (end.z - start.z) * elevation_ratio,
                    )
                )
            generated = elevation_adjusted
            points = tuple(generated)
        result.append(
            NavigationPath(
                edge_id=edge.id,
                u=edge.u,
                v=edge.v,
                points=points,
                lateral_half_width=lateral_half_width,
                vertical_radius=vertical_radius,
            )
        )
    return tuple(result)


def _path_segments(paths: tuple[NavigationPath, ...]) -> tuple[_SegmentGeometry, ...]:
    result: list[_SegmentGeometry] = []
    up = Vec3(0.0, 0.0, 1.0)
    for path in paths:
        for start, end in zip(path.points, path.points[1:]):
            delta = _sub(end, start)
            length = _norm(delta)
            tangent = _scale(delta, 1.0 / length)
            lateral_raw = _cross(up, tangent)
            if _norm(lateral_raw) <= 1e-8:
                lateral_raw = Vec3(1.0, 0.0, 0.0)
            lateral = _normalize(lateral_raw)
            normal = _normalize(_cross(tangent, lateral))
            if normal.z < 0:
                normal = _scale(normal, -1.0)
                lateral = _scale(lateral, -1.0)
            result.append(
                _SegmentGeometry(
                    edge_id=path.edge_id,
                    start=start,
                    end=end,
                    tangent=tangent,
                    lateral=lateral,
                    normal=normal,
                    length=length,
                    lateral_half_width=path.lateral_half_width,
                    vertical_radius=path.vertical_radius,
                )
            )
    return tuple(result)


def _inside_segment(
    point: Vec3,
    segment: _SegmentGeometry,
    floor_depth: float,
    floor_profile_mode: str,
) -> bool:
    relative = _sub(point, segment.start)
    projection = _dot(relative, segment.tangent)
    clamped = max(0.0, min(segment.length, projection))
    closest = _add(segment.start, _scale(segment.tangent, clamped))
    offset = _sub(point, closest)
    axial_residual = projection - clamped
    lateral = _dot(offset, segment.lateral)
    vertical = (
        point.z - closest.z
        if floor_profile_mode == "global_z"
        else _dot(offset, segment.normal)
    )
    radial = math.sqrt(lateral * lateral + axial_residual * axial_residual)
    ellipse = (
        (radial / segment.lateral_half_width) ** 2
        + (vertical / segment.vertical_radius) ** 2
    )
    return ellipse <= 1.0 and vertical >= -floor_depth


def _inside_chamber(
    point: Vec3,
    center: Vec3,
    lateral_radius: float,
    vertical_radius: float,
    floor_depth: float,
) -> bool:
    delta = _sub(point, center)
    horizontal = math.hypot(delta.x, delta.y)
    ellipse = (horizontal / lateral_radius) ** 2 + (
        delta.z / vertical_radius
    ) ** 2
    return ellipse <= 1.0 and delta.z >= -floor_depth


def _voxel_components(occupied: set[tuple[int, int, int]]) -> int:
    remaining = set(occupied)
    components = 0
    neighbours = ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1))
    while remaining:
        components += 1
        queue = [remaining.pop()]
        while queue:
            i, j, k = queue.pop()
            for di, dj, dk in neighbours:
                candidate = (i + di, j + dj, k + dk)
                if candidate in remaining:
                    remaining.remove(candidate)
                    queue.append(candidate)
    return components


def _extract_boundary(
    occupied: set[tuple[int, int, int]],
    origin: Vec3,
    voxel_size: float,
) -> tuple[tuple[Vec3, ...], tuple[tuple[int, int, int], ...]]:
    """Extract a topology-consistent isosurface with marching tetrahedra."""

    cube_offsets = (
        (0, 0, 0),
        (1, 0, 0),
        (0, 1, 0),
        (1, 1, 0),
        (0, 0, 1),
        (1, 0, 1),
        (0, 1, 1),
        (1, 1, 1),
    )
    # A global 0-to-7 body diagonal gives matching triangulations on shared faces.
    tetrahedra = (
        (0, 1, 3, 7),
        (0, 3, 2, 7),
        (0, 2, 6, 7),
        (0, 6, 4, 7),
        (0, 4, 5, 7),
        (0, 5, 1, 7),
    )
    tetra_edges = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))
    edge_vertex_indices: dict[
        tuple[tuple[int, int, int], tuple[int, int, int]], int
    ] = {}
    vertices: list[Vec3] = []
    faces: list[tuple[int, int, int]] = []

    def lattice_point(index: tuple[int, int, int]) -> Vec3:
        return Vec3(
            origin.x + (index[0] + 0.5) * voxel_size,
            origin.y + (index[1] + 0.5) * voxel_size,
            origin.z + (index[2] + 0.5) * voxel_size,
        )

    def edge_vertex_index(
        first: tuple[int, int, int], second: tuple[int, int, int]
    ) -> int:
        key = tuple(sorted((first, second)))
        existing = edge_vertex_indices.get(key)
        if existing is not None:
            return existing
        index = len(vertices)
        edge_vertex_indices[key] = index
        first_point = lattice_point(first)
        second_point = lattice_point(second)
        vertices.append(_scale(_add(first_point, second_point), 0.5))
        return index

    candidate_cubes: set[tuple[int, int, int]] = set()
    for i, j, k in occupied:
        for di in (-1, 0):
            for dj in (-1, 0):
                for dk in (-1, 0):
                    candidate_cubes.add((i + di, j + dj, k + dk))

    for cube_i, cube_j, cube_k in sorted(candidate_cubes):
        cube_vertices = tuple(
            (cube_i + di, cube_j + dj, cube_k + dk)
            for di, dj, dk in cube_offsets
        )
        cube_inside = tuple(vertex in occupied for vertex in cube_vertices)
        if all(cube_inside) or not any(cube_inside):
            continue
        for tetrahedron in tetrahedra:
            tetra_vertices = tuple(cube_vertices[index] for index in tetrahedron)
            tetra_inside = tuple(vertex in occupied for vertex in tetra_vertices)
            inside_count = sum(tetra_inside)
            if inside_count in {0, 4}:
                continue
            crossing_indices: list[int] = []
            crossing_points: list[Vec3] = []
            for first_local, second_local in tetra_edges:
                if tetra_inside[first_local] == tetra_inside[second_local]:
                    continue
                first = tetra_vertices[first_local]
                second = tetra_vertices[second_local]
                crossing_indices.append(edge_vertex_index(first, second))
                crossing_points.append(
                    _scale(_add(lattice_point(first), lattice_point(second)), 0.5)
                )
            if len(crossing_indices) not in {3, 4}:
                raise MeshGenerationError("invalid marching-tetrahedra intersection")

            inside_points = [
                lattice_point(vertex)
                for vertex, is_inside in zip(tetra_vertices, tetra_inside)
                if is_inside
            ]
            outside_points = [
                lattice_point(vertex)
                for vertex, is_inside in zip(tetra_vertices, tetra_inside)
                if not is_inside
            ]
            inside_centroid = _scale(
                Vec3(
                    sum(point.x for point in inside_points),
                    sum(point.y for point in inside_points),
                    sum(point.z for point in inside_points),
                ),
                1.0 / len(inside_points),
            )
            outside_centroid = _scale(
                Vec3(
                    sum(point.x for point in outside_points),
                    sum(point.y for point in outside_points),
                    sum(point.z for point in outside_points),
                ),
                1.0 / len(outside_points),
            )
            outward = _normalize(_sub(outside_centroid, inside_centroid))
            polygon_centroid = _scale(
                Vec3(
                    sum(point.x for point in crossing_points),
                    sum(point.y for point in crossing_points),
                    sum(point.z for point in crossing_points),
                ),
                1.0 / len(crossing_points),
            )
            basis = _normalize(_sub(crossing_points[0], polygon_centroid))
            perpendicular = _normalize(_cross(outward, basis))
            ordered = sorted(
                zip(crossing_indices, crossing_points),
                key=lambda item: math.atan2(
                    _dot(_sub(item[1], polygon_centroid), perpendicular),
                    _dot(_sub(item[1], polygon_centroid), basis),
                ),
            )
            ordered_indices = [item[0] for item in ordered]
            for index in range(1, len(ordered_indices) - 1):
                faces.append(
                    (
                        ordered_indices[0],
                        ordered_indices[index],
                        ordered_indices[index + 1],
                    )
                )
    return tuple(vertices), tuple(faces)


def _point_voxel_index(
    point: Vec3, origin: Vec3, voxel_size: float
) -> tuple[int, int, int]:
    return (
        math.floor((point.x - origin.x) / voxel_size),
        math.floor((point.y - origin.y) / voxel_size),
        math.floor((point.z - origin.z) / voxel_size),
    )


def _navigation_audit(
    graph: TunnelNetworkGraph,
    paths: tuple[NavigationPath, ...],
    occupied: set[tuple[int, int, int]],
    origin: Vec3,
    parameters: TunnelMeshParameters,
    required_clearance: float,
) -> dict[str, object]:
    if not parameters.navigation_audit_enabled:
        return {"status": "NOT_REQUESTED"}

    max_pitch = 0.0
    max_pitch_edge_id: str | None = None
    minimum_turn_radius = math.inf
    minimum_turn_radius_edge_id: str | None = None
    route_length = 0.0
    for path in paths:
        for first, second in zip(path.points, path.points[1:]):
            delta = _sub(second, first)
            horizontal = math.hypot(delta.x, delta.y)
            pitch = math.atan2(abs(delta.z), horizontal)
            if pitch > max_pitch:
                max_pitch = pitch
                max_pitch_edge_id = path.edge_id
            route_length += _norm(delta)
        for first, middle, third in zip(
            path.points, path.points[1:], path.points[2:]
        ):
            a = first.distance(middle)
            b = middle.distance(third)
            c = first.distance(third)
            double_area = _norm(_cross(_sub(middle, first), _sub(third, first)))
            if double_area > 1e-9:
                radius = a * b * c / (2.0 * double_area)
                if radius < minimum_turn_radius:
                    minimum_turn_radius = radius
                    minimum_turn_radius_edge_id = path.edge_id

    minimum_nonincident_path_distance = math.inf
    for first_index, first in enumerate(paths):
        first_nodes = {first.u, first.v}
        for second in paths[first_index + 1 :]:
            if first_nodes & {second.u, second.v}:
                continue
            for first_point in first.points:
                for second_point in second.points:
                    minimum_nonincident_path_distance = min(
                        minimum_nonincident_path_distance,
                        first_point.distance(second_point),
                    )

    node_map = graph.node_map()
    minimum_nonincident_node_path_distance = math.inf
    for node_id, node in node_map.items():
        for path in paths:
            if node_id in {path.u, path.v}:
                continue
            for point in path.points:
                minimum_nonincident_node_path_distance = min(
                    minimum_nonincident_node_path_distance,
                    node.position.distance(point),
                )

    probe_radius = (
        parameters.minimum_robot_radius_m + parameters.robot_safety_margin_m
    )
    robot_probe_count = 0
    robot_probe_failure_count = 0
    failed_robot_probes: list[dict[str, object]] = []
    for path in paths:
        for point_index, point in enumerate(path.points):
            if point_index == 0:
                tangent = _sub(path.points[1], point)
            elif point_index == len(path.points) - 1:
                tangent = _sub(point, path.points[-2])
            else:
                tangent = _sub(path.points[point_index + 1], path.points[point_index - 1])
            horizontal = max(math.hypot(tangent.x, tangent.y), 1e-9)
            local_pitch = math.atan2(abs(tangent.z), horizontal)
            slope_lift = probe_radius * math.tan(local_pitch)
            base_z = (
                point.z
                - parameters.floor_depth_m
                + parameters.robot_ground_clearance_m
                + slope_lift
            )
            for height_fraction in (0.0, 0.5, 1.0):
                z = base_z + height_fraction * parameters.robot_height_m
                offsets = [(0.0, 0.0)] + [
                    (
                        probe_radius * math.cos(2.0 * math.pi * index / 12.0),
                        probe_radius * math.sin(2.0 * math.pi * index / 12.0),
                    )
                    for index in range(12)
                ]
                for offset_x, offset_y in offsets:
                    robot_probe_count += 1
                    probe = Vec3(point.x + offset_x, point.y + offset_y, z)
                    if _point_voxel_index(
                        probe, origin, parameters.voxel_size_m
                    ) not in occupied:
                        robot_probe_failure_count += 1
                        if len(failed_robot_probes) < 20:
                            failed_robot_probes.append(
                                {
                                    "edge_id": path.edge_id,
                                    "path_point_index": point_index,
                                    "probe_m": [probe.x, probe.y, probe.z],
                                }
                            )

    minimum_side_clearance = min(
        path.lateral_half_width - probe_radius for path in paths
    )
    minimum_vertical_margin = min(
        path.vertical_radius
        + parameters.floor_depth_m
        - parameters.robot_ground_clearance_m
        - parameters.robot_height_m
        - parameters.robot_safety_margin_m
        for path in paths
    )
    turn_radius_value: float | None = (
        round(minimum_turn_radius, 6)
        if math.isfinite(minimum_turn_radius)
        else None
    )
    audit = {
        "status": "PASS",
        "centerline_mode": parameters.centerline_mode,
        "floor_profile_mode": parameters.floor_profile_mode,
        "curved_route_length_m": round(route_length, 6),
        "maximum_path_pitch_rad": round(max_pitch, 9),
        "maximum_path_pitch_edge_id": max_pitch_edge_id,
        "maximum_allowed_path_pitch_rad": parameters.maximum_path_pitch_rad,
        "minimum_path_turn_radius_m": turn_radius_value,
        "minimum_path_turn_radius_edge_id": minimum_turn_radius_edge_id,
        "minimum_allowed_path_turn_radius_m": parameters.minimum_path_turn_radius_m,
        "minimum_nonincident_path_distance_m": round(
            minimum_nonincident_path_distance, 6
        ),
        "minimum_nonincident_node_path_distance_m": round(
            minimum_nonincident_node_path_distance, 6
        ),
        "required_centerline_clearance_m": round(required_clearance, 6),
        "minimum_robot_side_clearance_m": round(minimum_side_clearance, 6),
        "minimum_robot_vertical_margin_m": round(minimum_vertical_margin, 6),
        "robot_probe_radius_m": probe_radius,
        "robot_probe_count": robot_probe_count,
        "robot_probe_failure_count": robot_probe_failure_count,
        "failed_robot_probe_examples": failed_robot_probes,
    }
    failures: list[str] = []
    if max_pitch > parameters.maximum_path_pitch_rad + 1e-12:
        failures.append("maximum path pitch exceeded")
    if (
        math.isfinite(minimum_turn_radius)
        and minimum_turn_radius + 1e-9 < parameters.minimum_path_turn_radius_m
    ):
        failures.append("minimum path turn radius violated")
    if minimum_nonincident_path_distance + 1e-9 < required_clearance:
        failures.append("curved nonincident path clearance violated")
    if minimum_nonincident_node_path_distance + 1e-9 < required_clearance:
        failures.append("curved node-path clearance violated")
    if robot_probe_failure_count:
        failures.append("robot footprint/body probes left tunnel free space")
    if minimum_side_clearance <= 0:
        failures.append("robot side clearance is nonpositive")
    if minimum_vertical_margin <= 0:
        failures.append("robot vertical margin is nonpositive")
    if failures:
        audit["status"] = "FAIL"
        audit["failures"] = failures
    return audit


def generate_tunnel_mesh(
    graph: TunnelNetworkGraph,
    seed_bundle: SeedBundle,
    parameters: TunnelMeshParameters,
) -> TriangleMesh:
    """Generate one deterministic watertight voxel-union tunnel boundary."""

    graph.validate()
    parameters.validate()
    max_cross_radius = max(
        parameters.lateral_half_width_m + parameters.lateral_half_width_noise_m,
        parameters.vertical_radius_m + parameters.vertical_radius_noise_m,
        parameters.lateral_half_width_m * parameters.junction_chamber_scale,
        parameters.vertical_radius_m * parameters.junction_chamber_scale,
    )
    required_clearance = 2.0 * max_cross_radius + parameters.geometry_guard_m
    graph_stats = graph.stats()
    for metric in (
        "sampled_min_nonincident_edge_distance",
        "sampled_min_nonincident_node_edge_distance",
    ):
        if float(graph_stats[metric]) < required_clearance:
            raise MeshGenerationError(
                f"{metric}={graph_stats[metric]} m is below geometry requirement "
                f"{required_clearance:.3f} m"
            )

    paths = _navigation_paths(graph, seed_bundle.geometry, parameters)
    segments = _path_segments(paths)
    voxel = parameters.voxel_size_m
    all_positions = [node.position for node in graph.nodes]
    margin = max_cross_radius + voxel
    origin = Vec3(
        math.floor((min(point.x for point in all_positions) - margin) / voxel) * voxel,
        math.floor((min(point.y for point in all_positions) - margin) / voxel) * voxel,
        math.floor((min(point.z for point in all_positions) - margin) / voxel) * voxel,
    )
    occupied: set[tuple[int, int, int]] = set()
    for segment in segments:
        radius = max(segment.lateral_half_width, segment.vertical_radius) + voxel
        min_x = min(segment.start.x, segment.end.x) - radius
        max_x = max(segment.start.x, segment.end.x) + radius
        min_y = min(segment.start.y, segment.end.y) - radius
        max_y = max(segment.start.y, segment.end.y) + radius
        min_z = min(segment.start.z, segment.end.z) - radius
        max_z = max(segment.start.z, segment.end.z) + radius
        index_bounds = (
            math.floor((min_x - origin.x) / voxel),
            math.ceil((max_x - origin.x) / voxel),
            math.floor((min_y - origin.y) / voxel),
            math.ceil((max_y - origin.y) / voxel),
            math.floor((min_z - origin.z) / voxel),
            math.ceil((max_z - origin.z) / voxel),
        )
        for i in range(index_bounds[0], index_bounds[1] + 1):
            x = origin.x + (i + 0.5) * voxel
            for j in range(index_bounds[2], index_bounds[3] + 1):
                y = origin.y + (j + 0.5) * voxel
                for k in range(index_bounds[4], index_bounds[5] + 1):
                    z = origin.z + (k + 0.5) * voxel
                    if _inside_segment(
                        Vec3(x, y, z),
                        segment,
                        parameters.floor_depth_m,
                        parameters.floor_profile_mode,
                    ):
                        occupied.add((i, j, k))

    if parameters.junction_chamber_scale > 1.0:
        adjacency = graph.adjacency()
        chamber_lateral = (
            parameters.lateral_half_width_m * parameters.junction_chamber_scale
        )
        chamber_vertical = (
            parameters.vertical_radius_m * parameters.junction_chamber_scale
        )
        chamber_radius = max(chamber_lateral, chamber_vertical) + voxel
        for node in graph.nodes:
            if len(adjacency[node.id]) < 3:
                continue
            index_bounds = (
                math.floor((node.position.x - chamber_radius - origin.x) / voxel),
                math.ceil((node.position.x + chamber_radius - origin.x) / voxel),
                math.floor((node.position.y - chamber_radius - origin.y) / voxel),
                math.ceil((node.position.y + chamber_radius - origin.y) / voxel),
                math.floor((node.position.z - chamber_radius - origin.z) / voxel),
                math.ceil((node.position.z + chamber_radius - origin.z) / voxel),
            )
            for i in range(index_bounds[0], index_bounds[1] + 1):
                x = origin.x + (i + 0.5) * voxel
                for j in range(index_bounds[2], index_bounds[3] + 1):
                    y = origin.y + (j + 0.5) * voxel
                    for k in range(index_bounds[4], index_bounds[5] + 1):
                        z = origin.z + (k + 0.5) * voxel
                        if _inside_chamber(
                            Vec3(x, y, z),
                            node.position,
                            chamber_lateral,
                            chamber_vertical,
                            parameters.floor_depth_m,
                        ):
                            occupied.add((i, j, k))

    if not occupied:
        raise MeshGenerationError("tunnel voxelization produced no free-space cells")
    component_count = _voxel_components(occupied)
    if component_count != 1:
        raise MeshGenerationError(
            f"tunnel free-space has {component_count} voxel components instead of 1"
        )
    navigation_audit = _navigation_audit(
        graph, paths, occupied, origin, parameters, required_clearance
    )
    if navigation_audit["status"] == "FAIL":
        raise MeshGenerationError(
            "navigation-grade audit failed: "
            + "; ".join(str(item) for item in navigation_audit["failures"])
            + "; metrics="
            + json.dumps(navigation_audit, sort_keys=True)
        )
    vertices, faces = _extract_boundary(occupied, origin, voxel)
    mesh = TriangleMesh(
        vertices=vertices,
        faces=faces,
        voxel_origin=origin,
        voxel_size_m=voxel,
        occupied_voxel_count=len(occupied),
        geometry_variant_id=parameters.geometry_variant_id,
        topology_parent_id=graph.topology_parent_id,
        geometry_seed=seed_bundle.geometry,
        navigation_paths=paths,
        navigation_audit=navigation_audit,
    )
    stats = mesh_stats(mesh)
    if stats["degenerate_face_count"] != 0:
        raise MeshGenerationError("mesh contains degenerate faces")
    if not stats["watertight_edge_incidence"]:
        raise MeshGenerationError("mesh is not a watertight 2-manifold")
    if stats["mesh_component_count"] != 1:
        raise MeshGenerationError("mesh surface is not one connected component")
    if stats["genus"] != graph_stats["cycle_rank"]:
        raise MeshGenerationError(
            f"mesh genus {stats['genus']} does not match graph cycle rank "
            f"{graph_stats['cycle_rank']}"
        )
    return mesh


def mesh_stats(mesh: TriangleMesh) -> dict[str, object]:
    if not mesh.vertices or not mesh.faces:
        raise ValueError("mesh is empty")
    edge_faces: dict[tuple[int, int], list[int]] = defaultdict(list)
    degenerate = 0
    surface_area = 0.0
    signed_volume = 0.0
    for face_index, (a, b, c) in enumerate(mesh.faces):
        if len({a, b, c}) != 3:
            degenerate += 1
            continue
        first, second, third = mesh.vertices[a], mesh.vertices[b], mesh.vertices[c]
        cross = _cross(_sub(second, first), _sub(third, first))
        double_area = _norm(cross)
        if double_area <= 1e-12:
            degenerate += 1
        surface_area += 0.5 * double_area
        signed_volume += _dot(first, _cross(second, third)) / 6.0
        for edge in ((a, b), (b, c), (c, a)):
            edge_faces[tuple(sorted(edge))].append(face_index)

    adjacency: list[set[int]] = [set() for _ in mesh.faces]
    for incident in edge_faces.values():
        for face_index in incident:
            adjacency[face_index].update(other for other in incident if other != face_index)
    remaining = set(range(len(mesh.faces)))
    component_count = 0
    while remaining:
        component_count += 1
        queue = [remaining.pop()]
        while queue:
            face_index = queue.pop()
            linked = adjacency[face_index] & remaining
            remaining.difference_update(linked)
            queue.extend(linked)

    edge_count = len(edge_faces)
    euler_characteristic = len(mesh.vertices) - edge_count + len(mesh.faces)
    genus_value = (2 * component_count - euler_characteristic) / 2
    genus = int(round(genus_value)) if abs(genus_value - round(genus_value)) < 1e-9 else genus_value
    x_values = [vertex.x for vertex in mesh.vertices]
    y_values = [vertex.y for vertex in mesh.vertices]
    z_values = [vertex.z for vertex in mesh.vertices]
    incidence_histogram: dict[str, int] = {}
    for incident in edge_faces.values():
        key = str(len(incident))
        incidence_histogram[key] = incidence_histogram.get(key, 0) + 1
    return {
        "vertex_count": len(mesh.vertices),
        "triangle_count": len(mesh.faces),
        "unique_edge_count": edge_count,
        "occupied_voxel_count": mesh.occupied_voxel_count,
        "degenerate_face_count": degenerate,
        "mesh_component_count": component_count,
        "edge_incidence_histogram": dict(sorted(incidence_histogram.items())),
        "watertight_edge_incidence": all(len(incident) == 2 for incident in edge_faces.values()),
        "euler_characteristic": euler_characteristic,
        "genus": genus,
        "surface_area_m2": round(surface_area, 6),
        "enclosed_volume_m3": round(abs(signed_volume), 6),
        "bounds_m": {
            "min": [min(x_values), min(y_values), min(z_values)],
            "max": [max(x_values), max(y_values), max(z_values)],
        },
        "mesh_hash": mesh.canonical_hash(),
    }


def write_obj(mesh: TriangleMesh, path: Path) -> None:
    with path.open("x", encoding="utf-8") as stream:
        stream.write("# mtare_topo deterministic tunnel mesh v1\n")
        stream.write(f"# topology_parent_id {mesh.topology_parent_id}\n")
        stream.write(f"# geometry_variant_id {mesh.geometry_variant_id}\n")
        stream.write(f"# mesh_hash {mesh.canonical_hash()}\n")
        for vertex in mesh.vertices:
            stream.write(f"v {vertex.x:.9f} {vertex.y:.9f} {vertex.z:.9f}\n")
        for a, b, c in mesh.faces:
            stream.write(f"f {a + 1} {b + 1} {c + 1}\n")


def _format_usd_array(values: Iterable[str], chunk: int = 24) -> str:
    items = list(values)
    return ",\n            ".join(
        ", ".join(items[index : index + chunk])
        for index in range(0, len(items), chunk)
    )


def write_usda(mesh: TriangleMesh, path: Path) -> None:
    stats = mesh_stats(mesh)
    minimum = stats["bounds_m"]["min"]
    maximum = stats["bounds_m"]["max"]
    points = _format_usd_array(
        f"({vertex.x:.9f}, {vertex.y:.9f}, {vertex.z:.9f})"
        for vertex in mesh.vertices
    )
    counts = _format_usd_array(("3" for _ in mesh.faces), chunk=64)
    indices = _format_usd_array(
        (str(index) for face in mesh.faces for index in face), chunk=48
    )
    navigation_prim = ""
    if mesh.navigation_audit.get("status") == "PASS":
        curve_counts = _format_usd_array(
            (str(len(route.points)) for route in mesh.navigation_paths), chunk=24
        )
        curve_points = _format_usd_array(
            (
                f"({point.x:.9f}, {point.y:.9f}, {point.z:.9f})"
                for route in mesh.navigation_paths
                for point in route.points
            )
        )
        navigation_prim = f'''\n    def BasisCurves "NavigationCenterlines"
    {{
        int[] curveVertexCounts = [
            {curve_counts}
        ]
        point3f[] points = [
            {curve_points}
        ]
        color3f[] primvars:displayColor = [(0.05, 0.35, 0.95)]
        uniform token primvars:displayColor:interpolation = "constant"
        uniform token type = "linear"
        float[] widths = [0.10]
        uniform token widths:interpolation = "constant"
        uniform token wrap = "nonperiodic"
    }}
'''
    content = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    custom string mtare:geometryVariantId = "{mesh.geometry_variant_id}"
    custom string mtare:meshHash = "{mesh.canonical_hash()}"
    custom string mtare:topologyParentId = "{mesh.topology_parent_id}"

    def Mesh "TunnelMesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform bool doubleSided = true
        float3[] extent = [({minimum[0]}, {minimum[1]}, {minimum[2]}), ({maximum[0]}, {maximum[1]}, {maximum[2]})]
        int[] faceVertexCounts = [
            {counts}
        ]
        int[] faceVertexIndices = [
            {indices}
        ]
        point3f[] points = [
            {points}
        ]
        uniform token physics:approximation = "none"
        color3f[] primvars:displayColor = [(0.34, 0.24, 0.15)]
        uniform token primvars:displayColor:interpolation = "constant"
        uniform token subdivisionScheme = "none"
    }}
{navigation_prim}
}}
'''
    with path.open("x", encoding="utf-8") as stream:
        stream.write(content)
