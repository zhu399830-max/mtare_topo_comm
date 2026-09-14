"""Deterministic geometry and labels for the fixed Cano LiDAR smoke.

This module is deliberately independent of Isaac Sim.  It turns one audited
Cano graph/spline bundle into a fixed list of diagnostic poses and computes
the 5 m structural-direction target directly from the exported splines.
Nothing here creates a training dataset or learns a parameter.
"""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


LABEL_RADIUS_M = 5.0
LABEL_SIGMA_DEG = 3.0
AZIMUTH_COLUMNS = 720
ELEVATION_DEG = np.arange(-15.0, 16.0, 2.0, dtype=np.float64)
MAX_RANGE_M = 50.0
NEAR_RANGE_M = 0.3
LIDAR_HEIGHT_ABOVE_FLOOR_M = 1.0
BRANCH_DEDUP_DEG = 8.0


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def _wrap_deg(value: float | np.ndarray) -> float | np.ndarray:
    return np.mod(value, 360.0)


def circular_distance_deg(first: float | np.ndarray, second: float) -> np.ndarray:
    first_array = np.asarray(first, dtype=np.float64)
    return np.abs((first_array - second + 180.0) % 360.0 - 180.0)


def _polyline_distances(points: np.ndarray) -> np.ndarray:
    return np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)))
    )


def interpolate_polyline(
    points: np.ndarray, distance_m: float
) -> tuple[np.ndarray, np.ndarray]:
    """Return a point and unit tangent at arc distance along a polyline."""

    distances = _polyline_distances(points)
    distance_m = float(np.clip(distance_m, 0.0, distances[-1]))
    index = int(np.searchsorted(distances, distance_m, side="right") - 1)
    index = min(max(index, 0), len(points) - 2)
    segment = points[index + 1] - points[index]
    length = float(np.linalg.norm(segment))
    if length <= 1e-12:
        raise ValueError("polyline contains a zero-length selected segment")
    ratio = (distance_m - distances[index]) / length
    return points[index] + ratio * segment, segment / length


def nearest_polyline_distance(points: np.ndarray, query: np.ndarray) -> float:
    segments = points[1:] - points[:-1]
    lengths_sq = np.sum(segments * segments, axis=1)
    safe = np.where(lengths_sq > 1e-15, lengths_sq, 1.0)
    ratios = np.sum((query - points[:-1]) * segments, axis=1) / safe
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = points[:-1] + ratios[:, None] * segments
    return float(np.min(np.linalg.norm(projections - query, axis=1)))


def sphere_polyline_intersections(
    center_xyz: Sequence[float], radius_m: float, points: np.ndarray
) -> list[np.ndarray]:
    """Intersect a sphere with every segment of a polyline."""

    center = np.asarray(center_xyz, dtype=np.float64)
    result: list[np.ndarray] = []
    for start, end in zip(points[:-1], points[1:]):
        delta = end - start
        offset = start - center
        a = float(np.dot(delta, delta))
        if a <= 1e-15:
            continue
        b = 2.0 * float(np.dot(offset, delta))
        c = float(np.dot(offset, offset) - radius_m * radius_m)
        discriminant = b * b - 4.0 * a * c
        if discriminant < -1e-10:
            continue
        root = math.sqrt(max(0.0, discriminant))
        for ratio in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if -1e-9 <= ratio <= 1.0 + 1e-9:
                point = start + np.clip(ratio, 0.0, 1.0) * delta
                if not any(np.linalg.norm(point - old) < 1e-5 for old in result):
                    result.append(point)
    return result


def deduplicate_headings(
    headings_deg: Iterable[float], tolerance_deg: float = BRANCH_DEDUP_DEG
) -> list[float]:
    """Deterministically merge headings that describe the same local branch."""

    clusters: list[list[float]] = []
    for heading in sorted(float(_wrap_deg(value)) for value in headings_deg):
        matching = None
        for cluster in clusters:
            mean = float(
                _wrap_deg(
                    math.degrees(
                        math.atan2(
                            sum(math.sin(math.radians(x)) for x in cluster),
                            sum(math.cos(math.radians(x)) for x in cluster),
                        )
                    )
                )
            )
            if float(circular_distance_deg(heading, mean)) <= tolerance_deg:
                matching = cluster
                break
        if matching is None:
            clusters.append([heading])
        else:
            matching.append(heading)
    # The 0/360 cluster can be split by sorted insertion; merge it once.
    if len(clusters) > 1:
        first_mean = float(np.mean(clusters[0]))
        last_mean = float(np.mean(clusters[-1]))
        if float(circular_distance_deg(first_mean, last_mean)) <= tolerance_deg:
            clusters[0] = clusters[-1] + clusters[0]
            clusters.pop()
    means = []
    for cluster in clusters:
        sine = sum(math.sin(math.radians(value)) for value in cluster)
        cosine = sum(math.cos(math.radians(value)) for value in cluster)
        means.append(float(_wrap_deg(math.degrees(math.atan2(sine, cosine)))))
    return sorted(means)


def circular_gaussian_label(
    headings_robot_deg: Sequence[float],
    columns: int = AZIMUTH_COLUMNS,
    sigma_deg: float = LABEL_SIGMA_DEG,
) -> np.ndarray:
    azimuth = np.arange(columns, dtype=np.float64) * 360.0 / columns
    label = np.zeros(columns, dtype=np.float64)
    for heading in headings_robot_deg:
        distance = circular_distance_deg(azimuth, float(heading))
        label = np.maximum(label, np.exp(-0.5 * (distance / sigma_deg) ** 2))
    return label.astype(np.float32)


def structural_label(
    axis_xyz: Sequence[float],
    yaw_deg: float,
    splines: Mapping[int, np.ndarray],
    radius_m: float = LABEL_RADIUS_M,
) -> dict[str, Any]:
    center = np.asarray(axis_xyz, dtype=np.float64)
    intersections: list[dict[str, Any]] = []
    headings_world: list[float] = []
    for tunnel_id in sorted(splines):
        for point in sphere_polyline_intersections(center, radius_m, splines[tunnel_id]):
            delta = point - center
            if np.linalg.norm(delta[:2]) <= 1e-8:
                continue
            heading = float(_wrap_deg(math.degrees(math.atan2(delta[1], delta[0]))))
            headings_world.append(heading)
            intersections.append(
                {
                    "tunnel_id": tunnel_id,
                    "xyz": point.tolist(),
                    "heading_world_deg": heading,
                }
            )
    unique_world = deduplicate_headings(headings_world)
    relative = sorted(float(_wrap_deg(value - yaw_deg)) for value in unique_world)
    return {
        "radius_m": radius_m,
        "headings_world_deg": unique_world,
        "headings_robot_deg": relative,
        "branch_count": len(unique_world),
        "raw_intersections": intersections,
        "label_720": circular_gaussian_label(relative).tolist(),
    }


def _node_maps(graph: Mapping[str, Any]) -> tuple[dict[str, dict[str, Any]], dict[str, list[str]]]:
    nodes = {node["id"]: node for node in graph["nodes"]}
    adjacency = {node_id: [] for node_id in nodes}
    for edge in graph["edges"]:
        first, second = edge["node_ids"]
        adjacency[first].append(second)
        adjacency[second].append(first)
    for neighbours in adjacency.values():
        neighbours.sort()
    return nodes, adjacency


def _nearest_spline_location(
    point: np.ndarray, splines: Mapping[int, np.ndarray], allowed_ids: Iterable[int]
) -> tuple[int, float]:
    best: tuple[float, int, float] | None = None
    for tunnel_id in sorted(set(int(value) for value in allowed_ids)):
        spline = splines[tunnel_id]
        distances = _polyline_distances(spline)
        segments = spline[1:] - spline[:-1]
        lengths_sq = np.sum(segments * segments, axis=1)
        ratios = np.sum((point - spline[:-1]) * segments, axis=1) / np.where(
            lengths_sq > 1e-15, lengths_sq, 1.0
        )
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = spline[:-1] + ratios[:, None] * segments
        errors = np.linalg.norm(projections - point, axis=1)
        index = int(np.argmin(errors))
        distance = float(distances[index] + ratios[index] * np.sqrt(lengths_sq[index]))
        candidate = (float(errors[index]), tunnel_id, distance)
        if best is None or candidate < best:
            best = candidate
    if best is None:
        raise ValueError("no permitted spline for graph point")
    return best[1], best[2]


def _pose_record(
    *,
    sample_id: str,
    role: str,
    axis_xyz: np.ndarray,
    tangent: np.ndarray,
    source_tunnel_id: int,
    source_detail: str,
    fta_distance_m: float,
    splines: Mapping[int, np.ndarray],
    expected_branch_count: int,
) -> dict[str, Any]:
    yaw_deg = float(_wrap_deg(math.degrees(math.atan2(tangent[1], tangent[0]))))
    label = structural_label(axis_xyz, yaw_deg, splines)
    if label["branch_count"] != expected_branch_count:
        raise ValueError(
            f"{sample_id}: expected {expected_branch_count} structural branches, "
            f"got {label['branch_count']} at {axis_xyz.tolist()}"
        )
    sensor_xyz = axis_xyz.copy()
    sensor_xyz[2] = axis_xyz[2] + fta_distance_m + LIDAR_HEIGHT_ABOVE_FLOOR_M
    return {
        "sample_id": sample_id,
        "role": role,
        "axis_xyz_m": axis_xyz.tolist(),
        "sensor_xyz_m": sensor_xyz.tolist(),
        "yaw_deg": yaw_deg,
        "source_tunnel_id": source_tunnel_id,
        "source_detail": source_detail,
        "expected_branch_count": expected_branch_count,
        "label": label,
    }


def select_role_stratified_poses(
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    fta_distance_m: float,
) -> list[dict[str, Any]]:
    """Select exactly 8 interior, 8 junction, and 8 terminal poses."""

    splines = {
        int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    nodes, adjacency = _node_maps(graph)
    poses: list[dict[str, Any]] = []

    # Two positions on every tunnel, far from all degree != 2 graph events.
    events = np.asarray(
        [node["xyz"] for node in graph["nodes"] if int(node["degree"]) != 2],
        dtype=np.float64,
    )
    for tunnel_id in sorted(splines):
        points = splines[tunnel_id]
        total = _polyline_distances(points)[-1]
        candidates = np.linspace(0.18, 0.82, 65) * total
        selected: list[tuple[np.ndarray, np.ndarray, float]] = []
        for distance in candidates:
            point, tangent = interpolate_polyline(points, float(distance))
            event_clearance = float(np.min(np.linalg.norm(events - point, axis=1)))
            if event_clearance < 12.0:
                continue
            if any(np.linalg.norm(point - prior[0]) < 20.0 for prior in selected):
                continue
            selected.append((point, tangent, float(distance)))
            if len(selected) == 2:
                break
        if len(selected) != 2:
            raise ValueError(f"tunnel {tunnel_id}: cannot select two interior poses")
        for local_index, (point, tangent, distance) in enumerate(selected):
            poses.append(
                _pose_record(
                    sample_id=f"interior_t{tunnel_id:02d}_{local_index:02d}",
                    role="tunnel_interior",
                    axis_xyz=point,
                    tangent=tangent,
                    source_tunnel_id=tunnel_id,
                    source_detail=f"arc_distance_m={distance:.6f}; event_clearance_m>=12",
                    fta_distance_m=fta_distance_m,
                    splines=splines,
                    expected_branch_count=2,
                )
            )

    # Every degree-3 structural intersection contributes center and 1 m along
    # the lexicographically first graph branch.  This is deterministic and is
    # not chosen based on LiDAR quality.
    junction_nodes = sorted(
        (node for node in graph["nodes"] if int(node["degree"]) == 3),
        key=lambda value: value["id"],
    )
    if len(junction_nodes) != 4:
        raise ValueError(f"expected four degree-3 junctions, got {len(junction_nodes)}")
    for junction in junction_nodes:
        node_id = junction["id"]
        center = np.asarray(junction["xyz"], dtype=np.float64)
        neighbour_id = adjacency[node_id][0]
        neighbour = np.asarray(nodes[neighbour_id]["xyz"], dtype=np.float64)
        incident = junction["incident_tunnel_ids"]
        tunnel_id, center_distance = _nearest_spline_location(center, splines, incident)
        _, neighbour_distance = _nearest_spline_location(neighbour, splines, incident)
        direction_sign = 1.0 if neighbour_distance >= center_distance else -1.0
        center_point, center_tangent = interpolate_polyline(splines[tunnel_id], center_distance)
        center_tangent = center_tangent * direction_sign
        offset_distance = center_distance + direction_sign * 1.0
        offset_point, offset_tangent = interpolate_polyline(splines[tunnel_id], offset_distance)
        offset_tangent = offset_tangent * direction_sign
        for local_index, (point, tangent, detail) in enumerate(
            (
                (center_point, center_tangent, f"junction={node_id}; offset_m=0"),
                (offset_point, offset_tangent, f"junction={node_id}; offset_m=1"),
            )
        ):
            poses.append(
                _pose_record(
                    sample_id=f"junction_{node_id}_{local_index:02d}",
                    role="junction_transition",
                    axis_xyz=point,
                    tangent=tangent,
                    source_tunnel_id=tunnel_id,
                    source_detail=detail,
                    fta_distance_m=fta_distance_m,
                    splines=splines,
                    expected_branch_count=3,
                )
            )

    # Every degree-1 endpoint contributes 1.5 m and 3.0 m inward.
    terminal_nodes = sorted(
        (node for node in graph["nodes"] if int(node["degree"]) == 1),
        key=lambda value: value["id"],
    )
    if len(terminal_nodes) != 4:
        raise ValueError(f"expected four terminals, got {len(terminal_nodes)}")
    for terminal in terminal_nodes:
        endpoint = np.asarray(terminal["xyz"], dtype=np.float64)
        tunnel_id, endpoint_distance = _nearest_spline_location(
            endpoint, splines, terminal["incident_tunnel_ids"]
        )
        total = _polyline_distances(splines[tunnel_id])[-1]
        direction_sign = 1.0 if endpoint_distance < total / 2.0 else -1.0
        for local_index, inward_m in enumerate((1.5, 3.0)):
            distance = endpoint_distance + direction_sign * inward_m
            point, tangent = interpolate_polyline(splines[tunnel_id], distance)
            tangent = tangent * direction_sign
            poses.append(
                _pose_record(
                    sample_id=f"terminal_{terminal['id']}_{local_index:02d}",
                    role="terminal_approach",
                    axis_xyz=point,
                    tangent=tangent,
                    source_tunnel_id=tunnel_id,
                    source_detail=f"terminal={terminal['id']}; inward_m={inward_m}",
                    fta_distance_m=fta_distance_m,
                    splines=splines,
                    expected_branch_count=1,
                )
            )

    role_order = {"tunnel_interior": 0, "junction_transition": 1, "terminal_approach": 2}
    poses.sort(key=lambda value: (role_order[value["role"]], value["sample_id"]))
    role_counts = {
        role: sum(pose["role"] == role for pose in poses) for role in role_order
    }
    if len(poses) != 24 or role_counts != {
        "tunnel_interior": 8,
        "junction_transition": 8,
        "terminal_approach": 8,
    }:
        raise ValueError(f"unexpected pose contract: total={len(poses)}, roles={role_counts}")
    return poses


def lidar_local_directions() -> np.ndarray:
    azimuth_deg = np.arange(AZIMUTH_COLUMNS, dtype=np.float64) * 0.5
    azimuth, elevation = np.meshgrid(
        np.radians(azimuth_deg), np.radians(ELEVATION_DEG)
    )
    cosine = np.cos(elevation)
    return np.stack(
        (cosine * np.cos(azimuth), cosine * np.sin(azimuth), np.sin(elevation)),
        axis=-1,
    ).astype(np.float32)


def world_directions(local_directions: np.ndarray, yaw_deg: float) -> np.ndarray:
    yaw = math.radians(yaw_deg)
    rotation = np.asarray(
        [[math.cos(yaw), -math.sin(yaw), 0.0],
         [math.sin(yaw), math.cos(yaw), 0.0],
         [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    return local_directions @ rotation.T


def rasterize_generic_model_output(
    azimuth_deg: Sequence[float],
    elevation_deg: Sequence[float],
    range_m: Sequence[float],
    *,
    rows_elevation_deg: np.ndarray = ELEVATION_DEG,
    columns: int = AZIMUTH_COLUMNS,
    max_range_m: float = MAX_RANGE_M,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Rasterize unordered Isaac GMO returns into the frozen 16x720 grid."""

    result = np.full((len(rows_elevation_deg), columns), max_range_m, dtype=np.float32)
    valid = np.zeros_like(result, dtype=np.uint8)
    counts = {"input": 0, "accepted": 0, "out_of_grid": 0, "duplicate_cells": 0}
    for azimuth, elevation, distance in zip(azimuth_deg, elevation_deg, range_m):
        counts["input"] += 1
        if not np.isfinite(azimuth + elevation + distance):
            counts["out_of_grid"] += 1
            continue
        row = int(np.argmin(np.abs(rows_elevation_deg - elevation)))
        if abs(float(rows_elevation_deg[row] - elevation)) > 0.25:
            counts["out_of_grid"] += 1
            continue
        column = int(round(float(_wrap_deg(azimuth)) / 360.0 * columns)) % columns
        if not (NEAR_RANGE_M <= distance <= max_range_m):
            counts["out_of_grid"] += 1
            continue
        if valid[row, column]:
            counts["duplicate_cells"] += 1
            result[row, column] = min(result[row, column], float(distance))
        else:
            result[row, column] = float(distance)
            valid[row, column] = 1
            counts["accepted"] += 1
    return result, valid, counts


def write_obj_usda(obj_path: Path, usda_path: Path) -> dict[str, int]:
    """Convert the original OBJ triangles to a collision-enabled ASCII USD stage."""

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    with obj_path.open("r", encoding="utf-8", errors="strict") as stream:
        for line in stream:
            if line.startswith("v "):
                fields = line.split()
                vertices.append(tuple(float(value) for value in fields[1:4]))
            elif line.startswith("f "):
                indices = [int(field.split("/")[0]) - 1 for field in line.split()[1:]]
                if len(indices) != 3:
                    raise ValueError("only triangular OBJ faces are accepted")
                faces.append(tuple(indices))
    if not vertices or not faces:
        raise ValueError("OBJ has no vertices or triangle faces")
    points_text = ",\n            ".join(
        ", ".join(
            f"({x:.9f}, {y:.9f}, {z:.9f})" for x, y, z in vertices[index:index + 24]
        )
        for index in range(0, len(vertices), 24)
    )
    counts_text = ",\n            ".join(
        ", ".join("3" for _ in faces[index:index + 64])
        for index in range(0, len(faces), 64)
    )
    flat = [value for face in faces for value in face]
    indices_text = ",\n            ".join(
        ", ".join(str(value) for value in flat[index:index + 48])
        for index in range(0, len(flat), 48)
    )
    minimum = np.min(np.asarray(vertices), axis=0)
    maximum = np.max(np.asarray(vertices), axis=0)
    content = f'''#usda 1.0
(
    defaultPrim = "World"
    metersPerUnit = 1
    upAxis = "Z"
)

def Xform "World"
{{
    def Mesh "TunnelMesh" (
        prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]
    )
    {{
        uniform bool doubleSided = true
        float3[] extent = [({minimum[0]}, {minimum[1]}, {minimum[2]}), ({maximum[0]}, {maximum[1]}, {maximum[2]})]
        int[] faceVertexCounts = [
            {counts_text}
        ]
        int[] faceVertexIndices = [
            {indices_text}
        ]
        point3f[] points = [
            {points_text}
        ]
        uniform token physics:approximation = "none"
        color3f[] primvars:displayColor = [(0.34, 0.24, 0.15)]
        uniform token primvars:displayColor:interpolation = "constant"
        uniform token subdivisionScheme = "none"
    }}
}}
'''
    usda_path.write_text(content, encoding="utf-8")
    return {"vertices": len(vertices), "triangles": len(faces)}
