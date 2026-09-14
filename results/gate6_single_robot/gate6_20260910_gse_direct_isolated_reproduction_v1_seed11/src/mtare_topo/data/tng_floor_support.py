"""Deterministic floor-only geometry derived from Cano TNG and splines.

This module deliberately does not build a collision world.  It creates one
3-D floor ribbon per source tunnel and narrow connector ribbons only between a
graph node and its explicitly incident tunnel splines.  Non-incident tunnels
are never merged, even when their XY projections overlap at different heights.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class FloorSupportGeometry:
    vertices: np.ndarray
    triangles: np.ndarray
    face_groups: tuple[str, ...]
    tunnel_half_width_m: Mapping[int, float]
    connector_count: int

    def obj_bytes(self) -> bytes:
        lines = ["# deterministic TNG-derived floor-support geometry v1\n"]
        lines.extend(
            f"v {x:.9f} {y:.9f} {z:.9f}\n" for x, y, z in self.vertices
        )
        current_group = None
        for triangle, group in zip(self.triangles, self.face_groups, strict=True):
            if group != current_group:
                lines.append(f"g {group}\n")
                current_group = group
            a, b, c = (int(value) + 1 for value in triangle)
            lines.append(f"f {a} {b} {c}\n")
        return "".join(lines).encode("ascii")

    def sha256(self) -> str:
        return hashlib.sha256(self.obj_bytes()).hexdigest()


def _polyline_projection(point: np.ndarray, polyline: np.ndarray) -> tuple[np.ndarray, float]:
    segments = np.diff(polyline, axis=0)
    squared = np.sum(segments * segments, axis=1)
    if len(polyline) < 2 or np.any(squared <= 1e-18):
        raise ValueError("spline must contain at least two non-degenerate segments")
    ratio = np.clip(
        np.sum((point - polyline[:-1]) * segments, axis=1) / squared,
        0.0,
        1.0,
    )
    candidates = polyline[:-1] + ratio[:, None] * segments
    errors = np.linalg.norm(candidates - point, axis=1)
    index = int(np.argmin(errors))
    return candidates[index], float(errors[index])


def _horizontal_normals(points: np.ndarray) -> np.ndarray:
    tangents = np.empty((len(points), 2), dtype=np.float64)
    tangents[0] = points[1, :2] - points[0, :2]
    tangents[-1] = points[-1, :2] - points[-2, :2]
    if len(points) > 2:
        tangents[1:-1] = points[2:, :2] - points[:-2, :2]
    lengths = np.linalg.norm(tangents, axis=1)
    if np.any(lengths <= 1e-9):
        raise ValueError("spline contains a locally vertical/degenerate XY tangent")
    tangents /= lengths[:, None]
    return np.column_stack((-tangents[:, 1], tangents[:, 0]))


def _append_ribbon(
    vertices: list[list[float]],
    triangles: list[list[int]],
    groups: list[str],
    centerline: np.ndarray,
    half_width_m: float,
    group: str,
) -> None:
    normals = _horizontal_normals(centerline)
    first = len(vertices)
    left = centerline.copy()
    right = centerline.copy()
    left[:, :2] += normals * half_width_m
    right[:, :2] -= normals * half_width_m
    for left_point, right_point in zip(left, right, strict=True):
        vertices.extend((left_point.tolist(), right_point.tolist()))
    for index in range(len(centerline) - 1):
        a = first + 2 * index
        b = a + 1
        c = a + 2
        d = a + 3
        triangles.extend(([a, b, c], [b, d, c]))
        groups.extend((group, group))


def build_tng_floor_support_geometry(
    graph: Mapping[str, Any],
    spline_document: Mapping[str, Any],
    geometry_parameters: Mapping[str, Any],
    *,
    safety_inset_m: float = 0.8,
    maximum_connector_m: float = 0.5,
    connector_half_width_m: float = 0.8,
) -> FloorSupportGeometry:
    """Build floor ribbons without reading a trajectory or native mesh."""

    if safety_inset_m <= 0 or maximum_connector_m <= 0 or connector_half_width_m <= 0:
        raise ValueError("floor-support dimensions must be positive")
    fta = float(geometry_parameters["fta_distance_m"])
    radius_by_tunnel = {
        int(item["tunnel_id"]): float(item["radius_m"])
        for item in geometry_parameters["tunnels"]
    }
    spline_by_tunnel = {
        int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in spline_document["tunnels"]
    }
    if set(radius_by_tunnel) != set(spline_by_tunnel):
        raise ValueError("geometry and spline tunnel identities differ")

    half_widths: dict[int, float] = {}
    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    groups: list[str] = []
    for tunnel_id in sorted(spline_by_tunnel):
        radius = radius_by_tunnel[tunnel_id]
        if radius <= abs(fta):
            raise ValueError(f"tunnel {tunnel_id} has no floor cross-section at FTA")
        half_width = float(np.sqrt(radius * radius - fta * fta) - safety_inset_m)
        if half_width < connector_half_width_m:
            raise ValueError(f"tunnel {tunnel_id} floor width is below connector width")
        half_widths[tunnel_id] = half_width
        centerline = spline_by_tunnel[tunnel_id].copy()
        centerline[:, 2] += fta
        _append_ribbon(
            vertices,
            triangles,
            groups,
            centerline,
            half_width,
            f"tunnel_{tunnel_id}",
        )

    connector_count = 0
    for node in sorted(graph["nodes"], key=lambda item: str(item["id"])):
        node_xyz = np.asarray(node["xyz"], dtype=np.float64)
        incident = sorted(int(value) for value in node["incident_tunnel_ids"])
        for tunnel_id in incident:
            if tunnel_id not in spline_by_tunnel:
                raise ValueError(f"node references absent tunnel {tunnel_id}")
            projection, error = _polyline_projection(node_xyz, spline_by_tunnel[tunnel_id])
            if error > maximum_connector_m + 1e-12:
                raise ValueError(
                    f"node {node['id']} to tunnel {tunnel_id} connector is {error:.6f}m"
                )
            if error <= 1e-9:
                continue
            connector = np.vstack((node_xyz, projection))
            connector[:, 2] += fta
            _append_ribbon(
                vertices,
                triangles,
                groups,
                connector,
                connector_half_width_m,
                f"connector_{node['id']}_tunnel_{tunnel_id}",
            )
            connector_count += 1

    vertex_array = np.asarray(vertices, dtype=np.float64)
    triangle_array = np.asarray(triangles, dtype=np.int64)
    if not np.isfinite(vertex_array).all() or len(triangle_array) == 0:
        raise ValueError("floor-support geometry is empty or non-finite")
    doubled_area = np.linalg.norm(
        np.cross(
            vertex_array[triangle_array[:, 1]] - vertex_array[triangle_array[:, 0]],
            vertex_array[triangle_array[:, 2]] - vertex_array[triangle_array[:, 0]],
        ),
        axis=1,
    )
    if np.any(doubled_area <= 1e-12):
        raise ValueError("floor-support geometry contains degenerate triangles")
    return FloorSupportGeometry(
        vertices=vertex_array,
        triangles=triangle_array,
        face_groups=tuple(groups),
        tunnel_half_width_m=half_widths,
        connector_count=connector_count,
    )


def write_floor_support_obj(geometry: FloorSupportGeometry, path: Path) -> None:
    path.write_bytes(geometry.obj_bytes())
