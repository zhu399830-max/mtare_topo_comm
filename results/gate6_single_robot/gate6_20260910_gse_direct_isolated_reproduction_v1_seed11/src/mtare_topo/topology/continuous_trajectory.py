"""Deterministic graph-edge to spline trajectories for Gate-4 replay."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class Projection:
    arc_m: float
    error_m: float
    xyz_m: np.ndarray
    segment_index: int


def project_to_polyline(point: Sequence[float], points: np.ndarray) -> Projection:
    """Project one point onto a non-degenerate 3-D polyline."""

    point_array = np.asarray(point, dtype=np.float64)
    polyline = np.asarray(points, dtype=np.float64)
    if polyline.ndim != 2 or polyline.shape[1] != 3 or len(polyline) < 2:
        raise ValueError("polyline must have shape [N>=2,3]")
    segments = np.diff(polyline, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    if np.any(lengths <= 1e-12):
        raise ValueError("polyline contains a degenerate segment")
    relative = point_array - polyline[:-1]
    ratio = np.clip(
        np.sum(relative * segments, axis=1) / np.sum(segments * segments, axis=1),
        0.0,
        1.0,
    )
    candidates = polyline[:-1] + ratio[:, None] * segments
    errors = np.linalg.norm(candidates - point_array, axis=1)
    index = int(np.argmin(errors))
    arc = float(np.sum(lengths[:index]) + ratio[index] * lengths[index])
    return Projection(arc, float(errors[index]), candidates[index], index)


def interpolate_polyline(points: np.ndarray, arc_m: float) -> np.ndarray:
    polyline = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    arc = float(np.clip(arc_m, 0.0, cumulative[-1]))
    index = min(int(np.searchsorted(cumulative, arc, side="right") - 1), len(lengths) - 1)
    ratio = (arc - cumulative[index]) / lengths[index]
    return polyline[index] + ratio * (polyline[index + 1] - polyline[index])


def polyline_between(points: np.ndarray, start_arc_m: float, end_arc_m: float) -> np.ndarray:
    """Return an oriented exact sub-polyline including interpolated endpoints."""

    polyline = np.asarray(points, dtype=np.float64)
    lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    low, high = sorted((float(start_arc_m), float(end_arc_m)))
    interior = polyline[(cumulative > low + 1e-12) & (cumulative < high - 1e-12)]
    result = np.vstack((interpolate_polyline(polyline, low), interior, interpolate_polyline(polyline, high)))
    if end_arc_m < start_arc_m:
        result = result[::-1]
    keep = np.concatenate(([True], np.linalg.norm(np.diff(result, axis=0), axis=1) > 1e-12))
    return result[keep]


def doubled_euler_traversals(graph: Mapping[str, Any]) -> list[dict[str, str]]:
    """Return a deterministic Euler circuit over two copies of every graph edge."""

    adjacency: dict[str, list[tuple[str, str, int]]] = {
        str(node["id"]): [] for node in graph["nodes"]
    }
    copy_id = 0
    for edge in sorted(graph["edges"], key=lambda item: str(item["id"])):
        first, second = map(str, edge["node_ids"])
        for _ in range(2):
            adjacency[first].append((second, str(edge["id"]), copy_id))
            adjacency[second].append((first, str(edge["id"]), copy_id))
            copy_id += 1
    for values in adjacency.values():
        values.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    terminals = sorted(str(node["id"]) for node in graph["nodes"] if int(node["degree"]) == 1)
    start = terminals[0] if terminals else min(adjacency)
    used: set[int] = set()
    node_stack = [start]
    edge_stack: list[tuple[str, int]] = []
    reversed_steps: list[dict[str, str]] = []
    while node_stack:
        current = node_stack[-1]
        while adjacency[current] and adjacency[current][-1][2] in used:
            adjacency[current].pop()
        if adjacency[current]:
            neighbour, edge_id, edge_copy = adjacency[current].pop()
            if edge_copy in used:
                continue
            used.add(edge_copy)
            node_stack.append(neighbour)
            edge_stack.append((edge_id, edge_copy))
            continue
        finished = node_stack.pop()
        if node_stack:
            edge_id, edge_copy = edge_stack.pop()
            reversed_steps.append(
                {"from_node": node_stack[-1], "to_node": finished, "edge_id": edge_id, "copy_id": str(edge_copy)}
            )
    traversals = list(reversed(reversed_steps))
    if len(traversals) != 2 * len(graph["edges"]) or len(used) != 2 * len(graph["edges"]):
        raise RuntimeError("doubled Euler traversal did not consume every edge copy")
    for previous, current in zip(traversals, traversals[1:]):
        if previous["to_node"] != current["from_node"]:
            raise RuntimeError("Euler traversal is not continuous")
    return traversals


def build_spline_route(
    graph: Mapping[str, Any], spline_document: Mapping[str, Any], maximum_connector_m: float
) -> tuple[np.ndarray, list[dict[str, Any]], list[dict[str, Any]]]:
    """Build node->spline->node edge traversals without hiding connector gaps."""

    nodes = {str(item["id"]): np.asarray(item["xyz"], dtype=np.float64) for item in graph["nodes"]}
    edges = {str(item["id"]): item for item in graph["edges"]}
    splines = {str(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64) for item in spline_document["tunnels"]}
    route_parts: list[np.ndarray] = []
    records: list[dict[str, Any]] = []
    connectors: list[dict[str, Any]] = []
    route_cursor = 0.0
    for traversal_index, traversal in enumerate(doubled_euler_traversals(graph)):
        edge = edges[traversal["edge_id"]]
        tunnel_ids = [str(value) for value in edge.get("tunnel_ids", [])]
        if len(tunnel_ids) != 1 or tunnel_ids[0] not in splines:
            raise RuntimeError(f"edge {edge['id']} must reference exactly one existing spline")
        tunnel_id = tunnel_ids[0]
        first = nodes[traversal["from_node"]]
        second = nodes[traversal["to_node"]]
        first_projection = project_to_polyline(first, splines[tunnel_id])
        second_projection = project_to_polyline(second, splines[tunnel_id])
        if max(first_projection.error_m, second_projection.error_m) > maximum_connector_m + 1e-12:
            raise RuntimeError(f"edge {edge['id']} exceeds connector contract")
        spline_part = polyline_between(splines[tunnel_id], first_projection.arc_m, second_projection.arc_m)
        part = np.vstack((first, first_projection.xyz_m, spline_part, second_projection.xyz_m, second))
        keep = np.concatenate(([True], np.linalg.norm(np.diff(part, axis=0), axis=1) > 1e-12))
        part = part[keep]
        length = float(np.sum(np.linalg.norm(np.diff(part, axis=0), axis=1)))
        if route_parts:
            part = part[1:]
        route_parts.append(part)
        records.append(
            {
                **traversal,
                "traversal_index": traversal_index,
                "tunnel_id": tunnel_id,
                "start_arc_m": first_projection.arc_m,
                "end_arc_m": second_projection.arc_m,
                "start_connector_m": first_projection.error_m,
                "end_connector_m": second_projection.error_m,
                "length_m": length,
                "route_start_m": route_cursor,
                "route_end_m": route_cursor + length,
            }
        )
        route_cursor += length
        for endpoint, node_id, projection in (
            ("start", traversal["from_node"], first_projection),
            ("end", traversal["to_node"], second_projection),
        ):
            connectors.append(
                {
                    "traversal_index": traversal_index,
                    "edge_id": traversal["edge_id"],
                    "tunnel_id": tunnel_id,
                    "endpoint": endpoint,
                    "node_id": node_id,
                    "node_xyz_m": nodes[node_id].astype(float).tolist(),
                    "projection_xyz_m": projection.xyz_m.astype(float).tolist(),
                    "length_m": projection.error_m,
                }
            )
    route = np.vstack(route_parts)
    if np.any(np.linalg.norm(np.diff(route, axis=0), axis=1) <= 1e-12):
        raise RuntimeError("assembled route contains zero-length segments")
    return route, records, connectors


def resample_route(points: np.ndarray, spacing_m: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample at 0, spacing, ... strictly before the route endpoint."""

    route = np.asarray(points, dtype=np.float64)
    segments = np.diff(route, axis=0)
    lengths = np.linalg.norm(segments, axis=1)
    cumulative = np.concatenate(([0.0], np.cumsum(lengths)))
    distances = np.arange(0.0, cumulative[-1], float(spacing_m), dtype=np.float64)
    indices = np.searchsorted(cumulative, distances, side="right") - 1
    indices = np.minimum(indices, len(lengths) - 1)
    ratios = (distances - cumulative[indices]) / lengths[indices]
    samples = route[indices] + ratios[:, None] * segments[indices]
    tangents = segments[indices] / lengths[indices, None]
    return samples, tangents, distances
