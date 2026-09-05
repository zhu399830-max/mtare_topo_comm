#!/usr/bin/env python3
"""Build a working LiDAR-to-exit-to-online-topometric-graph prototype."""

from __future__ import annotations

import argparse
import copy
import hashlib
import itertools
import json
import math
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import (
    ELEVATION_DEG,
    LIDAR_HEIGHT_ABOVE_FLOOR_M,
    MAX_RANGE_M,
    NEAR_RANGE_M,
    lidar_local_directions,
    load_json,
    structural_label,
    world_directions,
)
from mtare_topo.governance import write_json
from mtare_topo.semantics.range_exit_baseline import (
    RangeExitBaseline,
    circular_distance_deg,
)
from mtare_topo.topology.online_topometric import OnlineTopometricGraph


RUN_ID = "cano_seed0_lidar_to_topometric_vertical_slice_v2_graph_refined"
PREVIOUS_RUN = PROJECT_ROOT / "results/prototypes/cano_seed0_lidar_to_topometric_vertical_slice_v1"
WORLD_DIR = PROJECT_ROOT / "results/gate0_baseline/gate0_20260810_cano_readonly_audited_adapter_smoke_v1_seed0/artifacts/world_000"
FRAME_COUNT = 240
CONTACT_FRAME_COUNT = 24
GIF_FRAME_COUNT = 60


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    result = o3d.t.geometry.RaycastingScene()
    result.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return result


def _raycast(
    scene: o3d.t.geometry.RaycastingScene,
    origin: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float32), directions.shape)
    rays = np.concatenate((origins, directions.astype(np.float32)), axis=-1)
    raw = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"].numpy()
    raw = raw.reshape(directions.shape[:-1])
    valid = np.isfinite(raw) & (raw >= NEAR_RANGE_M) & (raw <= MAX_RANGE_M)
    return (
        np.where(valid, raw, MAX_RANGE_M).astype(np.float32),
        valid.astype(np.uint8),
    )


def _doubled_euler_route(graph: dict[str, Any]) -> list[str]:
    adjacency: dict[str, list[tuple[str, int]]] = {
        node["id"]: [] for node in graph["nodes"]
    }
    copy_id = 0
    for edge in sorted(graph["edges"], key=lambda item: item["id"]):
        first, second = edge["node_ids"]
        for _ in range(2):
            adjacency[first].append((second, copy_id))
            adjacency[second].append((first, copy_id))
            copy_id += 1
    for values in adjacency.values():
        values.sort(key=lambda item: (item[0], item[1]), reverse=True)
    terminals = sorted(
        node["id"] for node in graph["nodes"] if int(node["degree"]) == 1
    )
    start = terminals[0] if terminals else min(adjacency)
    used: set[int] = set()
    stack = [start]
    circuit: list[str] = []
    while stack:
        current = stack[-1]
        while adjacency[current] and adjacency[current][-1][1] in used:
            adjacency[current].pop()
        if not adjacency[current]:
            circuit.append(stack.pop())
            continue
        neighbour, edge_copy = adjacency[current].pop()
        if edge_copy in used:
            continue
        used.add(edge_copy)
        stack.append(neighbour)
    route = list(reversed(circuit))
    expected = 2 * len(graph["edges"]) + 1
    if len(route) != expected or len(used) != 2 * len(graph["edges"]):
        raise RuntimeError(
            f"doubled Euler route contract failed: nodes={len(route)}/{expected}, copies={len(used)}"
        )
    return route


def _resample_polyline(points: np.ndarray, count: int) -> tuple[np.ndarray, np.ndarray]:
    segment = np.diff(points, axis=0)
    length = np.linalg.norm(segment, axis=1)
    if np.any(length <= 1e-9):
        raise RuntimeError("route contains a zero-length graph edge")
    cumulative = np.concatenate(([0.0], np.cumsum(length)))
    distances = np.linspace(0.0, cumulative[-1], count, endpoint=False)
    sampled = np.zeros((count, 3), dtype=np.float64)
    directions = np.zeros((count, 3), dtype=np.float64)
    for index, distance in enumerate(distances):
        segment_index = min(
            int(np.searchsorted(cumulative, distance, side="right") - 1),
            len(segment) - 1,
        )
        ratio = (distance - cumulative[segment_index]) / length[segment_index]
        sampled[index] = points[segment_index] + ratio * segment[segment_index]
        directions[index] = segment[segment_index] / length[segment_index]
    return sampled, directions


def _spline_arrays(document: dict[str, Any]) -> dict[int, np.ndarray]:
    return {
        int(item["tunnel_id"]): np.asarray(item["points"], dtype=np.float64)
        for item in document["tunnels"]
    }


def _project_to_splines(
    query: np.ndarray,
    route_direction: np.ndarray,
    splines: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, int, float]:
    best: tuple[float, int, int, float, np.ndarray] | None = None
    for tunnel_id in sorted(splines):
        points = splines[tunnel_id]
        segments = points[1:] - points[:-1]
        lengths_sq = np.sum(segments * segments, axis=1)
        ratios = np.sum((query - points[:-1]) * segments, axis=1) / np.where(
            lengths_sq > 1e-12, lengths_sq, 1.0
        )
        ratios = np.clip(ratios, 0.0, 1.0)
        projections = points[:-1] + ratios[:, None] * segments
        errors = np.linalg.norm(projections - query, axis=1)
        index = int(np.argmin(errors))
        candidate = (
            float(errors[index]),
            tunnel_id,
            index,
            float(ratios[index]),
            projections[index],
        )
        if best is None or candidate[:4] < best[:4]:
            best = candidate
    if best is None:
        raise RuntimeError("no spline projection available")
    error, tunnel_id, segment_index, _, projection = best
    tangent = splines[tunnel_id][segment_index + 1] - splines[tunnel_id][segment_index]
    tangent /= np.linalg.norm(tangent)
    if np.dot(tangent[:2], route_direction[:2]) < 0:
        tangent = -tangent
    return projection, tangent, tunnel_id, error


def _match_headings(
    predicted: Sequence[float], truth: Sequence[float], tolerance_deg: float = 20.0
) -> tuple[int, list[float]]:
    candidates = sorted(
        (circular_distance_deg(first, second), i, j)
        for i, first in enumerate(predicted)
        for j, second in enumerate(truth)
    )
    used_predicted: set[int] = set()
    used_truth: set[int] = set()
    errors = []
    for error, first, second in candidates:
        if error > tolerance_deg or first in used_predicted or second in used_truth:
            continue
        used_predicted.add(first)
        used_truth.add(second)
        errors.append(float(error))
    return len(errors), errors


def _graph_comparison(
    predicted: dict[str, Any], oracle: dict[str, Any], radius_m: float = 6.0
) -> dict[str, Any]:
    candidates = []
    for first in predicted["nodes"]:
        for second in oracle["nodes"]:
            if first["role"] != second["role"]:
                continue
            distance = float(
                np.linalg.norm(
                    np.asarray(first["xyz_m"][:2]) - np.asarray(second["xyz_m"][:2])
                )
            )
            if distance <= radius_m:
                candidates.append((distance, int(first["id"]), int(second["id"])))
    candidates.sort()
    used_first: set[int] = set()
    used_second: set[int] = set()
    mapping: dict[int, int] = {}
    distances = []
    for distance, first, second in candidates:
        if first in used_first or second in used_second:
            continue
        used_first.add(first)
        used_second.add(second)
        mapping[first] = second
        distances.append(distance)
    oracle_edges = {
        frozenset((int(edge["from"]), int(edge["to"]))) for edge in oracle["edges"]
    }
    matched_edges = 0
    for edge in predicted["edges"]:
        first = mapping.get(int(edge["from"]))
        second = mapping.get(int(edge["to"]))
        if first is not None and second is not None and frozenset((first, second)) in oracle_edges:
            matched_edges += 1
    node_matches = len(mapping)
    predicted_nodes = {int(node["id"]): np.asarray(node["xyz_m"][:2]) for node in predicted["nodes"]}
    oracle_nodes = {int(node["id"]): np.asarray(node["xyz_m"][:2]) for node in oracle["nodes"]}
    geometric_candidates = []
    for first_index, first_edge in enumerate(predicted["edges"]):
        first_a = predicted_nodes[int(first_edge["from"])]
        first_b = predicted_nodes[int(first_edge["to"])]
        for second_index, second_edge in enumerate(oracle["edges"]):
            second_a = oracle_nodes[int(second_edge["from"])]
            second_b = oracle_nodes[int(second_edge["to"])]
            endpoint_error = min(
                max(float(np.linalg.norm(first_a - second_a)), float(np.linalg.norm(first_b - second_b))),
                max(float(np.linalg.norm(first_a - second_b)), float(np.linalg.norm(first_b - second_a))),
            )
            if endpoint_error <= 8.0:
                geometric_candidates.append((endpoint_error, first_index, second_index))
    used_predicted_edges: set[int] = set()
    used_oracle_edges: set[int] = set()
    geometric_errors = []
    for error, first_index, second_index in sorted(geometric_candidates):
        if first_index in used_predicted_edges or second_index in used_oracle_edges:
            continue
        used_predicted_edges.add(first_index)
        used_oracle_edges.add(second_index)
        geometric_errors.append(error)
    geometric_matched_edges = len(geometric_errors)
    geometric_edge_precision = geometric_matched_edges / max(1, predicted["edge_count"])
    geometric_edge_recall = geometric_matched_edges / max(1, oracle["edge_count"])
    geometric_edge_f1 = (
        2 * geometric_edge_precision * geometric_edge_recall
        / max(1e-12, geometric_edge_precision + geometric_edge_recall)
    )
    node_precision = node_matches / max(1, predicted["node_count"])
    node_recall = node_matches / max(1, oracle["node_count"])
    node_f1 = 2 * node_precision * node_recall / max(1e-12, node_precision + node_recall)
    return {
        "spatial_role_match_radius_m": radius_m,
        "matched_nodes": node_matches,
        "node_precision": node_precision,
        "node_recall": node_recall,
        "node_f1": node_f1,
        "mean_matched_node_distance_m": float(np.mean(distances)) if distances else None,
        "matched_edges": matched_edges,
        "edge_precision": matched_edges / max(1, predicted["edge_count"]),
        "edge_recall": matched_edges / max(1, oracle["edge_count"]),
        "predicted_to_oracle_node_mapping": {str(key): value for key, value in mapping.items()},
        "geometric_edge_match": {
            "maximum_unordered_endpoint_error_m": 8.0,
            "matched_edges": geometric_matched_edges,
            "precision": geometric_edge_precision,
            "recall": geometric_edge_recall,
            "f1": geometric_edge_f1,
            "mean_maximum_endpoint_error_m": float(np.mean(geometric_errors)) if geometric_errors else None,
            "reason": "This metric compares edge geometry directly and does not force all edges through one globally chosen node-ID mapping.",
        },
    }


def _replay_graph(
    frame_records: list[dict[str, Any]],
    config: Any,
    heading_key: str,
) -> dict[str, Any]:
    graph = OnlineTopometricGraph(config)
    for item in frame_records:
        graph.update(
            item["sensor_xyz_m"],
            item["yaw_deg"],
            item[heading_key],
            item["frame_index"],
        )
    return graph.snapshot()


def _association_sweep(frame_records: list[dict[str, Any]]) -> dict[str, Any]:
    from mtare_topo.topology.online_topometric import OnlineTopometricConfig

    rows = []
    for stable, travel, loop, merge in itertools.product(
        (1, 2, 3), (4.0, 6.0, 8.0), (3.0, 4.0, 6.0), (20.0, 25.0, 35.0)
    ):
        config = OnlineTopometricConfig(
            minimum_event_travel_m=travel,
            turn_event_deg=35.0,
            loop_merge_radius_m=loop,
            branch_heading_merge_deg=merge,
            stable_frames=stable,
        )
        predicted = _replay_graph(frame_records, config, "predicted_headings_robot_deg")
        oracle = _replay_graph(frame_records, config, "truth_headings_robot_deg")
        comparison = _graph_comparison(predicted, oracle)
        node_f1 = comparison["node_f1"]
        edge_f1 = comparison["geometric_edge_match"]["f1"]
        rows.append(
            {
                "config": config.to_dict(),
                "objective_node_f1_plus_geometric_edge_f1": node_f1 + edge_f1,
                "node_f1": node_f1,
                "geometric_edge_f1": edge_f1,
                "predicted_nodes": predicted["node_count"],
                "predicted_edges": predicted["edge_count"],
                "oracle_nodes": oracle["node_count"],
                "oracle_edges": oracle["edge_count"],
            }
        )
    rows.sort(
        key=lambda item: (
            -item["objective_node_f1_plus_geometric_edge_f1"],
            item["config"]["stable_frames"],
            abs(item["config"]["minimum_event_travel_m"] - 6.0),
            abs(item["config"]["loop_merge_radius_m"] - 4.0),
            item["config"]["branch_heading_merge_deg"],
        )
    )
    return {
        "schema_version": "vertical_slice_graph_association_sweep_v1",
        "role": "SAME_WORLD_DEVELOPMENT_TUNING_NOT_UNSEEN_EVALUATION",
        "candidate_count": len(rows),
        "selection_objective": "maximize node_f1 + geometric_edge_f1; deterministic tie preference stable_frames, |travel-6|, |loop-4|, heading tolerance",
        "selected": rows[0],
        "candidates": rows,
    }


ROLE_COLORS = {"terminal": "#009E73", "corridor": "#0072B2", "junction": "#D55E00"}


def _draw_splines(axis: plt.Axes, splines: dict[int, np.ndarray]) -> None:
    for points in splines.values():
        axis.plot(points[:, 0], points[:, 1], color="#999999", linewidth=0.8, alpha=0.65)


def _draw_graph(axis: plt.Axes, graph: dict[str, Any], title: str) -> None:
    nodes = {int(node["id"]): node for node in graph["nodes"]}
    for edge in graph["edges"]:
        first = nodes[int(edge["from"])]["xyz_m"]
        second = nodes[int(edge["to"])]["xyz_m"]
        axis.plot([first[0], second[0]], [first[1], second[1]], color="#202020", linewidth=1.4)
    for role, color in ROLE_COLORS.items():
        selected = [node for node in graph["nodes"] if node["role"] == role]
        if selected:
            xy = np.asarray([node["xyz_m"][:2] for node in selected])
            axis.scatter(xy[:, 0], xy[:, 1], s=32, color=color, label=f"{role} n={len(selected)}", zorder=4)
    for node in graph["nodes"]:
        x, y = node["xyz_m"][:2]
        for heading, state in zip(node["exit_headings_world_deg"], node["exit_stub_state"]):
            angle = math.radians(heading)
            color = "#CC79A7" if state == "unexplored" else "#555555"
            axis.arrow(x, y, 3 * math.cos(angle), 3 * math.sin(angle), color=color, width=0.08, alpha=0.8, length_includes_head=True)
    axis.set_title(title)
    axis.set_aspect("equal", adjustable="box")
    axis.grid(True, linewidth=0.25, alpha=0.35)
    axis.legend(fontsize=7, loc="best")


def _save_summary_figure(
    destination: Path,
    splines: dict[int, np.ndarray],
    trajectory: np.ndarray,
    predicted_graph: dict[str, Any],
    oracle_graph: dict[str, Any],
    frame_records: list[dict[str, Any]],
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(18, 14), constrained_layout=True)
    _draw_splines(axes[0, 0], splines)
    axes[0, 0].plot(trajectory[:, 0], trajectory[:, 1], color="#0072B2", linewidth=1.0)
    axes[0, 0].scatter(trajectory[0, 0], trajectory[0, 1], color="#009E73", s=60, label="start")
    axes[0, 0].set_title("Deterministic doubled-edge route: 240 causal LiDAR frames")
    axes[0, 0].set_aspect("equal", adjustable="box")
    axes[0, 0].grid(True, linewidth=0.25, alpha=0.35)
    axes[0, 0].legend()
    _draw_splines(axes[0, 1], splines)
    _draw_graph(axes[0, 1], oracle_graph, "Oracle online graph upper bound")
    _draw_splines(axes[1, 0], splines)
    _draw_graph(axes[1, 0], predicted_graph, "LiDAR-only rule baseline online graph")
    index = np.arange(len(frame_records))
    axes[1, 1].step(index, [item["truth_branch_count"] for item in frame_records], where="mid", label="oracle branches", color="#009E73")
    axes[1, 1].step(index, [item["predicted_branch_count"] for item in frame_records], where="mid", label="LiDAR rule branches", color="#CC79A7", alpha=0.9)
    axes[1, 1].set_xlabel("causal frame index")
    axes[1, 1].set_ylabel("outgoing branch count")
    axes[1, 1].set_title("Per-frame branch-count behavior (no temporal look-ahead)")
    axes[1, 1].grid(True, linewidth=0.25, alpha=0.35)
    axes[1, 1].legend()
    fig.suptitle("Working vertical slice: LiDAR → outgoing directions → online topometric graph", fontsize=15)
    fig.savefig(destination, dpi=170)
    plt.close(fig)


def _save_contact_sheet(
    destination: Path,
    ranges: np.ndarray,
    frame_records: list[dict[str, Any]],
) -> list[int]:
    selected = np.linspace(0, len(frame_records) - 1, CONTACT_FRAME_COUNT, dtype=int).tolist()
    fig, axes = plt.subplots(6, 4, figsize=(20, 20), sharex=True, sharey=True, constrained_layout=True)
    image = None
    for axis, frame_index in zip(axes.flat, selected):
        item = frame_records[frame_index]
        image = axis.imshow(
            ranges[frame_index], origin="lower", aspect="auto", interpolation="nearest",
            extent=(0, 360, float(ELEVATION_DEG[0]), float(ELEVATION_DEG[-1])),
            vmin=0, vmax=MAX_RANGE_M, cmap="viridis",
        )
        for heading in item["truth_headings_robot_deg"]:
            axis.axvline(heading, color="#00D060", linewidth=1.2)
        for heading in item["predicted_headings_robot_deg"]:
            axis.axvline(heading, color="#FF4FD8", linewidth=1.0, linestyle="--")
        axis.set_title(
            f"frame {frame_index:03d} | GT {item['truth_branch_count']} / rule {item['predicted_branch_count']}",
            fontsize=8,
        )
    for axis in axes[-1]:
        axis.set_xlabel("robot azimuth (deg)")
    for axis in axes[:, 0]:
        axis.set_ylabel("elevation (deg)")
    assert image is not None
    fig.colorbar(image, ax=axes, label="range (m), fixed 0–50 m", shrink=0.72)
    fig.suptitle("Deterministic 24-frame diagnostic: green=oracle, magenta dashed=LiDAR-only rule", fontsize=14)
    fig.savefig(destination, dpi=145)
    plt.close(fig)
    return selected


def _save_replay_gif(
    destination: Path,
    splines: dict[int, np.ndarray],
    trajectory: np.ndarray,
    ranges: np.ndarray,
    frame_records: list[dict[str, Any]],
    histories: list[dict[str, Any]],
) -> list[int]:
    selected = np.unique(np.linspace(0, len(frame_records) - 1, GIF_FRAME_COUNT, dtype=int)).tolist()
    all_points = np.concatenate(list(splines.values()), axis=0)
    bounds = (
        float(np.min(all_points[:, 0]) - 10), float(np.max(all_points[:, 0]) + 10),
        float(np.min(all_points[:, 1]) - 10), float(np.max(all_points[:, 1]) + 10),
    )
    fig, (map_axis, profile_axis) = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

    def render(frame_index: int) -> None:
        map_axis.clear()
        profile_axis.clear()
        _draw_splines(map_axis, splines)
        map_axis.plot(trajectory[: frame_index + 1, 0], trajectory[: frame_index + 1, 1], color="#56B4E9", linewidth=1.2)
        graph = histories[frame_index]
        _draw_graph(map_axis, graph, f"LiDAR-only online graph at frame {frame_index:03d}")
        map_axis.scatter(trajectory[frame_index, 0], trajectory[frame_index, 1], marker="*", s=130, color="#F0E442", edgecolor="black", zorder=8)
        map_axis.set_xlim(bounds[0], bounds[1]); map_axis.set_ylim(bounds[2], bounds[3])
        item = frame_records[frame_index]
        profile_axis.plot(np.arange(720) * 0.5, item["smoothed_range_profile_m"], color="#303030", linewidth=1.0)
        profile_axis.axhline(item["rule_threshold_m"], color="#999999", linestyle=":", label="rule threshold")
        for heading in item["truth_headings_robot_deg"]:
            profile_axis.axvline(heading, color="#00A650", linewidth=1.8)
        for heading in item["predicted_headings_robot_deg"]:
            profile_axis.axvline(heading, color="#CC33AA", linewidth=1.3, linestyle="--")
        profile_axis.set_xlim(0, 360); profile_axis.set_ylim(0, 52)
        profile_axis.set_xlabel("robot-frame azimuth (deg)"); profile_axis.set_ylabel("smoothed horizontal reach (m)")
        profile_axis.set_title(f"Current scan: GT={item['truth_branch_count']} rule={item['predicted_branch_count']}")
        profile_axis.grid(True, linewidth=0.25, alpha=0.35)

    movie = animation.FuncAnimation(fig, render, frames=selected, interval=180, repeat=True)
    movie.save(destination, writer=animation.PillowWriter(fps=5), dpi=95)
    plt.close(fig)
    return selected


def _seal(output: Path) -> int:
    destination = output / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in output.rglob("*") if path.is_file() and path != destination)
    destination.write_text(
        "".join(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def execute(output: Path) -> dict[str, Any]:
    if output.exists():
        raise FileExistsError(f"refusing to overwrite prototype: {output}")
    for child in ("artifacts", "metrics", "previews", "config", "logs"):
        (output / child).mkdir(parents=True, exist_ok=False)
    write_json(
        output / "RUN_STATE.json",
        {
            "run_id": RUN_ID,
            "state": "RUNNING",
            "role": "ENGINEERING_VERTICAL_SLICE_NOT_PHASE4_SCIENTIFIC_EVIDENCE",
        },
    )
    graph_document = load_json(WORLD_DIR / "graph.json")
    spline_document = load_json(WORLD_DIR / "splines.json")
    splines = _spline_arrays(spline_document)
    nodes = {node["id"]: np.asarray(node["xyz"], dtype=np.float64) for node in graph_document["nodes"]}
    route_ids = _doubled_euler_route(graph_document)
    raw_route = np.asarray([nodes[node_id] for node_id in route_ids], dtype=np.float64)
    sampled, route_directions = _resample_polyline(raw_route, FRAME_COUNT)
    projected = []
    tangents = []
    tunnel_ids = []
    projection_errors = []
    for point, direction in zip(sampled, route_directions):
        axis, tangent, tunnel_id, error = _project_to_splines(point, direction, splines)
        projected.append(axis); tangents.append(tangent); tunnel_ids.append(tunnel_id); projection_errors.append(error)
    trajectory = np.asarray(projected, dtype=np.float64)
    tangents_array = np.asarray(tangents, dtype=np.float64)
    yaw_deg = np.mod(np.degrees(np.arctan2(tangents_array[:, 1], tangents_array[:, 0])), 360.0)
    fta_distance = float((WORLD_DIR / "fta_dist.txt").read_text(encoding="utf-8"))
    origins = trajectory.copy()
    origins[:, 2] += fta_distance + LIDAR_HEIGHT_ABOVE_FLOOR_M

    mesh = o3d.io.read_triangle_mesh(str(WORLD_DIR / "mesh.obj"), enable_post_processing=False)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError("frozen Cano mesh could not be read")
    scene = _scene(mesh)
    local_directions = lidar_local_directions()
    baseline = RangeExitBaseline()
    predicted_graph = OnlineTopometricGraph()
    oracle_graph = OnlineTopometricGraph()
    ranges = np.empty((FRAME_COUNT, 16, 720), dtype=np.float32)
    valid_masks = np.empty((FRAME_COUNT, 16, 720), dtype=np.uint8)
    labels = np.empty((FRAME_COUNT, 720), dtype=np.float32)
    frame_records: list[dict[str, Any]] = []
    predicted_history: list[dict[str, Any]] = []
    oracle_history: list[dict[str, Any]] = []
    total_predicted = 0
    total_truth = 0
    total_matched = 0
    angular_errors: list[float] = []
    exact_count_frames = 0
    for frame_index in range(FRAME_COUNT):
        directions = world_directions(local_directions, float(yaw_deg[frame_index]))
        scan_range, scan_valid = _raycast(scene, origins[frame_index], directions)
        ranges[frame_index] = scan_range
        valid_masks[frame_index] = scan_valid
        teacher = structural_label(
            trajectory[frame_index], float(yaw_deg[frame_index]), splines
        )
        labels[frame_index] = np.asarray(teacher["label_720"], dtype=np.float32)
        prediction = baseline.predict(scan_range, scan_valid, ELEVATION_DEG)
        predicted_headings = prediction["headings_robot_deg"]
        truth_headings = teacher["headings_robot_deg"]
        matched, errors = _match_headings(predicted_headings, truth_headings)
        total_predicted += len(predicted_headings)
        total_truth += len(truth_headings)
        total_matched += matched
        angular_errors.extend(errors)
        exact_count_frames += int(len(predicted_headings) == len(truth_headings))
        predicted_update = predicted_graph.update(
            origins[frame_index], float(yaw_deg[frame_index]), predicted_headings, frame_index
        )
        oracle_update = oracle_graph.update(
            origins[frame_index], float(yaw_deg[frame_index]), truth_headings, frame_index
        )
        predicted_history.append(copy.deepcopy(predicted_graph.snapshot()))
        oracle_history.append(copy.deepcopy(oracle_graph.snapshot()))
        frame_records.append(
            {
                "frame_index": frame_index,
                "source_tunnel_id": int(tunnel_ids[frame_index]),
                "axis_xyz_m": trajectory[frame_index].tolist(),
                "sensor_xyz_m": origins[frame_index].tolist(),
                "yaw_deg": float(yaw_deg[frame_index]),
                "valid_ratio": float(np.mean(scan_valid)),
                "truth_headings_robot_deg": truth_headings,
                "predicted_headings_robot_deg": predicted_headings,
                "truth_branch_count": len(truth_headings),
                "predicted_branch_count": len(predicted_headings),
                "matched_branch_count_20deg": matched,
                "matched_angular_errors_deg": errors,
                "rule_threshold_m": prediction["threshold_m"],
                "smoothed_range_profile_m": prediction["smoothed_range_profile_m"].tolist(),
                "predicted_graph_update": predicted_update,
                "oracle_graph_update": oracle_update,
            }
        )

    predicted_snapshot = predicted_graph.snapshot()
    oracle_snapshot = oracle_graph.snapshot()
    graph_metrics = _graph_comparison(predicted_snapshot, oracle_snapshot)
    association_sweep = _association_sweep(frame_records)
    selected_config = association_sweep["selected"]["config"]
    if selected_config != predicted_graph.config.to_dict():
        raise RuntimeError(
            f"v2 default graph config is not the deterministic sweep selection: "
            f"default={predicted_graph.config.to_dict()}, selected={selected_config}"
        )
    previous_comparison = None
    if PREVIOUS_RUN.is_dir():
        previous_comparison = _graph_comparison(
            load_json(PREVIOUS_RUN / "artifacts/lidar_rule_online_graph.json"),
            load_json(PREVIOUS_RUN / "artifacts/oracle_online_graph.json"),
        )
    precision = total_matched / max(1, total_predicted)
    recall = total_matched / max(1, total_truth)
    branch_f1 = 2 * precision * recall / max(1e-12, precision + recall)
    metrics = {
        "schema_version": "cano_lidar_topometric_vertical_slice_metrics_v1",
        "status": "COMPLETED_PROTOTYPE",
        "scope": {
            "source_worlds": 1,
            "new_worlds": 0,
            "causal_frames": FRAME_COUNT,
            "rays_per_frame": 11520,
            "total_primary_rays": FRAME_COUNT * 11520,
            "models_trained": 0,
            "mtare_changes": 0,
        },
        "route": {
            "source_graph_nodes": len(graph_document["nodes"]),
            "source_graph_edges": len(graph_document["edges"]),
            "doubled_edge_circuit_nodes": len(route_ids),
            "sampled_frames": FRAME_COUNT,
            "path_length_m": float(np.sum(np.linalg.norm(np.diff(raw_route, axis=0), axis=1))),
            "maximum_graph_route_to_spline_projection_error_m": float(max(projection_errors)),
        },
        "lidar_rule_branch_metrics_at_20deg": {
            "matched": total_matched,
            "predicted": total_predicted,
            "truth": total_truth,
            "precision": precision,
            "recall": recall,
            "f1": branch_f1,
            "mean_matched_angular_error_deg": float(np.mean(angular_errors)) if angular_errors else None,
            "median_matched_angular_error_deg": float(np.median(angular_errors)) if angular_errors else None,
            "exact_branch_count_frame_fraction": exact_count_frames / FRAME_COUNT,
        },
        "online_graphs": {
            "lidar_rule": {
                "nodes": predicted_snapshot["node_count"],
                "edges": predicted_snapshot["edge_count"],
                "unexplored_exit_stubs": predicted_snapshot["unexplored_exit_stub_count"],
            },
            "oracle_upper_bound": {
                "nodes": oracle_snapshot["node_count"],
                "edges": oracle_snapshot["edge_count"],
                "unexplored_exit_stubs": oracle_snapshot["unexplored_exit_stub_count"],
            },
            "lidar_rule_vs_oracle_online_graph": graph_metrics,
        },
        "same_world_development_refinement": {
            "candidate_configurations_evaluated": association_sweep["candidate_count"],
            "selected_config": selected_config,
            "selected_node_f1": association_sweep["selected"]["node_f1"],
            "selected_geometric_edge_f1": association_sweep["selected"]["geometric_edge_f1"],
            "previous_v1_graph_comparison": previous_comparison,
            "warning": "The association parameters were selected on this same development trajectory and require validation on new topology before any generalization claim.",
        },
        "claim_boundary": [
            "This is a development vertical slice on one previously inspected world, not an unseen-topology scientific result.",
            "The range rule is transparent and non-learning; it exists to make the full interface runnable before model training.",
            "The oracle graph is an upper-bound plumbing check and is never reported as sensor prediction.",
            "The online graph is an engineering prototype and does not modify or replace M-TARE.",
            "V2 graph-association parameters were selected from 81 configurations on this same development trajectory.",
        ],
    }
    write_json(output / "metrics/summary.json", metrics)
    write_json(output / "metrics/per_frame.json", {"frames": frame_records})
    write_json(output / "metrics/association_sweep.json", association_sweep)
    write_json(output / "artifacts/lidar_rule_online_graph.json", predicted_snapshot)
    write_json(output / "artifacts/oracle_online_graph.json", oracle_snapshot)
    write_json(
        output / "artifacts/route_manifest.json",
        {
            "route_node_ids": route_ids,
            "frame_count": FRAME_COUNT,
            "axis_xyz_m": trajectory.tolist(),
            "sensor_xyz_m": origins.tolist(),
            "yaw_deg": yaw_deg.tolist(),
            "source_tunnel_id": tunnel_ids,
            "projection_error_m": projection_errors,
        },
    )
    np.savez_compressed(
        output / "artifacts/trajectory_scans.npz",
        range_m=ranges,
        valid_mask=valid_masks,
        label_720=labels,
        sensor_xyz_m=origins.astype(np.float32),
        axis_xyz_m=trajectory.astype(np.float32),
        yaw_deg=yaw_deg.astype(np.float32),
    )
    write_json(
        output / "config/prototype_scope.json",
        {
            "user_direction": "先把我们的东西做出来；哪里有问题再调整",
            "run_id": RUN_ID,
            "source_world": str(WORLD_DIR.relative_to(PROJECT_ROOT)),
            "frames": FRAME_COUNT,
            "method": "current-frame LiDAR-only range-sector rule and parallel spline-oracle upper bound feeding the same causal online topometric graph",
            "not_authorized_or_executed": ["new world generation", "training", "ROS", "M-TARE modification", "closed loop"],
            "baseline_config": baseline.config.to_dict(),
            "graph_config": predicted_graph.config.to_dict(),
        },
    )

    summary_figure = output / "previews/vertical_slice_summary.png"
    contact_figure = output / "previews/branch_diagnostic_24_frames.png"
    replay_gif = output / "previews/online_topometric_replay.gif"
    _save_summary_figure(summary_figure, splines, trajectory, predicted_snapshot, oracle_snapshot, frame_records)
    contact_indices = _save_contact_sheet(contact_figure, ranges, frame_records)
    gif_indices = _save_replay_gif(replay_gif, splines, trajectory, ranges, frame_records, predicted_history)
    write_json(
        output / "previews/provenance.json",
        {
            "summary": "Complete source splines, full 240-frame route, final oracle/rule graphs and all frame branch-count behavior.",
            "contact_sheet_indices": contact_indices,
            "contact_sheet_selection": "24 deterministic equally spaced frames; not selected by quality",
            "gif_indices": gif_indices,
            "gif_selection": "60 deterministic equally spaced causal states",
            "colors": "green=oracle branch direction; magenta=LiDAR-rule prediction/unexplored stub; gray=traversed stub/ source spline",
            "units": "world positions metres; headings degrees; ranges metres",
            "limitations": metrics["claim_boundary"],
        },
    )
    write_json(
        output / "RUN_STATE.json",
        {
            "run_id": RUN_ID,
            "state": "COMPLETED",
            "overall_status": "COMPLETED_WORKING_VERTICAL_SLICE",
            "note": "Engineering prototype only; quantitative weaknesses are retained in metrics/summary.json.",
        },
    )
    sealed = _seal(output)
    metrics["sealed_file_count"] = sealed
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "results/prototypes" / RUN_ID,
    )
    args = parser.parse_args()
    output = args.output.resolve()
    try:
        result = execute(output)
    except Exception as exc:
        if output.is_dir():
            write_json(
                output / "RUN_STATE.json",
                {
                    "run_id": RUN_ID,
                    "state": "FAILED",
                    "exception_type": type(exc).__name__,
                    "message": str(exc),
                    "traceback": traceback.format_exc(),
                },
            )
        raise
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
