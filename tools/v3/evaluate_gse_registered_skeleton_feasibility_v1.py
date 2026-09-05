#!/usr/bin/env python3
"""Zero-training ERCSS visible-skeleton Teacher/representation feasibility."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import replace
import csv
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
from scipy.spatial import cKDTree
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from mtare_topo.representation.gse_registered_structural_skeleton import relative_pose_window
from mtare_topo.teacher.gse_registered_structural_skeleton_teacher import (
    SKELETON_SAMPLE_SPACING_M,
    causal_mesh_visibility,
    ego_connected_component,
    sample_physical_skeleton,
)


PASS = "PASS_GSE_REGISTERED_SKELETON_FEASIBILITY_V1"
FAIL = "FAIL_GSE_REGISTERED_SKELETON_FEASIBILITY_V1"
EXPECTED_WORLDS = 80
EXPECTED_FRAMES = 252_430
EXPECTED_OBSERVATIONS = 188_126
EXPECTED_EVENT_ROWS = {
    "junction": 26_608,
    "terminal": 7_525,
    "turn": 1_998,
    "geometry_transition": 1_031,
}
EXPECTED_PARTITION_ROWS = {
    "fit": {"junction": 19_743, "terminal": 5_551, "turn": 1_482, "geometry_transition": 791},
    "selection": {"junction": 6_865, "terminal": 1_974, "turn": 516, "geometry_transition": 240},
}
EXPECTED_PARTITION_IDENTITIES = {
    "fit": {"junction": 417, "terminal": 375, "turn": 297, "geometry_transition": 59},
    "selection": {"junction": 146, "terminal": 128, "turn": 95, "geometry_transition": 17},
}
MINIMUM_IDENTITY_COVERAGE = 0.80
MAXIMUM_FIT_NODE_CAPACITY = 512
MAXIMUM_FIT_SEGMENT_CAPACITY = 1024
NODE_EVENT_RADIUS_M = 10.0
MINIMUM_LOCAL_EXTENT_M = 2.0
MINIMUM_TURN_CHANGE_DEG = 15.0
MINIMUM_GEOMETRY_CHANGE_M = 1.0
RAY_BATCH_SIZE = 500_000


def _partition(parent: str) -> str:
    suffix = int(parent.rsplit("_C", 1)[1])
    if 1 <= suffix <= 6:
        return "fit"
    if 7 <= suffix <= 8:
        return "selection"
    raise RuntimeError(f"forbidden world entered ERCSS proof: {parent}")


def _scene(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty native perception mesh: {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _caster(scene: o3d.t.geometry.RaycastingScene):
    def cast(origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
        result = np.empty(len(origins), dtype=np.float64)
        for start in range(0, len(origins), RAY_BATCH_SIZE):
            stop = min(start + RAY_BATCH_SIZE, len(origins))
            rays = np.concatenate((origins[start:stop], directions[start:stop]), axis=1)
            result[start:stop] = scene.cast_rays(
                o3d.core.Tensor(rays.astype(np.float32, copy=False))
            )["t_hit"].numpy().astype(np.float64)
        return result
    return cast


def _next_power_of_two(value: int) -> int:
    if value <= 0:
        return 1
    return 1 << (int(value) - 1).bit_length()


def _write_jsonl(stream, value: dict[str, Any]) -> None:
    stream.write(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")


def _attach_mesh_geometry(sampled, group, supervision: dict[str, np.ndarray]):
    references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
    axes = np.asarray(group["axis_xyz_m"][:], dtype=np.float64)[references[:, -1]]
    geometry = np.asarray(group["geometry"][:], dtype=np.float64)
    valid = np.asarray(group["geometry_valid_mask"][:], dtype=bool)
    traversal = np.asarray(supervision["traversal_id"]).astype(str)
    if len(traversal) != len(geometry) or geometry.shape != valid.shape or geometry.shape[1] != 4:
        raise RuntimeError("mesh geometry/supervision alignment drift")
    output = np.full((len(sampled.segment_node_indices), 4), np.nan, dtype=np.float64)
    midpoints = np.mean(sampled.node_xyz_world_m[sampled.segment_node_indices], axis=1)
    for edge_index, edge_id in enumerate(sampled.edge_identity):
        segment_mask = sampled.segment_edge_index == edge_index
        direction0 = np.char.endswith(traversal, f":{edge_id}:d0")
        direction1 = np.char.endswith(traversal, f":{edge_id}:d1")
        source = direction0 if np.any(direction0) else direction1
        if not np.any(source):
            continue
        _, nearest = cKDTree(axes[source]).query(midpoints[segment_mask], k=1, workers=1)
        selected_geometry = geometry[source][nearest].copy()
        selected_valid = valid[source][nearest]
        if not np.any(direction0):
            selected_geometry[:, 2] *= -1.0
        selected_geometry[~selected_valid] = np.nan
        output[segment_mask] = selected_geometry
    return replace(sampled, segment_geometry=output), int(np.sum(~np.all(np.isfinite(output), axis=1)))


def _component_evidence(sampled, component: np.ndarray, ego_xyz: np.ndarray, graph_degrees: np.ndarray, event: str, current_geometry: np.ndarray, current_valid: np.ndarray) -> dict[str, Any]:
    selected_nodes = np.flatnonzero(component)
    edge_mask = component[sampled.segment_node_indices[:, 0]] & component[sampled.segment_node_indices[:, 1]]
    selected_segments = sampled.segment_node_indices[edge_mask]
    node_count = int(len(selected_nodes)); segment_count = int(np.sum(edge_mask))
    if node_count == 0:
        return {"supported": False, "reason": "empty_component", "nodes": 0, "segments": 0, "extent_m": 0.0}
    distances = np.linalg.norm(sampled.node_xyz_world_m[selected_nodes] - ego_xyz, axis=1)
    extent = float(np.max(distances, initial=0.0))
    nearest = float(np.min(distances))
    local_degree = np.zeros(len(sampled.node_xyz_world_m), dtype=np.int16)
    if len(selected_segments):
        np.add.at(local_degree, selected_segments[:, 0], 1); np.add.at(local_degree, selected_segments[:, 1], 1)
    nearby = np.linalg.norm(sampled.node_xyz_world_m - ego_xyz, axis=1) <= NODE_EVENT_RADIUS_M + 1e-9
    base = nearest <= SKELETON_SAMPLE_SPACING_M + 1e-6 and extent >= MINIMUM_LOCAL_EXTENT_M and segment_count >= 2
    evidence: dict[str, Any] = {
        "nodes": node_count, "segments": segment_count, "extent_m": extent,
        "nearest_ego_node_m": nearest, "supported": False, "reason": "event_geometry_missing",
    }
    if event == "junction":
        structural = component & nearby & (graph_degrees >= 3) & (local_degree >= 3)
        evidence.update(structural_node_count=int(np.sum(structural)), supported=bool(base and np.any(structural)), reason="ok" if base and np.any(structural) else "no_visible_three_arm_junction")
    elif event == "terminal":
        structural = component & nearby & (graph_degrees == 1) & (local_degree >= 1)
        evidence.update(structural_node_count=int(np.sum(structural)), supported=bool(base and np.any(structural)), reason="ok" if base and np.any(structural) else "no_visible_terminal_endpoint")
    elif event == "turn":
        vectors = sampled.node_xyz_world_m[selected_segments[:, 1]] - sampled.node_xyz_world_m[selected_segments[:, 0]] if len(selected_segments) else np.empty((0, 3))
        horizontal = vectors[:, :2]
        norms = np.linalg.norm(horizontal, axis=1)
        horizontal = horizontal[norms > 1e-9] / norms[norms > 1e-9, None]
        maximum_angle = 0.0
        if len(horizontal) >= 2:
            cosine = np.clip(np.abs(horizontal @ horizontal.T), -1.0, 1.0)
            maximum_angle = float(np.degrees(np.max(np.arccos(cosine))))
        supported = base and maximum_angle >= MINIMUM_TURN_CHANGE_DEG
        evidence.update(maximum_visible_heading_change_deg=maximum_angle, supported=bool(supported), reason="ok" if supported else "visible_heading_change_below_teacher_threshold")
    elif event == "geometry_transition":
        segment_geometry = sampled.segment_geometry[edge_mask]
        finite = np.all(np.isfinite(segment_geometry[:, :2]), axis=1) if len(segment_geometry) else np.zeros(0, dtype=bool)
        width_change = float(np.ptp(segment_geometry[finite, 0])) if np.sum(finite) >= 2 else 0.0
        height_change = float(np.ptp(segment_geometry[finite, 1])) if np.sum(finite) >= 2 else 0.0
        current_complete = bool(np.all(current_valid) and np.all(np.isfinite(current_geometry)))
        supported = base and current_complete and max(width_change, height_change) >= MINIMUM_GEOMETRY_CHANGE_M
        evidence.update(visible_width_change_m=width_change, visible_height_change_m=height_change, current_geometry_complete=current_complete, supported=bool(supported), reason="ok" if supported else "visible_metric_change_below_teacher_threshold")
    else:
        raise ValueError(f"unsupported event audit type: {event}")
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--supervision", required=True, type=Path)
    parser.add_argument("--mesh-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    output = args.output_dir.resolve(); output.mkdir(parents=True, exist_ok=False)
    dataset_root = args.dataset.resolve() / "artifacts/dataset/train"
    supervision_root = args.supervision.resolve() / "artifacts/supervision/worlds"
    mesh_root = args.mesh_root.resolve()
    worlds = sorted(path.stem for path in supervision_root.glob("*.npz"))
    if len(worlds) != EXPECTED_WORLDS or any(int(world.rsplit("_C", 1)[1]) not in range(1, 9) for world in worlds):
        raise RuntimeError("ERCSS proof world population drift")

    total_frames = 0; total_observations = 0; relative_pose_rows = 0
    event_rows = Counter(); partition_rows: dict[str, Counter] = defaultdict(Counter)
    identity_supported: dict[tuple[str, str], dict[str, bool]] = defaultdict(dict)
    capacity_rows: dict[str, list[tuple[int, int]]] = defaultdict(list)
    world_records: list[dict[str, Any]] = []
    missing_segment_geometry = 0
    observation_path = output / "event_observation_audit.jsonl"
    with observation_path.open("w", encoding="utf-8") as observation_stream:
        for world_index, parent in enumerate(worlds, start=1):
            partition = _partition(parent)
            group = zarr.open_group(str(dataset_root / f"{parent}.zarr"), mode="r")
            with np.load(supervision_root / f"{parent}.npz", allow_pickle=False) as archive:
                supervision = {name: archive[name].copy() for name in archive.files}
            references = np.asarray(group["local_frame_references"][:], dtype=np.int64)
            sensors = np.asarray(group["sensor_xyz_m"][:], dtype=np.float64)
            yaws = np.asarray(group["yaw_deg"][:], dtype=np.float64)
            axes = np.asarray(group["axis_xyz_m"][:], dtype=np.float64)
            geometry = np.asarray(group["geometry"][:], dtype=np.float64)
            geometry_valid = np.asarray(group["geometry_valid_mask"][:], dtype=bool)
            events = np.asarray(supervision["event_name"]).astype(str)
            identities = np.asarray(supervision["identity"]).astype(str)
            if references.shape != (len(events), 5) or len(group["global_sequence_index"]) != len(events):
                raise RuntimeError(f"causal sequence alignment drift: {parent}")
            total_frames += len(sensors); total_observations += len(references)
            # Audit every relative pose window, including corridors, without
            # materializing their point clouds or using a learned model.
            for refs in references:
                relative_pose_window(sensors[refs], yaws[refs]); relative_pose_rows += 1

            primary = mesh_root / parent / "primary"
            graph = load_json(primary / "graph.json")
            sampled = sample_physical_skeleton(
                graph, load_json(primary / "splines.json"),
                load_json(primary / "geometry_parameters.json"),
            )
            sampled, missing = _attach_mesh_geometry(sampled, group, supervision)
            missing_segment_geometry += missing
            scene = _scene(primary / "mesh.obj"); cast = _caster(scene)
            spatial_index = cKDTree(sampled.node_xyz_world_m)
            graph_nodes_sorted = sorted(graph["nodes"], key=lambda row: str(row["id"]))
            graph_degrees = np.zeros(len(sampled.node_xyz_world_m), dtype=np.int16)
            graph_degrees[:len(graph_nodes_sorted)] = [int(row["degree"]) for row in graph_nodes_sorted]
            world_events = Counter(); world_supported = Counter(); world_max_nodes = 0; world_max_segments = 0
            event_indices = np.flatnonzero(np.isin(events, np.asarray(tuple(EXPECTED_EVENT_ROWS))))
            for row in event_indices:
                event = str(events[row]); identity = str(identities[row])
                if not identity:
                    raise RuntimeError(f"empty physical event identity: {parent}:{row}")
                refs = references[row]; current = int(refs[-1]); ego = axes[current]
                candidate_set: set[int] = set()
                for sensor in sensors[refs]:
                    candidate_set.update(spatial_index.query_ball_point(sensor, r=50.0 + 1e-9, workers=1))
                candidate = np.asarray(sorted(candidate_set), dtype=np.int64)
                visible = np.zeros(len(sampled.node_xyz_world_m), dtype=bool)
                if len(candidate):
                    visible[candidate] = causal_mesh_visibility(
                        sampled.node_xyz_world_m[candidate], sensors[refs], yaws[refs], cast,
                    )
                component = ego_connected_component(sampled, visible, ego)
                evidence = _component_evidence(
                    sampled, component, ego, graph_degrees, event,
                    geometry[row], geometry_valid[row],
                )
                supported = bool(evidence["supported"])
                event_rows[event] += 1; partition_rows[partition][event] += 1
                world_events[event] += 1; world_supported[event] += int(supported)
                identity_supported[(partition, event)][identity] = identity_supported[(partition, event)].get(identity, False) or supported
                capacity_rows[partition].append((int(evidence["nodes"]), int(evidence["segments"])))
                world_max_nodes = max(world_max_nodes, int(evidence["nodes"])); world_max_segments = max(world_max_segments, int(evidence["segments"]))
                _write_jsonl(observation_stream, {
                    "parent": parent, "partition": partition, "sequence_row": int(row),
                    "global_sequence_index": int(group["global_sequence_index"][row]),
                    "event": event, "identity": identity, **evidence,
                })
            record = {
                "parent": parent, "partition": partition, "frames": len(sensors),
                "observations": len(references), "sampled_nodes": len(sampled.node_xyz_world_m),
                "sampled_segments": len(sampled.segment_node_indices), "event_rows": dict(world_events),
                "supported_rows": dict(world_supported), "maximum_visible_nodes": world_max_nodes,
                "maximum_visible_segments": world_max_segments,
            }
            world_records.append(record)
            print(json.dumps({"world": f"{world_index}/{len(worlds)}", **record}, sort_keys=True), flush=True)

    fit_max_nodes = max(value[0] for value in capacity_rows["fit"])
    fit_max_segments = max(value[1] for value in capacity_rows["fit"])
    node_capacity = _next_power_of_two(fit_max_nodes); segment_capacity = _next_power_of_two(fit_max_segments)
    selection_max_nodes = max(value[0] for value in capacity_rows["selection"])
    selection_max_segments = max(value[1] for value in capacity_rows["selection"])
    identity_records = []
    for partition in ("fit", "selection"):
        for event in EXPECTED_EVENT_ROWS:
            values = identity_supported[(partition, event)]
            total = len(values); supported = sum(values.values()); coverage = supported / max(1, total)
            identity_records.append({"partition": partition, "event": event, "identities": total, "supported_identities": supported, "coverage": coverage})
    with (output / "identity_coverage.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(identity_records[0])); writer.writeheader(); writer.writerows(identity_records)
    with (output / "world_summary.jsonl").open("w", encoding="utf-8") as stream:
        for row in world_records: _write_jsonl(stream, row)

    exact_identity_population = all(
        next(row["identities"] for row in identity_records if row["partition"] == partition and row["event"] == event) == count
        for partition, values in EXPECTED_PARTITION_IDENTITIES.items() for event, count in values.items()
    )
    checks = {
        "exact_world_frame_observation_population": total_frames == EXPECTED_FRAMES and total_observations == EXPECTED_OBSERVATIONS and len(worlds) == EXPECTED_WORLDS,
        "all_relative_pose_windows_validated": relative_pose_rows == EXPECTED_OBSERVATIONS,
        "exact_event_row_population": dict(event_rows) == EXPECTED_EVENT_ROWS and all(dict(partition_rows[p]) == EXPECTED_PARTITION_ROWS[p] for p in EXPECTED_PARTITION_ROWS),
        "exact_event_identity_population": exact_identity_population,
        "all_partition_event_identity_coverage_at_least_0p80": all(row["coverage"] >= MINIMUM_IDENTITY_COVERAGE for row in identity_records),
        "fit_capacity_within_pre_registered_ceiling": node_capacity <= MAXIMUM_FIT_NODE_CAPACITY and segment_capacity <= MAXIMUM_FIT_SEGMENT_CAPACITY,
        "selection_does_not_overflow_fit_frozen_capacity": selection_max_nodes <= node_capacity and selection_max_segments <= segment_capacity,
        "some_mesh_geometry_attached": missing_segment_geometry < sum(row["sampled_segments"] for row in world_records),
        "zero_forbidden_work": True,
    }
    overall = PASS if all(checks.values()) else FAIL
    summary = {
        "schema_version": "gse_registered_skeleton_feasibility_v1", "overall_status": overall,
        "checks": checks, "worlds": len(worlds), "frames": total_frames,
        "observations": total_observations, "relative_pose_windows": relative_pose_rows,
        "event_rows": dict(event_rows), "partition_event_rows": {key: dict(value) for key, value in partition_rows.items()},
        "identity_coverage": identity_records,
        "capacity": {"fit_maximum_nodes": fit_max_nodes, "fit_maximum_segments": fit_max_segments,
                     "frozen_node_capacity": node_capacity, "frozen_segment_capacity": segment_capacity,
                     "selection_maximum_nodes": selection_max_nodes, "selection_maximum_segments": selection_max_segments,
                     "maximum_allowed_fit_nodes": MAXIMUM_FIT_NODE_CAPACITY, "maximum_allowed_fit_segments": MAXIMUM_FIT_SEGMENT_CAPACITY},
        "sample_spacing_m": SKELETON_SAMPLE_SPACING_M, "missing_segment_geometry_values": missing_segment_geometry,
        "optimizer_steps": 0, "model_inference_frames": 0, "c09_worlds_read": 0,
        "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0,
    }
    write_json(output / "summary.json", summary)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    labels = [f"{row['partition']}\n{row['event'].replace('_', ' ')}" for row in identity_records]
    coverage = [100.0 * row["coverage"] for row in identity_records]
    axes[0].bar(np.arange(len(labels)), coverage, color=["#2563eb"] * 4 + ["#f59e0b"] * 4)
    axes[0].axhline(80.0, color="#b91c1c", linestyle="--", linewidth=1)
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=35, ha="right", fontsize=8)
    axes[0].set_ylim(0, 105); axes[0].set_ylabel("supported physical identities [%]")
    axes[0].set_title("Causal visible-skeleton support")
    capacities = [fit_max_nodes, selection_max_nodes, fit_max_segments, selection_max_segments]
    axes[1].bar(["fit nodes", "selection nodes", "fit segments", "selection segments"], capacities, color=["#2563eb", "#f59e0b"] * 2)
    axes[1].axhline(node_capacity, color="#2563eb", linestyle="--", linewidth=1, label=f"node cap {node_capacity}")
    axes[1].axhline(segment_capacity, color="#6b7280", linestyle=":", linewidth=1, label=f"segment cap {segment_capacity}")
    axes[1].tick_params(axis="x", rotation=25); axes[1].set_ylabel("maximum visible items")
    axes[1].set_title("Fit-frozen representation capacity"); axes[1].legend(fontsize=8)
    fig.suptitle(f"ERCSS zero-training feasibility — {overall}")
    fig.tight_layout()
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(output / f"gse_registered_skeleton_feasibility_v1.{suffix}", dpi=180)
    plt.close(fig)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
