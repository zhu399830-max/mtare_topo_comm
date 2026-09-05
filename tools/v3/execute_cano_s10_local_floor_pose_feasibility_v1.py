#!/usr/bin/env python3
"""Audit floor-following sensor-height feasibility at S10 clearance failures."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.local_pose_feasibility import contiguous_feasible_intervals, physical_groups
from mtare_topo.governance import write_json


WORLD = "S10_3d_complex_C08"
MESH = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S10_3d_complex_C08/primary/mesh.obj"
FRAME_CSV = PROJECT_ROOT / "results/gate4_topology/gate4_20260813_cano_c08_complete_clearance_audit_v1_seed0/artifacts/complete_frame_clearance.csv"
OFFSETS_M = np.round(np.arange(-1.0, 2.0 + 1e-9, .05), 8)
HORIZONTAL_CLEARANCE_M = .8
SENSOR_ABOVE_FLOOR_M = 1.0
FLOOR_TOLERANCE_M = .05
UPWARD_CLEARANCE_M = .8
MAX_VERTICAL_QUERY_M = 8.0
PHYSICAL_GROUP_RADIUS_M = 2.0


def scene_from_mesh() -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(MESH), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError("S10 mesh is empty")
    scene = o3d.t.geometry.RaycastingScene(); scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def cast(scene: o3d.t.geometry.RaycastingScene, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    rays = np.concatenate((origins.astype(np.float32), directions.astype(np.float32)), axis=1)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)


def evaluate_candidate(scene: o3d.t.geometry.RaycastingScene, origin: np.ndarray) -> dict:
    azimuth = np.radians(np.arange(720, dtype=np.float32) * .5)
    directions = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros(720)), axis=1)
    horizontal = cast(scene, np.broadcast_to(origin, directions.shape), directions)
    horizontal_finite = np.isfinite(horizontal) & (horizontal >= 0)
    if not np.any(horizontal_finite):
        raise RuntimeError(f"no finite horizontal hit: {origin.tolist()}")
    horizontal_min = float(np.min(horizontal[horizontal_finite]))
    vertical_origins = np.broadcast_to(origin, (2, 3))
    vertical = cast(scene, vertical_origins, np.asarray([[0, 0, -1], [0, 0, 1]], dtype=np.float32))
    down, up = (float(value) if np.isfinite(value) and value >= 0 else float("inf") for value in vertical)
    vertical_evidence_complete = bool(down <= MAX_VERTICAL_QUERY_M and up <= MAX_VERTICAL_QUERY_M)
    floor_following = bool(vertical_evidence_complete and abs(down - SENSOR_ABOVE_FLOOR_M) <= FLOOR_TOLERANCE_M)
    feasible = bool(horizontal_min >= HORIZONTAL_CLEARANCE_M and floor_following and up >= UPWARD_CLEARANCE_M)
    return {"horizontal_clearance_m": horizontal_min, "downward_distance_m": down, "upward_distance_m": up, "vertical_evidence_complete": vertical_evidence_complete, "floor_following_passed": floor_following, "feasible": feasible}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); started = time.monotonic()
    with FRAME_CSV.open("r", encoding="utf-8") as stream:
        failures = [row for row in csv.DictReader(stream) if row["world"] == WORLD and row["clearance_passed"] == "False"]
    if len(failures) != 9:
        raise RuntimeError(f"expected 9 frozen S10 failure records, got {len(failures)}")
    for row in failures:
        row["axis_xyz_m"] = [float(row[key]) for key in ("axis_x_m", "axis_y_m", "axis_z_m")]
        row["sensor_xyz_m"] = [float(row[key]) for key in ("sensor_x_m", "sensor_y_m", "sensor_z_m")]
        row["frame_index"] = int(row["frame_index"]); row["route_arc_m"] = float(row["route_arc_m"])
    groups = physical_groups(failures, PHYSICAL_GROUP_RADIUS_M)
    scene = scene_from_mesh(); all_candidates = []; pose_summaries = []
    for pose_index, failure in enumerate(failures):
        origin = np.asarray(failure["sensor_xyz_m"], dtype=np.float64)
        records = []
        for offset in OFFSETS_M:
            candidate_origin = origin.copy(); candidate_origin[2] += float(offset)
            result = evaluate_candidate(scene, candidate_origin)
            record = {"pose_index": pose_index, "frame_index": failure["frame_index"], "route_arc_m": failure["route_arc_m"], "edge_id": failure["edge_id"], "tunnel_id": int(failure["tunnel_id"]), "z_offset_m": float(offset), "candidate_sensor_z_m": float(candidate_origin[2]), **result}
            records.append(record); all_candidates.append(record)
        intervals = contiguous_feasible_intervals(records, .05)
        recommended = min((item for item in records if item["feasible"]), key=lambda item: (abs(item["z_offset_m"]), item["z_offset_m"]), default=None)
        pose_summaries.append({
            "pose_index": pose_index, "frame_index": failure["frame_index"], "route_arc_m": failure["route_arc_m"],
            "edge_id": failure["edge_id"], "tunnel_id": int(failure["tunnel_id"]),
            "axis_xyz_m": failure["axis_xyz_m"], "original_sensor_xyz_m": failure["sensor_xyz_m"],
            "original_horizontal_clearance_m": float(failure["minimum_horizontal_clearance_m"]),
            "feasible_candidate_count": sum(item["feasible"] for item in records), "feasible_intervals": intervals,
            "recommended_candidate": recommended,
        })
        print(json.dumps({"pose": pose_index + 1, "of": len(failures), "frame": failure["frame_index"], "feasible_candidates": pose_summaries[-1]["feasible_candidate_count"]}), flush=True)

    for group_index, members in enumerate(groups):
        fig, axes = plt.subplots(3, 1, figsize=(11, 10), sharex=True, constrained_layout=True)
        for member in members:
            rows = [item for item in all_candidates if item["pose_index"] == member]
            offsets = [item["z_offset_m"] for item in rows]
            label = f"frame {failures[member]['frame_index']} {failures[member]['edge_id']}"
            axes[0].plot(offsets, [item["horizontal_clearance_m"] for item in rows], label=label)
            axes[1].plot(offsets, [item["downward_distance_m"] for item in rows], label=label)
            axes[2].plot(offsets, [item["upward_distance_m"] for item in rows], label=label)
        axes[0].axhline(.8, color="red", linestyle="--"); axes[0].set_ylabel("horizontal clearance (m)")
        axes[1].axhspan(.95, 1.05, color="green", alpha=.15); axes[1].set_ylabel("downward hit (m)")
        axes[2].axhline(.8, color="red", linestyle="--"); axes[2].set_ylabel("upward hit (m)"); axes[2].set_xlabel("sensor z offset from frozen pose (m)")
        for axis in axes: axis.grid(alpha=.2); axis.legend(fontsize=7)
        fig.suptitle(f"Gate 4 S10 local floor-following feasibility — physical group {group_index}")
        fig.savefig(run_dir / f"previews/physical_group_{group_index:02d}_height_feasibility.png", dpi=150); plt.close(fig)

    fields = list(all_candidates[0])
    with (run_dir / "artifacts/pose_height_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(all_candidates)
    write_json(run_dir / "artifacts/pose_feasibility.json", {"poses": pose_summaries})
    write_json(run_dir / "artifacts/physical_groups.json", {"group_radius_m": PHYSICAL_GROUP_RADIUS_M, "groups": [{"group_index": index, "pose_indices": members, "frame_indices": [failures[item]["frame_index"] for item in members]} for index, members in enumerate(groups)]})
    unresolved = sum(item["feasible_candidate_count"] == 0 for item in pose_summaries)
    summary = {
        "schema_version": "cano_s10_local_floor_pose_feasibility_v1",
        "overall_status": "PASS_CANO_S10_LOCAL_FLOOR_POSE_FEASIBILITY_AUDIT_V1",
        "route_failure_poses": len(failures), "physical_groups": len(groups),
        "offset_minimum_m": float(OFFSETS_M[0]), "offset_maximum_m": float(OFFSETS_M[-1]), "offset_step_m": .05,
        "pose_height_candidates": len(all_candidates), "horizontal_rays": len(all_candidates) * 720,
        "vertical_rays": len(all_candidates) * 2, "unresolved_route_poses": unresolved,
        "all_route_poses_have_feasible_height": unresolved == 0,
        "thresholds": {"horizontal_clearance_m": .8, "sensor_above_floor_m": 1.0, "floor_tolerance_m": .05, "upward_clearance_m": .8},
        "pose_summaries": pose_summaries, "inference_frames": 0, "graph_updates": 0, "training_samples_consumed": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "duration_seconds": time.monotonic() - started,
        "interpretation": "Diagnostic local pose feasibility only; not a full trajectory or dynamic-navigation qualification."
    }
    write_json(run_dir / "metrics/summary.json", summary); print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__": raise SystemExit(main())
