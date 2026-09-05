#!/usr/bin/env python3
"""Audit local lateral plus floor-following pose feasibility at S10 failures."""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.local_pose_feasibility import feasible_grid_components, physical_groups
from mtare_topo.governance import write_json


WORLD = "S10_3d_complex_C08"
MESH = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S10_3d_complex_C08/primary/mesh.obj"
FRAME_CSV = PROJECT_ROOT / "results/gate4_topology/gate4_20260813_cano_c08_complete_clearance_audit_v1_seed0/artifacts/complete_frame_clearance.csv"
XY_OFFSETS_M = np.round(np.arange(-1.0, 1.0 + 1e-9, .1), 8)
HORIZONTAL_CLEARANCE_M = .8
SENSOR_ABOVE_FLOOR_M = 1.0
FLOOR_TOLERANCE_M = .05
UPWARD_CLEARANCE_M = .8
FLOOR_QUERY_ABOVE_ORIGINAL_SENSOR_M = 2.0
MAX_VERTICAL_QUERY_M = 8.0
PHYSICAL_GROUP_RADIUS_M = 2.0
MIN_NONISOLATED_COMPONENT_CELLS = 3


def scene_from_mesh() -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(MESH), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError("S10 mesh is empty")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def cast(scene: o3d.t.geometry.RaycastingScene, origins: np.ndarray, directions: np.ndarray) -> np.ndarray:
    rays = np.concatenate((origins.astype(np.float32), directions.astype(np.float32)), axis=1)
    return scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)


def evaluate_xy(scene: o3d.t.geometry.RaycastingScene, original: np.ndarray, dx: float, dy: float) -> dict:
    query = np.asarray([original[0] + dx, original[1] + dy, original[2] + FLOOR_QUERY_ABOVE_ORIGINAL_SENSOR_M])
    query_down = float(cast(scene, query.reshape(1, 3), np.asarray([[0, 0, -1]], dtype=float))[0])
    if not np.isfinite(query_down) or query_down < 0 or query_down > MAX_VERTICAL_QUERY_M:
        return {
            "candidate_sensor_x_m": float(query[0]), "candidate_sensor_y_m": float(query[1]),
            "candidate_sensor_z_m": float("nan"), "floor_query_downward_distance_m": query_down,
            "horizontal_clearance_m": float("nan"), "candidate_downward_distance_m": float("nan"),
            "candidate_upward_distance_m": float("nan"), "vertical_evidence_complete": False,
            "floor_following_passed": False, "feasible": False,
        }
    candidate = np.asarray([query[0], query[1], query[2] - query_down + SENSOR_ABOVE_FLOOR_M])
    azimuth = np.radians(np.arange(720, dtype=np.float32) * .5)
    horizontal_directions = np.stack((np.cos(azimuth), np.sin(azimuth), np.zeros(720)), axis=1)
    horizontal = cast(scene, np.broadcast_to(candidate, horizontal_directions.shape), horizontal_directions)
    finite = np.isfinite(horizontal) & (horizontal >= 0)
    horizontal_min = float(np.min(horizontal[finite])) if np.any(finite) else float("nan")
    vertical = cast(scene, np.broadcast_to(candidate, (2, 3)), np.asarray([[0, 0, -1], [0, 0, 1]], dtype=float))
    down, up = (float(value) if np.isfinite(value) and value >= 0 else float("inf") for value in vertical)
    complete = bool(down <= MAX_VERTICAL_QUERY_M and up <= MAX_VERTICAL_QUERY_M and np.isfinite(horizontal_min))
    floor_following = bool(complete and abs(down - SENSOR_ABOVE_FLOOR_M) <= FLOOR_TOLERANCE_M)
    feasible = bool(floor_following and horizontal_min >= HORIZONTAL_CLEARANCE_M and up >= UPWARD_CLEARANCE_M)
    return {
        "candidate_sensor_x_m": float(candidate[0]), "candidate_sensor_y_m": float(candidate[1]),
        "candidate_sensor_z_m": float(candidate[2]), "floor_query_downward_distance_m": query_down,
        "horizontal_clearance_m": horizontal_min, "candidate_downward_distance_m": down,
        "candidate_upward_distance_m": up, "vertical_evidence_complete": complete,
        "floor_following_passed": floor_following, "feasible": feasible,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.monotonic()
    with FRAME_CSV.open("r", encoding="utf-8") as stream:
        failures = [row for row in csv.DictReader(stream) if row["world"] == WORLD and row["clearance_passed"] == "False"]
    if len(failures) != 9:
        raise RuntimeError(f"expected 9 frozen S10 failure records, got {len(failures)}")
    for row in failures:
        row["axis_xyz_m"] = [float(row[key]) for key in ("axis_x_m", "axis_y_m", "axis_z_m")]
        row["sensor_xyz_m"] = [float(row[key]) for key in ("sensor_x_m", "sensor_y_m", "sensor_z_m")]
        row["frame_index"] = int(row["frame_index"])
        row["route_arc_m"] = float(row["route_arc_m"])
    groups = physical_groups(failures, PHYSICAL_GROUP_RADIUS_M)
    scene = scene_from_mesh()
    all_candidates: list[dict] = []
    pose_summaries: list[dict] = []
    per_pose: list[list[dict]] = []
    for pose_index, failure in enumerate(failures):
        original = np.asarray(failure["sensor_xyz_m"], dtype=np.float64)
        records = []
        for x_index, dx in enumerate(XY_OFFSETS_M):
            for y_index, dy in enumerate(XY_OFFSETS_M):
                result = evaluate_xy(scene, original, float(dx), float(dy))
                record = {
                    "pose_index": pose_index, "frame_index": failure["frame_index"],
                    "route_arc_m": failure["route_arc_m"], "edge_id": failure["edge_id"],
                    "tunnel_id": int(failure["tunnel_id"]), "x_index": x_index, "y_index": y_index,
                    "x_offset_m": float(dx), "y_offset_m": float(dy), **result,
                }
                records.append(record)
                all_candidates.append(record)
        components = feasible_grid_components(records)
        component_summaries = []
        for component_index, members in enumerate(components):
            cells = [records[index] for index in members]
            component_summaries.append({
                "component_index": component_index, "cell_count": len(cells),
                "x_offset_minimum_m": min(cell["x_offset_m"] for cell in cells),
                "x_offset_maximum_m": max(cell["x_offset_m"] for cell in cells),
                "y_offset_minimum_m": min(cell["y_offset_m"] for cell in cells),
                "y_offset_maximum_m": max(cell["y_offset_m"] for cell in cells),
                "maximum_horizontal_clearance_m": max(cell["horizontal_clearance_m"] for cell in cells),
            })
        robust_members = {member for component in components if len(component) >= MIN_NONISOLATED_COMPONENT_CELLS for member in component}
        recommended = min(
            (record for index, record in enumerate(records) if index in robust_members),
            key=lambda record: (math.hypot(record["x_offset_m"], record["y_offset_m"]), -record["horizontal_clearance_m"], record["x_offset_m"], record["y_offset_m"]),
            default=None,
        )
        pose_summaries.append({
            "pose_index": pose_index, "frame_index": failure["frame_index"], "route_arc_m": failure["route_arc_m"],
            "edge_id": failure["edge_id"], "tunnel_id": int(failure["tunnel_id"]),
            "original_sensor_xyz_m": failure["sensor_xyz_m"],
            "original_horizontal_clearance_m": float(failure["minimum_horizontal_clearance_m"]),
            "feasible_cell_count": sum(record["feasible"] for record in records),
            "nonisolated_feasible_cell_count": len(robust_members), "components": component_summaries,
            "recommended_candidate": recommended,
        })
        per_pose.append(records)
        print(json.dumps({"pose": pose_index + 1, "of": 9, "frame": failure["frame_index"], "feasible_cells": pose_summaries[-1]["feasible_cell_count"], "nonisolated_cells": len(robust_members)}), flush=True)

    group_summaries = []
    for group_index, members in enumerate(groups):
        columns = min(2, len(members)); rows_count = math.ceil(len(members) / columns)
        fig, axes = plt.subplots(rows_count, columns, figsize=(7 * columns, 6 * rows_count), squeeze=False, constrained_layout=True)
        group_pose_results = []
        for plot_index, member in enumerate(members):
            axis = axes[plot_index // columns][plot_index % columns]
            grid = np.asarray([record["horizontal_clearance_m"] for record in per_pose[member]], dtype=float).reshape(21, 21)
            feasible_grid = np.asarray([record["feasible"] for record in per_pose[member]], dtype=bool).reshape(21, 21)
            image = axis.imshow(grid.T, origin="lower", extent=[-1.05, 1.05, -1.05, 1.05], vmin=0, vmax=max(1.5, float(np.nanmax(grid))), cmap="viridis", aspect="equal")
            yy, xx = np.where(feasible_grid.T)
            axis.scatter(XY_OFFSETS_M[xx], XY_OFFSETS_M[yy], marker="s", s=12, facecolors="none", edgecolors="white", linewidths=.6, label="feasible")
            axis.scatter([0], [0], c="red", marker="x", s=70, label="frozen pose")
            recommended = pose_summaries[member]["recommended_candidate"]
            if recommended:
                axis.scatter([recommended["x_offset_m"]], [recommended["y_offset_m"]], c="cyan", marker="*", s=110, label="nearest robust")
            axis.set_title(f"frame {failures[member]['frame_index']} {failures[member]['edge_id']}\nfeasible {pose_summaries[member]['feasible_cell_count']}/441")
            axis.set_xlabel("global x offset (m)"); axis.set_ylabel("global y offset (m)"); axis.legend(fontsize=7)
            fig.colorbar(image, ax=axis, label="horizontal clearance (m)")
            group_pose_results.append(bool(recommended))
        for unused in range(len(members), rows_count * columns):
            axes[unused // columns][unused % columns].axis("off")
        fig.suptitle(f"Gate 4 S10 lateral floor-pose feasibility — physical group {group_index}")
        fig.savefig(run_dir / f"previews/physical_group_{group_index:02d}_lateral_feasibility.png", dpi=150)
        plt.close(fig)
        group_summaries.append({
            "group_index": group_index, "pose_indices": members,
            "frame_indices": [failures[index]["frame_index"] for index in members],
            "all_route_poses_have_nonisolated_feasible_component": all(group_pose_results),
        })

    fields = list(all_candidates[0])
    with (run_dir / "artifacts/pose_xy_candidates.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(all_candidates)
    write_json(run_dir / "artifacts/pose_lateral_feasibility.json", {"poses": pose_summaries})
    write_json(run_dir / "artifacts/physical_groups.json", {"group_radius_m": PHYSICAL_GROUP_RADIUS_M, "groups": group_summaries})
    unresolved = sum(item["recommended_candidate"] is None for item in pose_summaries)
    summary = {
        "schema_version": "cano_s10_local_lateral_pose_feasibility_v1",
        "overall_status": "PASS_CANO_S10_LOCAL_LATERAL_POSE_FEASIBILITY_AUDIT_V1",
        "route_failure_poses": 9, "physical_groups": len(groups), "xy_grid_side": 21,
        "x_offset_minimum_m": -1.0, "x_offset_maximum_m": 1.0,
        "y_offset_minimum_m": -1.0, "y_offset_maximum_m": 1.0, "xy_step_m": .1,
        "pose_xy_candidates": len(all_candidates), "horizontal_rays": len(all_candidates) * 720,
        "vertical_rays": len(all_candidates) * 3, "unresolved_route_poses": unresolved,
        "all_route_poses_have_nonisolated_feasible_component": unresolved == 0,
        "minimum_nonisolated_component_cells": MIN_NONISOLATED_COMPONENT_CELLS,
        "thresholds": {"horizontal_clearance_m": .8, "sensor_above_floor_m": 1.0, "floor_tolerance_m": .05, "upward_clearance_m": .8},
        "pose_summaries": pose_summaries, "group_summaries": group_summaries,
        "inference_frames": 0, "graph_updates": 0, "training_samples_consumed": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
        "duration_seconds": time.monotonic() - started,
        "interpretation": "Diagnostic local pose feasibility only; not a complete trajectory or dynamic-navigation qualification.",
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
