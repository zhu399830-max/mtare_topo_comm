#!/usr/bin/env python3
"""Generate the approved C08 continuous sensor/teacher stream exactly once."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import zarr
from numcodecs import Blosc

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.c08_causal_replay import frame_contract
from mtare_topo.data.cano_phase2_dataset import cast_ranges, evaluate_frame, spline_arrays
from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG, lidar_local_directions, world_directions
from mtare_topo.governance import load_json, write_json
from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline


WORLDS = ("S01_flat_tree_small_C08", "S06_3d_branch_medium_C08", "S10_3d_complex_C08")
MESH_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
TEACHER_TRAJECTORY_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0"
QUALIFICATION_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260816_cano_c08_route_conditioned_support_corrective_v8r_seed0"
EXPECTED_FRAMES = 4773
FULL_SCAN_RAYS = 16 * 720


def scene_from_mesh(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty mesh: {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def create_shard(path: Path, frames: int) -> zarr.Group:
    group = zarr.open_group(str(path), mode="w")
    codec = Blosc(cname="zstd", clevel=5, shuffle=Blosc.BITSHUFFLE)
    group.create_dataset("range_m", shape=(frames, 16, 720), chunks=(16, 16, 720), dtype="f4", compressor=codec)
    group.create_dataset("valid_mask", shape=(frames, 16, 720), chunks=(32, 16, 720), dtype="u1", compressor=codec)
    group.create_dataset("axis_xyz_m", shape=(frames, 3), chunks=(1024, 3), dtype="f8", compressor=codec)
    group.create_dataset("teacher_axis_xyz_m", shape=(frames, 3), chunks=(1024, 3), dtype="f8", compressor=codec)
    group.create_dataset("sensor_xyz_m", shape=(frames, 3), chunks=(1024, 3), dtype="f8", compressor=codec)
    group.create_dataset("yaw_deg", shape=(frames,), chunks=(4096,), dtype="f8", compressor=codec)
    group.create_dataset("route_arc_m", shape=(frames,), chunks=(4096,), dtype="f8", compressor=codec)
    group.create_dataset("branch_count", shape=(frames,), chunks=(4096,), dtype="u1", compressor=codec)
    group.create_dataset("role_index", shape=(frames,), chunks=(4096,), dtype="u1", compressor=codec)
    return group


def preview(frames: list[dict], path: Path, world: str) -> None:
    xyz = np.asarray([item["axis_xyz_m"] for item in frames])
    roles = np.asarray([item["objective_role_index"] for item in frames])
    colors = np.asarray(["#5d7fa3", "#e17c32", "#c33c54"])
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, first, second, labels in (
        (axes[0], 0, 1, ("x (m)", "y (m)")),
        (axes[1], 0, 2, ("x (m)", "z (m)")),
    ):
        axis.plot(xyz[:, first], xyz[:, second], color="#bbbbbb", linewidth=.35)
        axis.scatter(xyz[:, first], xyz[:, second], c=colors[roles], s=3)
        axis.set_xlabel(labels[0]); axis.set_ylabel(labels[1]); axis.set_aspect("equal", adjustable="datalim")
    fig.suptitle(f"C08 causal replay objective roles — {world}\nblue interior, orange junction, red terminal")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    started = time.monotonic()
    sensor_root = run_dir / "artifacts/sensor"
    sensor_root.mkdir(parents=True, exist_ok=True)
    baseline = RangeExitBaseline()
    local_directions = lidar_local_directions()
    total_frames = total_primary = total_replay = total_clearance = total_los = 0
    world_metrics = []

    for world_index, world in enumerate(WORLDS, 1):
        source = MESH_RUN / f"artifacts/meshes/{world}/primary"
        graph = load_json(source / "graph.json")
        splines_document = load_json(source / "splines.json")
        splines = spline_arrays(splines_document)
        fta = float(load_json(source / "geometry_parameters.json")["fta_distance_m"])
        teacher_trajectory_path = TEACHER_TRAJECTORY_RUN / f"artifacts/{world}_trajectory.npz"
        qualified_trajectory_path = QUALIFICATION_RUN / f"artifacts/{world}_corrected_trajectory.npz"
        traversal_path = TEACHER_TRAJECTORY_RUN / f"artifacts/{world}_traversals.json"
        teacher_trajectory = np.load(teacher_trajectory_path)
        qualified_trajectory = np.load(qualified_trajectory_path)
        with traversal_path.open("r", encoding="utf-8") as stream:
            traversals = json.load(stream)
        frames = frame_contract(
            teacher_axis_xyz_m=teacher_trajectory["xyz_m"],
            graph_axis_xyz_m=qualified_trajectory["graph_axis_xyz_m"],
            sensor_xyz_m=qualified_trajectory["sensor_xyz_m"],
            tangent_world=teacher_trajectory["tangent_world"],
            route_arc_m=teacher_trajectory["route_arc_m"], traversals=traversals,
            graph=graph, fta_distance_m=fta,
        )
        shard = create_shard(sensor_root / f"{world}.zarr", len(frames))
        shard.attrs.update({
            "world": world, "student_fields": ["range_m", "valid_mask"],
            "evaluator_only_fields": ["teacher_axis_xyz_m", "axis_xyz_m", "sensor_xyz_m", "yaw_deg", "route_arc_m", "edge_id", "tunnel_id"],
            "teacher_only_fields": ["headings_robot_deg", "branch_count", "role_index"],
        })
        scene_a = scene_from_mesh(source / "mesh.obj")
        scene_b = scene_from_mesh(source / "mesh.obj")
        role_counts: Counter[str] = Counter()
        branch_counts: Counter[int] = Counter()
        b0_count_exact = 0
        min_clearance = float("inf")
        max_scene_difference = 0.0
        native_clearance_failed_frames = 0
        manifest_path = sensor_root / f"{world}_frames.jsonl"
        with manifest_path.open("w", encoding="utf-8") as manifest:
            for index, frame in enumerate(frames):
                evaluated = evaluate_frame(scene_a, frame, splines, o3d)
                directions = world_directions(local_directions, float(frame["yaw_deg"]))
                second_range, second_valid = cast_ranges(
                    scene_b, np.asarray(frame["sensor_xyz_m"]), directions, o3d
                )
                difference = float(np.max(np.abs(evaluated["range_m"] - second_range)))
                deterministic = bool(
                    np.array_equal(evaluated["valid_mask"], second_valid)
                    and difference <= 1e-6
                )
                finite_scan = bool(np.isfinite(evaluated["range_m"]).all())
                if not evaluated["teacher_eligible"] or not finite_scan or not deterministic:
                    failure = {
                        "world": world, "frame_index": index,
                        "route_arc_m": frame["route_arc_m"],
                        "eligible": bool(evaluated["eligible"]),
                        "teacher_eligible": bool(evaluated["teacher_eligible"]),
                        "finite_scan": finite_scan,
                        "deterministic": deterministic,
                        "scene_max_difference_m": difference,
                        "minimum_horizontal_clearance_m": evaluated["minimum_horizontal_clearance_m"],
                        "branch_los": evaluated["branch_los"],
                        "branch_count": evaluated["branch_count"],
                    }
                    write_json(run_dir / "metrics/sensor_failure.json", failure)
                    raise RuntimeError(f"C08 sensor/teacher contract failure: {failure}")
                predicted = baseline.predict(
                    evaluated["range_m"], evaluated["valid_mask"], ELEVATION_DEG
                )
                shard["range_m"][index] = evaluated["range_m"]
                shard["valid_mask"][index] = evaluated["valid_mask"]
                shard["axis_xyz_m"][index] = frame["axis_xyz_m"]
                shard["teacher_axis_xyz_m"][index] = frame["teacher_axis_xyz_m"]
                shard["sensor_xyz_m"][index] = frame["sensor_xyz_m"]
                shard["yaw_deg"][index] = frame["yaw_deg"]
                shard["route_arc_m"][index] = frame["route_arc_m"]
                shard["branch_count"][index] = evaluated["branch_count"]
                shard["role_index"][index] = frame["objective_role_index"]
                record = {
                    "frame_id": f"{world}_f{index:05d}", "world": world,
                    "frame_index": index, "route_arc_m": frame["route_arc_m"],
                    "yaw_deg": frame["yaw_deg"], "edge_id": frame["edge_id"],
                    "tunnel_id": frame["tunnel_id"], "objective_role": frame["objective_role"],
                    "objective_role_index": frame["objective_role_index"],
                    "junction_event_ids": frame["junction_event_ids"],
                    "terminal_event_ids": frame["terminal_event_ids"],
                    "headings_robot_deg": evaluated["headings_robot_deg"],
                    "branch_count": evaluated["branch_count"],
                    "minimum_horizontal_clearance_m": evaluated["minimum_horizontal_clearance_m"],
                    "branch_los": evaluated["branch_los"],
                    "native_horizontal_clearance_diagnostic_passed": evaluated["native_clearance_passed"],
                    "b0_headings_robot_deg": predicted["headings_robot_deg"],
                    "b0_branch_count": predicted["branch_count"],
                    "b0_threshold_m": predicted["threshold_m"],
                }
                manifest.write(json.dumps(record, separators=(",", ":")) + "\n")
                role_counts[frame["objective_role"]] += 1
                branch_counts[int(evaluated["branch_count"])] += 1
                b0_count_exact += int(predicted["branch_count"] == evaluated["branch_count"])
                min_clearance = min(min_clearance, float(evaluated["minimum_horizontal_clearance_m"]))
                native_clearance_failed_frames += int(not evaluated["native_clearance_passed"])
                max_scene_difference = max(max_scene_difference, difference)
                total_primary += FULL_SCAN_RAYS
                total_replay += FULL_SCAN_RAYS
                total_clearance += 720
                total_los += int(evaluated["branch_count"])
                if (index + 1) % 250 == 0 or index + 1 == len(frames):
                    print(json.dumps({"world": world, "world_index": world_index, "frame": index + 1, "frames": len(frames)}), flush=True)
        preview(frames, run_dir / f"previews/{world}_objective_role_trajectory.png", world)
        metric = {
            "world": world, "frames": len(frames), "role_counts": dict(role_counts),
            "branch_counts": {str(key): value for key, value in sorted(branch_counts.items())},
            "b0_branch_count_accuracy": b0_count_exact / len(frames),
            "minimum_horizontal_clearance_m": min_clearance,
            "native_horizontal_clearance_diagnostic_failed_frames": native_clearance_failed_frames,
            "dual_scene_max_difference_m": max_scene_difference,
            "primary_full_scan_rays": len(frames) * FULL_SCAN_RAYS,
            "replay_full_scan_rays": len(frames) * FULL_SCAN_RAYS,
            "primary_clearance_rays": len(frames) * 720,
            "primary_branch_los_rays": int(sum(key * value for key, value in branch_counts.items())),
        }
        write_json(run_dir / f"metrics/sensor_{world}.json", metric)
        world_metrics.append(metric)
        total_frames += len(frames)

    if total_frames != EXPECTED_FRAMES:
        raise RuntimeError(f"frame total mismatch: {total_frames} != {EXPECTED_FRAMES}")
    summary = {
        "schema_version": "cano_c08_causal_sensor_stage_v1",
        "overall_status": "PASS_C08_CAUSAL_SENSOR_STAGE_V1",
        "worlds": len(WORLDS), "frames": total_frames,
        "primary_full_scan_rays": total_primary, "replay_full_scan_rays": total_replay,
        "dual_scene_full_scan_rays": total_primary + total_replay,
        "auxiliary_primary_clearance_rays": total_clearance,
        "auxiliary_primary_branch_los_rays": total_los,
        "all_teacher_frames_eligible": True, "all_scans_finite": True,
        "all_branch_los_passed": True, "all_dual_scene_frames_equal": True,
        "world_metrics": world_metrics, "duration_seconds": time.monotonic() - started,
    }
    write_json(run_dir / "metrics/sensor_summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
