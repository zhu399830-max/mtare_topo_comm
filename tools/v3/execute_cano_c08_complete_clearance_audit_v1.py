#!/usr/bin/env python3
"""Audit horizontal clearance at every sealed C08 causal-replay pose."""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.c08_causal_replay import frame_contract, merge_clearance_risk_segments
from mtare_topo.governance import load_json, write_json


WORLDS = ("S01_flat_tree_small_C08", "S06_3d_branch_medium_C08", "S10_3d_complex_C08")
MESH_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
TRAJECTORY_RUN = PROJECT_ROOT / "results/gate4_topology/gate4_20260813_cano_c08_trajectory_mesh_contract_v1_seed0"
EXPECTED_FRAMES = 4773
RAYS_PER_FRAME = 720
MINIMUM_CLEARANCE_M = 0.8


def scene_from_mesh(path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(path), enable_post_processing=False)
    if not mesh.has_vertices() or not mesh.has_triangles():
        raise RuntimeError(f"empty mesh: {path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def horizontal_clearance(scene: o3d.t.geometry.RaycastingScene, origin: np.ndarray) -> tuple[float, int, float]:
    azimuth_deg = np.arange(RAYS_PER_FRAME, dtype=np.float32) * 0.5
    azimuth_rad = np.radians(azimuth_deg)
    directions = np.stack((np.cos(azimuth_rad), np.sin(azimuth_rad), np.zeros(RAYS_PER_FRAME)), axis=1).astype(np.float32)
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float32), directions.shape)
    rays = np.concatenate((origins, directions), axis=1)
    hit = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float64)
    finite = np.isfinite(hit) & (hit >= 0.0)
    if not np.any(finite):
        raise RuntimeError(f"no finite nonnegative horizontal hit at origin {origin.tolist()}")
    masked = np.where(finite, hit, np.inf)
    index = int(np.argmin(masked))
    return float(masked[index]), int(np.count_nonzero(finite)), float(azimuth_deg[index])


def make_previews(records: list[dict[str, Any]], run_dir: Path, world: str) -> None:
    xyz = np.asarray([item["axis_xyz_m"] for item in records], dtype=np.float64)
    arcs = np.asarray([item["route_arc_m"] for item in records], dtype=np.float64)
    clearance = np.asarray([item["minimum_horizontal_clearance_m"] for item in records], dtype=np.float64)
    risk = clearance < MINIMUM_CLEARANCE_M

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)
    for axis, first, second, labels in (
        (axes[0], 0, 1, ("x (m)", "y (m)")),
        (axes[1], 0, 2, ("x (m)", "z (m)")),
    ):
        axis.plot(xyz[:, first], xyz[:, second], color="#9ca3af", linewidth=.45, label="audited trajectory")
        if np.any(risk):
            axis.scatter(xyz[risk, first], xyz[risk, second], color="#d62728", s=18, label="clearance < 0.8 m", zorder=3)
        axis.set_xlabel(labels[0]); axis.set_ylabel(labels[1]); axis.set_aspect("equal", adjustable="datalim")
        axis.legend(loc="best", fontsize=8)
    fig.suptitle(f"Gate 4 C08 complete horizontal-clearance audit — {world}")
    fig.savefig(run_dir / f"previews/{world}_complete_risk_map_xy_xz.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(13, 4.5), constrained_layout=True)
    axis.plot(arcs, clearance, color="#315a7d", linewidth=.7)
    axis.axhline(MINIMUM_CLEARANCE_M, color="#d62728", linestyle="--", linewidth=1.2, label="frozen 0.8 m threshold")
    if np.any(risk):
        axis.scatter(arcs[risk], clearance[risk], color="#d62728", s=12, zorder=3)
    axis.set_xlabel("route arc (m)"); axis.set_ylabel("minimum horizontal clearance (m)")
    axis.set_ylim(bottom=0); axis.grid(alpha=.2); axis.legend()
    axis.set_title(f"All {len(records)} audited frames — {world}")
    fig.savefig(run_dir / f"previews/{world}_clearance_along_route.png", dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); started = time.monotonic()
    all_records: list[dict[str, Any]] = []; all_segments = []; world_metrics = []
    for world_index, world in enumerate(WORLDS, 1):
        source = MESH_RUN / f"artifacts/meshes/{world}/primary"
        graph = load_json(source / "graph.json")
        fta = float(load_json(source / "geometry_parameters.json")["fta_distance_m"])
        trajectory = np.load(TRAJECTORY_RUN / f"artifacts/{world}_trajectory.npz")
        with (TRAJECTORY_RUN / f"artifacts/{world}_traversals.json").open("r", encoding="utf-8") as stream:
            traversals = json.load(stream)
        frames = frame_contract(
            xyz_m=trajectory["xyz_m"], tangent_world=trajectory["tangent_world"],
            route_arc_m=trajectory["route_arc_m"], traversals=traversals,
            graph=graph, fta_distance_m=fta,
        )
        scene = scene_from_mesh(source / "mesh.obj")
        records = []
        for index, frame in enumerate(frames):
            clearance, finite_hits, azimuth = horizontal_clearance(scene, np.asarray(frame["sensor_xyz_m"]))
            record = {
                "world": world, "frame_index": index, "route_arc_m": frame["route_arc_m"],
                "axis_xyz_m": np.asarray(frame["axis_xyz_m"], dtype=float).tolist(),
                "sensor_xyz_m": np.asarray(frame["sensor_xyz_m"], dtype=float).tolist(),
                "edge_id": frame["edge_id"], "tunnel_id": frame["tunnel_id"],
                "minimum_horizontal_clearance_m": clearance,
                "minimum_hit_azimuth_world_deg": azimuth,
                "finite_horizontal_hit_count": finite_hits,
                "clearance_passed": bool(clearance >= MINIMUM_CLEARANCE_M),
            }
            records.append(record); all_records.append(record)
            if (index + 1) % 500 == 0 or index + 1 == len(frames):
                print(json.dumps({"world": world, "world_index": world_index, "frame": index + 1, "frames": len(frames)}), flush=True)
        segments = merge_clearance_risk_segments(records)
        for segment in segments:
            segment["world"] = world
        all_segments.extend(segments)
        values = np.asarray([item["minimum_horizontal_clearance_m"] for item in records])
        risky = values < MINIMUM_CLEARANCE_M
        edge_counts = Counter(item["edge_id"] for item in records if not item["clearance_passed"])
        tunnel_counts = Counter(str(item["tunnel_id"]) for item in records if not item["clearance_passed"])
        metric = {
            "world": world, "frames": len(records), "horizontal_rays": len(records) * RAYS_PER_FRAME,
            "clearance_passed_frames": int(np.count_nonzero(~risky)),
            "clearance_failed_frames": int(np.count_nonzero(risky)),
            "clearance_pass_rate": float(np.mean(~risky)), "risk_segment_count": len(segments),
            "minimum_clearance_m": float(values.min()),
            "clearance_quantiles_m": {name: float(np.quantile(values, q)) for name, q in (("p01", .01), ("p05", .05), ("p50", .5), ("p95", .95))},
            "failed_edge_counts": dict(sorted(edge_counts.items())),
            "failed_tunnel_counts": dict(sorted(tunnel_counts.items())),
        }
        write_json(run_dir / f"metrics/{world}.json", metric)
        with (run_dir / f"artifacts/{world}_risk_segments.json").open("w", encoding="utf-8") as stream:
            json.dump(segments, stream, indent=2); stream.write("\n")
        make_previews(records, run_dir, world)
        world_metrics.append(metric)

    if len(all_records) != EXPECTED_FRAMES:
        raise RuntimeError(f"frame total mismatch: {len(all_records)} != {EXPECTED_FRAMES}")
    fields = ["world", "frame_index", "route_arc_m", "edge_id", "tunnel_id", "minimum_horizontal_clearance_m", "minimum_hit_azimuth_world_deg", "finite_horizontal_hit_count", "clearance_passed", "axis_x_m", "axis_y_m", "axis_z_m", "sensor_x_m", "sensor_y_m", "sensor_z_m"]
    with (run_dir / "artifacts/complete_frame_clearance.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for item in all_records:
            writer.writerow({
                **{key: item[key] for key in fields[:9]},
                **dict(zip(fields[9:12], item["axis_xyz_m"])),
                **dict(zip(fields[12:15], item["sensor_xyz_m"])),
            })
    write_json(run_dir / "artifacts/all_risk_segments.json", {"segments": all_segments})
    values = np.asarray([item["minimum_horizontal_clearance_m"] for item in all_records])
    summary = {
        "schema_version": "cano_c08_complete_clearance_audit_v1",
        "overall_status": "PASS_CANO_C08_COMPLETE_CLEARANCE_AUDIT_V1",
        "worlds": len(WORLDS), "frames_audited": len(all_records),
        "horizontal_rays": len(all_records) * RAYS_PER_FRAME,
        "clearance_threshold_m": MINIMUM_CLEARANCE_M,
        "clearance_passed_frames": int(np.count_nonzero(values >= MINIMUM_CLEARANCE_M)),
        "clearance_failed_frames": int(np.count_nonzero(values < MINIMUM_CLEARANCE_M)),
        "risk_segment_count": len(all_segments), "minimum_clearance_m": float(values.min()),
        "world_metrics": world_metrics, "inference_frames": 0, "graph_updates": 0,
        "training_samples_consumed": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
        "mtare_worlds_read": 0, "duration_seconds": time.monotonic() - started,
        "interpretation": "PASS means complete diagnostic evidence, not that all trajectory frames meet 0.8m clearance.",
    }
    write_json(run_dir / "metrics/summary.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
