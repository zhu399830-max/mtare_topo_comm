#!/usr/bin/env python3
"""Prepare fixed poses, spline labels, USD, and CPU rays for the Cano smoke."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import (
    AZIMUTH_COLUMNS,
    ELEVATION_DEG,
    MAX_RANGE_M,
    lidar_local_directions,
    load_json,
    select_role_stratified_poses,
    sha256_file,
    world_directions,
    write_obj_usda,
)
from mtare_topo.governance import write_json


EXPECTED_WORLD_HASHES = {
    "mesh.obj": "f0ae7777d9d1dfe489075240692021bb88cc82fd8736ca1537def44b687e9f74",
    "graph.json": "0d325135d25555d513596ca73f1ceb5546ca6097ec75b47d298f174af9a0e077",
    "splines.json": "cc16d3b5b921bc1a4a44655eda84f6b35cfa84b814e5c7f6f4e8573321c74e5a",
    "fta_dist.txt": "e9146bf9372670070fe2957bc506be4501bb10a9942ab68b8800f7b3999dadf9",
    "metadata.json": "c520ec3a6af2f292bf5e2cc8f2c2b1275fa2bdccf0fe1ced751f9bf994d7d4d6",
}


def _raycast(
    scene: o3d.t.geometry.RaycastingScene,
    origin: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.broadcast_to(origin.astype(np.float32), directions.shape)
    rays = np.concatenate((origins, directions.astype(np.float32)), axis=-1)
    hits = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))
    ranges = hits["t_hit"].numpy().reshape(directions.shape[:-1])
    valid = np.isfinite(ranges) & (ranges <= MAX_RANGE_M)
    ranges = np.where(valid, ranges, MAX_RANGE_M).astype(np.float32)
    return ranges, valid.astype(np.uint8)


def _representative_branch_points(pose: dict) -> list[np.ndarray]:
    center = np.asarray(pose["axis_xyz_m"], dtype=np.float64)
    raw = pose["label"]["raw_intersections"]
    points: list[np.ndarray] = []
    for heading in pose["label"]["headings_world_deg"]:
        selected = min(
            raw,
            key=lambda item: abs(
                (float(item["heading_world_deg"]) - float(heading) + 180.0) % 360.0
                - 180.0
            ),
        )
        point = np.asarray(selected["xyz"], dtype=np.float64)
        if np.linalg.norm(point - center) > 1e-6:
            points.append(point)
    return points


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sensor-config", required=True, type=Path)
    args = parser.parse_args()
    world_dir = args.world_dir.resolve()
    run_dir = args.run_dir.resolve()
    sensor_config = args.sensor_config.resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    observed_hashes = {name: sha256_file(world_dir / name) for name in EXPECTED_WORLD_HASHES}
    if observed_hashes != EXPECTED_WORLD_HASHES:
        raise RuntimeError(f"audited seed-0 world hash mismatch: {observed_hashes}")

    graph = load_json(world_dir / "graph.json")
    splines = load_json(world_dir / "splines.json")
    fta_distance_m = float((world_dir / "fta_dist.txt").read_text(encoding="utf-8"))
    poses = select_role_stratified_poses(graph, splines, fta_distance_m)
    stage_path = run_dir / "artifacts/isaac_stage.usda"
    stage_stats = write_obj_usda(world_dir / "mesh.obj", stage_path)

    legacy_mesh = o3d.io.read_triangle_mesh(str(world_dir / "mesh.obj"), enable_post_processing=False)
    if legacy_mesh.is_empty() or len(legacy_mesh.triangles) == 0:
        raise RuntimeError("Open3D could not read the original mesh")
    tensor_mesh = o3d.t.geometry.TriangleMesh.from_legacy(legacy_mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tensor_mesh)

    cpu_dir = run_dir / "artifacts/cpu_reference"
    cpu_dir.mkdir(exist_ok=False)
    local = lidar_local_directions()
    pose_audits = []
    preparation_passed = True
    for pose in poses:
        origin = np.asarray(pose["sensor_xyz_m"], dtype=np.float32)
        directions = world_directions(local, float(pose["yaw_deg"]))
        ranges, valid = _raycast(scene, origin, directions)

        horizontal_azimuth = np.radians(np.arange(720, dtype=np.float32) * 0.5)
        horizontal = np.stack(
            (np.cos(horizontal_azimuth), np.sin(horizontal_azimuth), np.zeros(720)),
            axis=-1,
        ).astype(np.float32)
        clearance_ranges, clearance_valid = _raycast(scene, origin, horizontal)
        finite_clearance = clearance_ranges[clearance_valid.astype(bool)]
        minimum_clearance = float(np.min(finite_clearance)) if finite_clearance.size else MAX_RANGE_M

        branch_checks = []
        axis_center = np.asarray(pose["axis_xyz_m"], dtype=np.float64)
        for point in _representative_branch_points(pose):
            target = point.copy()
            # Structural targets live on the axis; preserve their slope while
            # starting at the actual LiDAR height.
            vector = target - origin
            target_distance = float(np.linalg.norm(vector))
            branch_range, branch_valid = _raycast(
                scene, origin, (vector / target_distance).reshape(1, 3).astype(np.float32)
            )
            observed = float(branch_range[0])
            clear = bool((not branch_valid[0]) or observed >= target_distance - 0.25)
            branch_checks.append(
                {
                    "target_xyz_m": target.tolist(),
                    "axis_center_xyz_m": axis_center.tolist(),
                    "target_distance_m": target_distance,
                    "first_hit_m": observed,
                    "clear_to_target_minus_0_25m": clear,
                }
            )

        clearance_passed = minimum_clearance >= 0.8
        branches_passed = all(item["clear_to_target_minus_0_25m"] for item in branch_checks)
        sample_passed = clearance_passed and branches_passed
        preparation_passed = preparation_passed and sample_passed
        np.savez_compressed(
            cpu_dir / f"{pose['sample_id']}.npz",
            range_m=ranges,
            valid_mask=valid,
            local_directions=local,
            sensor_xyz_m=origin,
            yaw_deg=np.asarray(float(pose["yaw_deg"]), dtype=np.float32),
            label_720=np.asarray(pose["label"]["label_720"], dtype=np.float32),
            label_headings_robot_deg=np.asarray(
                pose["label"]["headings_robot_deg"], dtype=np.float32
            ),
        )
        pose_audits.append(
            {
                "sample_id": pose["sample_id"],
                "role": pose["role"],
                "minimum_horizontal_clearance_m": minimum_clearance,
                "clearance_passed": clearance_passed,
                "branch_visibility": branch_checks,
                "branch_visibility_passed": branches_passed,
                "cpu_valid_ratio": float(np.mean(valid)),
                "passed": sample_passed,
            }
        )

    manifest = {
        "schema_version": "cano_lidar_pose_manifest_v1",
        "role": "DIAGNOSTIC_SENSOR_SMOKE_NOT_A_FORMAL_DATASET",
        "world_id": "cano_audited_adapter_seed0_world_000",
        "world_hashes": observed_hashes,
        "coordinate_frame": "cano_world_z_up_m",
        "fta_distance_m": fta_distance_m,
        "lidar_height_above_floor_m": 1.0,
        "sensor_position_rule": "sensor_z = spline_axis_z + fta_distance_m + 1.0 m",
        "label_rule": "all Cano spline intersections with a 5 m 3D sphere; 360-degree 720-bin max-composed circular Gaussians with sigma 3 degrees",
        "selection_rule": "fixed role stratification: 2 interior poses per tunnel, center+1m per degree-3 junction, 1.5m+3m inward per terminal",
        "counts": {
            "worlds": 1,
            "poses": len(poses),
            "tunnel_interior": sum(p["role"] == "tunnel_interior" for p in poses),
            "junction_transition": sum(p["role"] == "junction_transition" for p in poses),
            "terminal_approach": sum(p["role"] == "terminal_approach" for p in poses),
            "formal_dataset_samples": 0,
            "training_samples": 0,
        },
        "sensor": {
            "model": "frozen VLP16-like diagnostic profile",
            "elevation_deg": ELEVATION_DEG.tolist(),
            "azimuth_columns": AZIMUTH_COLUMNS,
            "azimuth_resolution_deg": 0.5,
            "near_range_m": 0.3,
            "far_range_m": MAX_RANGE_M,
            "sensor_config_path": str(sensor_config),
            "sensor_config_sha256": sha256_file(sensor_config),
        },
        "poses": poses,
    }
    write_json(run_dir / "artifacts/pose_manifest.json", manifest)
    write_json(
        run_dir / "metrics/preparation.json",
        {
            "schema_version": "cano_lidar_smoke_preparation_v1",
            "overall_status": "PASS" if preparation_passed else "FAIL",
            "acceptance": {
                "minimum_horizontal_clearance_m": 0.8,
                "branch_line_of_sight_margin_m": 0.25,
            },
            "stage": {
                **stage_stats,
                "path": str(stage_path),
                "sha256": sha256_file(stage_path),
            },
            "pose_audits": pose_audits,
        },
    )
    print(json.dumps({"passed": preparation_passed, "poses": len(poses)}, indent=2))
    return 0 if preparation_passed else 2


if __name__ == "__main__":
    raise SystemExit(main())
