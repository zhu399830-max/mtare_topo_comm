#!/usr/bin/env python3
"""Execute the frozen one-world Cano CPU raycast diagnostic contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import (
    AZIMUTH_COLUMNS,
    ELEVATION_DEG,
    MAX_RANGE_M,
    NEAR_RANGE_M,
    lidar_local_directions,
    load_json,
    select_role_stratified_poses,
    sha256_file,
    world_directions,
)
from mtare_topo.governance import write_json


EXPECTED_WORLD_HASHES = {
    "mesh.obj": "f0ae7777d9d1dfe489075240692021bb88cc82fd8736ca1537def44b687e9f74",
    "graph.json": "0d325135d25555d513596ca73f1ceb5546ca6097ec75b47d298f174af9a0e077",
    "splines.json": "cc16d3b5b921bc1a4a44655eda84f6b35cfa84b814e5c7f6f4e8573321c74e5a",
    "fta_dist.txt": "e9146bf9372670070fe2957bc506be4501bb10a9942ab68b8800f7b3999dadf9",
    "metadata.json": "c520ec3a6af2f292bf5e2cc8f2c2b1275fa2bdccf0fe1ced751f9bf994d7d4d6",
}
EXPECTED_SENSOR_SHA256 = "d9b45461b4f113c25fd5a2226bef50dd83d0fe2e3b1407cd93da9ff7798688fe"
EXPECTED_ROLES = {
    "tunnel_interior": 8,
    "junction_transition": 8,
    "terminal_approach": 8,
}


def _array_hash(arrays: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        array = np.ascontiguousarray(arrays[name])
        digest.update(name.encode("utf-8"))
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _build_scene(mesh: o3d.geometry.TriangleMesh) -> o3d.t.geometry.RaycastingScene:
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _cast(
    scene: o3d.t.geometry.RaycastingScene,
    origin: np.ndarray,
    directions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    origins = np.broadcast_to(np.asarray(origin, dtype=np.float32), directions.shape)
    rays = np.concatenate((origins, np.asarray(directions, dtype=np.float32)), axis=-1)
    raw = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"].numpy()
    raw = raw.reshape(directions.shape[:-1])
    valid = np.isfinite(raw) & (raw >= NEAR_RANGE_M) & (raw <= MAX_RANGE_M)
    ranges = np.where(valid, raw, MAX_RANGE_M).astype(np.float32)
    return ranges, valid.astype(np.uint8)


def _analytic_control() -> dict[str, Any]:
    box = o3d.geometry.TriangleMesh.create_box(width=2.0, height=4.0, depth=6.0)
    box.translate((-1.0, -2.0, -3.0))
    scene = _build_scene(box)
    directions = np.asarray(
        [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]],
        dtype=np.float32,
    )
    expected = np.asarray([1, 1, 2, 2, 3, 3], dtype=np.float32)
    origins = np.zeros_like(directions)
    rays = np.concatenate((origins, directions), axis=1)
    observed = scene.cast_rays(o3d.core.Tensor(rays))["t_hit"].numpy().astype(np.float32)
    errors = np.abs(observed - expected)
    passed = bool(np.all(np.isfinite(observed)) and np.max(errors) <= 1e-5)
    return {
        "geometry": "axis-aligned closed box x=[-1,1], y=[-2,2], z=[-3,3]; rays start at origin",
        "directions_xyz": directions.tolist(),
        "expected_distance_m": expected.tolist(),
        "observed_distance_m": observed.tolist(),
        "absolute_error_m": errors.tolist(),
        "maximum_absolute_error_m": float(np.max(errors)),
        "threshold_m": 1e-5,
        "passed": passed,
    }


def _representative_branch_points(pose: dict[str, Any]) -> list[np.ndarray]:
    raw = pose["label"]["raw_intersections"]
    points = []
    for heading in pose["label"]["headings_world_deg"]:
        selected = min(
            raw,
            key=lambda item: abs(
                (float(item["heading_world_deg"]) - float(heading) + 180.0) % 360.0
                - 180.0
            ),
        )
        points.append(np.asarray(selected["xyz"], dtype=np.float64))
    return points


def _horizontal_directions() -> np.ndarray:
    azimuth = np.radians(np.arange(AZIMUTH_COLUMNS, dtype=np.float32) * 0.5)
    return np.stack(
        (np.cos(azimuth), np.sin(azimuth), np.zeros(AZIMUTH_COLUMNS)), axis=-1
    ).astype(np.float32)


def _load_reference(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _max_valid_difference(
    first_range: np.ndarray,
    first_valid: np.ndarray,
    second_range: np.ndarray,
    second_valid: np.ndarray,
) -> tuple[bool, float]:
    masks_equal = bool(np.array_equal(first_valid, second_valid))
    common = first_valid.astype(bool) & second_valid.astype(bool)
    difference = (
        float(np.max(np.abs(first_range[common] - second_range[common])))
        if np.any(common)
        else 0.0
    )
    return masks_equal, difference


def _plot_world(
    mesh: o3d.geometry.TriangleMesh,
    spline_document: dict[str, Any],
    poses: list[dict[str, Any]],
    destination: Path,
) -> None:
    vertices = np.asarray(mesh.vertices)
    fig, axis = plt.subplots(figsize=(13, 11), constrained_layout=True)
    axis.scatter(vertices[:, 0], vertices[:, 1], s=0.12, c="#8b8b8b", alpha=0.18, rasterized=True)
    for tunnel in spline_document["tunnels"]:
        points = np.asarray(tunnel["points"], dtype=np.float64)
        axis.plot(points[:, 0], points[:, 1], color="#202020", linewidth=1.15, alpha=0.9)
    colors = {
        "tunnel_interior": "#0072B2",
        "junction_transition": "#D55E00",
        "terminal_approach": "#009E73",
    }
    for role, color in colors.items():
        selected = [pose for pose in poses if pose["role"] == role]
        xy = np.asarray([pose["sensor_xyz_m"][:2] for pose in selected])
        yaw = np.radians([pose["yaw_deg"] for pose in selected])
        axis.scatter(xy[:, 0], xy[:, 1], s=35, color=color, label=f"{role} (n={len(selected)})", zorder=4)
        axis.quiver(xy[:, 0], xy[:, 1], np.cos(yaw), np.sin(yaw), color=color, scale=28, width=0.003, zorder=5)
        for point, pose in zip(xy, selected):
            axis.annotate(pose["sample_id"], point, xytext=(3, 3), textcoords="offset points", fontsize=5.5)
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlabel("world x (m)")
    axis.set_ylabel("world y (m)")
    axis.set_title("Cano seed-0 full world: frozen mesh footprint, spline axes, and all 24 diagnostic poses")
    axis.legend(loc="best", fontsize=8)
    axis.grid(True, linewidth=0.25, alpha=0.35)
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def _plot_contact_sheet(
    pose_outputs: list[dict[str, Any]], destination: Path
) -> None:
    fig, axes = plt.subplots(6, 4, figsize=(20, 20), sharex=True, sharey=True, constrained_layout=True)
    image = None
    role_colors = {
        "tunnel_interior": "#0072B2",
        "junction_transition": "#D55E00",
        "terminal_approach": "#009E73",
    }
    for axis, item in zip(axes.flat, pose_outputs):
        image = axis.imshow(
            item["range_m"],
            origin="lower",
            aspect="auto",
            interpolation="nearest",
            extent=(0.0, 360.0, float(ELEVATION_DEG[0]), float(ELEVATION_DEG[-1])),
            vmin=0.0,
            vmax=MAX_RANGE_M,
            cmap="viridis",
        )
        for heading in item["headings_robot_deg"]:
            axis.axvline(float(heading), color="#F0E442", linewidth=1.0, alpha=0.95)
        axis.set_title(
            f"{item['sample_id']}\n{item['role']} | valid={item['valid_ratio']:.4f}",
            fontsize=8,
            color=role_colors[item["role"]],
        )
        axis.set_xlim(0.0, 360.0)
        axis.set_ylim(float(ELEVATION_DEG[0]), float(ELEVATION_DEG[-1]))
    for axis in axes[-1, :]:
        axis.set_xlabel("robot-frame azimuth (deg)")
    for axis in axes[:, 0]:
        axis.set_ylabel("elevation (deg)")
    assert image is not None
    fig.colorbar(image, ax=axes, label="first-return range (m), fixed scale 0–50 m", shrink=0.72)
    fig.suptitle(
        "All 24 frozen diagnostic scans; yellow lines are spline-oracle outgoing headings (teacher only)",
        fontsize=14,
    )
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _execute(world_dir: Path, reference_dir: Path, run_dir: Path, sensor_config: Path) -> dict[str, Any]:
    world_hashes = {name: sha256_file(world_dir / name) for name in EXPECTED_WORLD_HASHES}
    if world_hashes != EXPECTED_WORLD_HASHES:
        raise RuntimeError(f"frozen world identity mismatch: {world_hashes}")
    if sha256_file(sensor_config) != EXPECTED_SENSOR_SHA256:
        raise RuntimeError("frozen sensor config identity mismatch")

    analytic = _analytic_control()
    write_json(run_dir / "metrics/analytic_control.json", analytic)
    if not analytic["passed"]:
        raise RuntimeError("analytic ray-distance control failed")

    graph = load_json(world_dir / "graph.json")
    spline_document = load_json(world_dir / "splines.json")
    fta_distance_m = float((world_dir / "fta_dist.txt").read_text(encoding="utf-8"))
    poses = select_role_stratified_poses(graph, spline_document, fta_distance_m)
    role_counts = {role: sum(pose["role"] == role for pose in poses) for role in EXPECTED_ROLES}
    if len(poses) != 24 or role_counts != EXPECTED_ROLES:
        raise RuntimeError(f"pose contract drift: poses={len(poses)}, roles={role_counts}")

    mesh = o3d.io.read_triangle_mesh(str(world_dir / "mesh.obj"), enable_post_processing=False)
    if mesh.is_empty() or len(mesh.triangles) == 0:
        raise RuntimeError("Open3D could not read the frozen mesh")
    scene_first = _build_scene(mesh)
    scene_second = _build_scene(mesh)
    local = lidar_local_directions()
    if local.shape != (16, 720, 3) or local.dtype != np.float32:
        raise RuntimeError(f"ray-grid contract drift: {local.shape}, {local.dtype}")

    scan_dir = run_dir / "artifacts/scans"
    scan_dir.mkdir(parents=True, exist_ok=False)
    horizontal = _horizontal_directions()
    per_pose = []
    plot_records = []
    all_passed = True
    for pose in poses:
        sample_id = pose["sample_id"]
        origin = np.asarray(pose["sensor_xyz_m"], dtype=np.float32)
        directions = world_directions(local, float(pose["yaw_deg"]))
        first_range, first_valid = _cast(scene_first, origin, directions)
        second_range, second_valid = _cast(scene_second, origin, directions)
        reference_path = reference_dir / f"{sample_id}.npz"
        if not reference_path.is_file():
            raise FileNotFoundError(f"missing frozen regression reference: {reference_path}")
        reference = _load_reference(reference_path)

        shape_passed = first_range.shape == (16, 720) and first_valid.shape == (16, 720)
        dtype_passed = first_range.dtype == np.float32 and first_valid.dtype == np.uint8
        valid_boolean_passed = bool(np.all((first_valid == 0) | (first_valid == 1)))
        valid_ranges = first_range[first_valid.astype(bool)]
        invalid_ranges = first_range[~first_valid.astype(bool)]
        range_passed = bool(
            valid_ranges.size
            and np.all(np.isfinite(valid_ranges))
            and np.all(valid_ranges >= NEAR_RANGE_M)
            and np.all(valid_ranges <= MAX_RANGE_M)
            and np.all(invalid_ranges == MAX_RANGE_M)
        )
        replay_masks_equal, replay_max_diff = _max_valid_difference(
            first_range, first_valid, second_range, second_valid
        )
        replay_passed = replay_masks_equal and replay_max_diff <= 1e-6
        reference_masks_equal, reference_max_diff = _max_valid_difference(
            first_range,
            first_valid,
            np.asarray(reference["range_m"], dtype=np.float32),
            np.asarray(reference["valid_mask"], dtype=np.uint8),
        )
        reference_passed = reference_masks_equal and reference_max_diff <= 1e-5

        clearance_range, clearance_valid = _cast(scene_first, origin, horizontal)
        clearance_hits = clearance_range[clearance_valid.astype(bool)]
        minimum_clearance = float(np.min(clearance_hits)) if clearance_hits.size else MAX_RANGE_M
        clearance_passed = minimum_clearance >= 0.8

        branch_checks = []
        for target in _representative_branch_points(pose):
            vector = target - origin
            target_distance = float(np.linalg.norm(vector))
            branch_range, branch_valid = _cast(
                scene_first, origin, (vector / target_distance).reshape(1, 3).astype(np.float32)
            )
            observed = float(branch_range[0])
            clear = bool((not branch_valid[0]) or observed >= target_distance - 0.25)
            branch_checks.append(
                {
                    "target_xyz_m": target.tolist(),
                    "target_distance_m": target_distance,
                    "first_hit_m": observed,
                    "first_hit_valid": bool(branch_valid[0]),
                    "clear_to_target_minus_0_25m": clear,
                }
            )
        branch_passed = bool(branch_checks) and all(
            item["clear_to_target_minus_0_25m"] for item in branch_checks
        )

        arrays = {
            "range_m": first_range,
            "valid_mask": first_valid,
            "local_directions": local,
            "sensor_xyz_m": origin,
            "yaw_deg": np.asarray(float(pose["yaw_deg"]), dtype=np.float32),
            "label_720": np.asarray(pose["label"]["label_720"], dtype=np.float32),
            "label_headings_robot_deg": np.asarray(
                pose["label"]["headings_robot_deg"], dtype=np.float32
            ),
        }
        second_hash = _array_hash({"range_m": second_range, "valid_mask": second_valid})
        primary_hash = _array_hash({"range_m": first_range, "valid_mask": first_valid})
        np.savez_compressed(scan_dir / f"{sample_id}.npz", **arrays)
        sample_passed = bool(
            shape_passed
            and dtype_passed
            and valid_boolean_passed
            and range_passed
            and replay_passed
            and reference_passed
            and clearance_passed
            and branch_passed
        )
        all_passed = all_passed and sample_passed
        metric = {
            "sample_id": sample_id,
            "role": pose["role"],
            "shape_passed": shape_passed,
            "dtype_passed": dtype_passed,
            "valid_boolean_passed": valid_boolean_passed,
            "range_contract_passed": range_passed,
            "valid_ratio": float(np.mean(first_valid)),
            "minimum_valid_range_m": float(np.min(valid_ranges)),
            "maximum_valid_range_m": float(np.max(valid_ranges)),
            "primary_scan_array_sha256": primary_hash,
            "independent_scene_scan_array_sha256": second_hash,
            "determinism": {
                "valid_masks_equal": replay_masks_equal,
                "maximum_absolute_valid_range_difference_m": replay_max_diff,
                "threshold_m": 1e-6,
                "passed": replay_passed,
            },
            "prior_reference_regression": {
                "path": str(reference_path.relative_to(PROJECT_ROOT)),
                "valid_masks_equal": reference_masks_equal,
                "maximum_absolute_valid_range_difference_m": reference_max_diff,
                "threshold_m": 1e-5,
                "passed": reference_passed,
            },
            "minimum_horizontal_clearance_m": minimum_clearance,
            "clearance_threshold_m": 0.8,
            "clearance_passed": clearance_passed,
            "branch_visibility": branch_checks,
            "branch_visibility_passed": branch_passed,
            "passed": sample_passed,
        }
        per_pose.append(metric)
        plot_records.append(
            {
                "sample_id": sample_id,
                "role": pose["role"],
                "range_m": first_range,
                "valid_ratio": metric["valid_ratio"],
                "headings_robot_deg": pose["label"]["headings_robot_deg"],
            }
        )

    manifest = {
        "schema_version": "cano_cpu_raycast_pose_manifest_v1",
        "role": "DIAGNOSTIC_SENSOR_CONTRACT_NOT_A_FORMAL_DATASET",
        "world_id": "cano_audited_adapter_seed0_world_000",
        "world_hashes": world_hashes,
        "coordinate_frame": "world: right-handed metres Z-up; robot: x-forward y-left z-up",
        "selection_rule": "frozen deterministic 8 interior + 8 junction-transition + 8 terminal-approach poses; never selected by scan quality",
        "label_rule": "teacher-only 5 m sphere/spline intersections, 8 degree heading deduplication, 720-bin sigma-3-degree circular Gaussian",
        "counts": {
            "source_worlds": 1,
            "poses": 24,
            **role_counts,
            "primary_rays_per_pose": 11520,
            "primary_rays_per_pass": 276480,
            "independent_passes": 2,
            "formal_dataset_samples": 0,
            "training_samples": 0,
            "models": 0,
        },
        "sensor": {
            "shape": [16, 720],
            "elevation_deg": ELEVATION_DEG.tolist(),
            "azimuth_resolution_deg": 0.5,
            "near_range_m": NEAR_RANGE_M,
            "far_range_m": MAX_RANGE_M,
            "returns": 1,
            "noise": "NONE_IDEALIZED_CONTRACT_BASELINE",
            "missing_value": "range_m=50.0, valid_mask=0",
            "sensor_config": str(sensor_config.relative_to(PROJECT_ROOT)),
            "sensor_config_sha256": sha256_file(sensor_config),
        },
        "poses": poses,
    }
    write_json(run_dir / "artifacts/pose_manifest.json", manifest)
    write_json(
        run_dir / "metrics/per_pose.json",
        {"schema_version": "cano_cpu_raycast_per_pose_v1", "poses": per_pose},
    )

    world_preview = run_dir / "previews/full_world_pose_map.png"
    contact_preview = run_dir / "previews/all_24_range_label_contact_sheet.png"
    _plot_world(mesh, spline_document, poses, world_preview)
    _plot_contact_sheet(plot_records, contact_preview)
    common_limitations = [
        "The native Cano Poisson mesh is non-manifold, non-watertight, non-orientable and self-intersecting.",
        "Idealized CPU first-return raycasting does not establish Gazebo or real-LiDAR parity.",
        "Spline-oracle headings are teacher/provenance only and are not sensor input.",
        "This run is diagnostic evidence, not a formal train/validation/test dataset.",
    ]
    write_json(
        run_dir / "previews/full_world_pose_map.provenance.json",
        {
            "question": "Where are all fixed poses in the complete source world?",
            "source_world": "cano_audited_adapter_seed0_world_000",
            "included_sample_ids": [pose["sample_id"] for pose in poses],
            "counts": {"poses": 24, **role_counts},
            "encoding": "gray=all mesh vertices; black=spline axes; blue=interior; orange=junction; green=terminal; arrows=robot yaw",
            "coordinate_units": "world x/y in metres, Z-up world projected to XY",
            "limitations": common_limitations,
        },
    )
    write_json(
        run_dir / "previews/all_24_range_label_contact_sheet.provenance.json",
        {
            "question": "Do all 24 scans satisfy the same visible range/label contract without sample selection?",
            "included_sample_ids": [pose["sample_id"] for pose in poses],
            "ordering": "manifest order: 8 interior, 8 junction-transition, 8 terminal-approach",
            "range_encoding": "fixed 0--50 m viridis scale; missing rays are displayed at 50 m",
            "label_encoding": "yellow vertical lines are teacher-only robot-frame spline-oracle headings",
            "coordinate_units": "azimuth/elevation in degrees; range in metres",
            "limitations": common_limitations,
        },
    )

    valid_ratios = [item["valid_ratio"] for item in per_pose]
    clearances = [item["minimum_horizontal_clearance_m"] for item in per_pose]
    replay_diffs = [item["determinism"]["maximum_absolute_valid_range_difference_m"] for item in per_pose]
    reference_diffs = [item["prior_reference_regression"]["maximum_absolute_valid_range_difference_m"] for item in per_pose]
    contract = {
        "schema_version": "cano_cpu_raycast_lidar_contract_metrics_v1",
        "overall_status": "PASS" if all_passed else "FAIL",
        "analytic_control_passed": analytic["passed"],
        "counts": {
            "source_worlds": 1,
            "new_worlds_generated": 0,
            "diagnostic_poses": 24,
            "diagnostic_scan_npz": len(list(scan_dir.glob("*.npz"))),
            "primary_rays_per_pose": 11520,
            "primary_rays_per_pass": 276480,
            "independent_scene_passes": 2,
            "formal_dataset_worlds": 0,
            "formal_dataset_samples": 0,
            "labels_for_training": 0,
            "training_samples": 0,
            "models": 0,
            "isaac_runs": 0,
            "gazebo_runs": 0,
            "topology_changes": 0,
            "mtare_changes": 0,
        },
        "aggregate": {
            "passed_poses": sum(item["passed"] for item in per_pose),
            "minimum_valid_ratio": min(valid_ratios),
            "maximum_valid_ratio": max(valid_ratios),
            "mean_valid_ratio": float(np.mean(valid_ratios)),
            "minimum_horizontal_clearance_m": min(clearances),
            "maximum_independent_scene_difference_m": max(replay_diffs),
            "maximum_prior_reference_difference_m": max(reference_diffs),
            "branch_visibility_passed_poses": sum(item["branch_visibility_passed"] for item in per_pose),
        },
        "acceptance": {
            "exact_pose_and_role_counts": len(poses) == 24 and role_counts == EXPECTED_ROLES,
            "all_shape_dtype_range_contracts": all(
                item["shape_passed"]
                and item["dtype_passed"]
                and item["valid_boolean_passed"]
                and item["range_contract_passed"]
                for item in per_pose
            ),
            "two_fresh_scenes_deterministic": all(item["determinism"]["passed"] for item in per_pose),
            "prior_reference_regression": all(item["prior_reference_regression"]["passed"] for item in per_pose),
            "clearance_and_branch_los": all(
                item["clearance_passed"] and item["branch_visibility_passed"] for item in per_pose
            ),
            "complete_visual_evidence": world_preview.is_file() and contact_preview.is_file(),
        },
        "decision_if_pass": "Only a separately approved five-topology CPU-raycast contract pilot may be prepared; this run does not authorize data generation or training.",
        "claim_boundary": common_limitations,
    }
    write_json(run_dir / "metrics/contract.json", contract)
    return contract


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world-dir", required=True, type=Path)
    parser.add_argument("--reference-dir", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--sensor-config", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    try:
        contract = _execute(
            args.world_dir.resolve(),
            args.reference_dir.resolve(),
            run_dir,
            args.sensor_config.resolve(),
        )
    except Exception as exc:
        failure = {
            "schema_version": "cano_cpu_raycast_execution_failure_v1",
            "overall_status": "FAIL_IMPLEMENTATION_OR_ENVIRONMENT",
            "exception_type": type(exc).__name__,
            "message": str(exc),
            "traceback": traceback.format_exc(),
        }
        write_json(run_dir / "metrics/execution_failure.json", failure)
        print(json.dumps(failure, indent=2))
        return 2
    print(json.dumps(contract, indent=2))
    return 0 if contract["overall_status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
