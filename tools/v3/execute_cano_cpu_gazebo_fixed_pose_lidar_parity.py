#!/usr/bin/env python3
"""Execute the approved diagnostic-only CPU--Gazebo fixed-pose parity contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import traceback
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_gazebo_parity import (
    ROLE_COUNTS,
    analytic_box_ranges,
    gazebo_world_sdf,
    parity_metrics,
    select_parity_poses,
)
from mtare_topo.data.cano_sensor_smoke import (
    ELEVATION_DEG,
    MAX_RANGE_M,
    NEAR_RANGE_M,
    lidar_local_directions,
    world_directions,
)
from mtare_topo.governance import load_json, write_json


SOURCE_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
PARENT_IDS = list(ROLE_COUNTS)
IMAGE = "mtare-semantic-runtime:local"
IMAGE_ID = "sha256:9819eea672b2001ed18f12ee09c67f0da53a5ba6ce69192e2f21c128e5e78f9a"
PLUGIN_PATH = "/home/docker-user/mtare/autonomous_exploration_development_environment/devel/lib/libgazebo_ros_velodyne_laser.so"
PLUGIN_SHA256 = "325000c31a6af77114f0d7339781c83eb0c41644adce99f219a0b9ba536ca0d3"
EXPECTED_SCOPE = {
    "analytic_control_worlds": 1,
    "cano_parent_worlds": 3,
    "fixed_poses": 24,
    "cpu_reference_scans": 24,
    "gazebo_static_scans": 72,
    "rays_per_scan": 11520,
    "formal_dataset_samples": 0,
    "teacher_labels": 0,
    "training_samples": 0,
    "models": 0,
    "trajectories": 0,
    "mtare_changes": 0,
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _scene(mesh_path: Path) -> o3d.t.geometry.RaycastingScene:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        raise RuntimeError(f"Gazebo/CPU source mesh is empty: {mesh_path}")
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    return scene


def _cast(scene: o3d.t.geometry.RaycastingScene, pose: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    directions = world_directions(lidar_local_directions(), float(pose["yaw_deg"]))
    origins = np.broadcast_to(np.asarray(pose["sensor_xyz_m"], dtype=np.float32), directions.shape)
    rays = np.concatenate((origins, directions), axis=-1)
    raw = scene.cast_rays(o3d.core.Tensor(rays.reshape(-1, 6)))["t_hit"].numpy().reshape(16, 720)
    valid = np.isfinite(raw) & (raw >= NEAR_RANGE_M) & (raw <= MAX_RANGE_M)
    return np.where(valid, raw, MAX_RANGE_M).astype(np.float32), valid.astype(np.uint8)


def _run_gazebo(world: Path, output: Path, sensor_count: int, log_dir: Path, name: str) -> None:
    command = [
        "docker", "run", "--rm", "--name", name, "--network", "none", "--shm-size", "512m",
        "-v", "/etc/localtime:/etc/localtime:ro",
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{SOURCE_RUN}:/source:ro",
        "-v", f"{world.parent}:/parity:ro",
        "-v", f"{output.parents[2]}:/output:rw",
        IMAGE, "bash", "/workspace/tools/v3/gazebo/run_fixed_lidar_session.sh",
        f"/parity/{world.name}", f"/output/{output.relative_to(output.parents[2])}", str(sensor_count),
        f"/output/{log_dir.relative_to(output.parents[2])}",
    ]
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, timeout=180)


def _load_scans(path: Path, sensor_count: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    result = []
    with np.load(path, allow_pickle=False) as data:
        for index in range(sensor_count):
            ranges = data[f"range_{index:02d}"]
            valid = data[f"valid_{index:02d}"]
            stamps = data[f"stamp_{index:02d}"]
            if ranges.shape != (3, 16, 720) or valid.shape != (3, 16, 720) or stamps.shape != (3,):
                raise RuntimeError(f"unexpected Gazebo scan layout at sensor {index}")
            result.append((ranges, valid, stamps))
    return result


def _repeat_metrics(ranges: np.ndarray, valid: np.ndarray) -> dict[str, Any]:
    masks_equal = bool(np.array_equal(valid[0], valid[1]) and np.array_equal(valid[0], valid[2]))
    common = valid[0].astype(bool) & valid[1].astype(bool) & valid[2].astype(bool)
    maximum = 0.0
    for first, second in ((0, 1), (0, 2), (1, 2)):
        if np.any(common):
            maximum = max(maximum, float(np.max(np.abs(ranges[first][common] - ranges[second][common]))))
    return {
        "valid_masks_identical": masks_equal,
        "common_valid_count": int(np.count_nonzero(common)),
        "maximum_pairwise_range_difference_m": maximum,
        "passed": bool(masks_equal and maximum <= 0.0011),
    }


def _neighbour_diagnostic(cpu_range: np.ndarray, cpu_valid: np.ndarray, gazebo_range: np.ndarray, gazebo_valid: np.ndarray) -> dict[str, float]:
    values = {}
    for shift in (-1, 0, 1):
        shifted_range = np.roll(gazebo_range, shift, axis=1)
        shifted_valid = np.roll(gazebo_valid, shift, axis=1)
        common = cpu_valid.astype(bool) & shifted_valid.astype(bool)
        errors = np.abs(cpu_range[common].astype(np.float64) - shifted_range[common].astype(np.float64))
        values[str(shift)] = float(np.mean(errors)) if errors.size else math.inf
    return {"mae_shift_minus1_m": values["-1"], "mae_direct_m": values["0"], "mae_shift_plus1_m": values["1"]}


def _pose_figure(records: list[dict[str, Any]], raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]], destination: Path) -> None:
    for page in range(6):
        fig, axes = plt.subplots(4, 4, figsize=(15, 12), constrained_layout=True)
        for row, record in enumerate(records[page * 4:(page + 1) * 4]):
            cpu_range, cpu_valid, gazebo_range, gazebo_valid = raw[record["pose_id"]]
            mismatch = cpu_valid != gazebo_valid
            common = cpu_valid.astype(bool) & gazebo_valid.astype(bool)
            error = np.zeros_like(cpu_range)
            error[common] = np.abs(cpu_range[common] - gazebo_range[common])
            panels = (cpu_range, gazebo_range, mismatch.astype(float), error)
            titles = ("CPU range (m)", "Gazebo range (m)", "valid mismatch", "abs error (m)")
            for col, (panel, title) in enumerate(zip(panels, titles)):
                vmax = 50 if col < 2 else (1 if col == 2 else 0.25)
                axes[row, col].imshow(panel, aspect="auto", origin="lower", vmin=0, vmax=vmax)
                axes[row, col].set_title(f"{record['pose_id']}\n{title}", fontsize=7)
                axes[row, col].set_xlabel("azimuth bin")
                axes[row, col].set_ylabel("elevation row")
        fig.suptitle(f"Gate 0 CPU--Gazebo parity: complete page {page + 1}/6 (diagnostic train parents only)")
        fig.savefig(destination / f"parity_page_{page + 1:02d}.png", dpi=145)
        plt.close(fig)


def _pose_map(parent_inputs: list[tuple[str, dict, list[dict[str, Any]]]], destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
    colors = {"tunnel_interior": "#0072B2", "junction_transition": "#D55E00", "terminal_approach": "#009E73"}
    for axis, (parent_id, splines, poses) in zip(axes, parent_inputs):
        for tunnel in splines["tunnels"]:
            points = np.asarray(tunnel["points"])
            axis.plot(points[:, 0], points[:, 1], color="#777777", linewidth=0.8)
        for role, color in colors.items():
            selected = [pose for pose in poses if pose["role"] == role]
            xy = np.asarray([pose["sensor_xyz_m"][:2] for pose in selected])
            axis.scatter(xy[:, 0], xy[:, 1], color=color, label=role, s=28)
        axis.set_title(parent_id)
        axis.set_aspect("equal", adjustable="box")
        axis.set_xlabel("world x (m)")
        axis.set_ylabel("world y (m)")
    axes[0].legend(fontsize=8)
    fig.suptitle("Gate 0 frozen CPU--Gazebo parity poses: 8 interior / 8 junction / 8 terminal")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _analytic_figure(expected: np.ndarray, observed: np.ndarray, destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), constrained_layout=True)
    for axis, panel, title, vmax in zip(axes, (expected, observed, np.abs(expected - observed)), ("analytic/CPU range", "Gazebo range", "absolute error"), (20, 20, 0.001)):
        axis.imshow(panel, aspect="auto", origin="lower", vmin=0, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("azimuth bin")
        axis.set_ylabel("elevation row")
    fig.suptitle("Analytic closed-box control, meters")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def _stratified_errors(records: list[dict[str, Any]], raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]) -> dict[str, Any]:
    groups: dict[str, dict[str, list[float]]] = {"parent": {}, "role": {}, "elevation_deg": {}}
    for record in records:
        cpu_range, cpu_valid, gazebo_range, gazebo_valid = raw[record["pose_id"]]
        common = cpu_valid.astype(bool) & gazebo_valid.astype(bool)
        error = np.abs(cpu_range.astype(np.float64) - gazebo_range.astype(np.float64))
        for kind, key in (("parent", record["parent_id"]), ("role", record["role"])):
            groups[kind].setdefault(key, []).extend(error[common].tolist())
        for row, elevation in enumerate(ELEVATION_DEG):
            row_common = common[row]
            groups["elevation_deg"].setdefault(f"{elevation:.1f}", []).extend(error[row][row_common].tolist())
    result: dict[str, Any] = {}
    for kind, entries in groups.items():
        result[kind] = {
            key: {"count": len(values), "mae_m": float(np.mean(values)), "p95_m": float(np.quantile(values, 0.95)), "p99_m": float(np.quantile(values, 0.99))}
            for key, values in entries.items() if values
        }
    return result


def _distribution_figure(stratified: dict[str, Any], destination: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(17, 5), constrained_layout=True)
    for axis, kind, title in zip(axes, ("parent", "role", "elevation_deg"), ("By parent", "By pose role", "By elevation")):
        entries = stratified[kind]
        labels = list(entries)
        axis.bar(np.arange(len(labels)), [entries[key]["p95_m"] for key in labels], color="#0072B2", label="P95")
        axis.plot(np.arange(len(labels)), [entries[key]["mae_m"] for key in labels], color="#D55E00", marker="o", label="MAE")
        axis.set_xticks(np.arange(len(labels)), labels, rotation=70 if kind != "role" else 25, ha="right")
        axis.set_ylabel("direct-bin absolute range error (m)")
        axis.set_title(title)
        axis.legend()
    fig.suptitle("Gate 0 CPU--Gazebo direct-bin error distributions; all 24 poses")
    fig.savefig(destination, dpi=150)
    plt.close(fig)


def execute(run_dir: Path) -> dict[str, Any]:
    worlds = run_dir / "artifacts/worlds"
    raw_dir = run_dir / "artifacts/diagnostic_scans"
    sessions = run_dir / "logs/gazebo_sessions"
    pages = run_dir / "previews/parity_pages"
    for path in (worlds, raw_dir, sessions, pages):
        path.mkdir(parents=True, exist_ok=False)

    observed_image = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    observed_plugin = subprocess.check_output(["docker", "run", "--rm", "--network", "none", "-v", "/etc/localtime:/etc/localtime:ro", IMAGE, "sha256sum", PLUGIN_PATH], text=True).split()[0]
    if observed_image != IMAGE_ID or observed_plugin != PLUGIN_SHA256:
        raise RuntimeError("frozen container or plugin identity mismatch")

    analytic_world = worlds / "analytic_box.world"
    analytic_world.write_text(gazebo_world_sdf(None, [{"sensor_xyz_m": [0, 0, 0], "yaw_deg": 0}], analytic_box=True), encoding="utf-8")
    analytic_output = raw_dir / "analytic_gazebo.npz"
    _run_gazebo(analytic_world, analytic_output, 1, sessions / "analytic", f"mtare_parity_{run_dir.name}_analytic")
    analytic_scans = _load_scans(analytic_output, 1)[0]
    expected_range, expected_valid = analytic_box_ranges()
    analytic_repeat = _repeat_metrics(analytic_scans[0], analytic_scans[1])
    analytic_per_scan = [parity_metrics(expected_range, expected_valid, analytic_scans[0][index], analytic_scans[1][index]) for index in range(3)]
    analytic_passed = analytic_repeat["passed"] and all(item["passed"] for item in analytic_per_scan)
    analytic_metrics = {"passed": analytic_passed, "repeat": analytic_repeat, "per_scan": analytic_per_scan}
    write_json(run_dir / "metrics/analytic_control.json", analytic_metrics)
    _analytic_figure(expected_range, analytic_scans[0][0], run_dir / "previews/analytic_control.png")
    if not analytic_passed:
        raise RuntimeError("analytic box control failed; Cano worlds forbidden")

    manifest = load_json(SOURCE_RUN / "artifacts/mesh_manifest.json")
    source_records = {item["parent_id"]: item for item in manifest["parents"]}
    pose_records: list[dict[str, Any]] = []
    plot_inputs = []
    plot_raw: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for parent_index, parent_id in enumerate(PARENT_IDS):
        source = SOURCE_RUN / "artifacts/meshes" / parent_id / "primary"
        mesh_path = source / "mesh.obj"
        record = source_records[parent_id]
        if record["split"] != "train" or _sha256(mesh_path) != record["immutable_asset"]["mesh_sha256"]:
            raise RuntimeError(f"sealed train mesh identity failed: {parent_id}")
        graph = load_json(source / "graph.json")
        splines = load_json(source / "splines.json")
        geometry = load_json(source / "geometry_parameters.json")
        poses = select_parity_poses(parent_id, graph, splines, float(geometry["fta_distance_m"]))
        plot_inputs.append((parent_id, splines, poses))
        scene = _scene(mesh_path)
        cpu = [_cast(scene, pose) for pose in poses]
        np.savez_compressed(raw_dir / f"{parent_id}_cpu.npz", **{f"range_{i:02d}": item[0] for i, item in enumerate(cpu)}, **{f"valid_{i:02d}": item[1] for i, item in enumerate(cpu)})

        world = worlds / f"{parent_id}.world"
        mesh_uri = f"file:///source/artifacts/meshes/{parent_id}/primary/mesh.obj"
        world.write_text(gazebo_world_sdf(mesh_uri, poses), encoding="utf-8")
        gazebo_output = raw_dir / f"{parent_id}_gazebo.npz"
        _run_gazebo(world, gazebo_output, 8, sessions / parent_id, f"mtare_parity_{run_dir.name}_p{parent_index:02d}")
        gazebo = _load_scans(gazebo_output, 8)
        for pose_index, (pose, cpu_pair, gazebo_triplet) in enumerate(zip(poses, cpu, gazebo)):
            gazebo_ranges, gazebo_valid, stamps = gazebo_triplet
            repeat = _repeat_metrics(gazebo_ranges, gazebo_valid)
            per_scan = [parity_metrics(cpu_pair[0], cpu_pair[1], gazebo_ranges[index], gazebo_valid[index]) for index in range(3)]
            direct_pass = all(item["passed"] for item in per_scan)
            pose_record = {
                **pose,
                "parent_pose_index": pose_index,
                "source_mesh_sha256": _sha256(mesh_path),
                "gazebo_stamps_s": stamps.tolist(),
                "repeat": repeat,
                "per_scan": per_scan,
                "neighbour_diagnostic_first_scan": _neighbour_diagnostic(cpu_pair[0], cpu_pair[1], gazebo_ranges[0], gazebo_valid[0]),
                "passed": bool(repeat["passed"] and direct_pass),
            }
            write_json(run_dir / "metrics" / f"{pose['pose_id']}.json", pose_record)
            pose_records.append(pose_record)
            plot_raw[pose["pose_id"]] = (cpu_pair[0], cpu_pair[1], gazebo_ranges[0], gazebo_valid[0])
            if not pose_record["passed"]:
                raise RuntimeError(f"per-pose parity failed: {pose['pose_id']}")

    _pose_map(plot_inputs, run_dir / "previews/frozen_pose_map.png")
    _pose_figure(pose_records, plot_raw, pages)
    stratified = _stratified_errors(pose_records, plot_raw)
    write_json(run_dir / "metrics/stratified_error_statistics.json", stratified)
    _distribution_figure(stratified, run_dir / "previews/error_distributions.png")
    role_counts = {role: sum(item["role"] == role for item in pose_records) for role in ("tunnel_interior", "junction_transition", "terminal_approach")}
    summary = {
        "schema_version": "cano_cpu_gazebo_fixed_pose_lidar_parity_summary_v1",
        "overall_status": "PASS_CANO_CPU_GAZEBO_FIXED_POSE_LIDAR_PARITY_V1",
        "scope": EXPECTED_SCOPE,
        "container": {"image": IMAGE, "image_id": observed_image, "plugin_path": PLUGIN_PATH, "plugin_sha256": observed_plugin},
        "analytic_control": analytic_metrics,
        "counts": {"parents": len(PARENT_IDS), "poses": len(pose_records), "cpu_references": len(pose_records), "gazebo_scans": len(pose_records) * 3, "role_counts": role_counts},
        "aggregate": {
            "minimum_valid_mask_agreement": min(scan["valid_mask_agreement"] for item in pose_records for scan in item["per_scan"]),
            "maximum_range_mae_m": max(scan["range_mae_m"] for item in pose_records for scan in item["per_scan"]),
            "maximum_range_error_p95_m": max(scan["range_error_p95_m"] for item in pose_records for scan in item["per_scan"]),
            "maximum_range_error_p99_m": max(scan["range_error_p99_m"] for item in pose_records for scan in item["per_scan"]),
            "maximum_horizontal_minimum_range_difference_m": max(scan["horizontal_minimum_range_difference_m"] for item in pose_records for scan in item["per_scan"]),
            "maximum_repeat_difference_m": max(item["repeat"]["maximum_pairwise_range_difference_m"] for item in pose_records),
        },
        "stratified_error_statistics": stratified,
        "poses": pose_records,
        "claim_boundary": "Diagnostic sensor-domain parity only; zero formal data, labels, training, model, trajectory, online graph or M-TARE change.",
    }
    write_json(run_dir / "artifacts/pose_manifest.json", {"parents": PARENT_IDS, "poses": [{key: value for key, value in item.items() if key not in {"per_scan", "repeat"}} for item in pose_records]})
    write_json(run_dir / "previews/provenance.json", {"gate": 0, "split": "diagnostic_train_parents_only", "parents": PARENT_IDS, "pose_count": 24, "units": "meters", "complete_pages": 6, "selection": "all poses exactly once; no best-case selection"})
    write_json(run_dir / "metrics/summary.json", summary)
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    try:
        summary = execute(args.run_dir.resolve())
    except Exception as exc:
        write_json(args.run_dir.resolve() / "metrics/executor_failure.json", {"exception_type": type(exc).__name__, "message": str(exc), "traceback": traceback.format_exc()})
        raise
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
