#!/usr/bin/env python3
"""Execute and seal the P0 shape-generic swept-superellipse contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions, MAX_RANGE_M
from mtare_topo.governance import load_json, write_json
from mtare_topo.teacher.primitive_construction_supervisor import build_primitive_construction_graph
from mtare_topo.teacher.swept_superellipse_field import (
    SweptSuperellipsePrimitive,
    SweptSuperellipseProvenanceField,
    superellipse_primitives_from_circular_construction,
)


RUN_ID = "gate3_20260830_swept_superellipse_contract_v1_seed0"
PASS = "PASS_SWEPT_SUPERELLIPSE_CONTRACT_V1"
FAIL = "FAIL_SWEPT_SUPERELLIPSE_CONTRACT_V1"
MESH = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0/artifacts/meshes/S01_flat_tree_small_C01/primary"
TEST_PYTHON = Path("/home/zeng-workstation/anaconda3/bin/python")
SPACING_M = 0.05
POSE_PRIMITIVES = (0, 1, 2, 3, 4, 5)
RAY_INDICES = np.linspace(0, 16 * 720 - 1, 32, dtype=np.int64)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    destination = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != destination)
    with destination.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _primitive(identity, points, axes, exponent):
    return SweptSuperellipsePrimitive(identity, np.asarray(points, dtype=float), axes, exponent)


def _ray(field, origin, direction):
    hit = field.ray_exit_hits(np.asarray([origin], dtype=float), np.asarray([direction], dtype=float), maximum_m=20.0)[0]
    if hit is None:
        raise RuntimeError("analytic ray unexpectedly missed")
    return hit


def _analytic_metrics() -> dict:
    ellipse = SweptSuperellipseProvenanceField([_primitive("ellipse", [[-2, 0, 0], [2, 0, 0]], ((1.2, .8), (1.2, .8)), (2, 2))], spacing_m=.005)
    rounded = SweptSuperellipseProvenanceField([_primitive("rounded", [[-2, 0, 0], [2, 0, 0]], ((1, 1), (1, 1)), (8, 8))], spacing_m=.005)
    mixed_primitive = _primitive("mixed", [[0, 0, 0], [4, 0, 0]], ((.5, .6), (1.5, 1.2)), (2, 8))
    mixed = SweptSuperellipseProvenanceField([mixed_primitive], spacing_m=.005)
    diagonal = np.asarray([0., 1., 1.]) / np.sqrt(2.)
    measured = {
        "ellipse_lateral_m": _ray(ellipse, [0, 0, 0], [0, 1, 0]).distance_m,
        "ellipse_vertical_m": _ray(ellipse, [0, 0, 0], [0, 0, 1]).distance_m,
        "rounded_diagonal_m": _ray(rounded, [0, 0, 0], diagonal).distance_m,
        "mixed_midpoint_lateral_m": _ray(mixed, [2, 0, 0], [0, 1, 0]).distance_m,
    }
    expected = {"ellipse_lateral_m": 1.2, "ellipse_vertical_m": .8, "rounded_diagonal_m": 2 ** (.5 - 1/8), "mixed_midpoint_lateral_m": 1.0}
    error = {key: abs(measured[key] - value) for key, value in expected.items()}
    epsilon = 1e-5
    axes0, exp0 = mixed_primitive.parameters_at_fraction(0.)
    axes1, exp1 = mixed_primitive.parameters_at_fraction(1.)
    axes_start, exp_start = mixed_primitive.parameters_at_fraction(epsilon)
    axes_end, exp_end = mixed_primitive.parameters_at_fraction(1-epsilon)
    derivative = max(float(np.max(np.abs((axes_start-axes0)/epsilon))), float(abs((exp_start-exp0)/epsilon)), float(np.max(np.abs((axes1-axes_end)/epsilon))), float(abs((exp1-exp_end)/epsilon)))
    return {"measured": measured, "expected": expected, "absolute_error_m": error, "maximum_absolute_error_m": max(error.values()), "maximum_endpoint_parameter_derivative_numeric": derivative}


def _c01_rays(construction, primitives, fta_distance_m):
    local = lidar_local_directions().reshape(-1, 3)[RAY_INDICES]
    origins, directions, poses = [], [], []
    for primitive_index in POSE_PRIMITIVES:
        source = construction.primitives[primitive_index]
        points = source.centerline_xyz_m; middle = len(points)//2
        origin = points[middle].copy(); origin[2] += float(fta_distance_m) + 1.0
        tangent = points[min(middle+1, len(points)-1)] - points[max(middle-1, 0)]
        yaw = float(np.degrees(np.arctan2(tangent[1], tangent[0])) % 360.)
        origins.append(np.broadcast_to(origin, local.shape)); directions.append(world_directions(local, yaw))
        poses.append({"primitive_id": primitives[primitive_index].primitive_id, "origin_xyz_m": origin.tolist(), "yaw_deg": yaw})
    return np.stack(origins), np.stack(directions), poses


def _plot(destination: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    theta = np.linspace(0, 2*np.pi, 500)
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), constrained_layout=True)
    for exponent, label in ((2, "ellipse n=2"), (4, "mixed n=4"), (8, "rounded rectangle n=8")):
        x = np.sign(np.cos(theta)) * np.abs(np.cos(theta)) ** (2/exponent)
        y = np.sign(np.sin(theta)) * np.abs(np.sin(theta)) ** (2/exponent)
        axes[0].plot(x, y, label=label)
    axes[0].set_aspect("equal"); axes[0].set_title("One continuous cross-section family"); axes[0].set_xlabel("lateral / half-width"); axes[0].set_ylabel("vertical / half-height"); axes[0].legend(fontsize=8)
    fraction = np.linspace(0, 1, 300); smooth = fraction*fraction*(3-2*fraction)
    axes[1].plot(fraction, .5 + smooth, label="half-width 0.5→1.5 m")
    axes[1].plot(fraction, .6 + .6*smooth, label="half-height 0.6→1.2 m")
    axes[1].plot(fraction, 2 + 6*smooth, label="shape exponent 2→8")
    axes[1].set_title("C1 taper and shape transition"); axes[1].set_xlabel("normalized arc length"); axes[1].set_ylabel("parameter value"); axes[1].legend(fontsize=8); axes[1].grid(alpha=.25)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path); args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("shape contract executes exactly once")
    started = time.monotonic(); overall, error = FAIL, None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("user_authorization", {}).get("status") != "APPROVED": raise RuntimeError("scope/authorization mismatch")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"frozen tool drift: {record['path']}")
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src")
        with (run_dir / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            tests = subprocess.run([str(TEST_PYTHON), "-m", "pytest", "-q", "tests/v3/unit/test_primitive_construction_supervisor.py", "tests/v3/unit/test_primitive_provenance_field.py", "tests/v3/unit/test_swept_superellipse_field.py"], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, check=False, timeout=300)
        if tests.returncode: raise RuntimeError("shape contract tests failed")
        analytic = _analytic_metrics()
        graph = load_json(MESH / "graph.json"); splines = load_json(MESH / "splines.json"); geometry = load_json(MESH / "geometry_parameters.json")
        construction = build_primitive_construction_graph(graph, splines, geometry)
        primitives = superellipse_primitives_from_circular_construction(construction)
        field = SweptSuperellipseProvenanceField(primitives, spacing_m=SPACING_M)
        origins, directions, poses = _c01_rays(construction, primitives, geometry["fta_distance_m"])
        first = field.ray_exit_hits(origins.reshape(-1,3), directions.reshape(-1,3), maximum_m=MAX_RANGE_M)
        second = field.ray_exit_hits(origins.reshape(-1,3), directions.reshape(-1,3), maximum_m=MAX_RANGE_M)
        if first != second: raise RuntimeError("C01 shape-generic rays are nondeterministic")
        hits = [value for value in first if value is not None]
        write_json(run_dir / "artifacts/shape_contract.json", {"schema_version": "swept_superellipse_contract_v1", "interpolation": "cubic_smoothstep_c1_endpoint", "analytic": analytic, "c01_primitive_count": len(primitives), "c01_poses": poses, "c01_ray_indices": RAY_INDICES.tolist(), "primitive_schema_example": primitives[0].as_dict()})
        _plot(run_dir / "previews/swept_superellipse_contract")
        checks = {
            "twenty_unit_tests_pass": True,
            "analytic_max_error_le_0p02m": analytic["maximum_absolute_error_m"] <= .02,
            "c1_endpoint_derivative_le_0p01": analytic["maximum_endpoint_parameter_derivative_numeric"] <= .01,
            "c01_exact_55_identity_preserved": len(primitives) == 55 and tuple(x.primitive_id for x in primitives) == tuple(x.primitive_id for x in construction.primitives),
            "c01_circle_parameters_exact": all(x.endpoint_half_axes_m == ((y.radius_m,y.radius_m),(y.radius_m,y.radius_m)) and x.endpoint_shape_exponent == (2.,2.) for x,y in zip(primitives,construction.primitives)),
            "c01_exact_192_rays": len(first) == 192,
            "c01_all_hits_have_sources": all(value.source_primitive_ids for value in hits),
            "c01_deterministic_repeat": True,
        }
        overall = PASS if all(checks.values()) else FAIL
        write_json(run_dir / "metrics/summary.json", {"schema_version": "swept_superellipse_contract_v1", "overall_status": overall, "scientific_pass": overall == PASS, "checks": checks, "analytic": analytic, "c01_primitives": len(primitives), "c01_poses": 6, "c01_rays": len(first), "c01_hits": len(hits), "c01_misses": len(first)-len(hits), "optimizer_steps": 0, "model_inference_frames": 0, "c07_c10_worlds_read": 0, "graph_replays": 0, "duration_seconds": time.monotonic()-started, "error": None})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__": raise SystemExit(main())
