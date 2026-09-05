#!/usr/bin/env python3
"""Formal C01 accuracy and throughput proof for accelerated CSG provenance."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260830_csg_mesh_provenance_acceleration_v1_seed0"
PASS = "PASS_CSG_MESH_PROVENANCE_ACCELERATION_V1_P1_EXPORT_ENABLED"
FAIL = "FAIL_CSG_MESH_PROVENANCE_ACCELERATION_V1"
PLANNED_P1_RAYS = 8_723_980_800
SIDE_CAR = "/tmp/mtare_gate4_meshing_sidecar_v1/bin/python"
TEST_PYTHON = "/home/zeng-workstation/anaconda3/bin/python"


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


def _run(command: list[str], log: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(PROJECT_ROOT / "src")
    result = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    log.write_text(result.stdout, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"command failed ({result.returncode}): {' '.join(command)}")
    return result


def _plot(summary: dict, destination: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    full, sparse = summary["full_query"], summary["sparse_query"]
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.0), constrained_layout=True)
    axes[0].bar(["implicit\nreference", "CSG full\nquery", "CSG sparse\nquery"], [full["implicit_seconds"], full["csg_seconds"], sparse["csg_seconds"]], color=["#c45a54", "#d79a42", "#4f8f6b"])
    axes[0].set_ylabel("seconds / 6,144 rays"); axes[0].set_title("Teacher runtime")
    axes[1].bar(["mean", "95%", "99%", "maximum"], [100*full["range_mae_m"], 100*full["range_p95_m"], 100*full["range_p99_m"], 100*full["range_max_m"]], color="#4e79a7")
    axes[1].axhline(5, color="#a33", linestyle="--", linewidth=1, label="5 cm contract")
    axes[1].set_ylabel("range difference (cm)"); axes[1].set_title("CSG vs analytic reference"); axes[1].legend(frameon=False)
    for suffix in ("png", "pdf", "svg"):
        fig.savefig(destination.with_suffix(f".{suffix}"), dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("formal acceleration proof executes exactly once")
    started = time.monotonic(); overall, error = FAIL, None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("scope or authorization mismatch")
        for relative, expected in spec["frozen_inputs"].items():
            if _sha(PROJECT_ROOT / relative) != expected:
                raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool drift: {record['path']}")

        tests = [
            TEST_PYTHON, "-m", "pytest", "-q",
            "tests/v3/unit/test_primitive_construction_supervisor.py",
            "tests/v3/unit/test_primitive_provenance_field.py",
            "tests/v3/unit/test_swept_superellipse_field.py",
            "tests/v3/unit/test_geometry_variant_contract.py",
        ]
        test_result = _run(tests, run_dir / "logs/unit_tests.log")
        if "30 passed" not in test_result.stdout:
            raise RuntimeError("expected exactly 30 dependency-light unit tests")
        analytic_result = _run([SIDE_CAR, "tools/v3/check_csg_mesh_provenance_contract.py"], run_dir / "logs/analytic_contract.log")
        analytic = json.loads(analytic_result.stdout.strip().splitlines()[-1])
        if analytic.get("status") != "PASS_CSG_MESH_PROVENANCE_ANALYTIC_CONTRACT":
            raise RuntimeError("analytic CSG contract failed")

        benchmark = [SIDE_CAR, "tools/v3/benchmark_csg_mesh_provenance_backend.py", "--rays-per-pose", "1024"]
        full_path = run_dir / "metrics/full_query.json"
        sparse_path = run_dir / "metrics/sparse_query.json"
        _run(benchmark + ["--output", str(full_path)], run_dir / "logs/full_query_benchmark.log")
        _run(benchmark + ["--sparse", "--output", str(sparse_path)], run_dir / "logs/sparse_query_benchmark.log")
        full, sparse = load_json(full_path), load_json(sparse_path)
        serial_hours = PLANNED_P1_RAYS / (sparse["rays"] / sparse["csg_seconds"]) / 3600
        conservative_parallel_hours = 2.0 * serial_hours / 32.0
        checks = {
            "exact_30_unit_tests": True,
            "four_analytic_cases_pass": len(analytic["cases"]) == 4,
            "exact_6144_role_stratified_rays": full["rays"] == sparse["rays"] == 6144,
            "sparse_and_full_csg_exact_hit_digest": full["csg_hit_sha256"] == sparse["csg_hit_sha256"],
            "analytic_reference_exact_digest": full["implicit_hit_sha256"] == sparse["implicit_hit_sha256"],
            "valid_agreement_ge_0p98": full["valid_agreement"] >= .98,
            "qualified_coverage_ge_0p98": full["csg_qualified_coverage"] >= .98,
            "range_p95_le_0p05m": full["range_p95_m"] <= .05,
            "range_p99_le_0p05m": full["range_p99_m"] <= .05,
            "range_gt_0p05_fraction_le_0p005": full["range_gt_0p05_fraction"] <= .005,
            "identity_agreement_ge_0p995": full["identity_agreement"] >= .995,
            "sparse_speedup_over_implicit_ge_3": sparse["speedup"] >= 3.0,
            "conservative_32_worker_projection_le_72h": conservative_parallel_hours <= 72.0,
        }
        overall = PASS if all(checks.values()) else FAIL
        summary = {
            "schema_version": "csg_mesh_provenance_acceleration_v1",
            "overall_status": overall,
            "scientific_pass": overall == PASS,
            "checks": checks,
            "analytic_contract": analytic,
            "full_query": full,
            "sparse_query": sparse,
            "resource_projection": {
                "planned_p1_rays": PLANNED_P1_RAYS,
                "measured_sparse_rays_per_second": sparse["rays"] / sparse["csg_seconds"],
                "serial_hours": serial_hours,
                "edge_complexity_safety_factor": 2.0,
                "workers": 32,
                "conservative_parallel_hours": conservative_parallel_hours,
            },
            "decision": "ENABLE_P1_PRIMITIVE_RELATION_DATA_EXPORT" if overall == PASS else "STOP_P1_EXPORT_BACKEND_CONTRACT_FAILED",
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "c09_c10_worlds_read": 0,
            "graph_replays": 0,
            "duration_seconds": time.monotonic() - started,
            "error": None,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "artifacts/backend_schema.json", {
            "geometry": "one closed triangle mesh per swept superellipse primitive",
            "union": "ordered multi-hit CSG exit with analytic candidate and path validation",
            "provenance": "face operand identity enriched by all analytic co-active operands",
            "ambiguity": "multiple active identities retained and masked downstream",
            "acceleration": "expanded operand AABB rejection followed by exact local operand query",
        })
        write_json(run_dir / "config/environment.json", {
            "python": sys.version,
            "executable": sys.executable,
            "platform": platform.platform(),
            "numpy": np.__version__,
            "open3d": __import__("open3d").__version__,
            "test_python": TEST_PYTHON,
        })
        _plot(summary, run_dir / "previews/csg_provenance_accuracy_throughput")
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run_dir / "metrics/summary.json", {"overall_status": FAIL, "scientific_pass": False, "error": error})
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
