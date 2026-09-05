#!/usr/bin/env python3
"""Execute and seal one immutable spatial multi-event Teacher feasibility run."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time
import traceback

import matplotlib
import numpy
import open3d
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


PASS = "PASS_GSE_SPATIAL_MULTI_EVENT_TEACHER_FEASIBILITY_V1"
FAIL = "FAIL_GSE_SPATIAL_MULTI_EVENT_TEACHER_FEASIBILITY_V1"
SIDECAR = Path("/tmp/mtare_gate4_meshing_sidecar_v1/bin/python")
DATASET_RUN = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
MESH_RUN = PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_parent_perception_mesh_m1r_sanitized_assets_seed0"
LOW_RUN = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_low_support_endpoint_observability_audit_v1_seed0"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    with target.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run_dir.name != run_id or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("spatial multi-event feasibility may execute only once")

    started = time.monotonic()
    overall_status = FAIL
    error = None
    result: dict = {}
    source_before: dict[str, str] = {}
    source_after: dict[str, str] = {}
    executor_returncode = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_MULTI_EVENT_TEACHER_FEASIBILITY_V1":
            raise RuntimeError("spatial multi-event feasibility Data Card drift")
        for record in spec["frozen_tools"].values():
            path = PROJECT_ROOT / record["path"]
            if _sha256(path) != record["sha256"]:
                raise RuntimeError(f"spatial multi-event tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            path = PROJECT_ROOT / relative
            actual = _sha256(path)
            if actual != expected:
                raise RuntimeError(f"spatial multi-event input drift: {relative}")
            source_before[relative] = actual

        environment = {
            "executable": str(SIDECAR),
            "python": platform.python_version(),
            "numpy": numpy.__version__,
            "open3d": open3d.__version__,
            "zarr": zarr.__version__,
            "matplotlib": matplotlib.__version__,
            "gpu_used": False,
        }
        expected_environment = {
            "python": "3.12.3",
            "numpy": "1.26.4",
            "open3d": "0.19.0",
            "zarr": "2.18.7",
            "matplotlib": "3.11.1",
        }
        if {key: environment[key] for key in expected_environment} != expected_environment:
            raise RuntimeError(f"spatial multi-event sidecar drift: {environment}")
        write_json(run_dir / "config/source_integrity_before.json", source_before)
        write_json(run_dir / "config/environment.json", environment)
        write_json(
            run_dir / "RUN_STATE.json",
            {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"},
        )

        command = [
            str(SIDECAR),
            str(PROJECT_ROOT / "tools/v3/execute_gse_spatial_multi_event_teacher_feasibility_v1.py"),
            "--dataset-root",
            str(DATASET_RUN / "artifacts/dataset"),
            "--shard-manifest",
            str(DATASET_RUN / "artifacts/shard_manifest.json"),
            "--mesh-root",
            str(MESH_RUN / "artifacts/meshes"),
            "--mesh-manifest",
            str(MESH_RUN / "artifacts/mesh_manifest.json"),
            "--low-observability",
            str(LOW_RUN / "artifacts/audit/low_selection_rows.json"),
            "--output-dir",
            str(run_dir / "artifacts/audit"),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        environment_variables = os.environ.copy()
        environment_variables["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        environment_variables["OMP_NUM_THREADS"] = "2"
        with (run_dir / "logs/audit.log").open("w", encoding="utf-8") as log_stream:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment_variables,
                stdout=log_stream,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=3300,
                check=False,
            )
        executor_returncode = completed.returncode
        if executor_returncode not in (0, 2):
            raise RuntimeError(f"spatial multi-event executor system failure: {executor_returncode}")

        result = load_json(run_dir / "artifacts/audit/summary.json")
        if result.get("status") not in (PASS, FAIL):
            raise RuntimeError("spatial multi-event result status drift")
        if (
            result.get("population", {}).get("worlds") != 80
            or result.get("population", {}).get("observations") != 188126
            or result.get("rare_failure_rows") != 4
            or any(
                result.get(key) != 0
                for key in (
                    "optimizer_steps",
                    "model_inference_frames",
                    "model_updates",
                    "threshold_selection_steps",
                    "c09_worlds_read",
                    "c10_worlds_read",
                    "mtare_worlds_read",
                    "student_inputs_read",
                    "full_teacher_export_rows",
                )
            )
        ):
            raise RuntimeError("spatial multi-event population/isolation drift")
        if bool(result.get("gates", {}).get("all_passed")) != bool(result["status"] == PASS):
            raise RuntimeError("spatial multi-event gate/status mismatch")
        required = [
            run_dir / "artifacts/audit/world_feasibility_summary.jsonl",
            run_dir / "artifacts/audit/event_identity_coverage.jsonl",
            run_dir / "artifacts/audit/rare_failure_los_evidence.json",
            run_dir / "artifacts/audit/figure_source.json",
            run_dir / "artifacts/audit/gse_spatial_multi_event_teacher_feasibility.png",
            run_dir / "artifacts/audit/gse_spatial_multi_event_teacher_feasibility.pdf",
            run_dir / "artifacts/audit/gse_spatial_multi_event_teacher_feasibility.svg",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("spatial multi-event evidence incomplete")
        source_after = {relative: _sha256(PROJECT_ROOT / relative) for relative in source_before}
        if source_before != source_after:
            raise RuntimeError("spatial multi-event frozen sources changed")
        overall_status = str(result["status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "gse_spatial_multi_event_teacher_feasibility_outer_v1",
            "overall_status": overall_status,
            "error": error,
            "executor_returncode": executor_returncode,
            "result": result,
            "duration_seconds": time.monotonic() - started,
            "source_unchanged": bool(source_before and source_before == source_after),
            "optimizer_steps": 0,
            "model_inference_frames": 0,
            "c09_worlds_read": 0,
            "c10_worlds_read": 0,
            "mtare_worlds_read": 0,
        },
    )
    completed_state = "COMPLETED" if overall_status == PASS and error is None else "FAILED"
    write_json(
        run_dir / "RUN_STATE.json",
        {
            "schema_version": "v3_run_state_v1",
            "run_id": run_id,
            "state": completed_state,
            "overall_status": overall_status,
            "error": error,
        },
    )
    seal_entries = _seal(run_dir)
    print(
        json.dumps(
            {
                "overall_status": overall_status,
                "error": error,
                "executor_returncode": executor_returncode,
                "seal_entries": seal_entries,
            },
            indent=2,
        )
    )
    return 0 if overall_status == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
