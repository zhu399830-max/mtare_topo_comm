#!/usr/bin/env python3
"""Requalify JCGS after the attributed circular-boundary index correction."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json
from run_gse_joint_cyclic_gap_simplex_readiness_v3 import DATASET, PYTHON, TEACHER, seal, sha256


PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V5"
FAIL = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V5"
INNER_PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V2"
BOUNDARY_PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_BOUNDARY_INDEX_READINESS_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V5"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("gap-simplex V5 readiness executes exactly once")
    started = time.monotonic()
    overall, error, readiness, boundary = FAIL, None, {}, {}
    before, after, commands = {}, {}, []
    test_code = boundary_code = readiness_code = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"gap-simplex V5 Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        environment = json.loads(subprocess.check_output([
            str(PYTHON), "-c",
            "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__,'cuda':torch.cuda.is_available()},sort_keys=True))"
        ], text=True))
        expected_environment = {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "zarr": "2.18.7", "cuda": True}
        if environment != expected_environment:
            raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "device": "cuda"})
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        process_environment = os.environ.copy()
        process_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")

        test_command = [str(PYTHON), "-m", "pytest", "-q", "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py"]
        commands.append(test_command)
        with (run / "logs/00_boundary_regression.log").open("w", encoding="utf-8") as stream:
            test_code = subprocess.run(test_command, cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=300, check=False).returncode
        if test_code != 0:
            raise RuntimeError("circular-boundary regression failed")

        boundary_command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_joint_cyclic_gap_simplex_boundary_index_readiness_v1.py"),
            "--teacher-root", str(TEACHER / "artifacts/export/teacher"),
            "--source-root", str(DATASET / "artifacts/dataset/train"),
            "--output", str(run / "artifacts/boundary_index_summary.json"),
        ]
        commands.append(boundary_command)
        with (run / "logs/01_attributed_boundary_cuda_replay.log").open("w", encoding="utf-8") as stream:
            boundary_code = subprocess.run(boundary_command, cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=900, check=False).returncode
        if boundary_code not in (0, 2):
            raise RuntimeError(f"boundary replay system failure: {boundary_code}")
        boundary = load_json(run / "artifacts/boundary_index_summary.json")
        boundary_passed = boundary_code == 0 and boundary.get("status") == BOUNDARY_PASS and boundary.get("scientific_pass") is True
        if not boundary_passed:
            raise RuntimeError("attributed circular-boundary CUDA replay failed")

        readiness_command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v2.py"),
            "--teacher-root", str(TEACHER / "artifacts/export/teacher"),
            "--source-root", str(DATASET / "artifacts/dataset/train"),
            "--output-dir", str(run / "artifacts/readiness"),
        ]
        commands.append(readiness_command)
        write_json(run / "config/commands.json", commands)
        with (run / "logs/02_readiness.log").open("w", encoding="utf-8") as stream:
            readiness_code = subprocess.run(readiness_command, cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1200, check=False).returncode
        if readiness_code not in (0, 2):
            raise RuntimeError(f"gap-simplex V5 readiness system failure: {readiness_code}")
        readiness = load_json(run / "artifacts/readiness/summary.json")
        readiness_passed = readiness_code == 0 and readiness.get("status") == INNER_PASS and readiness.get("scientific_pass") is True
        if not readiness_passed:
            raise RuntimeError("prior JCGS method readiness regressed")
        forbidden = ("optimizer_steps", "checkpoint_writes", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read", "graph_replays", "planner_calls")
        if any(boundary.get(name) != 0 for name in forbidden):
            raise RuntimeError("boundary replay performed a forbidden operation")
        required = [
            run / "artifacts/boundary_index_summary.json",
            run / "artifacts/readiness/summary.json",
            run / "artifacts/readiness/gse_joint_cyclic_gap_simplex_readiness_v2.png",
            run / "artifacts/readiness/gse_joint_cyclic_gap_simplex_readiness_v2.pdf",
            run / "artifacts/readiness/gse_joint_cyclic_gap_simplex_readiness_v2.svg",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("gap-simplex V5 evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("gap-simplex V5 frozen input changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run / "metrics/summary.json", {
        "schema_version": "gse_joint_cyclic_gap_simplex_readiness_outer_v5",
        "overall_status": overall, "scientific_pass": overall == PASS, "error": error,
        "boundary_regression_pass": test_code == 0,
        "attributed_boundary_cuda_replay_pass": boundary_code == 0 and boundary.get("status") == BOUNDARY_PASS,
        "prior_method_readiness_pass": readiness_code == 0 and readiness.get("status") == INNER_PASS,
        "boundary": boundary, "readiness": readiness,
        "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "checkpoint_writes": 0, "c09_worlds_read": 0,
        "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": run_id,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
