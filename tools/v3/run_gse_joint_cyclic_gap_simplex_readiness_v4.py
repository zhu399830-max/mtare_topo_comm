#!/usr/bin/env python3
"""Requalify JCGS stable phase backward and fail-closed training path."""
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


PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V4"
FAIL = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V4"
INNER_PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V2"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_JOINT_CYCLIC_GAP_SIMPLEX_READINESS_V4"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("gap-simplex V4 readiness executes exactly once")
    started = time.monotonic()
    overall, error, result = FAIL, None, {}
    before, after, commands = {}, {}, []
    regression_returncode = executor_returncode = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"gap-simplex V4 Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        environment = json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'zarr':zarr.__version__},sort_keys=True))"], text=True))
        if environment != {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "zarr": "2.18.7"}:
            raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "device": "cpu"})
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        process_environment = os.environ.copy()
        process_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        regression = [str(PYTHON), "-m", "pytest", "-q", "tests/v3/unit/test_gse_joint_cyclic_gap_simplex.py"]
        commands.append(regression)
        with (run / "logs/00_stable_phase_regression.log").open("w", encoding="utf-8") as stream:
            regression_returncode = subprocess.run(regression, cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=300, check=False).returncode
        if regression_returncode != 0:
            raise RuntimeError("stable phase regression failed")
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_joint_cyclic_gap_simplex_readiness_v2.py"), "--teacher-root", str(TEACHER / "artifacts/export/teacher"), "--source-root", str(DATASET / "artifacts/dataset/train"), "--output-dir", str(run / "artifacts/readiness")]
        commands.append(command)
        write_json(run / "config/commands.json", commands)
        with (run / "logs/01_readiness.log").open("w", encoding="utf-8") as stream:
            executor_returncode = subprocess.run(command, cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1200, check=False).returncode
        if executor_returncode not in (0, 2):
            raise RuntimeError(f"gap-simplex V4 executor system failure: {executor_returncode}")
        result = load_json(run / "artifacts/readiness/summary.json")
        passed = result.get("status") == INNER_PASS and result.get("scientific_pass") is True
        if (executor_returncode == 0) != passed:
            raise RuntimeError("gap-simplex V4 status/return mismatch")
        forbidden = ("optimizer_steps", "checkpoint_writes", "threshold_selection_steps", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read", "graph_replays", "planner_calls")
        if any(result.get(name) != 0 for name in forbidden):
            raise RuntimeError("gap-simplex V4 forbidden operation")
        required = [run / "artifacts/readiness" / name for name in ("summary.json", "figure_source.json", "selected_real_rows.csv", "gse_joint_cyclic_gap_simplex_readiness_v2.png", "gse_joint_cyclic_gap_simplex_readiness_v2.pdf", "gse_joint_cyclic_gap_simplex_readiness_v2.svg")]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("gap-simplex V4 evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("gap-simplex V4 source changed")
        overall = PASS if passed else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run / "metrics/summary.json", {
        "schema_version": "gse_joint_cyclic_gap_simplex_readiness_outer_v4",
        "overall_status": overall, "scientific_pass": overall == PASS, "error": error,
        "stable_phase_regression_pass": regression_returncode == 0,
        "finite_bearing_guard_pass": regression_returncode == 0,
        "finite_gradient_guard_pass": regression_returncode == 0,
        "regression_returncode": regression_returncode, "executor_returncode": executor_returncode,
        "result": result, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after), "optimizer_steps": 0,
        "checkpoint_writes": 0, "c09_worlds_read": 0, "c10_worlds_read": 0,
        "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0
    })
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED" if overall == PASS and error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "stable_phase_regression_pass": regression_returncode == 0, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
