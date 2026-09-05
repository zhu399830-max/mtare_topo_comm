#!/usr/bin/env python3
"""Run and seal the local composition-slot Teacher feasibility audit."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260903_local_composition_slot_teacher_feasibility_v1_seed0"
PASS = "PASS_SYSTEM_LOCAL_COMPOSITION_SLOT_TEACHER_FEASIBILITY_V1"
FAIL = "FAIL_SYSTEM_LOCAL_COMPOSITION_SLOT_TEACHER_FEASIBILITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_LOCAL_COMPOSITION_SLOT_TEACHER_FEASIBILITY_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
    ), encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall = FAIL; error = None; audit = {}; before = {}; after = {}; subprocesses = []; peak_rss_bytes = 0
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("local composition-slot feasibility executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"local composition-slot Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            path = PROJECT_ROOT / relative
            if not path.is_file() or _sha(path) != expected:
                raise RuntimeError(f"local composition-slot frozen input drift: {relative}")
            before[relative] = expected
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"local composition-slot tool drift: {record['path']}")
        versions = {"python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__, "zarr": zarr.__version__}
        expected_versions = {"python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3", "zarr": "2.18.7"}
        if versions != expected_versions:
            raise RuntimeError(f"local composition-slot environment drift: {versions}")
        write_json(run / "config/environment.json", {"versions": versions, "platform": platform.platform(), "device": "CPU_ONLY"})
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT))); env["PYTHONHASHSEED"] = "0"
        tests = ["tests/v3/unit/test_local_composition_slot_teacher.py", "tests/v3/unit/test_execute_local_composition_slot_teacher_feasibility_v1.py"]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run([PYTHON, "-m", "pytest", "-q", *tests], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False)
        subprocesses.append({"stage": "unit_tests", "returncode": completed.returncode})
        if completed.returncode:
            raise RuntimeError("local composition-slot tests failed")
        output = run / "metrics/teacher_feasibility"
        command = [
            PYTHON, str(PROJECT_ROOT / "tools/v3/execute_local_composition_slot_teacher_feasibility_v1.py"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--observability-root", str(OBS / "artifacts/endpoint_observability"),
            "--output-dir", str(output),
        ]
        (run / "config/audit_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        with (run / "logs/01_teacher_feasibility.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=3600, check=False)
        subprocesses.append({"stage": "teacher_feasibility", "returncode": completed.returncode})
        log = (run / "logs/01_teacher_feasibility.log").read_text(encoding="utf-8")
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log); peak_rss_bytes = int(match.group(1)) * 1024 if match else 0
        if completed.returncode:
            raise RuntimeError("local composition-slot audit process failed")
        audit = load_json(output / "summary.json")
        if (
            audit.get("teacher_rows_read") != 491_196 or audit.get("sensor_range_rows_read") != 0
            or audit.get("model_forward_rows") != 0 or audit.get("optimizer_steps") != 0
            or audit.get("c08_rows_read") != 0 or audit.get("decision") not in {
                "ALLOW_LOCAL_COMPOSITION_SLOT_MODEL_READINESS", "STOP_LOCAL_COMPOSITION_SLOT_AND_REASSESS_RELATION_REPRESENTATION",
            }
        ):
            raise RuntimeError("local composition-slot evidence contract drift")
        if peak_rss_bytes <= 0 or peak_rss_bytes > 4 * 1024**3:
            raise RuntimeError("local composition-slot host RAM missing or exceeded")
        required = [output / "summary.json", output / "fit.json", output / "c07.json", output / "per_task.json", output / "per_task.csv", output / "figure_source.json", *[output / f"local_composition_slot_teacher_feasibility.{suffix}" for suffix in ("png", "pdf", "svg")]]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("local composition-slot evidence incomplete")
        for suffix in ("png", "pdf", "svg"):
            shutil.copy2(output / f"local_composition_slot_teacher_feasibility.{suffix}", run / "previews")
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("local composition-slot frozen inputs changed")
        write_json(run / "config/source_integrity_after.json", after)
        if sum(path.stat().st_size for path in run.rglob("*") if path.is_file()) > 200 * 1024**2:
            raise RuntimeError("local composition-slot output exceeded 200 MiB")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    duration = time.monotonic() - started
    write_json(run / "metrics/summary.json", {
        "schema_version": "local_composition_slot_teacher_feasibility_outer_v1", "overall_status": overall,
        "system_evidence_pass": overall == PASS, "error": error, "subprocesses": subprocesses,
        "audit": audit, "scientific_pass": audit.get("scientific_pass"), "decision": audit.get("decision"),
        "peak_host_rss_bytes": peak_rss_bytes, "duration_seconds": duration,
        "teacher_rows_read": int(audit.get("teacher_rows_read", 0)), "sensor_range_rows_read": 0,
        "model_forward_rows": 0, "optimizer_steps": 0, "c08_rows_read": 0, "c09_c10_worlds_read": 0,
        "graph_replays": 0, "mtare_worlds_read": 0, "source_unchanged": bool(before and before == after),
    })
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS and error is None else "FAILED", "overall_status": overall, "error": error, "duration_seconds": duration})
    expected_entries = len([path for path in run.rglob("*") if path.is_file()]) + 1
    write_json(run / "artifacts/seal_summary.json", {"schema_version": "local_composition_slot_teacher_feasibility_seal_v1", "expected_evidence_entries": expected_entries, "overall_status": overall, "error": error})
    entries = _seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "scientific_pass": audit.get("scientific_pass"), "decision": audit.get("decision"), "evidence_files": entries, "expected_evidence_files": expected_entries}, indent=2))
    return 0 if overall == PASS and error is None and entries == expected_entries else 2


if __name__ == "__main__":
    raise SystemExit(main())
