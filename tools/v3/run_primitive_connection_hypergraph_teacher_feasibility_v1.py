#!/usr/bin/env python3
"""Run and seal one immutable fit/C07 connection-hypergraph Teacher audit."""

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

import matplotlib
import numpy as np
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


RUN_ID = "gate3_20260902_primitive_connection_hypergraph_teacher_feasibility_v1_seed0"
PASS = "PASS_PRIMITIVE_CONNECTION_HYPERGRAPH_TEACHER_FEASIBILITY_V1"
FAIL = "FAIL_PRIMITIVE_CONNECTION_HYPERGRAPH_TEACHER_FEASIBILITY_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_CONNECTION_HYPERGRAPH_TEACHER_FEASIBILITY_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
SIDECAR = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_hash(path: Path) -> str:
    digest = hashlib.sha256()
    files = sorted(value for value in path.rglob("*") if value.is_file())
    for value in files:
        relative = value.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "little"))
        digest.update(relative)
        digest.update(bytes.fromhex(_sha(value)))
    return digest.hexdigest()


def _dataset_trees() -> dict[str, str]:
    result = {}
    roots = {
        "p1b": P1B / "artifacts/teacher",
        "sidecar": SIDECAR / "artifacts/endpoint_observability",
    }
    for source, root in roots.items():
        for split in ("fit", "c07"):
            for path in sorted(value for value in (root / split).glob("*.zarr") if value.is_dir()):
                result[f"{source}/{split}/{path.name}"] = _tree_hash(path)
    return result


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("".join(
        f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files
    ), encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run = args.run_dir.resolve()
    started = time.monotonic()
    overall = FAIL
    error = None
    audit = {}
    before_files = {}
    after_files = {}
    before_trees = {}
    after_trees = {}
    subprocesses = []
    peak_rss_kib = None
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("connection hypergraph Teacher audit executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"connection hypergraph Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            path = PROJECT_ROOT / relative
            if not path.is_file():
                raise RuntimeError(f"frozen input missing: {relative}")
            before_files[relative] = _sha(path)
            if before_files[relative] != expected:
                raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool drift: {record['path']}")
        if load_json(P1B / "RUN_STATE.json").get("state") != "COMPLETED":
            raise RuntimeError("P1b Teacher source is not sealed complete")
        if load_json(SIDECAR / "RUN_STATE.json").get("state") != "COMPLETED":
            raise RuntimeError("endpoint-observability sidecar source is not sealed complete")

        versions = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "zarr": zarr.__version__,
            "matplotlib": matplotlib.__version__,
        }
        expected_versions = {
            "python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3",
            "zarr": "2.18.7", "matplotlib": "3.10.0",
        }
        if versions != expected_versions:
            raise RuntimeError(f"connection hypergraph audit environment drift: {versions}")
        write_json(run / "config/environment.json", {
            "versions": versions, "platform": platform.platform(),
            "compute": "CPU_ONLY_ZERO_MODEL_INFERENCE",
        })
        write_json(run / "config/source_integrity_before.json", before_files)
        write_json(run / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })

        before_trees = _dataset_trees()
        if before_trees != spec["frozen_dataset_trees"]:
            raise RuntimeError("fit/C07 Teacher or sidecar shard tree drift")
        write_json(run / "config/dataset_tree_integrity_before.json", before_trees)

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join((
            str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT),
        ))
        env["PYTHONHASHSEED"] = "0"
        tests = [
            "tests/v3/unit/test_primitive_connection_hypergraph_teacher.py",
            "tests/v3/unit/test_primitive_relation_storage.py",
            "tests/v3/unit/test_primitive_attachment_observability_sidecar.py",
            "tests/v3/unit/test_primitive_relation_targets.py",
        ]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                [PYTHON, "-m", "pytest", "-q", *tests], cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False,
            )
        subprocesses.append({"stage": "unit_tests", "returncode": result.returncode})
        if result.returncode:
            raise RuntimeError("connection hypergraph audit tests failed")

        output = run / "metrics/teacher_feasibility"
        command = [
            PYTHON,
            str(PROJECT_ROOT / "tools/v3/execute_primitive_connection_hypergraph_teacher_feasibility_v1.py"),
            "--teacher-root", str(P1B / "artifacts/teacher"),
            "--sidecar-root", str(SIDECAR / "artifacts/endpoint_observability"),
            "--output-dir", str(output),
        ]
        (run / "config/audit_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        with (run / "logs/01_teacher_feasibility.log").open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                ["/usr/bin/time", "-v", *command], cwd=PROJECT_ROOT, env=env,
                stdout=stream, stderr=subprocess.STDOUT, text=True,
                timeout=3600, check=False,
            )
        subprocesses.append({"stage": "teacher_feasibility", "returncode": result.returncode})
        log = (run / "logs/01_teacher_feasibility.log").read_text(encoding="utf-8")
        match = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", log)
        peak_rss_kib = int(match.group(1)) if match else None
        if result.returncode:
            raise RuntimeError("connection hypergraph Teacher feasibility failed")
        audit = load_json(output / "summary.json")
        if (
            audit.get("scientific_pass") is not True
            or audit.get("decision") != "PROCEED_TO_ENDPOINT_CLUSTER_DECODER_READINESS"
            or audit.get("rows_read") != 491_196
            or audit.get("fit_rows_read") != 426_552
            or audit.get("c07_rows_read") != 64_644
            or audit.get("c08_rows_read") != 0
            or audit.get("model_forward_rows") != 0
            or audit.get("optimizer_steps") != 0
        ):
            raise RuntimeError("connection hypergraph scientific contract did not pass")
        if peak_rss_kib is None or peak_rss_kib > 4 * 1024 * 1024:
            raise RuntimeError("connection hypergraph audit host RAM evidence missing or exceeded")
        required = [
            output / "summary.json", output / "per_task.json", output / "per_task.csv",
            output / "figure_source.json",
            *[
                output / f"primitive_connection_hypergraph_teacher_feasibility.{suffix}"
                for suffix in ("png", "pdf", "svg")
            ],
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("connection hypergraph Teacher evidence incomplete")
        for suffix in ("png", "pdf", "svg"):
            shutil.copy2(
                output / f"primitive_connection_hypergraph_teacher_feasibility.{suffix}",
                run / "previews",
            )

        after_trees = _dataset_trees()
        if after_trees != before_trees:
            raise RuntimeError("fit/C07 Teacher inputs changed during audit")
        write_json(run / "config/dataset_tree_integrity_after.json", after_trees)
        after_files = {relative: _sha(PROJECT_ROOT / relative) for relative in before_files}
        if after_files != before_files:
            raise RuntimeError("frozen inputs changed during connection hypergraph audit")
        write_json(run / "config/source_integrity_after.json", after_files)
        if time.monotonic() - started > 3600:
            raise RuntimeError("connection hypergraph audit wall-time exceeded")
        if sum(path.stat().st_size for path in run.rglob("*") if path.is_file()) > 100 * 1024**2:
            raise RuntimeError("connection hypergraph audit output exceeded 100 MiB")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")

    duration = time.monotonic() - started
    write_json(run / "metrics/summary.json", {
        "schema_version": "primitive_connection_hypergraph_teacher_feasibility_outer_v1",
        "overall_status": overall,
        "evidence_scientific_pass": overall == PASS,
        "primitive_connection_hypergraph_teacher_feasible": (
            overall == PASS and audit.get("scientific_pass") is True
        ),
        "error": error,
        "subprocesses": subprocesses,
        "audit": audit,
        "decision": audit.get("decision"),
        "selected_cluster_capacity": audit.get("capacity", {}).get("selected_capacity"),
        "peak_host_rss_kib": peak_rss_kib,
        "duration_seconds": duration,
        "fit_rows_read": int(audit.get("fit_rows_read", 0)),
        "c07_rows_read": int(audit.get("c07_rows_read", 0)),
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
        "model_forward_rows": 0,
        "optimizer_steps": 0,
        "graph_replays": 0,
        "mtare_worlds_read": 0,
        "source_unchanged": bool(
            before_files and before_files == after_files
            and before_trees and before_trees == after_trees
        ),
    })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error, "duration_seconds": duration,
    })
    expected_entries = len([path for path in run.rglob("*") if path.is_file()]) + 1
    write_json(run / "artifacts/seal_summary.json", {
        "schema_version": "primitive_connection_hypergraph_teacher_feasibility_seal_v1",
        "expected_evidence_entries": expected_entries,
        "overall_status": overall, "error": error,
    })
    entries = _seal(run)
    print(json.dumps({
        "overall_status": overall, "error": error,
        "decision": audit.get("decision"),
        "selected_cluster_capacity": audit.get("capacity", {}).get("selected_capacity"),
        "evidence_files": entries, "expected_evidence_files": expected_entries,
    }, indent=2))
    return 0 if overall == PASS and error is None and entries == expected_entries else 2


if __name__ == "__main__":
    raise SystemExit(main())
