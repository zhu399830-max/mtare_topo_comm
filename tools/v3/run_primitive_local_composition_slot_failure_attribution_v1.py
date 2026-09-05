#!/usr/bin/env python3
"""Formal wrapper for the frozen C07 local-slot factor attribution."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import traceback

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json
from run_primitive_relation_sparse_port_three_seed_training_v1r import _run_training_monitored


RUN_ID = "gate3_20260904_primitive_local_composition_slot_failure_attribution_v1_seed0"
PASS = "PASS_PRIMITIVE_LOCAL_COMPOSITION_SLOT_FAILURE_ATTRIBUTION_V1"
FAIL = "FAIL_PRIMITIVE_LOCAL_COMPOSITION_SLOT_FAILURE_ATTRIBUTION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_PRIMITIVE_LOCAL_COMPOSITION_SLOT_FAILURE_ATTRIBUTION_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
EXPECTED_TESTS = 16
MAXIMUM_HOST_RSS_BYTES = 4 * 1024**3
MAXIMUM_RESULT_BYTES = 1 * 1024**3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _directory_size(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(value for value in run.rglob("*") if value.is_file() and value != target)
    target.write_text("".join(
        f"{_sha(value)}  {value.relative_to(PROJECT_ROOT)}\n" for value in files
    ), encoding="utf-8")
    return len(files)


def _run(command: list[str], log: Path, env: dict[str, str], timeout: int) -> int:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, env=env, stdout=stream,
            stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False,
        )
    return int(completed.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve()
    started = time.monotonic(); overall = FAIL; error = None; checks = {}; summary = {}; subprocesses = []
    before = {}; peak_host = 0
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("local-slot attribution executes exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"local-slot attribution Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"frozen tool drift: {record['path']}")
        versions = {
            "python": sys.version.split()[0], "executable": sys.executable,
            "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda,
            "zarr": zarr.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }
        expected_versions = {
            "python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3",
            "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7",
            "gpu": "NVIDIA GeForce RTX 5090 D",
        }
        if versions != expected_versions: raise RuntimeError(f"environment drift: {versions}")
        write_json(run / "config/environment.json", {"versions": versions, "platform": platform.platform()})
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = os.pathsep.join((
            str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT),
        ))
        env["PYTHONHASHSEED"] = "0"; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = [
            "tests/v3/unit/test_primitive_local_composition_slot_failure_attribution.py",
            "tests/v3/unit/test_primitive_local_composition_slot_decoding.py",
            "tests/v3/unit/test_primitive_relation_sparse_port_failure_attribution.py",
        ]
        rc = _run([PYTHON, "-m", "pytest", "-q", *tests], run / "logs/00_unit_tests.log", env, 600)
        subprocesses.append({"stage": "unit_tests", "returncode": rc})
        if rc or f"{EXPECTED_TESTS} passed" not in (run / "logs/00_unit_tests.log").read_text(encoding="utf-8"):
            raise RuntimeError(f"expected exactly {EXPECTED_TESTS} attribution tests")
        command = list(spec["diagnostic_command"])
        (run / "config/diagnostic_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        monitor = _run_training_monitored(
            command, log=run / "logs/01_c07_factor_attribution.log", environment=env,
            timeout_seconds=14_400, maximum_host_rss_bytes=MAXIMUM_HOST_RSS_BYTES,
        )
        write_json(run / "metrics/resource_monitor.json", monitor); subprocesses.append({"stage": "attribution", **monitor})
        peak_host = int(monitor["peak_host_rss_bytes"])
        if monitor["returncode"] or monitor["killed_for_host_limit"]:
            raise RuntimeError(f"C07 factor attribution failed: {monitor}")
        output = run / "metrics/attribution"; summary = load_json(output / "summary.json")
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        checks = {
            "attribution_pass": summary.get("overall_status") == PASS and summary.get("scientific_pass") is True,
            "all_frozen_inputs_unchanged": after == before,
            "source_metrics_reproduced": summary.get("checks", {}).get("three_source_metrics_exactly_reproduced") is True,
            "complete_population": summary.get("checks", {}).get("complete_c07_population") is True,
            "zero_optimizer_c08_graph_mtare": summary.get("optimizer_steps") == 0
                and summary.get("c08_rows_read") == 0 and summary.get("graph_replays") == 0
                and summary.get("mtare_worlds_read") == 0,
            "resource_cap": peak_host <= MAXIMUM_HOST_RSS_BYTES,
        }
        if not all(checks.values()): raise RuntimeError(f"attribution outer evidence drift: {checks}")
        write_json(run / "config/source_integrity_after.json", after)
        for suffix in ("png", "pdf", "svg"):
            source = output / f"primitive_local_composition_slot_failure_attribution.{suffix}"
            shutil.copy2(source, run / "previews" / source.name)
        size = _directory_size(run)
        if size > MAXIMUM_RESULT_BYTES: raise RuntimeError("attribution result exceeds 1 GiB")
        overall = PASS
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_local_composition_slot_failure_attribution_outer_v1",
            "overall_status": overall, "scientific_pass": True,
            "decision": summary["attribution_diagnosis"], "recommendation": summary["recommendation"],
            "checks": checks, "optimizer_steps": 0, "c08_rows_read": 0,
            "graph_replays": 0, "mtare_worlds_read": 0,
            "peak_host_rss_bytes": peak_host, "result_bytes_before_seal": size,
            "subprocesses": subprocesses, "duration_seconds": time.monotonic() - started,
            "error": None,
        })
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {
            "schema_version": "primitive_local_composition_slot_failure_attribution_outer_v1",
            "overall_status": FAIL, "scientific_pass": False,
            "decision": "STOP_LOCAL_SLOT_ATTRIBUTION_SYSTEM_FAILURE", "checks": checks,
            "optimizer_steps": 0, "c08_rows_read": 0, "graph_replays": 0,
            "mtare_worlds_read": 0, "subprocesses": subprocesses,
            "duration_seconds": time.monotonic() - started, "error": error,
        })
    write_json(run / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if error is None else "FAILED",
        "overall_status": overall, "error": error,
        "duration_seconds": time.monotonic() - started,
    })
    entries = _seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "evidence_files": entries}, indent=2))
    return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
