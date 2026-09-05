#!/usr/bin/env python3
"""Run and seal the structural-node tiny-overfit gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import sys
import time
import traceback

import numpy as np
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json
from run_primitive_relation_sparse_port_three_seed_training_v1r import _run_training_monitored


RUN_ID = "gate3_20260904_gse_structural_node_tiny_overfit_v1_seed0"
PASS = "PASS_GSE_STRUCTURAL_NODE_TINY_OVERFIT_V1"
FAIL = "FAIL_GSE_STRUCTURAL_NODE_TINY_OVERFIT_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_STRUCTURAL_NODE_TINY_OVERFIT_V1"
PYTHON = "/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python"
P1A = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1a_lossless_storage_corrective_v1_seed0"
P1B = PROJECT_ROOT / "results/gate3_semantics/gate3_20260830_primitive_relation_p1b_teacher_materialization_v1r_seed0"
OBS = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_attachment_observability_sidecar_v1_seed0"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0/artifacts/models/seed0/selected.pt"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260904_gse_structural_node_graph_consistency_attribution_v1_seed0"
MAX_GPU = 16 * 1024**3
MAX_HOST = 4 * 1024**3


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(value for value in run.rglob("*") if value.is_file() and value != target)
    target.write_text("".join(f"{_sha(value)}  {value.relative_to(PROJECT_ROOT)}\n" for value in files), encoding="utf-8")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve(); started = time.monotonic()
    overall, error, scientific_pass, result = FAIL, None, False, {}
    before: dict[str, str] = {}; subprocesses = []; monitor = {}
    try:
        if run.name != RUN_ID or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
            raise RuntimeError("tiny-overfit may execute exactly once")
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"tiny-overfit Data Card invalid: {validation.errors}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"tiny-overfit frozen input drift: {relative}")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"tiny-overfit frozen tool drift: {record['path']}")
        capacity = load_json(CAPACITY / "metrics/attribution/summary.json")
        if capacity.get("status") != "PASS_GSE_STRUCTURAL_NODE_GRAPH_CONSISTENCY_ATTRIBUTION_V1" or not capacity.get("scientific_pass"):
            raise RuntimeError("tiny-overfit graph-consistency prerequisite drift")
        versions = {"python": sys.version.split()[0], "executable": sys.executable, "numpy": np.__version__, "torch": torch.__version__, "cuda": torch.version.cuda, "zarr": zarr.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        expected = {"python": "3.13.5", "executable": PYTHON, "numpy": "2.1.3", "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7", "gpu": "NVIDIA GeForce RTX 5090 D"}
        if versions != expected: raise RuntimeError(f"tiny-overfit environment drift: {versions}")
        write_json(run / "config/environment.json", {"versions": versions, "platform": platform.platform()}); write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = os.pathsep.join((str(PROJECT_ROOT / "src"), str(PROJECT_ROOT / "tools/v3"), str(PROJECT_ROOT))); env["PYTHONHASHSEED"] = "0"; env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        tests = ["tests/v3/unit/test_gse_structural_node_evidence.py", "tests/v3/unit/test_gse_structural_node_student.py"]
        test_monitor = _run_training_monitored([PYTHON, "-m", "pytest", "-q", *tests], log=run / "logs/00_unit_tests.log", environment=env, timeout_seconds=600, maximum_host_rss_bytes=MAX_HOST)
        subprocesses.append({"stage": "unit_tests", **test_monitor})
        if test_monitor["returncode"] or "9 passed" not in (run / "logs/00_unit_tests.log").read_text(encoding="utf-8"):
            raise RuntimeError("expected exactly 9 tiny-overfit tests")
        output = run / "artifacts/tiny_overfit"
        command = [PYTHON, str(PROJECT_ROOT / "tools/v3/train_gse_structural_node_tiny_overfit_v1.py"), "--sensor-root", str(P1A / "artifacts/dataset/fit"), "--teacher-root", str(P1B / "artifacts/teacher/fit"), "--observability-root", str(OBS / "artifacts/endpoint_observability/fit"), "--construction-root", str(P1A / "artifacts/constructions/fit"), "--traversal-manifest", str(P1A / "artifacts/traversal_manifest.jsonl"), "--source-checkpoint", str(SOURCE), "--output-dir", str(output), "--steps", "500"]
        (run / "config/training_command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        monitor = _run_training_monitored(command, log=run / "logs/01_tiny_overfit.log", environment=env, timeout_seconds=1800, maximum_host_rss_bytes=MAX_HOST)
        subprocesses.append({"stage": "tiny_overfit", **monitor})
        if monitor["returncode"] not in (0, 2) or monitor["killed_for_host_limit"]:
            raise RuntimeError("tiny-overfit program/resource failure")
        result = load_json(output / "summary.json"); scientific_pass = result.get("status") == PASS
        if scientific_pass != (monitor["returncode"] == 0): raise RuntimeError("tiny-overfit status/return mismatch")
        checks = {
            "population": result.get("population", {}).get("observations") == 180 and result.get("population", {}).get("nodes") == 100,
            "steps": result.get("optimizer_steps") == 500,
            "resources": int(result.get("peak_cuda_allocated_bytes", MAX_GPU + 1)) <= MAX_GPU and int(result.get("peak_cuda_reserved_bytes", MAX_GPU + 1)) <= MAX_GPU and int(monitor["peak_host_rss_bytes"]) <= MAX_HOST,
            "zero_c07_test_graph": result.get("c07_rows_read") == 0 and result.get("c08_c09_c10_worlds_read") == 0 and result.get("graph_replays") == 0 and result.get("mtare_runs") == 0,
            "artifacts": all((output / name).is_file() for name in ("frozen_tiny_features.npz", "sample_manifest.json", "tiny_overfit_head.pt", "tiny_overfit_diagnostic.png", "tiny_overfit_diagnostic.pdf", "tiny_overfit_diagnostic.svg")),
        }
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}; checks["source_unchanged"] = after == before
        if not all(checks.values()): raise RuntimeError(f"tiny-overfit outer contract drift: {checks}")
        for suffix in ("png", "pdf", "svg"): shutil.copy2(output / f"tiny_overfit_diagnostic.{suffix}", run / "previews" / f"tiny_overfit_diagnostic.{suffix}")
        overall = PASS if scientific_pass else FAIL
        write_json(run / "config/source_integrity_after.json", after)
        write_json(run / "metrics/summary.json", {"schema_version": "gse_structural_node_tiny_overfit_outer_v1", "overall_status": overall, "scientific_pass": scientific_pass, "error": None, "checks": checks, "tiny_overfit": result, "resource_monitor": monitor, "subprocesses": subprocesses, "optimizer_steps": 500, "c07_rows_read": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0, "duration_seconds": time.monotonic() - started})
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
        write_json(run / "metrics/summary.json", {"schema_version": "gse_structural_node_tiny_overfit_outer_v1", "overall_status": FAIL, "scientific_pass": False, "error": error, "tiny_overfit": result, "resource_monitor": monitor, "subprocesses": subprocesses, "c07_rows_read": 0, "c08_c09_c10_worlds_read": 0, "graph_replays": 0, "mtare_runs": 0, "duration_seconds": time.monotonic() - started})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error, "duration_seconds": time.monotonic() - started})
    entries = _seal(run); print(json.dumps({"overall_status": overall, "scientific_pass": scientific_pass, "error": error, "evidence_files": entries}, indent=2)); return 0 if error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
