#!/usr/bin/env python3
"""Execute and seal structured-polar multi-depth zero-training readiness."""
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

import numpy
import torch
import zarr

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_structured_polar_multidepth_readiness_v1_seed0"
PASS = "PASS_GSE_STRUCTURED_POLAR_MULTIDEPTH_READINESS_V1"
FAIL = "FAIL_GSE_STRUCTURED_POLAR_MULTIDEPTH_READINESS_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0/artifacts/export/teacher"
SOURCE = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/dataset"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("structured-polar readiness executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("structured-polar readiness scope drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"structured-polar readiness tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"structured-polar readiness input drift: {relative}")
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "python": platform.python_version(), "torch": torch.__version__, "numpy": numpy.__version__, "zarr": zarr.__version__, "cuda": torch.version.cuda})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_structured_polar_multidepth_readiness_v1.py"), "--teacher-root", str(TEACHER), "--source-root", str(SOURCE), "--output-dir", str(run_dir / "artifacts/audit")]
        (run_dir / "config/commands.json").write_text(json.dumps([command], indent=2) + "\n", encoding="utf-8")
        with (run_dir / "logs/01_readiness.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1800, check=False)
        returncode = int(completed.returncode)
        if returncode not in (0, 2):
            raise RuntimeError(f"structured-polar readiness executor system failure: {returncode}")
        result = load_json(run_dir / "artifacts/audit/summary.json")
        population = result.get("population", {})
        if (
            result.get("status") not in (PASS, FAIL)
            or population.get("worlds") != 80
            or population.get("observations") != 188126
            or population.get("tokens") != 131424
            or population.get("event_identities") != 1076
            or population.get("slot1_tokens") != 277
            or result.get("model", {}).get("parameters") != 172430
            or any(result.get(name) != 0 for name in ("optimizer_steps", "trained_model_inference_frames", "threshold_selection_steps", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("structured-polar readiness result drift")
        if not (run_dir / "artifacts/audit/summary.json").is_file():
            raise RuntimeError("structured-polar readiness evidence incomplete")
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("structured-polar readiness source changed")
        overall = str(result["status"])
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {"schema_version": "gse_structured_polar_multidepth_readiness_outer_v1", "overall_status": overall, "error": error, "returncode": returncode, "result": result, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "optimizer_steps": 0, "trained_model_inference_frames": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS and error is None else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
