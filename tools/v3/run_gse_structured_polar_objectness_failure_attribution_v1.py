#!/usr/bin/env python3
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

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_STRUCTURED_POLAR_OBJECTNESS_FAILURE_ATTRIBUTION_V1"
FAIL = "FAIL_GSE_STRUCTURED_POLAR_OBJECTNESS_FAILURE_ATTRIBUTION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0/artifacts/export/teacher"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_structured_polar_multidepth_capacity_v1_seed0"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    with target.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n")
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
        raise RuntimeError("structured-polar attribution executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    returncode = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if (
            not validation.passed
            or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_STRUCTURED_POLAR_OBJECTNESS_FAILURE_ATTRIBUTION_V1"
            or card.get("approval", {}).get("authorized_operations") != ["audit"]
            or card.get("approval", {}).get("authorized_gates") != [3]
        ):
            raise RuntimeError(f"attribution Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "python": platform.python_version(), "numpy": numpy.__version__, "gpu": "none"})
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_structured_polar_objectness_failure_attribution_v1.py"), "--teacher-root", str(TEACHER), "--capacity-run", str(CAPACITY), "--output-dir", str(run_dir / "artifacts/audit")]
        write_json(run_dir / "config/commands.json", [command])
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/00_attribution.log").open("w", encoding="utf-8") as stream:
            returncode = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1200, check=False).returncode
        if returncode != 0:
            raise RuntimeError(f"attribution executor failure: {returncode}")
        result = load_json(run_dir / "artifacts/audit/summary.json")
        if (
            result.get("status") != PASS
            or result.get("population", {}).get("observations") != 45942
            or result.get("population", {}).get("target_tokens") != 33145
            or result.get("population", {}).get("second_depth_targets") != 79
            or result.get("decision") not in {
                "STOP_DENSE_PROPOSAL_ROUTE",
                "ONE_PER_BIN_CORRECTIVE_JUSTIFIED",
                "CARDINALITY_RANKING_CORRECTIVE_JUSTIFIED",
                "FROZEN_GEOMETRY_OBJECTNESS_REFIT_REQUIRED",
            }
            or any(result.get(name) != 0 for name in ("optimizer_steps", "model_inference_frames", "threshold_selection_steps", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("attribution result drift")
        required = [
            run_dir / "artifacts/audit/summary.json",
            run_dir / "artifacts/audit/figure_source.json",
            run_dir / "artifacts/audit/seed_attribution.csv",
            run_dir / "artifacts/audit/gse_structured_polar_objectness_failure_attribution_v1.png",
            run_dir / "artifacts/audit/gse_structured_polar_objectness_failure_attribution_v1.pdf",
            run_dir / "artifacts/audit/gse_structured_polar_objectness_failure_attribution_v1.svg",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("attribution evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("attribution source changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {"schema_version": "gse_structured_polar_objectness_failure_attribution_outer_v1", "overall_status": overall, "error": error, "executor_returncode": returncode, "result": result, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "optimizer_steps": 0, "model_inference_frames": 0, "threshold_selection_steps": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED" if overall == PASS and error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "decision": result.get("decision"), "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
