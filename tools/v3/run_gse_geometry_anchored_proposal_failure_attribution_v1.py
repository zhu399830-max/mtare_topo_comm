#!/usr/bin/env python3
"""Execute and seal one immutable geometry-anchored proposal attribution."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_geometry_anchored_proposal_failure_attribution_v1_seed0"
PASS = "PASS_GSE_GEOMETRY_ANCHORED_PROPOSAL_FAILURE_ATTRIBUTION_V1"
FAIL = "FAIL_GSE_GEOMETRY_ANCHORED_PROPOSAL_FAILURE_ATTRIBUTION_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_observable_spatial_event_teacher_v2_seed0/artifacts/export/teacher"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_geometry_anchored_joint_capacity_v1_seed0"
EXECUTOR = PROJECT_ROOT / "tools/v3/execute_gse_geometry_anchored_proposal_failure_attribution_v1.py"


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
        raise RuntimeError("proposal attribution executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    result: dict[str, object] = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    returncode = None
    try:
        if (
            spec.get("gate") != 3
            or spec.get("operation") != "audit"
            or spec.get("user_authorization", {}).get("status") != "APPROVED"
        ):
            raise RuntimeError("proposal attribution scope drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"proposal attribution tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = _sha(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"proposal attribution input drift: {relative}")
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        environment = os.environ.copy()
        environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        command = [
            str(PYTHON),
            str(EXECUTOR),
            "--teacher-root", str(TEACHER),
            "--capacity-run", str(CAPACITY),
            "--output-dir", str(run_dir / "artifacts/audit"),
        ]
        (run_dir / "config/commands.json").write_text(json.dumps([command], indent=2) + "\n", encoding="utf-8")
        with (run_dir / "logs/01_attribution.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1200, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError(f"proposal attribution executor failure: {returncode}")
        result = load_json(run_dir / "artifacts/audit/summary.json")
        if (
            result.get("status") != PASS
            or result.get("population") != {
                "worlds": 20,
                "observations": 45942,
                "target_tokens": 33145,
                "target_types": [7578, 25567],
                "cardinality": [19159, 20710, 5786, 285, 2, 0],
                "event_identities": 275,
            }
            or len(result.get("seed_records", [])) != 3
            or result.get("decision") not in {
                "DUPLICATE_PROPOSAL_CORRECTIVE_REQUIRED",
                "EXPLICIT_OBJECTNESS_CARDINALITY_CORRECTIVE_REQUIRED",
                "TYPE_CONDITIONING_CORRECTIVE_REQUIRED",
                "STRUCTURED_POLAR_PROPOSAL_REQUIRED",
            }
            or any(result.get(name) != 0 for name in ("optimizer_steps", "model_inference_frames", "threshold_selection_steps", "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read"))
        ):
            raise RuntimeError("proposal attribution evidence drift")
        required = [
            run_dir / "artifacts/audit/summary.json",
            run_dir / "artifacts/audit/figure_source.json",
            run_dir / "artifacts/audit/gse_geometry_anchored_proposal_failure_attribution_v1.png",
            run_dir / "artifacts/audit/gse_geometry_anchored_proposal_failure_attribution_v1.pdf",
            run_dir / "artifacts/audit/gse_geometry_anchored_proposal_failure_attribution_v1.svg",
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("proposal attribution evidence incomplete")
        after = {relative: _sha(PROJECT_ROOT / relative) for relative in before}
        if after != before:
            raise RuntimeError("proposal attribution source changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_geometry_anchored_proposal_failure_attribution_outer_v1",
        "overall_status": overall,
        "error": error,
        "returncode": returncode,
        "result": result,
        "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0,
        "model_inference_frames": 0,
        "threshold_selection_steps": 0,
        "c09_worlds_read": 0,
        "c10_worlds_read": 0,
        "mtare_worlds_read": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1",
        "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall,
        "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries, "decision": result.get("decision")}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
