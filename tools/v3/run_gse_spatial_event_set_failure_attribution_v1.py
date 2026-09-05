#!/usr/bin/env python3
"""Execute and seal one read-only spatial event-set failure attribution."""

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


RUN_ID = "gate3_20260828_gse_spatial_event_set_failure_attribution_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_SPATIAL_EVENT_SET_FAILURE_ATTRIBUTION_V1"
FAIL = "FAIL_GSE_SPATIAL_EVENT_SET_FAILURE_ATTRIBUTION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_SET_FAILURE_ATTRIBUTION_V1"
TEACHER = PROJECT_ROOT / "results/gate2_representation/gate2_20260828_gse_spatial_multi_event_teacher_export_v1_seed0/artifacts/export/teacher"
CAPACITY = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_event_set_capacity_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    values = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"failure attribution frozen input drift: {relative}")
        values[relative] = actual
    return values


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
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("failure attribution formal run may execute only once")
    started = time.monotonic(); overall = FAIL; error = None; result = {}; before = {}; after = {}; returncode = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if (
            spec.get("gate") != 3 or spec.get("operation") != "audit"
            or spec.get("user_authorization", {}).get("status") != "APPROVED"
            or card.get("approval", {}).get("status") != "APPROVED"
            or card.get("status") != CARD_STATUS
        ):
            raise RuntimeError("failure attribution authorization/scope drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"failure attribution tool drift: {record['path']}")
        before = _verify(spec); write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        output = run_dir / "artifacts/audit"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_spatial_event_set_failure_attribution_v1.py"),
            "--teacher-root", str(TEACHER), "--capacity-run", str(CAPACITY),
            "--output-dir", str(output),
        ]
        with (run_dir / "logs/01_audit.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=900, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError("failure attribution executor failed")
        result = load_json(output / "summary.json")
        if (
            result.get("status") != PASS
            or result.get("population") != {"worlds": 20, "observations": 45942, "target_tokens": 33563, "multi_event_rows": 6153}
            or len(result.get("seed_records", [])) != 3
            or result.get("decision") not in {
                "DUPLICATE_QUERY_DOMINANT_MINIMAL_REPULSION_CORRECTIVE_ALLOWED",
                "CONFIDENCE_CARDINALITY_DOMINANT_CALIBRATION_CORRECTIVE_ALLOWED",
                "ENCODER_SPATIAL_REPRESENTATION_INSUFFICIENT_NEW_ENCODER_OBJECTIVE_REQUIRED",
            }
            or any(result.get(key) != 0 for key in (
                "optimizer_steps", "model_inference_frames", "threshold_selection_steps",
                "c09_worlds_read", "c10_worlds_read", "mtare_worlds_read",
            ))
        ):
            raise RuntimeError("failure attribution evidence contract drift")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("failure attribution sources changed")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_spatial_event_set_failure_attribution_outer_v1",
        "overall_status": overall, "error": error, "executor_returncode": returncode,
        "result": result, "optimizer_steps": 0, "model_inference_frames": 0,
        "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
