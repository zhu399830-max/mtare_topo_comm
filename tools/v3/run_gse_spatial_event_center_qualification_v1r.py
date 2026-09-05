#!/usr/bin/env python3
"""Run and seal the evaluation-only spatial event-center qualification."""

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


RUN_ID = "gate3_20260828_gse_spatial_event_center_qualification_v1r_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_SPATIAL_EVENT_CENTER_QUALIFICATION_V1R"
FAIL = "FAIL_GSE_SPATIAL_EVENT_CENTER_QUALIFICATION_V1R"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_spatial_event_center_training_v1_seed0"
ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
CENTER = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
PROJECTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict[str, str]:
    result = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"qualification frozen input drift: {relative}")
        result[relative] = actual
    return result


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
    run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("spatial qualification may execute only once")
    started = time.monotonic(); overall = FAIL; error = None; ensemble = {}; before = after = {}; returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "checkpoint_selection":
            raise RuntimeError("spatial qualification scope drift")
        card = load_json(PROJECT_ROOT / spec["data_card"])
        if card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPATIAL_EVENT_CENTER_QUALIFICATION_V1R" or card.get("approval", {}).get("status") != "APPROVED":
            raise RuntimeError("spatial qualification Data Card drift")
        predecessor = load_json(TRAINING / "metrics/summary.json")
        if (
            predecessor.get("overall_status") != "FAIL_GSE_SPATIAL_EVENT_CENTER_TRAINING_V1"
            or predecessor.get("optimizer_steps") != 11880
            or predecessor.get("error") != "RuntimeError: spatial event-center ensemble evaluator program failure"
        ):
            raise RuntimeError("spatial qualification predecessor boundary drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"spatial qualification frozen tool drift: {record['path']}")
        before = _verify(spec); write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        output = run_dir / "metrics/ensemble"
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_spatial_event_center_ensemble_v1.py")]
        for seed in (0, 1, 2):
            command.extend((f"--seed{seed}", str(TRAINING / f"artifacts/models/seed{seed}")))
        command.extend((
            "--baseline-projection", str(PROJECTION / "artifacts/projection/event_center_projection.npz"),
            "--action-cache", str(ACTION / "scratch/action_set_cache"),
            "--teacher", str(CENTER / "artifacts/teacher/event_center_teacher.npz"),
            "--output-dir", str(output),
        ))
        with (run_dir / "logs/01_ensemble.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=120, check=False)
        returncode = int(completed.returncode)
        if returncode not in (0, 2):
            raise RuntimeError("spatial qualification evaluator program failure")
        ensemble = load_json(output / "summary.json")
        scientific_pass = ensemble.get("status") == "PASS_GSE_SPATIAL_EVENT_CENTER_ENSEMBLE_V1"
        if (returncode == 0) != scientific_pass or ensemble.get("selection_rows") != 8839:
            raise RuntimeError("spatial qualification status/count drift")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("spatial qualification sources changed")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_spatial_event_center_qualification_outer_v1r",
        "overall_status": overall, "scientific_pass": overall == PASS,
        "error": error, "evaluator_returncode": returncode, "ensemble": ensemble,
        "new_optimizer_steps": 0, "reused_sealed_optimizer_steps": 11880,
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
