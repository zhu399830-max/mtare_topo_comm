#!/usr/bin/env python3
"""Execute and seal the event-center teacher generation."""

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


RUN_ID = "gate2_20260828_gse_event_center_teacher_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_EVENT_CENTER_TEACHER_V1"
FAIL = "FAIL_GSE_EVENT_CENTER_TEACHER_V1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict:
    result = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"event-center frozen input drift: {relative}")
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
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("event-center teacher may execute only once")
    started = time.monotonic()
    overall, error, result, returncode = FAIL, None, {}, None
    before = after = {}
    try:
        if spec.get("gate") != 2 or spec.get("operation") != "teacher_generation":
            raise RuntimeError("event-center teacher scope drift")
        for value in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / value["path"]) != value["sha256"]:
                raise RuntimeError(f"event-center frozen tool drift: {value['path']}")
        before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/generate_gse_event_center_teacher_v1.py"),
            "--teacher", str(PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0/artifacts/teacher_observations.jsonl"),
            "--pair-cache", str(PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0/artifacts/pair_cache/pairs.npz"),
            "--parent-manifest", str(PROJECT_ROOT / "results/gate0_baseline/gate0_20260811_cano_100_topology_parent_recipe_reclassification_v2r_seed0/artifacts/accepted_parent_manifest.json"),
            "--output-dir", str(run_dir / "artifacts/teacher"),
        ]
        (run_dir / "config/command.txt").write_text(" ".join(command) + "\n", encoding="utf-8")
        with (run_dir / "logs/generator.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=600, check=False)
        returncode = int(completed.returncode)
        if returncode != 0:
            raise RuntimeError("event-center teacher generator failed")
        result = load_json(run_dir / "artifacts/teacher/summary.json")
        if result.get("status") != PASS:
            raise RuntimeError("event-center teacher scientific status drift")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("event-center teacher inputs changed during generation")
        overall = PASS
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_event_center_teacher_outer_v1", "overall_status": overall,
        "scientific_pass": overall == PASS, "error": error, "returncode": returncode,
        "teacher": result, "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after), "optimizer_steps": 0,
        "model_inference_frames": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "COMPLETED" if overall == PASS else "FAILED", "overall_status": overall, "error": error,
    })
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
