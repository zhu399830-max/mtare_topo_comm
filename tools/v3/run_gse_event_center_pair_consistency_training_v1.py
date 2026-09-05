#!/usr/bin/env python3
"""Train, evaluate and seal three cross-traversal event-center heads."""

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


RUN_ID = "gate3_20260828_gse_event_center_vector_training_v2_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_EVENT_CENTER_VECTOR_TRAINING_V2"
FAIL = "FAIL_GSE_EVENT_CENTER_VECTOR_TRAINING_V2"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""): digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict:
    result = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected: raise RuntimeError(f"pair-consistency frozen input drift: {relative}")
        result[relative] = actual
    return result


def _run(command, log, env, timeout):
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False)
    return int(completed.returncode)


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"; files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files: stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); run_dir = args.run_dir.resolve(); spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("pair-consistency training may execute only once")
    started = time.monotonic(); overall, error, ensemble = FAIL, None, {}; processes = []; before = after = {}
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "training": raise RuntimeError("pair-consistency training scope drift")
        for value in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / value["path"]) != value["sha256"]: raise RuntimeError(f"pair-consistency frozen tool drift: {value['path']}")
        before = _verify(spec); write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy(); env.update({"PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"), "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "CUBLAS_WORKSPACE_CONFIG": ":4096:8"})
        action = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
        prior = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
        scalar_prior = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_pair_consistency_training_v1r3_seed0"
        graph = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
        teacher = prior / "artifacts/teacher/event_center_teacher.npz"
        for seed in range(3):
            command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/train_gse_event_center_vector_v1.py"), "--action-cache", str(action / "scratch/action_set_cache"), "--teacher", str(teacher), "--action-checkpoint", str(action / f"artifacts/models/seed{seed}/best.pt"), "--scalar-checkpoint", str(scalar_prior / f"artifacts/models/seed{seed}/best.pt"), "--output-dir", str(run_dir / f"artifacts/models/seed{seed}"), "--seed", str(seed)]
            code = _run(command, run_dir / f"logs/0{seed + 1}_seed{seed}.log", env, 1800); processes.append({"stage": f"seed{seed}", "returncode": code})
            if code != 0: raise RuntimeError(f"pair-consistency seed{seed} training failed")
        evaluate = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/evaluate_gse_event_center_vector_ensemble_v1.py")]
        for seed in range(3): evaluate.extend((f"--seed{seed}", str(run_dir / f"artifacts/models/seed{seed}")))
        evaluate.extend(("--baseline-projection", str(graph / "artifacts/projection/event_center_projection.npz"), "--output-dir", str(run_dir / "metrics/ensemble")))
        code = _run(evaluate, run_dir / "logs/04_ensemble.log", env, 600); processes.append({"stage": "ensemble", "returncode": code})
        if code not in (0, 2): raise RuntimeError("pair-consistency evaluator program failure")
        ensemble = load_json(run_dir / "metrics/ensemble/summary.json")
        scientific_pass = ensemble.get("status") == "PASS_GSE_EVENT_CENTER_VECTOR_ENSEMBLE_V1"
        if (code == 0) != scientific_pass: raise RuntimeError("pair-consistency status/return mismatch")
        after = _verify(spec)
        if before != after: raise RuntimeError("pair-consistency sources changed during training")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    optimizer_steps = sum(int(load_json(path).get("optimizer_steps", 0)) for path in [run_dir / f"artifacts/models/seed{seed}/summary.json" for seed in range(3)] if path.exists())
    summary = {"schema_version": "gse_event_center_pair_consistency_training_outer_v1", "overall_status": overall, "scientific_pass": overall == PASS, "error": error, "ensemble": ensemble, "subprocesses": processes, "optimizer_steps": optimizer_steps, "backbone_optimizer_steps": 0, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0}
    write_json(run_dir / "metrics/summary.json", summary); write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir); print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2)); return 0 if overall == PASS else 2


if __name__ == "__main__": raise SystemExit(main())
