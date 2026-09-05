#!/usr/bin/env python3
"""Execute and seal the frozen learned-center trace-commit graph audit."""

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


RUN_ID = "gate3_20260828_gse_projected_trace_commit_capacity_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
PASS = "PASS_GSE_PROJECTED_TRACE_COMMIT_CAPACITY_V1"
FAIL = "FAIL_GSE_PROJECTED_TRACE_COMMIT_CAPACITY_V1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify(spec: dict) -> dict:
    observed = {}
    for relative, expected in spec["frozen_inputs"].items():
        actual = _sha(PROJECT_ROOT / relative)
        if actual != expected:
            raise RuntimeError(f"projected trace-commit input drift: {relative}")
        observed[relative] = actual
    return observed


def _seal(run_dir: Path) -> int:
    seal = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != seal)
    with seal.open("w", encoding="utf-8") as stream:
        for path in files:
            stream.write(f"{_sha(path)}  {path.relative_to(PROJECT_ROOT)}\n")
    return len(files)


def _run(command: list[str], log: Path, env: dict[str, str], allowed: tuple[int, ...]) -> int:
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, cwd=PROJECT_ROOT, env=env, text=True,
            stdout=stream, stderr=subprocess.STDOUT, timeout=3600, check=False,
        )
    code = int(completed.returncode)
    if code not in allowed:
        raise RuntimeError(f"subprocess failed with return code {code}: {command[1]}")
    return code


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    spec = load_json(args.spec.resolve())
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("projected trace-commit audit may execute only once")
    started = time.monotonic()
    overall, error, result = FAIL, None, {}
    before = after = {}
    inference_code = replay_code = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit":
            raise RuntimeError("projected trace-commit operation scope drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"projected trace-commit frozen tool drift: {record['path']}")
        before = _verify(spec)
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        env = os.environ.copy()
        env.update({
            "PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"),
            "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        })
        teacher = PROJECT_ROOT / "results/gate2_representation/gate2_20260827_gse_corrected_causal_teacher_manifest_v1r_seed0"
        dataset = PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0"
        source = PROJECT_ROOT / "results/gate3_semantics/gate3_20260826_gse_exit_token_association_corrective_v2_seed0"
        capacity = PROJECT_ROOT / "results/gate3_semantics/gate3_20260827_gse_factorized_association_capacity_v1r_seed0"
        action = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
        center = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_event_center_offset_training_v1r3_seed0"
        projection = run_dir / "artifacts/projection/event_center_projection.npz"
        projection.parent.mkdir(parents=True, exist_ok=False)
        inference = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/infer_gse_event_center_offset_all_v1.py"),
            "--action-cache", str(action / "scratch/action_set_cache"),
            "--action-model-run", str(action), "--center-model-run", str(center),
            "--pair-cache", str(source / "artifacts/pair_cache/pairs.npz"),
            "--output", str(projection),
        ]
        replay = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_trace_commit_capacity_v1.py"),
            "--teacher", str(teacher / "artifacts/teacher_observations.jsonl"),
            "--sequence-manifest", str(dataset / "artifacts/sequence_manifest.jsonl"),
            "--pair-cache", str(source / "artifacts/pair_cache/pairs.npz"),
            "--action-cache", str(action / "scratch/action_set_cache"),
            "--action-model-run", str(action), "--association-capacity-run", str(capacity),
            "--center-offsets", str(projection),
        ]
        for seed in range(3):
            replay.extend((f"--observation{seed}", str(capacity / f"artifacts/unified_observation/seed{seed}_unified_observation_features.npy")))
            replay.extend((f"--tokens{seed}", str(source / f"artifacts/models/seed{seed}/frozen_exit_token_outputs.npz")))
        replay.extend(("--output-dir", str(run_dir / "artifacts/replay")))
        (run_dir / "config/inference_command.txt").write_text(" ".join(inference) + "\n", encoding="utf-8")
        (run_dir / "config/replay_command.txt").write_text(" ".join(replay) + "\n", encoding="utf-8")
        inference_code = _run(inference, run_dir / "logs/inference.log", env, (0,))
        replay_code = _run(replay, run_dir / "logs/replay.log", env, (0, 2))
        result = load_json(run_dir / "artifacts/replay/summary.json")
        scientific_pass = result.get("status") == "PASS_GSE_TRACE_COMMIT_CAPACITY_V1"
        if (replay_code == 0) != scientific_pass:
            raise RuntimeError("projected trace-commit status/return mismatch")
        after = _verify(spec)
        if before != after:
            raise RuntimeError("projected trace-commit sources changed during replay")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_projected_trace_commit_capacity_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS,
        "error": error, "inference_returncode": inference_code, "replay_returncode": replay_code,
        "duration_seconds": time.monotonic() - started, "result": result,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "model_updates": 0,
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
