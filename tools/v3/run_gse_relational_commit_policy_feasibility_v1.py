#!/usr/bin/env python3
"""Execute and seal the frozen relational commit-policy feasibility audit."""

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
from mtare_topo.governance import load_json, validate_data_card, write_json


PASS = "PASS_GSE_RELATIONAL_COMMIT_POLICY_FEASIBILITY_V1"
FAIL = "FAIL_GSE_RELATIONAL_COMMIT_POLICY_FEASIBILITY_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
ACTION = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
TRAINING = PROJECT_ROOT / "results/gate3_semantics/gate3_20260829_gse_relational_exit_transport_training_v1_seed0"


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
        raise RuntimeError("commit-policy audit executes exactly once")
    started = time.monotonic()
    overall = FAIL
    error = None
    result = {}
    before: dict[str, str] = {}
    after: dict[str, str] = {}
    returncode = None
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != "APPROVED_FOR_ONE_IMMUTABLE_GSE_RELATIONAL_COMMIT_POLICY_FEASIBILITY_V1" or card.get("approval", {}).get("authorized_operations") != ["audit"]:
            raise RuntimeError(f"commit-policy Data Card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        environment = json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,numpy,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__},sort_keys=True))"], text=True))
        if environment != {"python": "3.13.5", "numpy": "2.1.3"}:
            raise RuntimeError(f"commit-policy environment drift: {environment}")
        write_json(run_dir / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "gpu": "none"})
        write_json(run_dir / "config/source_integrity_before.json", before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        output = run_dir / "metrics/feasibility"
        command = [
            str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_relational_commit_policy_feasibility_v1.py"),
            "--cache-dir", str(ACTION / "scratch/action_set_cache"),
            "--training-summary", str(TRAINING / "metrics/summary.json"),
            "--output-dir", str(output),
        ]
        for seed in range(3):
            command.extend((f"--seed{seed}", str(TRAINING / f"artifacts/models/seed{seed}/selection_outputs.npz")))
        process_environment = os.environ.copy()
        process_environment["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        with (run_dir / "logs/00_feasibility.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=process_environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False)
        returncode = int(completed.returncode)
        if returncode not in (0, 2):
            raise RuntimeError("commit-policy executor program failure")
        result = load_json(output / "summary.json")
        expected_status = PASS if returncode == 0 else FAIL
        if result.get("status") != expected_status or bool(result.get("scientific_pass")) != (returncode == 0):
            raise RuntimeError("commit-policy status/return mismatch")
        required = [
            output / "summary.json", output / "candidate_grid.csv", output / "figure_source.json",
            *[output / f"gse_relational_commit_policy_feasibility_v1.{suffix}" for suffix in ("png", "pdf", "svg")],
        ]
        if any(not path.is_file() or path.stat().st_size == 0 for path in required):
            raise RuntimeError("commit-policy evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("commit-policy audit changed a frozen source")
        overall = expected_status
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "gse_relational_commit_policy_feasibility_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS,
        "error": error, "executor_returncode": returncode, "result": result,
        "duration_seconds": time.monotonic() - started,
        "source_unchanged": bool(before and before == after),
        "optimizer_steps": 0, "model_inference_frames": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": run_id,
        "state": "COMPLETED" if overall == PASS and error is None else "FAILED",
        "overall_status": overall, "error": error,
    })
    entries = seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
