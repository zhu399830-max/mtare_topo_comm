#!/usr/bin/env python3
"""Seal the scientific selection conclusion omitted by the V1 evaluator error."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate3_20260828_gse_action_set_node_selection_corrective_v1_seed0"
SOURCE = PROJECT_ROOT / "results/gate3_semantics/gate3_20260828_gse_action_set_node_training_v1_seed0"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")
EVALUATOR = PROJECT_ROOT / "tools/v3/evaluate_gse_action_set_node_selection_v1.py"
PASS = "PASS_GSE_ACTION_SET_NODE_SELECTION_CORRECTIVE_V1"
FAIL = "FAIL_GSE_ACTION_SET_NODE_SELECTION_CORRECTIVE_V1"


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_source() -> dict:
    state = load_json(SOURCE / "RUN_STATE.json")
    summary = load_json(SOURCE / "metrics/summary.json")
    if state.get("overall_status") != "FAIL_GSE_ACTION_SET_NODE_TRAINING_V1" or summary.get("optimizer_steps") != 39996:
        raise RuntimeError("action-set V1 source identity/status drift")
    seal = SOURCE / "artifacts/evidence_sha256.txt"
    sealed = set()
    for line in seal.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        target = PROJECT_ROOT / relative
        if not target.is_file() or _sha(target) != expected:
            raise RuntimeError(f"action-set V1 seal mismatch: {relative}")
        sealed.add(target.resolve())
    actual = {path.resolve() for path in SOURCE.rglob("*") if path.is_file() and path != seal}
    if len(sealed) != 45 or sealed != actual:
        raise RuntimeError("action-set V1 seal coverage drift")
    return {"run": str(SOURCE.relative_to(PROJECT_ROOT)), "seal_entries": len(sealed), "seal_sha256": _sha(seal)}


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
        raise RuntimeError("action-set selection corrective may execute only once")
    overall = FAIL
    error = None
    selection = {}
    source_before = {}
    source_after = {}
    returncode = None
    try:
        if spec.get("gate") != 3 or spec.get("operation") != "audit":
            raise RuntimeError("action-set corrective scope drift")
        for record in spec["frozen_tools"].values():
            if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"action-set corrective tool drift: {record['path']}")
        source_before = _verify_source()
        write_json(run_dir / "config/source_integrity_before.json", source_before)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3")
        command = [str(PYTHON), str(EVALUATOR), "--cache-dir", str(SOURCE / "scratch/action_set_cache")]
        for seed in (0, 1, 2):
            command.extend((f"--seed{seed}", str(SOURCE / f"artifacts/models/seed{seed}/selection_outputs.npz")))
        command.extend(("--output-dir", str(run_dir / "metrics/selection")))
        with (run_dir / "logs/evaluation.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, text=True, stdout=stream, stderr=subprocess.STDOUT, timeout=300, check=False)
        returncode = int(completed.returncode)
        if returncode not in (0, 2):
            raise RuntimeError("corrected action-set evaluator failed as a program")
        selection = load_json(run_dir / "metrics/selection/summary.json")
        scientific_pass = selection.get("status") == "PASS_GSE_ACTION_SET_NODE_SELECTION_V1"
        if (returncode == 0) != scientific_pass:
            raise RuntimeError("corrected action-set status/return mismatch")
        source_after = _verify_source()
        if source_before != source_after:
            raise RuntimeError("action-set V1 source changed during corrective audit")
        overall = PASS if scientific_pass else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run_dir / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {
        "schema_version": "gse_action_set_node_selection_corrective_outer_v1",
        "overall_status": overall, "scientific_pass": overall == PASS, "error": error,
        "source": source_after or source_before, "source_unchanged": bool(source_before and source_before == source_after),
        "evaluator_returncode": returncode, "selection": selection,
        "optimizer_steps": 0, "model_inference_frames": 0,
        "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED" if overall == PASS else "FAILED", "overall_status": overall, "error": error})
    entries = _seal(run_dir)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS else 2


if __name__ == "__main__":
    raise SystemExit(main())
