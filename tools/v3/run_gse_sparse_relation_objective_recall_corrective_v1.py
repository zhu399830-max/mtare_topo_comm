#!/usr/bin/env python3
"""Execute and seal the objective-recall metric corrective."""
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

PASS = "PASS_GSE_SPARSE_RELATION_OBJECTIVE_RECALL_CORRECTIVE_V1"
FAIL = "FAIL_GSE_SPARSE_RELATION_OBJECTIVE_RECALL_CORRECTIVE_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_SPARSE_RELATION_OBJECTIVE_RECALL_CORRECTIVE_V1"
PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run: Path) -> int:
    target = run / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run.rglob("*") if path.is_file() and path != target)
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
    run = args.run_dir.resolve()
    run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("objective-recall corrective executes exactly once")
    started = time.monotonic(); overall = FAIL; error = None; result = {}; before = {}; after = {}
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"])
        validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS:
            raise RuntimeError(f"corrective card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]:
                raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected:
                raise RuntimeError(f"input drift: {relative}")
        write_json(run / "config/source_integrity_before.json", before)
        write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        environment = os.environ.copy()
        environment.update({"PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"), "CUDA_VISIBLE_DEVICES": "", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"})
        tests = [
            "tests/v3/unit/test_evaluate_gse_sparse_circular_relation_transport_v2.py",
            "tests/v3/unit/test_train_gse_sparse_circular_relation_transport_v2.py",
            "tests/v3/unit/test_gse_sparse_circular_relation_transport.py",
            "tests/v3/unit/test_gse_sparse_relation_cardinality_corrective.py",
        ]
        with (run / "logs/00_unit_tests.log").open("w", encoding="utf-8") as stream:
            unit = subprocess.run([str(PYTHON), "-m", "pytest", "-q", *tests], cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=600, check=False)
        if unit.returncode:
            raise RuntimeError("objective-recall corrective unit tests failed")
        command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/execute_gse_sparse_relation_objective_recall_corrective_v1.py"), "--dataset-root", str(PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/dataset/train"), "--sequence-manifest", str(PROJECT_ROOT / "results/gate2_representation/gate2_20260824_gse_deduplicated_dataset_export_v1_seed0/artifacts/sequence_manifest.jsonl"), "--output-dir", str(run / "metrics/corrective")]
        write_json(run / "config/commands.json", [command])
        with (run / "logs/01_corrective.log").open("w", encoding="utf-8") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, env=environment, stdout=stream, stderr=subprocess.STDOUT, text=True, timeout=1200, check=False)
        if completed.returncode not in (0, 2):
            raise RuntimeError("objective-recall corrective program failure")
        result = load_json(run / "metrics/corrective/summary.json")
        passed = completed.returncode == 0 and result.get("status") == PASS and result.get("scientific_pass") is True
        if (completed.returncode == 0) != passed:
            raise RuntimeError("objective-recall corrective status mismatch")
        required = [run / f"metrics/corrective/gse_sparse_relation_objective_recall_corrective_v1.{suffix}" for suffix in ("png", "pdf", "svg")]
        required += [run / "metrics/corrective/per_world_counts.json", run / "metrics/corrective/figure_source.json"]
        if any(not path.is_file() or not path.stat().st_size for path in required):
            raise RuntimeError("objective-recall corrective evidence incomplete")
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after:
            raise RuntimeError("frozen source changed during objective-recall corrective")
        overall = PASS if passed else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    summary = {"schema_version": "gse_sparse_relation_objective_recall_corrective_outer_v1", "overall_status": overall, "scientific_pass": overall == PASS, "error": error, "result": result, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "optimizer_steps": 0, "model_forward_observations": 0, "checkpoints_written": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0}
    write_json(run / "metrics/summary.json", summary)
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED" if error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run)
    print(json.dumps({"overall_status": overall, "error": error, "seal_entries": entries}, indent=2))
    return 0 if overall == PASS and error is None else 2


if __name__ == "__main__":
    raise SystemExit(main())
