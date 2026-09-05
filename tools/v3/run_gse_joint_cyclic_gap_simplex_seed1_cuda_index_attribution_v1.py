#!/usr/bin/env python3
"""Synchronously attribute the repeatable JCGS seed1 CUDA gather failure."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time
import traceback

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, validate_data_card, write_json
from run_gse_joint_cyclic_gap_simplex_three_seed_training_v1 import DATASET, PYTHON, TEACHER, execute, seal, sha256

PASS = "PASS_GSE_JOINT_CYCLIC_GAP_SIMPLEX_SEED1_CUDA_INDEX_ATTRIBUTION_V1"
FAIL = "FAIL_GSE_JOINT_CYCLIC_GAP_SIMPLEX_SEED1_CUDA_INDEX_ATTRIBUTION_V1"
CARD_STATUS = "APPROVED_FOR_ONE_IMMUTABLE_GSE_JOINT_CYCLIC_GAP_SIMPLEX_SEED1_CUDA_INDEX_ATTRIBUTION_V1"

def main() -> int:
    parser = argparse.ArgumentParser(); parser.add_argument("--spec", required=True, type=Path); parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args(); spec = load_json(args.spec.resolve()); run = args.run_dir.resolve(); run_id = f"gate{spec['gate']}_{spec['date']}_{spec['slug']}_seed{spec['seed']}"
    if run.name != run_id or load_json(run / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED": raise RuntimeError("seed1 CUDA attribution executes exactly once")
    started = time.monotonic(); overall, error, diagnosis = FAIL, None, {}; before, after = {}, {}
    try:
        card = load_json(PROJECT_ROOT / spec["data_card"]); validation = validate_data_card(card)
        if not validation.passed or card.get("status") != CARD_STATUS: raise RuntimeError(f"diagnostic card invalid: {validation.errors}")
        for record in spec["frozen_tools"].values():
            if sha256(PROJECT_ROOT / record["path"]) != record["sha256"]: raise RuntimeError(f"tool drift: {record['path']}")
        for relative, expected in spec["frozen_inputs"].items():
            before[relative] = sha256(PROJECT_ROOT / relative)
            if before[relative] != expected: raise RuntimeError(f"input drift: {relative}")
        environment = json.loads(subprocess.check_output([str(PYTHON), "-c", "import json,numpy,torch,zarr,sys;print(json.dumps({'python':sys.version.split()[0],'numpy':numpy.__version__,'torch':torch.__version__,'cuda':torch.version.cuda,'zarr':zarr.__version__,'gpu':torch.cuda.get_device_name(0)},sort_keys=True))"], text=True))
        if environment != {"python": "3.13.5", "numpy": "2.1.3", "torch": "2.9.0+cu129", "cuda": "12.9", "zarr": "2.18.7", "gpu": "NVIDIA GeForce RTX 5090 D"}: raise RuntimeError(f"environment drift: {environment}")
        write_json(run / "config/environment.json", {"executable": str(PYTHON), "versions": environment, "CUDA_LAUNCH_BLOCKING": "1", "diagnostic_batch_trace": True})
        write_json(run / "config/source_integrity_before.json", before); write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "RUNNING"})
        output = run / "artifacts/diagnostic_seed1"; command = [str(PYTHON), str(PROJECT_ROOT / "tools/v3/train_gse_joint_cyclic_gap_simplex_v1.py"), "--teacher-root", str(TEACHER / "artifacts/export/teacher"), "--source-root", str(DATASET / "artifacts/dataset/train"), "--output-dir", str(output), "--seed", "1", "--epochs", "5", "--batch-size", "128", "--evaluation-batch-size", "256", "--learning-rate", "0.0003", "--weight-decay", "0.0001"]
        write_json(run / "config/commands.json", [command]); process_environment = os.environ.copy(); process_environment.update({"PYTHONPATH": str(PROJECT_ROOT / "src") + os.pathsep + str(PROJECT_ROOT / "tools/v3"), "CUBLAS_WORKSPACE_CONFIG": ":4096:8", "OMP_NUM_THREADS": "4", "MKL_NUM_THREADS": "4", "CUDA_LAUNCH_BLOCKING": "1", "GSE_DIAGNOSTIC_BATCH_TRACE": "1", "GSE_DIAGNOSTIC_STOP_AFTER_TRAIN": "1"})
        log = run / "logs/00_seed1_cuda_blocking.log"; code, rss = execute(command, log, process_environment, 3600); text = log.read_text(encoding="utf-8")
        contexts = []
        for line in text.splitlines():
            if '"diagnostic_batch": true' in line:
                try: contexts.append(json.loads(line))
                except json.JSONDecodeError: pass
        stack = re.findall(r'File "([^"]+)", line (\d+), in ([^\n]+)', text)
        reproduced = code != 0 and "index out of bounds" in text and bool(contexts)
        diagnosis = {
            "schema_version": "gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_v1",
            "status": PASS if reproduced else FAIL,
            "reproduced": reproduced, "subprocess_returncode": code, "peak_host_rss_kib": rss,
            "traced_batches": len(contexts), "optimizer_steps_before_failure": max(len(contexts) - 1, 0) if reproduced else min(len(contexts), 5685),
            "last_batch_context": contexts[-1] if contexts else None,
            "python_stack": [{"path": path, "line": int(line), "function": function.strip()} for path, line, function in stack],
            "cuda_scatter_gather_assert": "ScatterGatherKernel.cu" in text,
            "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0,
        }
        write_json(run / "artifacts/diagnosis.json", diagnosis)
        after = {relative: sha256(PROJECT_ROOT / relative) for relative in before}
        if before != after: raise RuntimeError("diagnostic source changed")
        overall = PASS if reproduced else FAIL
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"; (run / "logs/failure_traceback.log").write_text(traceback.format_exc(), encoding="utf-8")
    write_json(run / "metrics/summary.json", {"schema_version": "gse_joint_cyclic_gap_simplex_seed1_cuda_index_attribution_outer_v1", "overall_status": overall, "scientific_pass": False, "attribution_pass": overall == PASS, "error": error, "diagnosis": diagnosis, "duration_seconds": time.monotonic() - started, "source_unchanged": bool(before and before == after), "c08_worlds_read": 0, "c09_worlds_read": 0, "c10_worlds_read": 0, "mtare_worlds_read": 0, "graph_replays": 0, "planner_calls": 0})
    write_json(run / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": run_id, "state": "COMPLETED" if overall == PASS and error is None else "FAILED", "overall_status": overall, "error": error})
    entries = seal(run); print(json.dumps({"overall_status": overall, "error": error, "last_batch_context": diagnosis.get("last_batch_context"), "python_stack": diagnosis.get("python_stack"), "seal_entries": entries}, indent=2)); return 0 if overall == PASS and error is None else 2

if __name__ == "__main__": raise SystemExit(main())
