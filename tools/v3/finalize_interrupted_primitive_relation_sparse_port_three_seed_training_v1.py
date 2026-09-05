#!/usr/bin/env python3
"""Seal the operator-interrupted V2 run as a system failure."""

from __future__ import annotations

import json
import os
from pathlib import Path
import time

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json
from run_primitive_relation_nonlearning_readiness_v1 import _seal, _sha


RUN_ID = "gate3_20260831_primitive_relation_sparse_port_three_seed_training_v1_seed0"
RUN = PROJECT_ROOT / "results/gate3_semantics" / RUN_ID


def _matching_processes() -> list[dict[str, str | int]]:
    matches: list[dict[str, str | int]] = []
    needles = (
        "run_primitive_relation_sparse_port_three_seed_training_v1.py",
        "train_primitive_relation_sparse_port_v1.py",
    )
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (FileNotFoundError, PermissionError, UnicodeDecodeError):
            continue
        if any(needle in command for needle in needles):
            matches.append({"pid": int(entry.name), "command": command.strip()})
    return matches


def main() -> None:
    if load_json(RUN / "RUN_STATE.json").get("state") != "RUNNING":
        raise RuntimeError("interrupted source is not the expected stale RUNNING run")
    if (RUN / "artifacts/evidence_sha256.txt").exists():
        raise RuntimeError("interrupted source is already sealed")
    processes = _matching_processes()
    if processes:
        raise RuntimeError(f"interrupted processes still exist: {processes}")
    checkpoints = sorted((RUN / "artifacts/models").rglob("*.pt"))
    histories = sorted((RUN / "artifacts/models").rglob("history.json"))
    if checkpoints or histories:
        raise RuntimeError("interrupted run unexpectedly completed model evidence")
    test_log = (RUN / "logs/00_unit_tests.log").read_text(encoding="utf-8")
    if "44 passed" not in test_log:
        raise RuntimeError("interrupted run did not complete frozen tests")
    spec = load_json(RUN / "config/run_spec.json")
    after = {
        relative: _sha(PROJECT_ROOT / relative)
        for relative in spec["frozen_inputs"]
    }
    if after != spec["frozen_inputs"]:
        raise RuntimeError("interrupted run frozen input drift")
    for record in spec["frozen_tools"].values():
        if _sha(PROJECT_ROOT / record["path"]) != record["sha256"]:
            raise RuntimeError(f"interrupted run frozen tool drift: {record['path']}")
    write_json(RUN / "config/source_integrity_after.json", after)
    summary = {
        "schema_version": "primitive_relation_sparse_port_interrupted_training_finalization_v1",
        "overall_status": "FAIL_SYSTEM_HOST_RSS_MONITOR_CONTRACT_V1",
        "scientific_pass": False,
        "error": (
            "The trainer's process_memory metric queried nvidia-smi GPU process "
            "memory, while the outer runner treated it as host memory evidence. "
            "The run was deliberately interrupted before epoch0 completion; "
            "host peak RSS and exact partial optimizer steps are not recoverable."
        ),
        "failure_class": "SYSTEM_RESOURCE_EVIDENCE_CONTRACT",
        "model_or_teacher_conclusion": "NONE",
        "unit_tests_passed": 44,
        "completed_epochs": 0,
        "checkpoint_files": 0,
        "history_files": 0,
        "optimizer_steps_exact": None,
        "optimizer_steps_note": "Unknown partial nonzero work; no checkpoint retained.",
        "c07_model_rows_scored": 0,
        "c08_rows_read": 0,
        "c09_c10_worlds_read": 0,
        "graph_replays": 0,
        "mtare_worlds_read": 0,
        "processes_after_interruption": processes,
        "corrective_boundary": (
            "Add independent GPU-process-memory and host peak-RSS monitors only; "
            "do not change data, model, losses, schedule, seeds, thresholds or gates."
        ),
        "finalized_at_unix_seconds": time.time(),
        "result_bytes_before_seal": sum(
            path.stat().st_size for path in RUN.rglob("*") if path.is_file()
        ),
    }
    write_json(RUN / "metrics/summary.json", summary)
    write_json(RUN / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1",
        "run_id": RUN_ID,
        "state": "FAILED",
        "overall_status": summary["overall_status"],
        "error": summary["error"],
    })
    entries = _seal(RUN)
    print(json.dumps({
        "run_id": RUN_ID,
        "overall_status": summary["overall_status"],
        "evidence_files": entries,
    }, indent=2))


if __name__ == "__main__":
    main()
