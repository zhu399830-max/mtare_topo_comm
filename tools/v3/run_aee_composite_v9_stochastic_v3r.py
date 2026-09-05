#!/home/zeng-workstation/anaconda3/bin/python
"""Recover one clock-faulted block and the unexecuted tail of Gate-6 V9 V3."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE
from mtare_topo.evaluation.stochastic_closed_loop import analyze_stochastic_cases
from mtare_topo.governance import load_json, write_json
import run_mtare_single_robot_stochastic_v2 as base


RUN_ID = "gate6_20260823_aee_composite_v9_stochastic_v3r_seed20260820"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_STOCHASTIC_V3R_SYSTEM_RECOVERY"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3R_SYSTEM_RECOVERY"
SOURCE_STATUS = "FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3"
SOURCE_COMPLETED_CASES = 77
FAILED_CASE_INDEX = 77
AFFECTED_BLOCK = "tunnel_env23"
EXPECTED_RECOVERY_CASES = 21
EXPECTED_COMBINED_CASES = 90
EXPECTED_SOURCE_REUSED_CASES = 69
EXPECTED_REEXECUTED_SOURCE_CASES = 8


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def recovery_cases(cases: tuple[Any, ...]) -> tuple[Any, ...]:
    """Select a whole replacement block plus the previously unexecuted tail."""
    selected = tuple(
        case for case in cases
        if case.block_id == AFFECTED_BLOCK or case.index >= FAILED_CASE_INDEX
    )
    if len(selected) != EXPECTED_RECOVERY_CASES:
        raise RuntimeError("recovery case count drift")
    if len({case.case_id for case in selected}) != EXPECTED_RECOVERY_CASES:
        raise RuntimeError("recovery cases are not unique")
    if sum(case.block_id == AFFECTED_BLOCK for case in selected) != 9:
        raise RuntimeError("affected block is not replaced in full")
    if [case.index for case in selected if case.index >= FAILED_CASE_INDEX] != list(range(77, 90)):
        raise RuntimeError("unexecuted schedule tail drift")
    return selected


def source_cases(cases: tuple[Any, ...]) -> tuple[Any, ...]:
    selected = tuple(
        case for case in cases
        if case.index < FAILED_CASE_INDEX and case.block_id != AFFECTED_BLOCK
    )
    if len(selected) != EXPECTED_SOURCE_REUSED_CASES:
        raise RuntimeError("source reuse count drift")
    return selected


def _seal_entries(seal_path: Path) -> dict[str, str]:
    entries: dict[str, str] = {}
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in entries or len(expected) != 64:
            raise RuntimeError("source seal is malformed or duplicated")
        entries[relative] = expected
    return entries


def _verify_sealed_path(path: Path, entries: dict[str, str]) -> str:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    expected = entries.get(relative)
    if expected is None or _sha256(path) != expected:
        raise RuntimeError(f"source evidence drift: {relative}")
    return expected


def verify_failed_source(spec: dict[str, Any], all_cases: tuple[Any, ...]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    source = (PROJECT_ROOT / spec["failed_source_run"]).resolve()
    source.relative_to(PROJECT_ROOT)
    seal_path = source / "artifacts/evidence_sha256.txt"
    if _sha256(seal_path) != spec["failed_source_seal_sha256"]:
        raise RuntimeError("failed source seal identity drift")
    entries = _seal_entries(seal_path)
    state_path = source / "RUN_STATE.json"
    root_summary_path = source / "metrics/summary.json"
    progress_path = source / "metrics/progress.json"
    schedule_path = source / "config/case_schedule.json"
    for path in (state_path, root_summary_path, progress_path, schedule_path):
        _verify_sealed_path(path, entries)
    state = load_json(state_path)
    summary = load_json(root_summary_path)
    progress = load_json(progress_path)
    schedule = load_json(schedule_path)
    if state.get("state") != "FAILED" or state.get("overall_status") != SOURCE_STATUS:
        raise RuntimeError("source is not the immutable expected failed run")
    if summary.get("completed_case_count") != SOURCE_COMPLETED_CASES:
        raise RuntimeError("source completed-case count drift")
    expected_reason = "case container failed with exit code 1: mtare-v9-v3-077"
    if summary.get("failure_reason") != expected_reason:
        raise RuntimeError("source failure identity drift")
    if progress.get("completed_case_count") != SOURCE_COMPLETED_CASES:
        raise RuntimeError("source progress count drift")
    if schedule.get("cases") != [case.to_dict() for case in all_cases]:
        raise RuntimeError("source schedule drift")

    summaries: list[dict[str, Any]] = []
    verified_files = 4
    for case in source_cases(all_cases):
        case_dir = source / "artifacts/cases" / case.case_id
        paths = sorted(path for path in case_dir.rglob("*") if path.is_file())
        if not paths:
            raise RuntimeError(f"source case evidence absent: {case.case_id}")
        for path in paths:
            _verify_sealed_path(path, entries)
            verified_files += 1
        item = load_json(case_dir / "summary.json")
        if item.get("status") != "PASS_SINGLE_ROBOT_CASE_V2" or item.get("case", {}).get("case_id") != case.case_id:
            raise RuntimeError(f"source case is not an exact V2 PASS: {case.case_id}")
        summaries.append(item)
    return summaries, {
        "schema_version": "aee_composite_v9_failed_source_reuse_v1",
        "source_run": source.relative_to(PROJECT_ROOT).as_posix(),
        "source_status": SOURCE_STATUS,
        "source_seal_sha256": spec["failed_source_seal_sha256"],
        "source_seal_entries": len(entries),
        "verified_source_files": verified_files,
        "source_completed_cases": SOURCE_COMPLETED_CASES,
        "combined_reused_cases": len(summaries),
        "excluded_affected_block_cases": EXPECTED_REEXECUTED_SOURCE_CASES,
        "failed_case_index": FAILED_CASE_INDEX,
        "failure_reason": expected_reason,
        "clock_failure_evidence": spec["clock_failure_evidence"],
    }


def failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "aee_composite_v9_stochastic_v3r_summary_v1",
        "overall_status": STATUS_FAIL,
        "failure_reason": message,
        "completed_recovery_case_count": completed,
        "planned_recovery_case_count": EXPECTED_RECOVERY_CASES,
        "training_steps": 0,
        "optimizer_steps": 0,
        "c09_reads": 0,
        "c10_reads": 0,
        "duration_seconds": time.monotonic() - started,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "FAILED", "overall_status": STATUS_FAIL,
    })
    base.base.seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("V3R recovery run identity/state mismatch")
    started = time.monotonic()
    recovery_summaries: list[dict[str, Any]] = []
    try:
        base.RUN_ID = RUN_ID
        base.STATUS_PASS = STATUS_PASS
        base.STATUS_FAIL = STATUS_FAIL
        base.EXPECTED_READINESS_STATUS = "PASS_AEE_COMPOSITE_V9_READINESS_V1R3"
        base.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
        base.TOPIC_CONTRACT = "configs/v3/gate6/closed_loop_recording_topics_aee_v2.json"
        base.CONTAINER_PREFIX = "mtare-v9-v3r"
        matrix, all_cases, full_audit = base.base.validate_frozen_inputs(spec)
        readiness = base.verify_readiness_source(spec)
        reused_summaries, source_audit = verify_failed_source(spec, all_cases)
        selected = recovery_cases(all_cases)
        write_json(run_dir / "config/readiness_source.json", readiness)
        write_json(run_dir / "config/full_matrix_audit.json", full_audit)
        write_json(run_dir / "config/failed_source_audit.json", source_audit)
        write_json(run_dir / "config/recovery_case_schedule.json", {
            "schema_version": "aee_composite_v9_recovery_schedule_v1",
            "selection_rule": "whole tunnel_env23 block union original indices 77..89",
            "cases": [case.to_dict() for case in selected],
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        (run_dir / "artifacts/cases").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/cases").mkdir(parents=True, exist_ok=False)
        for case in selected:
            if shutil.disk_usage(run_dir).free < base.MINIMUM_FREE_BYTES:
                raise RuntimeError(f"free disk below 150 GiB before case {case.case_id}")
            command, name = base.case_command(run_dir, case, matrix)
            base.base.run_case(command, name, run_dir / f"logs/cases/{case.case_id}.log")
            item = base.finalize_case(run_dir / f"artifacts/cases/{case.case_id}", case.case_id)
            recovery_summaries.append(item)
            write_json(run_dir / "metrics/progress.json", {
                "schema_version": "aee_composite_v9_stochastic_v3r_progress_v1",
                "completed_recovery_case_count": len(recovery_summaries),
                "planned_recovery_case_count": EXPECTED_RECOVERY_CASES,
                "last_case_id": case.case_id,
                "last_original_schedule_index": case.index,
                "elapsed_wall_sec": time.monotonic() - started,
                "free_disk_bytes": shutil.disk_usage(run_dir).free,
            })

        combined_by_id = {item["case"]["case_id"]: item for item in reused_summaries + recovery_summaries}
        combined = [combined_by_id[case.case_id] for case in all_cases]
        if len(combined_by_id) != EXPECTED_COMBINED_CASES or len(combined) != EXPECTED_COMBINED_CASES:
            raise RuntimeError("combined 90-case evidence is incomplete")
        analysis = analyze_stochastic_cases(combined)
        write_json(run_dir / "metrics/stochastic_analysis.json", analysis)
        write_json(run_dir / "config/combined_case_sources.json", {
            "schema_version": "aee_composite_v9_combined_case_sources_v1",
            "cases": [
                {"case_id": case.case_id, "original_schedule_index": case.index,
                 "source": "recovery_run" if case.case_id in {x["case"]["case_id"] for x in recovery_summaries} else "failed_source_run"}
                for case in all_cases
            ],
        })
        fallback_rates = [
            item["method_identity"]["post_warmup_fallback_rate"]
            for item in combined if item["case"]["method_family"] == "m1d_topology"
        ]
        summary = {
            "schema_version": "aee_composite_v9_stochastic_v3r_summary_v1",
            "overall_status": STATUS_PASS,
            "completed_recovery_case_count": len(recovery_summaries),
            "combined_case_count": len(combined),
            "combined_block_count": 10,
            "source_reused_case_count": len(reused_summaries),
            "whole_block_reexecuted_case_count": 9,
            "previously_unexecuted_case_count": 12,
            "failed_source_preserved": True,
            "simulated_recovery_runtime_sec": 12600,
            "primary_metric": analysis["primary_metric"],
            "primary_m1d_comparison": analysis["m1d_comparisons"][analysis["primary_metric"]],
            "m1d_post_warmup_fallback_rate_mean": sum(fallback_rates) / len(fallback_rates),
            "m1d_post_warmup_fallback_rate_maximum": max(fallback_rates),
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0,
            "optimizer_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "COMPLETED", "overall_status": STATUS_PASS,
        })
        sealed = base.base.seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure(run_dir, started, len(recovery_summaries), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
