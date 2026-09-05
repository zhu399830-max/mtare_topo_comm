#!/home/zeng-workstation/anaconda3/bin/python
"""Analyze sealed V4 cases against sealed original M-TARE evidence."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.corrected_stochastic_comparison import (
    analyze_corrected_stochastic_cases,
)
from mtare_topo.evaluation.stochastic_closed_loop_v2_bridge import SOURCE_STATUS
from mtare_topo.governance import load_json, write_json
from run_stochastic_v2_compatibility_audit_v1 import (
    _parse_seal,
    _validate_case_summary,
    schedule_content_sha256,
    seal,
    sha256,
)


STATUS_PASS = "PASS_CORRECTED_STOCHASTIC_COMPARISON_V1"
STATUS_FAIL = "FAIL_CORRECTED_STOCHASTIC_COMPARISON_V1"


def _verify_bound_input(path: Path, entries: dict[str, str]) -> str:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    if relative not in entries:
        raise RuntimeError(f"sealed source file is absent from seal: {relative}")
    observed = sha256(path)
    if observed != entries[relative]:
        raise RuntimeError(f"sealed source file hash drift: {relative}")
    return observed


def load_sealed_cases(
    identity: dict[str, Any], *, expected_count: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    run = (PROJECT_ROOT / identity["run"]).resolve()
    run.relative_to(PROJECT_ROOT)
    state_path = run / "RUN_STATE.json"
    summary_path = run / "metrics/summary.json"
    schedule_path = run / identity["schedule_relative_path"]
    state, summary = load_json(state_path), load_json(summary_path)
    if state.get("state") != identity["expected_state"]:
        raise RuntimeError(f"sealed source state mismatch: {identity['run']}")
    if state.get("overall_status") != identity["expected_status"]:
        raise RuntimeError(f"sealed source status mismatch: {identity['run']}")
    completed = summary.get("completed_case_count", summary.get("completed_cases"))
    if completed != expected_count:
        raise RuntimeError(f"sealed source completed-case count mismatch: {identity['run']}")
    entries = _parse_seal(run, identity["seal_sha256"])
    bound = {}
    for path in (state_path, summary_path, schedule_path):
        bound[path.relative_to(PROJECT_ROOT).as_posix()] = _verify_bound_input(path, entries)
    schedule_file_sha256 = sha256(schedule_path)
    if schedule_file_sha256 != identity["schedule_file_sha256"]:
        raise RuntimeError(f"sealed source schedule file identity drift: {identity['run']}")
    for relative in identity.get("identity_files", []):
        path = run / relative
        bound[path.relative_to(PROJECT_ROOT).as_posix()] = _verify_bound_input(path, entries)

    schedule = load_json(schedule_path).get("cases")
    if not isinstance(schedule, list) or len(schedule) != expected_count:
        raise RuntimeError(f"sealed source schedule count mismatch: {identity['run']}")
    schedule_canonical_sha256 = schedule_content_sha256(schedule)
    if schedule_canonical_sha256 != identity["schedule_content_sha256"]:
        raise RuntimeError(
            f"sealed source schedule canonical-content identity drift: {identity['run']}"
        )
    case_ids = [str(item["case_id"]) for item in schedule]
    if len(set(case_ids)) != expected_count:
        raise RuntimeError(f"sealed source schedule identities are not unique: {identity['run']}")
    discovered = {
        path.parent.name: path for path in (run / "artifacts/cases").glob("*/summary.json")
    }
    if set(discovered) != set(case_ids):
        raise RuntimeError(f"sealed source case set differs from schedule: {identity['run']}")
    summaries = []
    cases = []
    for scheduled in schedule:
        path = discovered[scheduled["case_id"]]
        digest = _verify_bound_input(path, entries)
        bound[path.relative_to(PROJECT_ROOT).as_posix()] = digest
        item = load_json(path)
        if item.get("status") != SOURCE_STATUS:
            raise RuntimeError(f"sealed source has a non-finalized case: {scheduled['case_id']}")
        if item.get("schema_version") != "mtare_single_robot_case_summary_v1":
            raise RuntimeError(f"sealed source summary schema drift: {scheduled['case_id']}")
        if item.get("case", {}).get("schema_version") != "mtare_single_robot_case_contract_v1":
            raise RuntimeError(f"sealed source case schema drift: {scheduled['case_id']}")
        _validate_case_summary(item, scheduled)
        for field in (
            "case_id", "block_id", "method_family", "method_id", "world",
            "environment_seed", "execution_repeat", "checkpoint_seed", "runtime_sec",
        ):
            if item.get("case", {}).get(field) != scheduled.get(field):
                raise RuntimeError(f"sealed case schedule mismatch: {scheduled['case_id']}: {field}")
        summaries.append(item)
        cases.append({
            "index": scheduled["index"],
            "case_id": scheduled["case_id"],
            "block_id": scheduled["block_id"],
            "method_family": scheduled["method_family"],
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": digest,
        })
    manifest = {
        "schema_version": "sealed_stochastic_case_manifest_v1",
        "run": identity["run"],
        "state": identity["expected_state"],
        "status": identity["expected_status"],
        "seal_sha256": identity["seal_sha256"],
        "seal_entry_count": len(entries),
        "verified_bound_file_count": len(bound),
        "case_count": expected_count,
        "schedule_path": schedule_path.relative_to(PROJECT_ROOT).as_posix(),
        "schedule_file_sha256": schedule_file_sha256,
        "schedule_content_sha256": schedule_canonical_sha256,
        "cases": cases,
        "source_mutation_permitted": False,
    }
    return summaries, manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != spec["run_id"] or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("corrected comparison run identity/state mismatch")
    started = time.monotonic()
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": spec["run_id"], "state": "RUNNING",
    })
    try:
        if spec.get("gate") != 6 or spec.get("operation") != "audit":
            raise RuntimeError("Gate-6 read-only comparison scope required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen comparison tool drift: {name}")
        source, source_manifest = load_sealed_cases(spec["source_v9_run"], expected_count=90)
        corrected, corrected_manifest = load_sealed_cases(spec["corrected_v4_run"], expected_count=30)
        analysis = analyze_corrected_stochastic_cases(source, corrected)
        write_json(run_dir / "config/source_v9_manifest.json", source_manifest)
        write_json(run_dir / "config/corrected_v4_manifest.json", corrected_manifest)
        write_json(run_dir / "metrics/corrected_stochastic_analysis.json", analysis)
        primary = analysis["corrected_main_analysis"]["primary_metric"]
        summary = {
            "schema_version": "corrected_stochastic_comparison_summary_v1",
            "overall_status": STATUS_PASS,
            "combined_case_count": 90,
            "corrected_case_count": 30,
            "reused_original_mtare_case_count": 30,
            "reused_oracle_diagnostic_case_count": 30,
            "block_count": 10,
            "primary_metric": primary,
            "primary_v4_minus_original": analysis["corrected_main_analysis"]["m1d_comparisons"][primary],
            "primary_v4_minus_defective_v9": analysis["v4_vs_defective_v9"]["metrics"][primary],
            "source_mutation_count": 0,
            "raw_bag_reads": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "COMPLETED", "overall_status": STATUS_PASS,
        })
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        write_json(run_dir / "metrics/summary.json", {
            "schema_version": "corrected_stochastic_comparison_summary_v1",
            "overall_status": STATUS_FAIL,
            "failure_reason": str(exc),
            "source_mutation_count": 0,
            "raw_bag_reads": 0,
            "training_steps": 0,
            "optimizer_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
            "duration_seconds": time.monotonic() - started,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "FAILED", "overall_status": STATUS_FAIL,
        })
        seal(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
