#!/home/zeng-workstation/anaconda3/bin/python
"""Run a sealed no-GT audit of V9 frontier execution outcomes."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.topology_frontier_attempt_evidence import (
    audit_frontier_attempt_evidence,
)
from mtare_topo.governance import load_json, write_json
from run_stochastic_v2_compatibility_audit_v1 import (
    _parse_seal,
    _verify_bound_file,
    load_source,
    sha256,
    validate_execution_scope,
)


STATUS_PASS = "PASS_FRONTIER_ATTEMPT_OUTCOME_AUDIT_V1"
STATUS_FAIL = "FAIL_FRONTIER_ATTEMPT_OUTCOME_AUDIT_V1"


def summarize_frontier_attempt_cases(cases: list[dict[str, Any]]) -> dict[str, Any]:
    if len(cases) != 30:
        raise RuntimeError("frontier attempt outcome audit requires exactly 30 cases")
    outcome_counts: Counter[str] = Counter()
    total_nonmatching = 0
    at_least_lookahead = 0
    at_least_event_travel = 0
    cases_with_nonmatching = 0
    for case in cases:
        attempt = case["frontier_attempt"]
        outcomes = attempt["outcome_counts"]
        outcome_counts.update({str(key): int(value) for key, value in outcomes.items()})
        nonmatching = int(attempt["nonmatching_attempt_event_count"])
        total_nonmatching += nonmatching
        cases_with_nonmatching += int(nonmatching > 0)
        lookahead = float(case["waypoint_lookahead_m"])
        event_travel = float(case["minimum_event_travel_m"])
        if not all(math.isfinite(value) and value > 0 for value in (lookahead, event_travel)):
            raise RuntimeError("frontier attempt source thresholds must be finite and positive")
        for event in attempt["events"]:
            if event["outcome"] == "matched_verified_departure":
                continue
            travel = float(event["target_run_route_arc_m_before_event"])
            at_least_lookahead += int(travel >= lookahead)
            at_least_event_travel += int(travel >= event_travel)
    return {
        "schema_version": "frontier_attempt_outcome_aggregate_v1",
        "case_count": len(cases),
        "case_count_with_nonmatching_attempt": cases_with_nonmatching,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "nonmatching_attempt_event_count": total_nonmatching,
        "nonmatching_at_least_frozen_waypoint_lookahead_count": at_least_lookahead,
        "nonmatching_at_least_frozen_minimum_event_travel_count": at_least_event_travel,
        "uses_new_tuned_threshold": False,
        "uses_evaluator_gt": False,
    }


def audit_source_frontier_attempts(
    spec: dict[str, Any], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    source_run = (PROJECT_ROOT / spec["source_run"]).resolve()
    source_run.relative_to(PROJECT_ROOT)
    entries = _parse_seal(source_run, spec["source_seal_sha256"])
    verified_files: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
    for item in summaries:
        case = item["case"]
        if case["method_family"] != "m1d_topology":
            continue
        case_dir = source_run / "artifacts/cases" / case["case_id"]
        planner = item["planner_evidence"]
        trace_path = (case_dir / planner["decision_trace"]).resolve()
        snapshot_path = (case_dir / planner["topology_snapshot"]).resolve()
        trace_path.relative_to(case_dir.resolve())
        snapshot_path.relative_to(case_dir.resolve())
        for path in (trace_path, snapshot_path):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            verified_files[relative] = _verify_bound_file(path, entries)
        decision_rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        snapshot = load_json(snapshot_path)
        runtime = snapshot.get("runtime")
        graph = None if not isinstance(runtime, dict) else runtime.get("graph")
        if not isinstance(graph, dict):
            raise RuntimeError(f"invalid V9 graph snapshot: {case['case_id']}")
        graph_config = graph.get("config")
        planner_config = snapshot.get("planner_config")
        if not isinstance(graph_config, dict) or not isinstance(planner_config, dict):
            raise RuntimeError(f"missing frozen planner configuration: {case['case_id']}")
        attempt = audit_frontier_attempt_evidence(decision_rows, graph)
        metrics = item["metrics"]
        cases.append({
            "case_id": case["case_id"],
            "block_id": case["block_id"],
            "world": case["world"],
            "environment_seed": case["environment_seed"],
            "checkpoint_seed": case["checkpoint_seed"],
            "traveling_distance_m": float(metrics["traveling_distance_m"]),
            "final_explored_volume_m3": float(metrics["final_explored_volume_m3"]),
            "waypoint_lookahead_m": float(planner_config["waypoint_lookahead_m"]),
            "minimum_event_travel_m": float(graph_config["minimum_event_travel_m"]),
            "frontier_attempt": attempt,
        })
    aggregate = summarize_frontier_attempt_cases(cases)
    return {
        "schema_version": "frontier_attempt_outcome_audit_v1",
        **aggregate,
        "verified_trace_snapshot_file_count": len(verified_files),
        "verified_trace_snapshot_files": dict(sorted(verified_files.items())),
        "raw_bag_reads": 0,
        "c09_reads": 0,
        "c10_reads": 0,
        "cases": cases,
    }


def seal(run_dir: Path) -> int:
    seal_path = run_dir / "artifacts/evidence_sha256.txt"
    lines = []
    for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item != seal_path):
        lines.append(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}")
    seal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != spec["run_id"]:
        raise RuntimeError("frontier attempt audit run identity mismatch")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("frontier attempt audit run is not executable")
    started = time.monotonic()
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1",
        "run_id": spec["run_id"],
        "state": "RUNNING",
    })
    try:
        frozen_tools = validate_execution_scope(spec)
        summaries, provenance, manifest = load_source(spec)
        provenance["verified_frozen_tools"] = frozen_tools
        evidence = audit_source_frontier_attempts(spec, summaries)
        write_json(run_dir / "config/source_provenance.json", provenance)
        write_json(run_dir / "config/source_manifest.json", manifest)
        write_json(run_dir / "metrics/frontier_attempt_outcomes.json", evidence)
        summary = {
            "schema_version": "frontier_attempt_outcome_audit_summary_v1",
            "overall_status": STATUS_PASS,
            "source_case_count": len(summaries),
            "v9_case_count": evidence["case_count"],
            "case_count_with_nonmatching_attempt": evidence[
                "case_count_with_nonmatching_attempt"
            ],
            "outcome_counts": evidence["outcome_counts"],
            "nonmatching_attempt_event_count": evidence[
                "nonmatching_attempt_event_count"
            ],
            "nonmatching_at_least_frozen_waypoint_lookahead_count": evidence[
                "nonmatching_at_least_frozen_waypoint_lookahead_count"
            ],
            "nonmatching_at_least_frozen_minimum_event_travel_count": evidence[
                "nonmatching_at_least_frozen_minimum_event_travel_count"
            ],
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0,
            "optimizer_steps": 0,
            "raw_bag_reads": 0,
            "c09_reads": 0,
            "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1",
            "run_id": spec["run_id"],
            "state": "COMPLETED",
            "overall_status": STATUS_PASS,
        })
        sealed_files = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed_files}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        write_json(run_dir / "metrics/summary.json", {
            "schema_version": "frontier_attempt_outcome_audit_summary_v1",
            "overall_status": STATUS_FAIL,
            "failure_reason": str(exc),
            "duration_seconds": time.monotonic() - started,
            "training_steps": 0,
            "optimizer_steps": 0,
            "raw_bag_reads": 0,
            "c09_reads": 0,
            "c10_reads": 0,
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1",
            "run_id": spec["run_id"],
            "state": "FAILED",
            "overall_status": STATUS_FAIL,
        })
        seal(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
