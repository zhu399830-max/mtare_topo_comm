#!/home/zeng-workstation/anaconda3/bin/python
"""Run the fail-closed V1R frontier execution-outcome audit."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import math
import shlex
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r import (
    audit_frontier_attempt_evidence_v1r,
)
from mtare_topo.governance import load_json, write_json
from run_stochastic_v2_compatibility_audit_v1 import (
    _parse_seal,
    _verify_bound_file,
    load_source,
    sha256,
    validate_execution_scope,
)


STATUS_PASS = "PASS_FRONTIER_ATTEMPT_OUTCOME_AUDIT_V1R"
STATUS_FAIL = "FAIL_FRONTIER_ATTEMPT_OUTCOME_AUDIT_V1R"
VALID_OUTCOMES = {
    "matched_verified_departure",
    "divergent_verified_departure",
    "same_node_loop_merge",
}


def validate_material_run_identity(
    spec_path: Path,
    spec: dict[str, Any],
    run_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    root = project_root.resolve()
    expected_spec = (root / spec["config_path"]).resolve()
    try:
        expected_spec.relative_to(root)
    except ValueError as exc:
        raise RuntimeError("frontier attempt audit spec escapes project root") from exc
    if spec_path.resolve() != expected_spec:
        raise RuntimeError("frontier attempt audit spec path identity mismatch")
    expected_run = (root / "results/gate6_single_robot" / spec["run_id"]).resolve()
    if run_dir.resolve() != expected_run:
        raise RuntimeError("frontier attempt audit run directory identity mismatch")
    required_dirs = ("config", "logs", "metrics", "previews", "artifacts")
    if any(not (run_dir / child).is_dir() for child in required_dirs):
        raise RuntimeError("frontier attempt audit run lacks create_run directory contract")
    state = load_json(run_dir / "RUN_STATE.json")
    if state.get("state") != "CREATED_NOT_EXECUTED" or state.get("run_id") != spec["run_id"]:
        raise RuntimeError("frontier attempt audit RUN_STATE identity mismatch")
    if load_json(run_dir / "config/run_spec.json") != spec:
        raise RuntimeError("frontier attempt audit run-spec snapshot drift")
    if sha256(run_dir / "config/source_config") != sha256(expected_spec):
        raise RuntimeError("frontier attempt audit source-config snapshot drift")
    card_path = (root / spec["data_card"]).resolve()
    card_path.relative_to(root)
    if load_json(run_dir / "config/data_card.json") != load_json(card_path):
        raise RuntimeError("frontier attempt audit data-card snapshot drift")
    expected_command = shlex.join(spec["command"]) + "\n"
    if (run_dir / "config/command.txt").read_text(encoding="utf-8") != expected_command:
        raise RuntimeError("frontier attempt audit command snapshot drift")
    for required in ("config/status_snapshot.json", "config/environment.json"):
        if not (run_dir / required).is_file():
            raise RuntimeError("frontier attempt audit run lacks create_run provenance")


def validate_append_only_source_identity(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, str]:
    root = project_root.resolve()
    source_spec_path = root / spec["source_run"] / "config/run_spec.json"
    source_spec = load_json(source_spec_path)
    causal = source_spec.get("frozen_tools", {}).get("causal_graph")
    expected = {
        "path": spec["source_causal_graph_path"],
        "sha256": spec["source_causal_graph_sha256"],
    }
    if causal != expected:
        raise RuntimeError("source append-only causal-graph identity drift")
    if sha256(root / expected["path"]) != expected["sha256"]:
        raise RuntimeError("local append-only causal-graph implementation drift")
    return expected


def summarize_frontier_attempt_cases_v1r(
    cases: list[dict[str, Any]],
) -> dict[str, Any]:
    if len(cases) != 30:
        raise RuntimeError("frontier attempt outcome audit requires exactly 30 cases")
    aggregate: Counter[str] = Counter()
    total_nonmatching = 0
    at_least_lookahead = 0
    at_least_event_travel = 0
    cases_with_nonmatching = 0
    for case in cases:
        attempt = case["frontier_attempt"]
        events = attempt.get("events")
        if not isinstance(events, list):
            raise RuntimeError("frontier attempt events must be a list")
        event_counts = Counter(str(event.get("outcome")) for event in events)
        if set(event_counts) - VALID_OUTCOMES:
            raise RuntimeError("frontier attempt events contain an invalid outcome")
        declared = {str(key): int(value) for key, value in attempt["outcome_counts"].items()}
        if dict(sorted(event_counts.items())) != dict(sorted(declared.items())):
            raise RuntimeError("frontier attempt event/outcome counts drift")
        if int(attempt["classified_attempt_event_count"]) != len(events):
            raise RuntimeError("frontier attempt classified-event count drift")
        nonmatching = sum(
            count for outcome, count in event_counts.items()
            if outcome != "matched_verified_departure"
        )
        if int(attempt["nonmatching_attempt_event_count"]) != nonmatching:
            raise RuntimeError("frontier attempt nonmatching count drift")
        if len({int(event["frame_index"]) for event in events}) != len(events):
            raise RuntimeError("frontier attempt event frames are not unique")
        aggregate.update(event_counts)
        total_nonmatching += nonmatching
        cases_with_nonmatching += int(nonmatching > 0)
        lookahead = float(case["waypoint_lookahead_m"])
        event_travel = float(case["minimum_event_travel_m"])
        if not all(math.isfinite(value) and value > 0 for value in (lookahead, event_travel)):
            raise RuntimeError("frozen source distance scales must be finite and positive")
        for event in events:
            if event["outcome"] == "matched_verified_departure":
                continue
            travel = float(event["target_run_route_arc_m_before_event"])
            if not math.isfinite(travel) or travel < 0:
                raise RuntimeError("frontier attempt persistence travel is invalid")
            at_least_lookahead += int(travel >= lookahead)
            at_least_event_travel += int(travel >= event_travel)
    return {
        "schema_version": "frontier_attempt_outcome_aggregate_v1r",
        "case_count": len(cases),
        "case_count_with_nonmatching_attempt": cases_with_nonmatching,
        "outcome_counts": dict(sorted(aggregate.items())),
        "nonmatching_attempt_event_count": total_nonmatching,
        "nonmatching_at_least_frozen_waypoint_lookahead_count": at_least_lookahead,
        "nonmatching_at_least_frozen_minimum_event_travel_count": at_least_event_travel,
        "uses_new_tuned_threshold": False,
        "uses_evaluator_gt": False,
    }


def audit_source_frontier_attempts_v1r(
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
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        snapshot = load_json(snapshot_path)
        runtime = snapshot.get("runtime")
        graph = None if not isinstance(runtime, dict) else runtime.get("graph")
        graph_config, planner_config = (
            None if not isinstance(graph, dict) else graph.get("config"),
            snapshot.get("planner_config"),
        )
        if not all(isinstance(value, dict) for value in (graph, graph_config, planner_config)):
            raise RuntimeError(f"invalid V9 planner snapshot: {case['case_id']}")
        attempt = audit_frontier_attempt_evidence_v1r(rows, graph)
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
    aggregate = summarize_frontier_attempt_cases_v1r(cases)
    return {
        "schema_version": "frontier_attempt_outcome_audit_v1r",
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
    lines = [
        f"{sha256(path)}  {path.relative_to(PROJECT_ROOT).as_posix()}"
        for path in sorted(item for item in run_dir.rglob("*") if item.is_file() and item != seal_path)
    ]
    seal_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path, run_dir = args.spec.resolve(), args.run_dir.resolve()
    spec = load_json(spec_path)
    validate_material_run_identity(spec_path, spec, run_dir)
    started = time.monotonic()
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": spec["run_id"], "state": "RUNNING"
    })
    try:
        frozen_tools = validate_execution_scope(spec)
        append_only_identity = validate_append_only_source_identity(spec)
        summaries, provenance, manifest = load_source(spec)
        provenance["verified_frozen_tools"] = frozen_tools
        provenance["append_only_causal_graph_identity"] = append_only_identity
        evidence = audit_source_frontier_attempts_v1r(spec, summaries)
        write_json(run_dir / "config/source_provenance.json", provenance)
        write_json(run_dir / "config/source_manifest.json", manifest)
        write_json(run_dir / "metrics/frontier_attempt_outcomes.json", evidence)
        summary = {
            "schema_version": "frontier_attempt_outcome_audit_summary_v1r",
            "overall_status": STATUS_PASS,
            "source_case_count": len(summaries),
            "v9_case_count": evidence["case_count"],
            "case_count_with_nonmatching_attempt": evidence["case_count_with_nonmatching_attempt"],
            "outcome_counts": evidence["outcome_counts"],
            "nonmatching_attempt_event_count": evidence["nonmatching_attempt_event_count"],
            "nonmatching_at_least_frozen_waypoint_lookahead_count": evidence["nonmatching_at_least_frozen_waypoint_lookahead_count"],
            "nonmatching_at_least_frozen_minimum_event_travel_count": evidence["nonmatching_at_least_frozen_minimum_event_travel_count"],
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0, "optimizer_steps": 0, "raw_bag_reads": 0,
            "c09_reads": 0, "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "COMPLETED", "overall_status": STATUS_PASS,
        })
        sealed_files = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed_files}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "frontier_attempt_outcome_audit_summary_v1r",
            "overall_status": STATUS_FAIL, "failure_reason": str(exc),
            "duration_seconds": time.monotonic() - started,
            "training_steps": 0, "optimizer_steps": 0, "raw_bag_reads": 0,
            "c09_reads": 0, "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", failure)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "FAILED", "overall_status": STATUS_FAIL,
        })
        seal(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
