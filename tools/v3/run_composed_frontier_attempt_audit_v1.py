#!/home/zeng-workstation/anaconda3/bin/python
"""Audit all V9 frontier outcomes across a sealed 69+21 composed source."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.stochastic_closed_loop import METRICS
from mtare_topo.evaluation.topology_reanchor_evidence import audit_reanchor_evidence
from mtare_topo.evaluation.stochastic_closed_loop_v2_bridge import (
    analyze_finalized_v2_cases,
)
from mtare_topo.evaluation.combined_correction_probe_selection import (
    select_combined_correction_probe_case,
)
from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r3 import (
    audit_frontier_attempt_evidence_v1r3,
)
from mtare_topo.governance import load_json, write_json
import run_frontier_attempt_outcome_audit_v1r as audit_base
import run_frontier_attempt_outcome_audit_v1r3 as audit_v1r3
from run_stochastic_v2_compatibility_audit_v1 import (
    SOURCE_STATUS,
    _parse_seal,
    _validate_case_summary,
    _verify_bound_file,
    schedule_content_sha256,
    sha256,
    validate_execution_scope,
)


STATUS_PASS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1"
STATUS_FAIL = "FAIL_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1"
EXPECTED_RECOVERY_STATUS = "PASS_AEE_COMPOSITE_V9_STOCHASTIC_V3R_SYSTEM_RECOVERY"
EXPECTED_PREDECESSOR_STATUS = "FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3"
SOURCE_NAMES = {"failed_source_run", "recovery_run"}


def _root(relative: str, project_root: Path = PROJECT_ROOT) -> Path:
    path = (project_root / relative).resolve()
    path.relative_to(project_root.resolve())
    return path


def _verify_bound_file_local(
    path: Path, entries: dict[str, str], project_root: Path
) -> str:
    relative = path.relative_to(project_root.resolve()).as_posix()
    if relative not in entries:
        raise RuntimeError(f"source file absent from evidence seal: {relative}")
    observed = sha256(path)
    if observed != entries[relative]:
        raise RuntimeError(f"source file hash drift: {relative}")
    return observed


def _verify_root_identity(
    root: Path,
    entries: dict[str, str],
    expected_state: str,
    expected_status: str,
    project_root: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, str]]:
    state_path, summary_path, spec_path = (
        root / "RUN_STATE.json",
        root / "metrics/summary.json",
        root / "config/run_spec.json",
    )
    bound = {}
    for path in (state_path, summary_path, spec_path):
        relative = path.relative_to(project_root.resolve()).as_posix()
        bound[relative] = _verify_bound_file_local(path, entries, project_root)
    state, summary = load_json(state_path), load_json(summary_path)
    if state.get("state") != expected_state or state.get("overall_status") != expected_status:
        raise RuntimeError(f"composed source state/status mismatch: {root.name}")
    if summary.get("overall_status") != expected_status:
        raise RuntimeError(f"composed source summary status mismatch: {root.name}")
    return summary, load_json(spec_path), bound


def validate_composed_generator_identity(
    spec: dict[str, Any], source_specs: dict[str, dict[str, Any]], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    expected = {
        "causal_graph": {
            "path": spec["source_causal_graph_path"],
            "sha256": spec["source_causal_graph_sha256"],
        },
        "frontier_planner": {
            "path": spec["source_frontier_planner_path"],
            "sha256": spec["source_frontier_planner_sha256"],
        },
        "online_runtime": {
            "path": spec["source_online_runtime_path"],
            "sha256": spec["source_online_runtime_sha256"],
        },
        "v9_node": {
            "path": spec["source_v9_node_path"],
            "sha256": spec["source_v9_node_sha256"],
        },
    }
    for name, item in expected.items():
        if sha256(project_root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"local composed-source generator drift: {name}")
    predecessor_tools = source_specs["failed_source_run"].get("frozen_tools", {})
    for name, item in expected.items():
        if predecessor_tools.get(name) != item:
            raise RuntimeError(f"predecessor generator identity drift: {name}")
    recovery_spec = source_specs["recovery_run"]
    if recovery_spec.get("replacement_contract", {}).get("scientific_contract_change") is not False:
        raise RuntimeError("recovery source does not declare unchanged scientific contract")
    for name in ("online_runtime", "v9_node"):
        if recovery_spec.get("frozen_tools", {}).get(name) != expected[name]:
            raise RuntimeError(f"recovery directly frozen generator drift: {name}")
    probe_selection = select_combined_correction_probe_case(cases)
    return {
        "schema_version": "composed_source_generator_identity_v1",
        "direct_local_hashes_verified": expected,
        "predecessor_direct_bindings_verified": list(expected),
        "recovery_direct_bindings_verified": ["online_runtime", "v9_node"],
        "recovery_indirect_predecessor_bindings_verified": ["causal_graph", "frontier_planner"],
        "scientific_contract_change": False,
    }


def load_composed_source(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> tuple[list[dict[str, Any]], dict[str, Path], dict[str, dict[str, str]], dict[str, Any], dict[str, Any]]:
    predecessor = _root(spec["predecessor_run"], project_root)
    recovery = _root(spec["source_run"], project_root)
    predecessor_entries = _parse_seal(predecessor, spec["predecessor_seal_sha256"])
    recovery_entries = _parse_seal(recovery, spec["source_seal_sha256"])
    predecessor_summary, predecessor_spec, predecessor_bound = _verify_root_identity(
        predecessor, predecessor_entries, "FAILED", EXPECTED_PREDECESSOR_STATUS, project_root
    )
    recovery_summary, recovery_spec, recovery_bound = _verify_root_identity(
        recovery, recovery_entries, "COMPLETED", EXPECTED_RECOVERY_STATUS, project_root
    )
    if predecessor_summary.get("completed_case_count") != 77:
        raise RuntimeError("predecessor completed-case count drift")
    if (
        recovery_summary.get("completed_recovery_case_count") != 21
        or recovery_summary.get("combined_case_count") != 90
        or recovery_summary.get("source_reused_case_count") != 69
    ):
        raise RuntimeError("recovery aggregate counts drift")

    schedule_path = predecessor / "config/case_schedule.json"
    predecessor_bound[schedule_path.relative_to(project_root.resolve()).as_posix()] = _verify_bound_file_local(
        schedule_path, predecessor_entries, project_root
    )
    schedule = load_json(schedule_path).get("cases")
    if not isinstance(schedule, list) or len(schedule) != 90:
        raise RuntimeError("composed source schedule is not exactly 90 cases")
    if sha256(schedule_path) != spec["source_schedule_file_sha256"]:
        raise RuntimeError("composed source schedule file drift")
    if schedule_content_sha256(schedule) != spec["source_schedule_content_sha256"]:
        raise RuntimeError("composed source schedule content drift")

    map_path = recovery / "config/combined_case_sources.json"
    recovery_bound[map_path.relative_to(project_root.resolve()).as_posix()] = _verify_bound_file_local(
        map_path, recovery_entries, project_root
    )
    mapping_rows = load_json(map_path).get("cases")
    if not isinstance(mapping_rows, list) or len(mapping_rows) != 90:
        raise RuntimeError("composed source map is not exactly 90 cases")
    mapping = {row.get("case_id"): row.get("source") for row in mapping_rows}
    scheduled_ids = [case["case_id"] for case in schedule]
    if len(mapping) != len(mapping_rows):
        raise RuntimeError("composed source-map case identities are not unique")
    if set(mapping) != set(scheduled_ids) or set(mapping.values()) != SOURCE_NAMES:
        raise RuntimeError("composed source-map identities drift")
    if Counter(mapping.values()) != Counter({"failed_source_run": 69, "recovery_run": 21}):
        raise RuntimeError("composed source-map quota drift")
    mapping_rows_by_id = {row["case_id"]: row for row in mapping_rows}
    for case in schedule:
        mapped_index = mapping_rows_by_id[case["case_id"]].get("original_schedule_index")
        if type(mapped_index) is not int or mapped_index != int(case["index"]):
            raise RuntimeError(
                f"composed case original schedule index drift: {case['case_id']}"
            )
        expected_source = (
            "recovery_run"
            if case["block_id"] == "tunnel_env23" or int(case["index"]) >= 77
            else "failed_source_run"
        )
        if mapping[case["case_id"]] != expected_source:
            raise RuntimeError(f"composed case source rule drift: {case['case_id']}")

    roots = {"failed_source_run": predecessor, "recovery_run": recovery}
    seals = {"failed_source_run": predecessor_entries, "recovery_run": recovery_entries}
    summaries = []
    case_roots: dict[str, Path] = {}
    manifest_cases = []
    for scheduled in schedule:
        source_name = mapping[scheduled["case_id"]]
        case_root = roots[source_name] / "artifacts/cases" / scheduled["case_id"]
        summary_path = case_root / "summary.json"
        observed = _verify_bound_file_local(summary_path, seals[source_name], project_root)
        item = load_json(summary_path)
        if item.get("status") != SOURCE_STATUS:
            raise RuntimeError(f"composed case is not V2 PASS: {scheduled['case_id']}")
        _validate_case_summary(item, scheduled)
        summaries.append(item)
        case_roots[scheduled["case_id"]] = case_root
        manifest_cases.append({
            "index": scheduled["index"], "case_id": scheduled["case_id"],
            "block_id": scheduled["block_id"], "method_family": scheduled["method_family"],
            "source": source_name,
            "summary_path": summary_path.relative_to(project_root.resolve()).as_posix(),
            "summary_sha256": observed,
        })
    quotas = Counter(item["case"]["method_family"] for item in summaries)
    blocks = Counter(item["case"]["block_id"] for item in summaries)
    if quotas != Counter({"original_mtare": 30, "m1d_topology": 30, "layered_gt_map_oracle": 30}):
        raise RuntimeError("composed method-family quotas are not 30/30/30")
    if len(blocks) != 10 or set(blocks.values()) != {9}:
        raise RuntimeError("composed block design is not ten blocks of nine")
    source_specs = {"failed_source_run": predecessor_spec, "recovery_run": recovery_spec}
    provenance = {
        "schema_version": "composed_stochastic_source_provenance_v1",
        "predecessor_run": spec["predecessor_run"],
        "predecessor_status": EXPECTED_PREDECESSOR_STATUS,
        "predecessor_seal_sha256": spec["predecessor_seal_sha256"],
        "predecessor_seal_entries": len(predecessor_entries),
        "source_run": spec["source_run"],
        "source_status": EXPECTED_RECOVERY_STATUS,
        "source_seal_sha256": spec["source_seal_sha256"],
        "source_seal_entries": len(recovery_entries),
        "verified_bound_files": {**predecessor_bound, **recovery_bound},
        "source_mutation_permitted": False,
        "raw_bag_reads": 0, "gt_or_map_reads": 0, "c09_reads": 0, "c10_reads": 0,
    }
    manifest = {
        "schema_version": "composed_stochastic_source_manifest_v1",
        "case_count": 90, "block_count": 10,
        "family_case_counts": dict(sorted(quotas.items())),
        "source_counts": dict(sorted(Counter(mapping.values()).items())),
        "schedule_file_sha256": sha256(schedule_path),
        "schedule_content_sha256": schedule_content_sha256(schedule),
        "cases": manifest_cases,
    }
    return summaries, case_roots, seals, source_specs, {"provenance": provenance, "manifest": manifest}


def audit_composed_frontier_attempts(
    summaries: list[dict[str, Any]],
    case_roots: dict[str, Path],
    seals: dict[str, dict[str, str]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    source_by_case = {row["case_id"]: row["source"] for row in manifest["cases"]}
    verified_files: dict[str, str] = {}
    cases = []
    for item in summaries:
        case = item["case"]
        if case["method_family"] != "m1d_topology":
            continue
        case_id = case["case_id"]
        source_name, case_dir = source_by_case[case_id], case_roots[case_id]
        planner = item["planner_evidence"]
        trace_path = (case_dir / planner["decision_trace"]).resolve()
        snapshot_path = (case_dir / planner["topology_snapshot"]).resolve()
        trace_path.relative_to(case_dir.resolve())
        snapshot_path.relative_to(case_dir.resolve())
        for path in (trace_path, snapshot_path):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            verified_files[relative] = _verify_bound_file(path, seals[source_name])
        rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        snapshot = load_json(snapshot_path)
        runtime = snapshot.get("runtime")
        graph = None if not isinstance(runtime, dict) else runtime.get("graph")
        graph_config = None if not isinstance(graph, dict) else graph.get("config")
        planner_config = snapshot.get("planner_config")
        if not all(isinstance(value, dict) for value in (graph, graph_config, planner_config)):
            raise RuntimeError(f"invalid composed V9 planner snapshot: {case_id}")
        attempt = audit_frontier_attempt_evidence_v1r3(rows, graph)
        reanchor = audit_reanchor_evidence(rows, graph)
        cases.append({
            "case_id": case_id, "source": source_name,
            "block_id": case["block_id"], "world": case["world"],
            "environment_seed": case["environment_seed"],
            "checkpoint_seed": case["checkpoint_seed"],
            "traveling_distance_m": float(item["metrics"]["traveling_distance_m"]),
            "final_explored_volume_m3": float(item["metrics"]["final_explored_volume_m3"]),
            "waypoint_lookahead_m": float(planner_config["waypoint_lookahead_m"]),
            "minimum_event_travel_m": float(graph_config["minimum_event_travel_m"]),
            "frontier_attempt": attempt,
            "reanchor_evidence": reanchor,
        })
    aggregate = audit_base.summarize_frontier_attempt_cases_v1r(cases)
    proxy_cases = [
        case for case in cases
        if int(case["reanchor_evidence"]["exact_arrival_proxy_frame_count"]) > 0
    ]
    return {
        "schema_version": "composed_frontier_attempt_outcome_audit_v1",
        **aggregate,
        "historical_stub_creation_frame_directly_reconstructable": False,
        "historical_target_state_bound_to_frozen_source_planner": True,
        "traversal_event_bijection_required": True,
        "case_count_with_exact_arrival_proxy": len(proxy_cases),
        "total_exact_arrival_proxy_frames": sum(
            int(case["reanchor_evidence"]["exact_arrival_proxy_frame_count"])
            for case in cases
        ),
        "maximum_constant_proxy_run_frames": max(
            int(case["reanchor_evidence"]["longest_constant_proxy_run_frames"])
            for case in cases
        ),
        "verified_trace_snapshot_file_count": len(verified_files),
        "verified_trace_snapshot_files": dict(sorted(verified_files.items())),
        "combined_correction_probe_selection": probe_selection,
        "raw_bag_reads": 0, "gt_or_map_reads": 0, "c09_reads": 0, "c10_reads": 0,
        "cases": cases,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path, run_dir = args.spec.resolve(), args.run_dir.resolve()
    spec = load_json(spec_path)
    audit_v1r3.validate_material_run_identity_v1r3(spec_path, spec, run_dir)
    started = time.monotonic()
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": spec["run_id"], "state": "RUNNING"})
    try:
        frozen = validate_execution_scope(spec)
        summaries, case_roots, seals, source_specs, source = load_composed_source(spec)
        generators = validate_composed_generator_identity(spec, source_specs)
        stochastic_analysis = analyze_finalized_v2_cases(summaries)
        evidence = audit_composed_frontier_attempts(summaries, case_roots, seals, source["manifest"])
        source["provenance"]["verified_frozen_tools"] = frozen
        source["provenance"]["source_generator_identities"] = generators
        write_json(run_dir / "config/source_provenance.json", source["provenance"])
        write_json(run_dir / "config/source_manifest.json", source["manifest"])
        write_json(run_dir / "metrics/stochastic_analysis.json", stochastic_analysis)
        write_json(run_dir / "metrics/frontier_attempt_outcomes.json", evidence)
        primary_metric = stochastic_analysis["primary_metric"]
        summary = {
            "schema_version": "composed_frontier_attempt_audit_summary_v1",
            "overall_status": STATUS_PASS,
            "source_case_count": 90, "v9_case_count": evidence["case_count"],
            "case_count_with_nonmatching_attempt": evidence["case_count_with_nonmatching_attempt"],
            "case_count_with_exact_arrival_proxy": evidence["case_count_with_exact_arrival_proxy"],
            "total_exact_arrival_proxy_frames": evidence["total_exact_arrival_proxy_frames"],
            "mapped_in_memory_field": "status",
            "mapped_in_memory_field_count": 90,
            "primary_metric": primary_metric,
            "primary_m1d_comparison": stochastic_analysis["m1d_comparisons"][primary_metric],
            "outcome_counts": evidence["outcome_counts"],
            "nonmatching_attempt_event_count": evidence["nonmatching_attempt_event_count"],
            "nonmatching_at_least_frozen_waypoint_lookahead_count": evidence["nonmatching_at_least_frozen_waypoint_lookahead_count"],
            "nonmatching_at_least_frozen_minimum_event_travel_count": evidence["nonmatching_at_least_frozen_minimum_event_travel_count"],
            "combined_correction_probe_selected_case": evidence["combined_correction_probe_selection"]["selected_case"],
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0, "optimizer_steps": 0, "raw_bag_reads": 0,
            "gt_or_map_reads": 0, "c09_reads": 0, "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": spec["run_id"], "state": "COMPLETED", "overall_status": STATUS_PASS})
        sealed = audit_base.seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "composed_frontier_attempt_audit_summary_v1",
            "overall_status": STATUS_FAIL, "failure_reason": str(exc),
            "duration_seconds": time.monotonic() - started,
            "training_steps": 0, "optimizer_steps": 0, "raw_bag_reads": 0,
            "gt_or_map_reads": 0, "c09_reads": 0, "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", failure)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": spec["run_id"], "state": "FAILED", "overall_status": STATUS_FAIL})
        audit_base.seal(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
