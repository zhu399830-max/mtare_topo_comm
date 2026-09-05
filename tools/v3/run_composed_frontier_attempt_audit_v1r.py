#!/home/zeng-workstation/anaconda3/bin/python
"""Audit the exact 69+21 source after the sealed V3R aggregate-only failure."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.combined_correction_probe_selection import (
    select_combined_correction_probe_case,
)
from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r3 import (
    audit_frontier_attempt_evidence_v1r3,
)
from mtare_topo.evaluation.topology_reanchor_evidence import audit_reanchor_evidence
from mtare_topo.governance import load_json
import run_composed_frontier_attempt_audit_v1 as base


STATUS_PASS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R"
STATUS_FAIL = "FAIL_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1R"
EXPECTED_RECOVERY_STATE = "FAILED"
EXPECTED_RECOVERY_STATUS = "FAIL_AEE_COMPOSITE_V9_STOCHASTIC_V3R_SYSTEM_RECOVERY"
EXPECTED_RECOVERY_FAILURE = "analysis cannot include a failed case"
EXPECTED_RECOVERY_CASES = 21


def _verify_all_case_files(
    case_root: Path,
    entries: dict[str, str],
    project_root: Path,
) -> dict[str, str]:
    files = sorted(path for path in case_root.rglob("*") if path.is_file())
    if not files:
        raise RuntimeError(f"recovery case evidence absent: {case_root.name}")
    return {
        path.relative_to(project_root.resolve()).as_posix(): base._verify_bound_file_local(
            path, entries, project_root
        )
        for path in files
    }


def load_composed_source(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> tuple[
    list[dict[str, Any]],
    dict[str, Path],
    dict[str, dict[str, str]],
    dict[str, dict[str, Any]],
    dict[str, Any],
]:
    """Load 69 predecessor and 21 recovery cases without promoting V3R to PASS."""
    predecessor = base._root(spec["predecessor_run"], project_root)
    recovery = base._root(spec["source_run"], project_root)
    predecessor_entries = base._parse_seal(
        predecessor, spec["predecessor_seal_sha256"]
    )
    recovery_entries = base._parse_seal(recovery, spec["source_seal_sha256"])
    predecessor_summary, predecessor_spec, predecessor_bound = base._verify_root_identity(
        predecessor,
        predecessor_entries,
        "FAILED",
        base.EXPECTED_PREDECESSOR_STATUS,
        project_root,
    )
    recovery_summary, recovery_spec, recovery_bound = base._verify_root_identity(
        recovery,
        recovery_entries,
        EXPECTED_RECOVERY_STATE,
        EXPECTED_RECOVERY_STATUS,
        project_root,
    )
    if predecessor_summary.get("completed_case_count") != 77:
        raise RuntimeError("predecessor completed-case count drift")
    if recovery_summary.get("failure_reason") != EXPECTED_RECOVERY_FAILURE:
        raise RuntimeError("recovery aggregate-only failure reason drift")
    if (
        recovery_summary.get("completed_recovery_case_count")
        != EXPECTED_RECOVERY_CASES
        or recovery_summary.get("planned_recovery_case_count")
        != EXPECTED_RECOVERY_CASES
    ):
        raise RuntimeError("recovery completed/planned case count drift")

    schedule_path = predecessor / "config/case_schedule.json"
    schedule_relative = schedule_path.relative_to(project_root.resolve()).as_posix()
    predecessor_bound[schedule_relative] = base._verify_bound_file_local(
        schedule_path, predecessor_entries, project_root
    )
    schedule = load_json(schedule_path).get("cases")
    if not isinstance(schedule, list) or len(schedule) != 90:
        raise RuntimeError("composed source schedule is not exactly 90 cases")
    if base.sha256(schedule_path) != spec["source_schedule_file_sha256"]:
        raise RuntimeError("composed source schedule file drift")
    if base.schedule_content_sha256(schedule) != spec["source_schedule_content_sha256"]:
        raise RuntimeError("composed source schedule content drift")
    scheduled_ids = [item.get("case_id") for item in schedule]
    if len(set(scheduled_ids)) != 90:
        raise RuntimeError("composed schedule case identities are not unique")

    recovery_schedule_path = recovery / "config/recovery_case_schedule.json"
    progress_path = recovery / "metrics/progress.json"
    for path in (recovery_schedule_path, progress_path):
        relative = path.relative_to(project_root.resolve()).as_posix()
        recovery_bound[relative] = base._verify_bound_file_local(
            path, recovery_entries, project_root
        )
    recovery_schedule = load_json(recovery_schedule_path).get("cases")
    expected_recovery = [
        item
        for item in schedule
        if item.get("block_id") == "tunnel_env23" or int(item.get("index", -1)) >= 77
    ]
    if recovery_schedule != expected_recovery or len(expected_recovery) != 21:
        raise RuntimeError("recovery schedule is not the exact whole-block plus tail selection")
    progress = load_json(progress_path)
    if (
        progress.get("completed_recovery_case_count") != 21
        or progress.get("planned_recovery_case_count") != 21
        or progress.get("last_original_schedule_index") != 89
        or progress.get("last_case_id") != expected_recovery[-1]["case_id"]
    ):
        raise RuntimeError("recovery progress identity drift")

    recovery_ids = {item["case_id"] for item in expected_recovery}
    mapping = {
        item["case_id"]: (
            "recovery_run" if item["case_id"] in recovery_ids else "failed_source_run"
        )
        for item in schedule
    }
    if Counter(mapping.values()) != Counter(
        {"failed_source_run": 69, "recovery_run": 21}
    ):
        raise RuntimeError("composed source quota drift")

    roots = {"failed_source_run": predecessor, "recovery_run": recovery}
    seals = {
        "failed_source_run": predecessor_entries,
        "recovery_run": recovery_entries,
    }
    summaries: list[dict[str, Any]] = []
    case_roots: dict[str, Path] = {}
    manifest_cases: list[dict[str, Any]] = []
    for scheduled in schedule:
        case_id = scheduled["case_id"]
        source_name = mapping[case_id]
        case_root = roots[source_name] / "artifacts/cases" / case_id
        summary_path = case_root / "summary.json"
        observed = base._verify_bound_file_local(
            summary_path, seals[source_name], project_root
        )
        if source_name == "recovery_run":
            recovery_bound.update(
                _verify_all_case_files(case_root, recovery_entries, project_root)
            )
        item = load_json(summary_path)
        if item.get("status") != base.SOURCE_STATUS:
            raise RuntimeError(f"composed case is not V2 PASS: {case_id}")
        base._validate_case_summary(item, scheduled)
        summaries.append(item)
        case_roots[case_id] = case_root
        manifest_cases.append(
            {
                "index": scheduled["index"],
                "case_id": case_id,
                "block_id": scheduled["block_id"],
                "method_family": scheduled["method_family"],
                "source": source_name,
                "summary_path": summary_path.relative_to(
                    project_root.resolve()
                ).as_posix(),
                "summary_sha256": observed,
            }
        )

    quotas = Counter(item["case"]["method_family"] for item in summaries)
    blocks = Counter(item["case"]["block_id"] for item in summaries)
    if quotas != Counter(
        {"original_mtare": 30, "m1d_topology": 30, "layered_gt_map_oracle": 30}
    ):
        raise RuntimeError("composed method-family quotas are not 30/30/30")
    if len(blocks) != 10 or set(blocks.values()) != {9}:
        raise RuntimeError("composed block design is not ten blocks of nine")

    source_specs = {
        "failed_source_run": predecessor_spec,
        "recovery_run": recovery_spec,
    }
    provenance = {
        "schema_version": "composed_stochastic_source_provenance_v1r",
        "predecessor_run": spec["predecessor_run"],
        "predecessor_status": base.EXPECTED_PREDECESSOR_STATUS,
        "predecessor_seal_sha256": spec["predecessor_seal_sha256"],
        "predecessor_seal_entries": len(predecessor_entries),
        "source_run": spec["source_run"],
        "source_state": EXPECTED_RECOVERY_STATE,
        "source_status": EXPECTED_RECOVERY_STATUS,
        "source_failure_reason": EXPECTED_RECOVERY_FAILURE,
        "source_failure_class": "aggregate_status_compatibility_only",
        "source_case_pass_count": 21,
        "source_seal_sha256": spec["source_seal_sha256"],
        "source_seal_entries": len(recovery_entries),
        "verified_bound_files": {**predecessor_bound, **recovery_bound},
        "source_status_promoted_to_pass": False,
        "source_mutation_permitted": False,
        "raw_bag_reads": 0,
        "gt_or_map_reads": 0,
        "c09_reads": 0,
        "c10_reads": 0,
    }
    manifest = {
        "schema_version": "composed_stochastic_source_manifest_v1r",
        "case_count": 90,
        "block_count": 10,
        "family_case_counts": dict(sorted(quotas.items())),
        "source_counts": dict(sorted(Counter(mapping.values()).items())),
        "schedule_file_sha256": base.sha256(schedule_path),
        "schedule_content_sha256": base.schedule_content_sha256(schedule),
        "cases": manifest_cases,
    }
    return summaries, case_roots, seals, source_specs, {
        "provenance": provenance,
        "manifest": manifest,
    }


def validate_composed_generator_identity(
    spec: dict[str, Any],
    source_specs: dict[str, dict[str, Any]],
    *,
    project_root: Path = PROJECT_ROOT,
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
        if base.sha256(project_root / item["path"]) != item["sha256"]:
            raise RuntimeError(f"local composed-source generator drift: {name}")
    predecessor_tools = source_specs["failed_source_run"].get("frozen_tools", {})
    for name, item in expected.items():
        if predecessor_tools.get(name) != item:
            raise RuntimeError(f"predecessor generator identity drift: {name}")
    recovery_spec = source_specs["recovery_run"]
    if recovery_spec.get("replacement_contract", {}).get(
        "scientific_contract_change"
    ) is not False:
        raise RuntimeError("recovery source does not declare unchanged scientific contract")
    for name in ("online_runtime", "v9_node"):
        if recovery_spec.get("frozen_tools", {}).get(name) != expected[name]:
            raise RuntimeError(f"recovery directly frozen generator drift: {name}")
    return {
        "schema_version": "composed_source_generator_identity_v1r",
        "direct_local_hashes_verified": expected,
        "predecessor_direct_bindings_verified": list(expected),
        "recovery_direct_bindings_verified": ["online_runtime", "v9_node"],
        "recovery_indirect_predecessor_bindings_verified": [
            "causal_graph",
            "frontier_planner",
        ],
        "scientific_contract_change": False,
    }


def audit_composed_frontier_attempts(
    summaries: list[dict[str, Any]],
    case_roots: dict[str, Path],
    seals: dict[str, dict[str, str]],
    manifest: dict[str, Any],
) -> dict[str, Any]:
    source_by_case = {row["case_id"]: row["source"] for row in manifest["cases"]}
    verified_files: dict[str, str] = {}
    cases: list[dict[str, Any]] = []
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
            verified_files[relative] = base._verify_bound_file(path, seals[source_name])
        rows = [
            json.loads(line)
            for line in trace_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        snapshot = load_json(snapshot_path)
        runtime = snapshot.get("runtime")
        graph = None if not isinstance(runtime, dict) else runtime.get("graph")
        graph_config = None if not isinstance(graph, dict) else graph.get("config")
        planner_config = snapshot.get("planner_config")
        if not all(
            isinstance(value, dict)
            for value in (graph, graph_config, planner_config)
        ):
            raise RuntimeError(f"invalid composed V9 planner snapshot: {case_id}")
        cases.append(
            {
                "case_id": case_id,
                "source": source_name,
                "block_id": case["block_id"],
                "world": case["world"],
                "environment_seed": case["environment_seed"],
                "checkpoint_seed": case["checkpoint_seed"],
                "traveling_distance_m": float(item["metrics"]["traveling_distance_m"]),
                "final_explored_volume_m3": float(
                    item["metrics"]["final_explored_volume_m3"]
                ),
                "waypoint_lookahead_m": float(
                    planner_config["waypoint_lookahead_m"]
                ),
                "minimum_event_travel_m": float(
                    graph_config["minimum_event_travel_m"]
                ),
                "frontier_attempt": audit_frontier_attempt_evidence_v1r3(rows, graph),
                "reanchor_evidence": audit_reanchor_evidence(rows, graph),
            }
        )
    if len(cases) != 30:
        raise RuntimeError("composed V9 audit did not contain exactly 30 cases")
    aggregate = base.audit_base.summarize_frontier_attempt_cases_v1r(cases)
    proxy_cases = [
        case
        for case in cases
        if int(case["reanchor_evidence"]["exact_arrival_proxy_frame_count"]) > 0
    ]
    probe_selection = select_combined_correction_probe_case(cases)
    return {
        "schema_version": "composed_frontier_attempt_outcome_audit_v1r",
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
        "raw_bag_reads": 0,
        "gt_or_map_reads": 0,
        "c09_reads": 0,
        "c10_reads": 0,
        "cases": cases,
    }


def main() -> int:
    base.STATUS_PASS = STATUS_PASS
    base.STATUS_FAIL = STATUS_FAIL
    base.load_composed_source = load_composed_source
    base.validate_composed_generator_identity = validate_composed_generator_identity
    base.audit_composed_frontier_attempts = audit_composed_frontier_attempts
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
