#!/home/zeng-workstation/anaconda3/bin/python
"""Run the source-bound, traversal-complete V1R2 frontier audit."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r2 import (
    audit_frontier_attempt_evidence_v1r2,
)
from mtare_topo.governance import load_json, write_json
import run_frontier_attempt_outcome_audit_v1r as v1r
from run_stochastic_v2_compatibility_audit_v1 import (
    load_source,
    sha256,
    validate_execution_scope,
)


STATUS_PASS = "PASS_FRONTIER_ATTEMPT_OUTCOME_AUDIT_V1R2"
STATUS_FAIL = "FAIL_FRONTIER_ATTEMPT_OUTCOME_AUDIT_V1R2"


def validate_material_run_identity_v1r2(
    spec_path: Path,
    spec: dict[str, Any],
    run_dir: Path,
    *,
    project_root: Path = PROJECT_ROOT,
) -> None:
    v1r.validate_material_run_identity(
        spec_path, spec, run_dir, project_root=project_root
    )
    root = project_root.resolve()
    status_snapshot = load_json(run_dir / "config/status_snapshot.json")
    if status_snapshot != load_json(root / "results/project_status.json"):
        raise RuntimeError("frontier attempt audit project-status snapshot drift")
    environment = load_json(run_dir / "config/environment.json")
    if set(environment) != {
        "created_at_utc", "hostname", "platform", "python", "python_executable"
    }:
        raise RuntimeError("frontier attempt audit environment schema drift")
    try:
        created = datetime.fromisoformat(str(environment["created_at_utc"]))
    except ValueError as exc:
        raise RuntimeError("frontier attempt audit environment time is invalid") from exc
    now = datetime.now(timezone.utc)
    if (
        created.tzinfo is None
        or created > now
        or (now - created).total_seconds() > 3600
        or environment["hostname"] != platform.node()
        or environment["platform"] != platform.platform()
        or environment["python"] != sys.version
        or Path(environment["python_executable"]).resolve() != Path(sys.executable).resolve()
    ):
        raise RuntimeError("frontier attempt audit environment identity drift")


def validate_source_generator_identity(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, dict[str, str]]:
    root = project_root.resolve()
    source_spec = load_json(root / spec["source_run"] / "config/run_spec.json")
    identities = {
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
    frozen = source_spec.get("frozen_tools", {})
    for name, identity in identities.items():
        if frozen.get(name) != identity:
            raise RuntimeError(f"source generator identity drift: {name}")
        if sha256(root / identity["path"]) != identity["sha256"]:
            raise RuntimeError(f"local source generator implementation drift: {name}")
    return identities


def audit_source_frontier_attempts_v1r2(
    spec: dict[str, Any], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    original = v1r.audit_frontier_attempt_evidence_v1r
    v1r.audit_frontier_attempt_evidence_v1r = audit_frontier_attempt_evidence_v1r2
    try:
        evidence = v1r.audit_source_frontier_attempts_v1r(spec, summaries)
    finally:
        v1r.audit_frontier_attempt_evidence_v1r = original
    evidence["schema_version"] = "frontier_attempt_outcome_audit_v1r2"
    evidence["historical_stub_creation_frame_directly_reconstructable"] = False
    evidence["historical_target_state_bound_to_frozen_source_planner"] = True
    return evidence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec_path, run_dir = args.spec.resolve(), args.run_dir.resolve()
    spec = load_json(spec_path)
    validate_material_run_identity_v1r2(spec_path, spec, run_dir)
    started = time.monotonic()
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": spec["run_id"], "state": "RUNNING"
    })
    try:
        frozen_tools = validate_execution_scope(spec)
        generator_identity = validate_source_generator_identity(spec)
        summaries, provenance, manifest = load_source(spec)
        provenance["verified_frozen_tools"] = frozen_tools
        provenance["source_generator_identities"] = generator_identity
        provenance["historical_stub_creation_frame_directly_reconstructable"] = False
        evidence = audit_source_frontier_attempts_v1r2(spec, summaries)
        write_json(run_dir / "config/source_provenance.json", provenance)
        write_json(run_dir / "config/source_manifest.json", manifest)
        write_json(run_dir / "metrics/frontier_attempt_outcomes.json", evidence)
        summary = {
            "schema_version": "frontier_attempt_outcome_audit_summary_v1r2",
            "overall_status": STATUS_PASS,
            "source_case_count": len(summaries),
            "v9_case_count": evidence["case_count"],
            "case_count_with_nonmatching_attempt": evidence["case_count_with_nonmatching_attempt"],
            "outcome_counts": evidence["outcome_counts"],
            "nonmatching_attempt_event_count": evidence["nonmatching_attempt_event_count"],
            "nonmatching_at_least_frozen_waypoint_lookahead_count": evidence["nonmatching_at_least_frozen_waypoint_lookahead_count"],
            "nonmatching_at_least_frozen_minimum_event_travel_count": evidence["nonmatching_at_least_frozen_minimum_event_travel_count"],
            "historical_stub_creation_frame_directly_reconstructable": False,
            "historical_target_state_bound_to_frozen_source_planner": True,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0, "optimizer_steps": 0, "raw_bag_reads": 0,
            "gt_or_map_reads": 0, "c09_reads": 0, "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "COMPLETED", "overall_status": STATUS_PASS,
        })
        sealed_files = v1r.seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed_files}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure = {
            "schema_version": "frontier_attempt_outcome_audit_summary_v1r2",
            "overall_status": STATUS_FAIL, "failure_reason": str(exc),
            "duration_seconds": time.monotonic() - started,
            "training_steps": 0, "optimizer_steps": 0, "raw_bag_reads": 0,
            "gt_or_map_reads": 0, "c09_reads": 0, "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", failure)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": spec["run_id"],
            "state": "FAILED", "overall_status": STATUS_FAIL,
        })
        v1r.seal(run_dir)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
