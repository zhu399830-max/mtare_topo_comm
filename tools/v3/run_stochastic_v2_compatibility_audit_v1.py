#!/home/zeng-workstation/anaconda3/bin/python
"""Run a sealed, read-only statistical audit of finalized V2 cases.

The source run remains immutable.  Only the case ``status`` field is mapped in
deep in-memory copies by ``analyze_finalized_v2_cases`` before delegating to the
pre-registered V1 statistics implementation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.stochastic_closed_loop_v2_bridge import (
    SOURCE_STATUS,
    analyze_finalized_v2_cases,
)
from mtare_topo.evaluation.stochastic_closed_loop import METRICS
from mtare_topo.evaluation.topology_frontier_lifecycle_evidence import (
    audit_frontier_lifecycle_evidence,
)
from mtare_topo.evaluation.topology_reanchor_evidence import audit_reanchor_evidence
from mtare_topo.governance import load_json, write_json


STATUS_PASS = "PASS_STOCHASTIC_V2_COMPATIBILITY_AUDIT_V1"
STATUS_FAIL = "FAIL_STOCHASTIC_V2_COMPATIBILITY_AUDIT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def schedule_content_sha256(cases: list[dict[str, Any]]) -> str:
    payload = json.dumps(cases, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def validate_execution_scope(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, str]:
    if spec.get("gate") != 6 or spec.get("operation") != "audit":
        raise RuntimeError("Gate-6 audit scope required")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("approved autonomous audit scope required")
    tools = spec.get("frozen_tools")
    if not isinstance(tools, dict) or not tools:
        raise RuntimeError("compatibility audit requires frozen tools")
    verified = {}
    root = project_root.resolve()
    for name, item in tools.items():
        path = (root / item["path"]).resolve()
        path.relative_to(root)
        observed = sha256(path)
        if observed != item["sha256"]:
            raise RuntimeError(f"frozen compatibility audit tool drift: {name}")
        verified[name] = observed
    return verified


def _source_path(relative: str) -> Path:
    path = (PROJECT_ROOT / relative).resolve()
    path.relative_to(PROJECT_ROOT)
    return path


def _parse_seal(source_run: Path, expected_sha256: str) -> dict[str, str]:
    seal_path = source_run / "artifacts/evidence_sha256.txt"
    if sha256(seal_path) != expected_sha256:
        raise RuntimeError("source evidence seal identity drift")
    entries: dict[str, str] = {}
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if relative in entries:
            raise RuntimeError(f"duplicate source seal entry: {relative}")
        entries[relative] = expected
    if not entries:
        raise RuntimeError("source evidence seal is empty")
    return entries


def _verify_bound_file(path: Path, entries: dict[str, str]) -> str:
    relative = path.relative_to(PROJECT_ROOT).as_posix()
    if relative not in entries:
        raise RuntimeError(f"source file absent from evidence seal: {relative}")
    observed = sha256(path)
    if observed != entries[relative]:
        raise RuntimeError(f"source file hash drift: {relative}")
    return observed


def _validate_case_summary(item: dict[str, Any], scheduled: dict[str, Any]) -> None:
    case = item.get("case", {})
    for field in (
        "case_id", "block_id", "method_family", "method_id", "world",
        "environment_seed", "execution_repeat", "checkpoint_seed", "runtime_sec",
    ):
        if case.get(field) != scheduled.get(field):
            raise RuntimeError(f"case schedule mismatch for {scheduled['case_id']}: {field}")
    metrics = item.get("metrics", {})
    for field in METRICS:
        try:
            value = float(metrics[field])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"invalid metric {field}: {scheduled['case_id']}") from exc
        if not math.isfinite(value):
            raise RuntimeError(f"non-finite metric {field}: {scheduled['case_id']}")
    for field in ("final_pose_xyz_m", "final_waypoint_xyz_m"):
        vector = metrics.get(field)
        if not isinstance(vector, list) or len(vector) != 3 or not all(
            isinstance(value, (int, float)) and math.isfinite(float(value)) for value in vector
        ):
            raise RuntimeError(f"invalid {field}: {scheduled['case_id']}")
    recording = metrics.get("recording_audit")
    if not isinstance(recording, dict) or recording.get("passed") is not True:
        raise RuntimeError(f"recording audit is not PASS: {scheduled['case_id']}")
    planner = item.get("planner_evidence")
    if not isinstance(planner, dict):
        raise RuntimeError(f"planner evidence is invalid: {scheduled['case_id']}")
    if case["method_family"] != "original_mtare" and planner.get("failed_cycles") != 0:
        raise RuntimeError(f"planner evidence is invalid: {scheduled['case_id']}")
    identity = item.get("method_identity")
    if not isinstance(identity, dict) or identity.get("method_family") != case["method_family"]:
        raise RuntimeError(f"method identity is invalid: {scheduled['case_id']}")
    storage = item.get("storage")
    required_storage = ("archive_sha256", "decompressed_sha256", "original_bag_sha256")
    if not isinstance(storage, dict) or any(not storage.get(field) for field in required_storage):
        raise RuntimeError(f"storage identity is invalid: {scheduled['case_id']}")


def load_source(
    spec: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    source_run = _source_path(spec["source_run"])
    state_path = source_run / "RUN_STATE.json"
    summary_path = source_run / "metrics/summary.json"
    state = load_json(state_path)
    summary = load_json(summary_path)
    if state.get("state") != spec["expected_source_state"]:
        raise RuntimeError("source run state mismatch")
    if state.get("overall_status") != spec["expected_source_status"]:
        raise RuntimeError("source run status mismatch")
    if summary.get("completed_case_count") != 90:
        raise RuntimeError("source run does not contain 90 completed cases")

    entries = _parse_seal(source_run, spec["source_seal_sha256"])
    bound = {
        state_path.relative_to(PROJECT_ROOT).as_posix(): _verify_bound_file(state_path, entries),
        summary_path.relative_to(PROJECT_ROOT).as_posix(): _verify_bound_file(summary_path, entries),
    }
    for relative in spec.get("source_identity_files", []):
        path = source_run / relative
        bound[path.relative_to(PROJECT_ROOT).as_posix()] = _verify_bound_file(path, entries)

    schedule_path = source_run / "config/case_schedule.json"
    bound[schedule_path.relative_to(PROJECT_ROOT).as_posix()] = _verify_bound_file(schedule_path, entries)
    schedule_file_sha256 = sha256(schedule_path)
    if schedule_file_sha256 != spec["source_schedule_file_sha256"]:
        raise RuntimeError("source case schedule file identity drift")
    schedule = load_json(schedule_path).get("cases")
    if not isinstance(schedule, list) or len(schedule) != 90:
        raise RuntimeError("source schedule does not contain exactly 90 cases")
    schedule_canonical_sha256 = schedule_content_sha256(schedule)
    if schedule_canonical_sha256 != spec["source_schedule_content_sha256"]:
        raise RuntimeError("source case schedule canonical-content identity drift")
    scheduled_ids = [str(case["case_id"]) for case in schedule]
    if len(set(scheduled_ids)) != 90:
        raise RuntimeError("source schedule case IDs are not unique")

    case_root = source_run / "artifacts/cases"
    discovered = {path.parent.name: path for path in case_root.glob("*/summary.json")}
    if len(discovered) != 90 or set(discovered) != set(scheduled_ids):
        raise RuntimeError("source case summary count is not exactly 90")
    summaries: list[dict[str, Any]] = []
    manifest_cases = []
    for scheduled in schedule:
        path = discovered[scheduled["case_id"]]
        bound[path.relative_to(PROJECT_ROOT).as_posix()] = _verify_bound_file(path, entries)
        item = load_json(path)
        if item.get("status") != SOURCE_STATUS:
            raise RuntimeError(f"source case is not finalized V2 PASS: {path.parent.name}")
        _validate_case_summary(item, scheduled)
        summaries.append(item)
        manifest_cases.append({
            "index": scheduled["index"],
            "case_id": scheduled["case_id"],
            "block_id": scheduled["block_id"],
            "method_family": scheduled["method_family"],
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "sha256": bound[path.relative_to(PROJECT_ROOT).as_posix()],
        })
    quotas = Counter(item["case"]["method_family"] for item in summaries)
    if quotas != Counter({"original_mtare": 30, "m1d_topology": 30, "layered_gt_map_oracle": 30}):
        raise RuntimeError("source method-family quotas are not 30/30/30")
    blocks = Counter(item["case"]["block_id"] for item in summaries)
    if len(blocks) != 10 or set(blocks.values()) != {9}:
        raise RuntimeError("source block design is not ten blocks of nine cases")
    provenance = {
        "schema_version": "stochastic_v2_compatibility_source_v1",
        "source_run": spec["source_run"],
        "source_state": state["state"],
        "source_status": state["overall_status"],
        "source_seal_sha256": spec["source_seal_sha256"],
        "source_seal_entry_count": len(entries),
        "verified_bound_file_count": len(bound),
        "verified_bound_files": bound,
        "source_mutation_permitted": False,
        "raw_bag_reads": 0,
        "c09_reads": 0,
        "c10_reads": 0,
    }
    manifest = {
        "schema_version": "stochastic_v2_compatibility_source_manifest_v1",
        "case_count": 90,
        "block_count": 10,
        "family_case_counts": dict(sorted(quotas.items())),
        "schedule_path": schedule_path.relative_to(PROJECT_ROOT).as_posix(),
        "schedule_file_sha256": schedule_file_sha256,
        "schedule_content_sha256": schedule_canonical_sha256,
        "cases": manifest_cases,
    }
    return summaries, provenance, manifest


def audit_source_reanchor_mechanisms(
    spec: dict[str, Any], summaries: list[dict[str, Any]]
) -> dict[str, Any]:
    """Audit all sealed V9 planner traces without reading archived sensor bags."""

    source_run = _source_path(spec["source_run"])
    entries = _parse_seal(source_run, spec["source_seal_sha256"])
    cases = []
    verified_files: dict[str, str] = {}
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
        if not isinstance(runtime, dict) or not isinstance(runtime.get("graph"), dict):
            raise RuntimeError(f"invalid V9 topology snapshot: {case['case_id']}")
        audit = audit_reanchor_evidence(rows, runtime["graph"])
        lifecycle = audit_frontier_lifecycle_evidence(rows, runtime["graph"])
        metrics = item["metrics"]
        cases.append({
            "case_id": case["case_id"],
            "block_id": case["block_id"],
            "world": case["world"],
            "environment_seed": case["environment_seed"],
            "checkpoint_seed": case["checkpoint_seed"],
            "traveling_distance_m": float(metrics["traveling_distance_m"]),
            "final_explored_volume_m3": float(metrics["final_explored_volume_m3"]),
            "post_warmup_fallback_rate": float(
                item["method_identity"]["post_warmup_fallback_rate"]
            ),
            "mechanism": audit,
            "frontier_lifecycle": lifecycle,
        })
    if len(cases) != 30:
        raise RuntimeError("V9 mechanism audit requires exactly 30 sealed M1D cases")
    proxy_cases = [
        value for value in cases
        if value["mechanism"]["exact_arrival_proxy_frame_count"] > 0
    ]
    return {
        "schema_version": "v9_reanchor_mechanism_audit_v1",
        "case_count": len(cases),
        "case_count_with_exact_arrival_proxy": len(proxy_cases),
        "case_fraction_with_exact_arrival_proxy": len(proxy_cases) / len(cases),
        "total_exact_arrival_proxy_frames": sum(
            value["mechanism"]["exact_arrival_proxy_frame_count"] for value in cases
        ),
        "maximum_constant_proxy_run_frames": max(
            value["mechanism"]["longest_constant_proxy_run_frames"] for value in cases
        ),
        "maximum_tail_without_route_arc_growth_frames": max(
            value["mechanism"]["tail_without_route_arc_growth_frames"] for value in cases
        ),
        "maximum_constant_frontier_run_frames": max(
            value["frontier_lifecycle"]["longest_constant_frontier_run_frames"]
            for value in cases
        ),
        "maximum_node_exit_stub_minus_branch_count": max(
            value["frontier_lifecycle"]["maximum_node_exit_stub_minus_branch_count"]
            for value in cases
        ),
        "verified_trace_snapshot_file_count": len(verified_files),
        "verified_trace_snapshot_files": dict(sorted(verified_files.items())),
        "raw_bag_reads": 0,
        "uses_evaluator_gt": False,
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
        raise RuntimeError("compatibility audit run identity mismatch")
    if load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("compatibility audit run is not executable")
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
        analysis = analyze_finalized_v2_cases(summaries)
        mechanism = audit_source_reanchor_mechanisms(spec, summaries)
        write_json(run_dir / "config/source_provenance.json", provenance)
        write_json(run_dir / "config/source_manifest.json", manifest)
        write_json(run_dir / "metrics/stochastic_analysis.json", analysis)
        write_json(run_dir / "metrics/v9_reanchor_mechanism_audit.json", mechanism)
        summary = {
            "schema_version": "stochastic_v2_compatibility_audit_summary_v1",
            "overall_status": STATUS_PASS,
            "source_case_count": len(summaries),
            "source_mutation_count": 0,
            "mapped_in_memory_field": "status",
            "mapped_in_memory_field_count": len(summaries),
            "primary_metric": analysis["primary_metric"],
            "primary_m1d_comparison": analysis["m1d_comparisons"][analysis["primary_metric"]],
            "v9_case_count_with_exact_arrival_proxy": mechanism[
                "case_count_with_exact_arrival_proxy"
            ],
            "v9_total_exact_arrival_proxy_frames": mechanism[
                "total_exact_arrival_proxy_frames"
            ],
            "v9_maximum_constant_frontier_run_frames": mechanism[
                "maximum_constant_frontier_run_frames"
            ],
            "v9_maximum_node_exit_stub_minus_branch_count": mechanism[
                "maximum_node_exit_stub_minus_branch_count"
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
            "schema_version": "stochastic_v2_compatibility_audit_summary_v1",
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
