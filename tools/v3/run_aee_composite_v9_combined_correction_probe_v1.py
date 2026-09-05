#!/home/zeng-workstation/anaconda3/bin/python
"""Run one source-selected live probe for the V5 combined graph correction."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE
from mtare_topo.evaluation.host_bag_archive import finalize_handed_off_bag
from mtare_topo.governance import load_json, write_json
import run_aee_adapted_readiness_v1 as readiness
import run_aee_composite_v9_reanchor_probe_v1 as reanchor_probe
from run_mtare_single_robot_stochastic_v1 import image_file_hash, seal, sha256


RUN_ID = "gate6_20260823_aee_composite_v9_combined_correction_probe_v1_seed20260823"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_COMBINED_CORRECTION_PROBE_V1"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_COMBINED_CORRECTION_PROBE_V1"
AUDIT_STATUS_PASS = "PASS_COMPOSED_FRONTIER_ATTEMPT_AUDIT_V1"
DEPLOYMENT_STATUS_PASS = reanchor_probe.DEPLOYMENT_STATUS_PASS
IMAGE = readiness.IMAGE
HOST_UID = readiness.HOST_UID
HOST_GID = readiness.HOST_GID
MINIMUM_FREE_BYTES = 20 * 1024**3


def _seal_entries(seal_path: Path) -> dict[str, str]:
    rows = seal_path.read_text(encoding="utf-8").splitlines()
    entries: dict[str, str] = {}
    for row in rows:
        digest, relative = row.split("  ", 1)
        if relative in entries:
            raise RuntimeError("combined-correction audit seal has duplicate entries")
        entries[relative] = digest
    return entries


def validate_composed_audit_source(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    root = project_root.resolve()
    source = (root / spec["mechanism_audit_run"]).resolve()
    source.relative_to(root)
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    evidence_path = source / "metrics/frontier_attempt_outcomes.json"
    manifest_path = source / "config/source_manifest.json"
    provenance_path = source / "config/source_provenance.json"
    seal_path = source / "artifacts/evidence_sha256.txt"
    state, summary, evidence, manifest, provenance = (
        load_json(state_path), load_json(summary_path), load_json(evidence_path),
        load_json(manifest_path), load_json(provenance_path),
    )
    expected_status = spec.get("mechanism_audit_status")
    if expected_status != AUDIT_STATUS_PASS:
        raise RuntimeError("combined-correction probe audit status contract drift")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != expected_status
        or summary.get("overall_status") != expected_status
        or summary.get("source_case_count") != 90
        or summary.get("v9_case_count") != 30
    ):
        raise RuntimeError("combined-correction mechanism audit is not the exact PASS")
    if (
        manifest.get("case_count") != 90
        or manifest.get("family_case_counts")
        != {"layered_gt_map_oracle": 30, "m1d_topology": 30, "original_mtare": 30}
        or provenance.get("source_mutation_permitted") is not False
        or evidence.get("case_count") != 30
        or evidence.get("raw_bag_reads") != 0
        or evidence.get("gt_or_map_reads") != 0
    ):
        raise RuntimeError("combined-correction mechanism audit content drift")
    selection = evidence.get("combined_correction_probe_selection")
    if (
        not isinstance(selection, dict)
        or selection.get("selection_rule")
        != "minimize_max_first_mechanism_frame_then_case_id"
        or selection.get("performance_outcomes_used_for_selection") != []
        or not isinstance(selection.get("selected_case"), dict)
        or not isinstance(selection.get("ranked_candidates"), list)
        or not selection["ranked_candidates"]
        or selection.get("eligible_case_count") != len(selection["ranked_candidates"])
        or selection["selected_case"] != selection["ranked_candidates"][0]
    ):
        raise RuntimeError("combined-correction probe selection evidence drift")
    selected = selection["selected_case"]
    matching = [case for case in evidence["cases"] if case.get("case_id") == selected.get("case_id")]
    if len(matching) != 1:
        raise RuntimeError("combined-correction selected source case is not unique")
    for key in ("case_id", "world", "environment_seed", "checkpoint_seed"):
        if matching[0].get(key) != selected.get(key):
            raise RuntimeError(f"combined-correction selected source identity drift: {key}")
    if sha256(seal_path) != spec["mechanism_audit_seal_sha256"]:
        raise RuntimeError("combined-correction mechanism audit seal identity drift")
    entries = _seal_entries(seal_path)
    bound = {}
    for path in (state_path, summary_path, evidence_path, manifest_path, provenance_path):
        relative = path.relative_to(root).as_posix()
        observed = sha256(path)
        if entries.get(relative) != observed:
            raise RuntimeError(f"combined-correction audit bound evidence drift: {relative}")
        bound[relative] = observed
    return {
        "schema_version": "aee_composite_v9_combined_correction_audit_source_v1",
        "run": spec["mechanism_audit_run"],
        "status": expected_status,
        "seal_sha256": spec["mechanism_audit_seal_sha256"],
        "selected_source_case": selected,
        "selection_rule": selection["selection_rule"],
        "performance_outcomes_used_for_selection": [],
        "verified_bound_files": bound,
        "source_mutation_permitted": False,
    }


def validate_probe_case(value: Any, selected: dict[str, Any]) -> dict[str, Any]:
    expected = {
        "case_id": (
            f"{selected['world']}_env{selected['environment_seed']}_"
            f"m1d_seed{selected['checkpoint_seed']}_combined_v5_probe"
        ),
        "source_case_id": selected["case_id"],
        "world": selected["world"],
        "environment_seed": selected["environment_seed"],
        "checkpoint_seed": selected["checkpoint_seed"],
        "runtime_sec": 180.0,
    }
    if value != expected:
        raise RuntimeError("combined-correction probe case identity drift")
    return value


def case_command(
    run_dir: Path,
    case: dict[str, Any],
    checkpoint: dict[str, Any],
    topic_contract: str,
) -> tuple[list[str], str]:
    name = "aee-v5-combined-correction-probe"
    inner = [
        "python3", "/workspace/tools/v3/run_mtare_single_robot_case_v3.py",
        "--case-dir", f"/evidence/cases/{case['case_id']}",
        "--case-id", case["case_id"],
        "--world", case["world"],
        "--environment-seed", str(case["environment_seed"]),
        "--method-id", f"m1d_aee_combined_v5_seed{case['checkpoint_seed']}",
        "--method-family", "m1d_topology",
        "--execution-repeat", "0",
        "--checkpoint-seed", str(case["checkpoint_seed"]),
        "--block-id", f"combined_v5_probe_{case['world']}_env{case['environment_seed']}",
        "--checkpoint", checkpoint["path"],
        "--checkpoint-sha256", checkpoint["sha256"],
        "--runtime-sec", "180.0",
        "--topic-contract", f"/workspace/{topic_contract}",
        "--archive-mode", "host",
        "--host-uid", str(HOST_UID),
        "--host-gid", str(HOST_GID),
    ]
    shell = (
        "source /opt/ros/noetic/setup.bash && "
        "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && "
        "source /home/docker-user/mtare/tare_system/devel/setup.bash --extend && "
        "export PYTHONPATH=/workspace/src:$PYTHONPATH && " + shlex.join(inner)
    )
    return [
        "docker", "run", "--rm", "--network", "none", "--hostname", "localhost",
        "--name", name, "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw", "--entrypoint", "/bin/bash",
        IMAGE, "-lc", shell,
    ], name


def audit_combined_correction(
    case_dir: Path, *, require_correction: bool = True
) -> dict[str, Any]:
    summary = load_json(case_dir / "summary.json")
    planner = summary["planner_evidence"]
    snapshot = load_json(case_dir / planner["topology_snapshot"])
    if snapshot.get("schema_version") != "semantic_topology_global_node_v5_snapshot_v1":
        raise RuntimeError("probe did not use the V5 ROS node")
    runtime = snapshot.get("runtime", {})
    if runtime.get("schema_version") != "online_topology_planner_runtime_v3":
        raise RuntimeError("probe did not use the V3 combined runtime")
    reanchor_count = int(runtime.get("verified_reanchor_count", -1))
    rejection_count = int(runtime.get("frontier_execution_rejection_count", -1))
    outcomes = runtime.get("frontier_execution_outcomes")
    if reanchor_count < 0 or rejection_count < 0 or not isinstance(outcomes, dict):
        raise RuntimeError("combined-correction snapshot counters are invalid")
    rows = [
        json.loads(line) for line in
        (case_dir / planner["decision_trace"]).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    reanchor_rows = [
        row for row in rows
        if row.get("graph_update", {}).get("reason") == "verified_backtrack_reanchor"
    ]
    feedback_rows = [
        row for row in rows
        if isinstance(row.get("graph_update", {}).get("frontier_execution_feedback"), dict)
    ]
    rejection_rows = [
        row for row in feedback_rows
        if row["graph_update"]["frontier_execution_feedback"].get("record_rejection") is True
    ]
    if len(reanchor_rows) != reanchor_count or len(rejection_rows) != rejection_count:
        raise RuntimeError("combined-correction trace and snapshot counters disagree")
    trace_outcomes = Counter(
        str(row["graph_update"]["frontier_execution_feedback"].get("outcome"))
        for row in feedback_rows
    )
    if dict(sorted(trace_outcomes.items())) != outcomes:
        raise RuntimeError("combined-correction feedback outcome totals disagree")
    permitted = {
        "matched_verified_departure": False,
        "divergent_verified_departure": True,
        "same_node_loop_merge": True,
    }
    for row in feedback_rows:
        feedback = row["graph_update"]["frontier_execution_feedback"]
        if feedback.get("outcome") not in permitted or feedback.get("record_rejection") is not permitted[feedback["outcome"]]:
            raise RuntimeError("combined-correction feedback outcome contract drift")
    corrective_rows = sorted(
        [*reanchor_rows, *rejection_rows], key=lambda row: int(row["frame_index"])
    )
    if require_correction and not corrective_rows:
        raise RuntimeError("source-selected probe did not activate either graph correction")
    first_frame = None if not corrective_rows else int(corrective_rows[0]["frame_index"])
    first_arc = None if not corrective_rows else float(corrective_rows[0]["route_arc_m"])
    later_growth = [
        row for row in rows
        if first_frame is not None
        and int(row["frame_index"]) > first_frame
        and float(row["route_arc_m"]) > first_arc
    ]
    if corrective_rows and not later_growth:
        raise RuntimeError("route arc did not grow after combined graph correction")
    return {
        "schema_version": "aee_composite_v9_combined_correction_probe_audit_v1",
        "verified_reanchor_count": reanchor_count,
        "frontier_execution_rejection_count": rejection_count,
        "frontier_execution_outcomes": outcomes,
        "first_corrective_frame": first_frame,
        "first_corrective_route_arc_m": first_arc,
        "first_post_correction_growth_frame": (
            None if not later_growth else int(later_growth[0]["frame_index"])
        ),
        "final_route_arc_m": float(rows[-1]["route_arc_m"]),
        "post_correction_route_growth": bool(later_growth),
        "both_correction_types_activated": reanchor_count > 0 and rejection_count > 0,
        "uses_evaluator_gt": False,
    }


def failure(run_dir: Path, started: float, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "aee_composite_v9_combined_correction_probe_summary_v1",
        "overall_status": STATUS_FAIL, "failure_reason": message,
        "completed_cases": 0, "planned_cases": 1,
        "training_steps": 0, "optimizer_steps": 0,
        "c09_reads": 0, "c10_reads": 0,
        "duration_seconds": time.monotonic() - started,
    })
    write_json(run_dir / "RUN_STATE.json", {
        "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
        "state": "FAILED", "overall_status": STATUS_FAIL,
    })
    seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("combined-correction probe run identity/state mismatch")
    started = time.monotonic()
    try:
        if spec.get("gate") != 6 or spec.get("operation") != "closed_loop_single":
            raise RuntimeError("Gate-6 closed-loop scope required")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("approved autonomous execution scope required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen combined-correction probe tool drift: {name}")
        audit_source = validate_composed_audit_source(spec)
        case = validate_probe_case(spec["probe_case"], audit_source["selected_source_case"])
        readiness.DEPLOYMENT_STATUS_PASS = DEPLOYMENT_STATUS_PASS
        readiness.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
        checkpoints = readiness.validate_deployment_source(spec)
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
        ).strip()
        if image_id != spec["derived_ros_image_id"]:
            raise RuntimeError("combined-correction probe image identity drift")
        planner_path = "/home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node"
        if image_file_hash(planner_path) != spec["planner_binary_sha256"]:
            raise RuntimeError("combined-correction probe planner binary identity drift")
        write_json(run_dir / "config/probe_case.json", case)
        write_json(run_dir / "config/mechanism_audit_source.json", audit_source)
        write_json(run_dir / "config/input_integrity.json", {
            "image_id": image_id, "checkpoint": checkpoints[case["checkpoint_seed"]],
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        (run_dir / "artifacts/cases").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/cases").mkdir(parents=True, exist_ok=False)
        if shutil.disk_usage(run_dir).free < MINIMUM_FREE_BYTES:
            raise RuntimeError("free disk below 20 GiB before combined-correction probe")
        checkpoint = checkpoints[case["checkpoint_seed"]]
        command, name = case_command(run_dir, case, checkpoint, str(spec["readiness_topic_contract"]))
        readiness.run_case(command, name, run_dir / f"logs/cases/{case['case_id']}.log")
        case_dir = run_dir / f"artifacts/cases/{case['case_id']}"
        qualification = readiness.audit_case(case_dir, case)
        mechanism = audit_combined_correction(case_dir)
        pending = load_json(case_dir / "summary.json")
        storage = finalize_handed_off_bag(
            case_dir, expected_uid=HOST_UID, expected_gid=HOST_GID,
            expected_raw_sha256=pending["storage"]["original_bag_sha256"],
        )
        pending["storage"] = storage
        pending["status"] = "PASS_SINGLE_ROBOT_CASE_V2"
        pending["readiness_qualification"] = qualification
        pending["combined_correction_mechanism_audit"] = mechanism
        write_json(case_dir / "summary.json", pending)
        summary = {
            "schema_version": "aee_composite_v9_combined_correction_probe_summary_v1",
            "overall_status": STATUS_PASS, "completed_cases": 1, "planned_cases": 1,
            "source_selected_case": audit_source["selected_source_case"],
            "case_result": qualification, "mechanism_audit": mechanism,
            "training_steps": 0, "optimizer_steps": 0,
            "c09_reads": 0, "c10_reads": 0,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID,
            "state": "COMPLETED", "overall_status": STATUS_PASS,
        })
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure(run_dir, started, str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
