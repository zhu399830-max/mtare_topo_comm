#!/home/zeng-workstation/anaconda3/bin/python
"""Run one matched V4 live probe for the verified-backtrack re-anchor fix."""

from __future__ import annotations

import argparse
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
from mtare_topo.evaluation.topology_reanchor_evidence import audit_reanchor_evidence
from mtare_topo.governance import load_json, write_json
import run_aee_adapted_readiness_v1 as readiness
from run_mtare_single_robot_stochastic_v1 import image_file_hash, seal, sha256


RUN_ID = "gate6_20260823_aee_composite_v9_reanchor_probe_v1_seed20260822"
STATUS_PASS = "PASS_AEE_COMPOSITE_V9_REANCHOR_PROBE_V1"
STATUS_FAIL = "FAIL_AEE_COMPOSITE_V9_REANCHOR_PROBE_V1"
DEPLOYMENT_STATUS_PASS = "PASS_AEE_COMPOSITE_V9_ROS_DEPLOYMENT_EXPORT"
IMAGE = readiness.IMAGE
HOST_UID = readiness.HOST_UID
HOST_GID = readiness.HOST_GID
MINIMUM_FREE_BYTES = 20 * 1024**3


def validate_compatibility_audit_source(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    root = project_root.resolve()
    source = (root / spec["compatibility_audit_run"]).resolve()
    source.relative_to(root)
    state_path = source / "RUN_STATE.json"
    summary_path = source / "metrics/summary.json"
    provenance_path = source / "config/source_provenance.json"
    analysis_path = source / "metrics/stochastic_analysis.json"
    seal_path = source / "artifacts/evidence_sha256.txt"
    state, summary, provenance, analysis = (
        load_json(state_path), load_json(summary_path),
        load_json(provenance_path), load_json(analysis_path),
    )
    if state.get("state") != "COMPLETED" or state.get("overall_status") != spec["compatibility_audit_status"]:
        raise RuntimeError("re-anchor compatibility audit is not the exact completed PASS")
    if (
        summary.get("overall_status") != spec["compatibility_audit_status"]
        or summary.get("source_case_count") != 90
        or summary.get("source_mutation_count") != 0
        or summary.get("raw_bag_reads") != 0
    ):
        raise RuntimeError("re-anchor compatibility audit summary drift")
    if (
        provenance.get("source_run") != spec["mechanism_source_run"]
        or provenance.get("source_seal_sha256") != spec["mechanism_source_seal_sha256"]
        or provenance.get("source_mutation_permitted") is not False
    ):
        raise RuntimeError("compatibility audit does not bind the mechanism source")
    bridge = analysis.get("compatibility_bridge", {})
    if (
        bridge.get("source_case_count") != 90
        or bridge.get("mapped_field") != "status"
        or bridge.get("mapped_field_count") != 90
        or bridge.get("source_mutation_permitted") is not False
    ):
        raise RuntimeError("compatibility audit bridge contract drift")
    if sha256(seal_path) != spec["compatibility_audit_seal_sha256"]:
        raise RuntimeError("compatibility audit seal identity drift")
    entries = {
        path: digest
        for digest, path in (
            line.split("  ", 1) for line in seal_path.read_text(encoding="utf-8").splitlines()
        )
    }
    bound = {}
    for path in (state_path, summary_path, provenance_path, analysis_path):
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        if entries.get(relative) != digest:
            raise RuntimeError(f"compatibility audit bound evidence drift: {relative}")
        bound[relative] = digest
    return {
        "schema_version": "aee_composite_v9_compatibility_audit_source_v1",
        "run": spec["compatibility_audit_run"],
        "status": spec["compatibility_audit_status"],
        "seal_sha256": spec["compatibility_audit_seal_sha256"],
        "source_run": spec["mechanism_source_run"],
        "source_seal_sha256": spec["mechanism_source_seal_sha256"],
        "source_case_count": 90,
        "source_mutation_count": 0,
        "verified_bound_files": bound,
    }


def validate_mechanism_source(
    spec: dict[str, Any], *, project_root: Path = PROJECT_ROOT
) -> dict[str, Any]:
    root = project_root.resolve()
    source = (root / spec["mechanism_source_run"]).resolve()
    source.relative_to(root)
    state_path = source / "RUN_STATE.json"
    run_summary_path = source / "metrics/summary.json"
    case_id = spec["mechanism_source_case_id"]
    case_dir = source / f"artifacts/cases/{case_id}"
    case_summary_path = case_dir / "summary.json"
    seal_path = source / "artifacts/evidence_sha256.txt"
    state, run_summary, case_summary = (
        load_json(state_path), load_json(run_summary_path), load_json(case_summary_path)
    )
    if state.get("state") != "FAILED" or state.get("overall_status") != spec["mechanism_source_status"]:
        raise RuntimeError("re-anchor mechanism source is not the expected sealed FAIL")
    if run_summary.get("completed_case_count") != 90:
        raise RuntimeError("re-anchor mechanism source did not complete all 90 simulations")
    case = case_summary.get("case", {})
    if (
        case_summary.get("status") != "PASS_SINGLE_ROBOT_CASE_V2"
        or case.get("case_id") != case_id
        or case.get("world") != "tunnel"
        or case.get("environment_seed") != 23
        or case.get("checkpoint_seed") != 2
        or case.get("method_family") != "m1d_topology"
    ):
        raise RuntimeError("re-anchor mechanism source case identity drift")
    trace_path = case_dir / case_summary["planner_evidence"]["decision_trace"]
    snapshot_path = case_dir / case_summary["planner_evidence"]["topology_snapshot"]
    if sha256(seal_path) != spec["mechanism_source_seal_sha256"]:
        raise RuntimeError("re-anchor mechanism source seal identity drift")
    entries = {
        path: digest
        for digest, path in (
            line.split("  ", 1) for line in seal_path.read_text(encoding="utf-8").splitlines()
        )
    }
    bound = {}
    for path in (state_path, run_summary_path, case_summary_path, trace_path, snapshot_path):
        relative = path.relative_to(root).as_posix()
        digest = sha256(path)
        if entries.get(relative) != digest:
            raise RuntimeError(f"re-anchor mechanism source evidence drift: {relative}")
        bound[relative] = digest
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    snapshot = load_json(snapshot_path)
    audit = audit_reanchor_evidence(rows, snapshot["runtime"]["graph"])
    if audit != spec["expected_source_mechanism"]:
        raise RuntimeError("re-anchor mechanism source audit drift")
    return {
        "schema_version": "aee_composite_v9_reanchor_mechanism_source_v1",
        "run": spec["mechanism_source_run"],
        "status": spec["mechanism_source_status"],
        "seal_sha256": spec["mechanism_source_seal_sha256"],
        "case_id": case_id,
        "case_identity": {
            "world": "tunnel", "environment_seed": 23, "checkpoint_seed": 2,
        },
        "mechanism_audit": audit,
        "verified_bound_files": bound,
        "uses_evaluator_gt": False,
    }


def validate_probe_case(value: Any) -> dict[str, Any]:
    expected = {
        "case_id": "tunnel_v9_seed2_env23_reanchor_probe",
        "world": "tunnel",
        "environment_seed": 23,
        "checkpoint_seed": 2,
        "runtime_sec": 180.0,
    }
    if value != expected:
        raise RuntimeError("re-anchor probe case identity drift")
    return value


def case_command(
    run_dir: Path,
    case: dict[str, Any],
    checkpoint: dict[str, Any],
    topic_contract: str,
) -> tuple[list[str], str]:
    name = "aee-v4-reanchor-probe"
    inner = [
        "python3", "/workspace/tools/v3/run_mtare_single_robot_case_v2.py",
        "--case-dir", f"/evidence/cases/{case['case_id']}",
        "--case-id", case["case_id"],
        "--world", case["world"],
        "--environment-seed", str(case["environment_seed"]),
        "--method-id", f"m1d_aee_v4_seed{case['checkpoint_seed']}",
        "--method-family", "m1d_topology",
        "--execution-repeat", "0",
        "--checkpoint-seed", str(case["checkpoint_seed"]),
        "--block-id", "reanchor_probe_tunnel_env23",
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


def audit_reanchor_mechanism(
    case_dir: Path, *, require_reanchor: bool = True
) -> dict[str, Any]:
    summary = load_json(case_dir / "summary.json")
    snapshot = load_json(case_dir / summary["planner_evidence"]["topology_snapshot"])
    if snapshot.get("schema_version") != "semantic_topology_global_node_v4_snapshot_v1":
        raise RuntimeError("probe did not use the V4 ROS node")
    runtime = snapshot.get("runtime", {})
    if runtime.get("schema_version") != "online_topology_planner_runtime_v2":
        raise RuntimeError("probe did not use the V2 online runtime")
    snapshot_count = int(runtime.get("verified_reanchor_count", -1))
    trace_path = case_dir / summary["planner_evidence"]["decision_trace"]
    rows = [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
    transition_rows = [
        row for row in rows
        if row.get("graph_update", {}).get("reason") == "verified_backtrack_reanchor"
    ]
    if snapshot_count < int(require_reanchor) or len(transition_rows) != snapshot_count:
        raise RuntimeError("probe did not record a consistent verified re-anchor transition")
    first_index = None
    first_arc = None
    first_growth_frame = None
    later_growth: list[dict[str, Any]] = []
    if transition_rows:
        first = transition_rows[0]
        first_index = int(first["frame_index"])
        first_arc = float(first["route_arc_m"])
        later_growth = [
            row for row in rows
            if int(row["frame_index"]) > first_index and float(row["route_arc_m"]) > first_arc
        ]
        if not later_growth:
            raise RuntimeError("route arc did not grow after verified re-anchor")
        first_growth_frame = int(later_growth[0]["frame_index"])
    graph = runtime.get("graph", {})
    evidence = audit_reanchor_evidence(rows, graph)
    return {
        "schema_version": "aee_composite_v9_reanchor_probe_audit_v1",
        "verified_reanchor_count": snapshot_count,
        "first_reanchor_frame": first_index,
        "first_reanchor_route_arc_m": first_arc,
        "first_post_reanchor_growth_frame": first_growth_frame,
        "final_route_arc_m": float(rows[-1]["route_arc_m"]),
        "current_node_after_run": graph.get("current_node"),
        "trace_transition_count_matches_snapshot": True,
        "post_reanchor_route_growth": bool(later_growth),
        "uses_evaluator_gt": False,
        "stale_anchor_diagnostic": evidence,
    }


def failure(run_dir: Path, started: float, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {
        "schema_version": "aee_composite_v9_reanchor_probe_summary_v1",
        "overall_status": STATUS_FAIL,
        "failure_reason": message,
        "completed_cases": 0,
        "planned_cases": 1,
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
    seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("re-anchor probe run identity/state mismatch")
    started = time.monotonic()
    try:
        if spec.get("gate") != 6 or spec.get("operation") != "closed_loop_single":
            raise RuntimeError("Gate-6 closed-loop scope required")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("approved autonomous execution scope required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen probe tool drift: {name}")
        compatibility_source = validate_compatibility_audit_source(spec)
        mechanism_source = validate_mechanism_source(spec)
        case = validate_probe_case(spec["probe_case"])
        topic_contract = str(spec["readiness_topic_contract"])
        readiness.DEPLOYMENT_STATUS_PASS = DEPLOYMENT_STATUS_PASS
        readiness.EXPECTED_CHECKPOINT_MODE = COMPOSITE_V9_MODE
        checkpoints = readiness.validate_deployment_source(spec)
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
        ).strip()
        if image_id != spec["derived_ros_image_id"]:
            raise RuntimeError("probe image identity drift")
        planner_path = "/home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node"
        if image_file_hash(planner_path) != spec["planner_binary_sha256"]:
            raise RuntimeError("probe planner binary identity drift")
        write_json(run_dir / "config/probe_case.json", case)
        write_json(run_dir / "config/compatibility_audit_source.json", compatibility_source)
        write_json(run_dir / "config/mechanism_source.json", mechanism_source)
        write_json(run_dir / "config/input_integrity.json", {
            "image_id": image_id, "checkpoint": checkpoints[case["checkpoint_seed"]],
        })
        write_json(run_dir / "RUN_STATE.json", {
            "schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING",
        })
        (run_dir / "artifacts/cases").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/cases").mkdir(parents=True, exist_ok=False)
        if shutil.disk_usage(run_dir).free < MINIMUM_FREE_BYTES:
            raise RuntimeError("free disk below 20 GiB before probe")
        checkpoint = checkpoints[case["checkpoint_seed"]]
        command, name = case_command(run_dir, case, checkpoint, topic_contract)
        readiness.run_case(command, name, run_dir / f"logs/cases/{case['case_id']}.log")
        case_dir = run_dir / f"artifacts/cases/{case['case_id']}"
        readiness_qualification = readiness.audit_case(case_dir, case)
        mechanism = audit_reanchor_mechanism(case_dir)
        pending = load_json(case_dir / "summary.json")
        storage = finalize_handed_off_bag(
            case_dir, expected_uid=HOST_UID, expected_gid=HOST_GID,
            expected_raw_sha256=pending["storage"]["original_bag_sha256"],
        )
        pending["storage"] = storage
        pending["status"] = "PASS_SINGLE_ROBOT_CASE_V2"
        pending["readiness_qualification"] = readiness_qualification
        pending["reanchor_mechanism_audit"] = mechanism
        write_json(case_dir / "summary.json", pending)
        summary = {
            "schema_version": "aee_composite_v9_reanchor_probe_summary_v1",
            "overall_status": STATUS_PASS,
            "completed_cases": 1,
            "planned_cases": 1,
            "case_result": readiness_qualification,
            "mechanism_audit": mechanism,
            "training_steps": 0,
            "optimizer_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
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
