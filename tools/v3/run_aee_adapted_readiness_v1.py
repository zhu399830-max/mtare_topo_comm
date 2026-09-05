#!/home/zeng-workstation/anaconda3/bin/python
"""Run and seal six AEE-adapted closed-loop readiness cases."""

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
from mtare_topo.evaluation.host_bag_archive import finalize_handed_off_bag
from mtare_topo.governance import load_json, write_json
from run_mtare_single_robot_stochastic_v1 import image_file_hash, seal, sha256


RUN_ID = "gate6_20260820_aee_adapted_readiness_v1_seed20260820"
IMAGE = "mtare-semantic-runtime:planner-seed-v1-runnable"
STATUS_PASS = "PASS_AEE_ADAPTED_READINESS_V1"
STATUS_FAIL = "FAIL_AEE_ADAPTED_READINESS_V1"
DEPLOYMENT_STATUS_PASS = "PASS_AEE_ADAPTED_ROS_DEPLOYMENT_EXPORT_V1"
EXPECTED_CHECKPOINT_MODE = "M1D_AEE_HEAD_ADAPTED_V1"
DEPLOYMENT_CHECKPOINT_PATTERN = "m1d_aee_seed{seed}_ros.pt"
HOST_UID = 1000
HOST_GID = 1000
MINIMUM_FREE_BYTES = 20 * 1024**3


def validate_schedule(cases: Any) -> list[dict[str, Any]]:
    if not isinstance(cases, list) or len(cases) != 6:
        raise RuntimeError("readiness schedule must contain exactly six cases")
    observed = []
    identifiers = set()
    for index, case in enumerate(cases):
        if not isinstance(case, dict):
            raise RuntimeError("readiness case must be an object")
        required = {"case_id", "world", "environment_seed", "checkpoint_seed", "runtime_sec"}
        if set(case) != required:
            raise RuntimeError(f"readiness case field drift at index {index}")
        if case["case_id"] in identifiers:
            raise RuntimeError("duplicate readiness case ID")
        identifiers.add(case["case_id"])
        if case["world"] not in ("tunnel", "garage") or case["checkpoint_seed"] not in (0, 1, 2):
            raise RuntimeError("readiness world/checkpoint identity drift")
        if float(case["runtime_sec"]) != 180.0 or int(case["environment_seed"]) < 0:
            raise RuntimeError("readiness runtime/environment seed drift")
        observed.append((case["world"], int(case["checkpoint_seed"])))
    expected = [(world, seed) for world in ("tunnel", "garage") for seed in (0, 1, 2)]
    if sorted(observed) != sorted(expected):
        raise RuntimeError("readiness schedule must be 2 worlds x 3 checkpoints")
    return cases


def case_command(
    run_dir: Path,
    case: dict[str, Any],
    checkpoint: dict[str, Any],
    topic_contract: str = "configs/v3/gate5/closed_loop_recording_topics_v1.json",
) -> tuple[list[str], str]:
    name = f"aee-readiness-{case['case_id']}"
    inner = [
        "python3", "/workspace/tools/v3/run_mtare_single_robot_case_v1.py",
        "--case-dir", f"/evidence/cases/{case['case_id']}",
        "--case-id", case["case_id"],
        "--world", case["world"],
        "--environment-seed", str(case["environment_seed"]),
        "--method-id", f"m1d_aee_seed{case['checkpoint_seed']}",
        "--method-family", "m1d_topology",
        "--execution-repeat", "0",
        "--checkpoint-seed", str(case["checkpoint_seed"]),
        "--block-id", f"readiness_{case['world']}",
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
        "docker", "run", "--rm", "--network", "none", "--hostname", "localhost", "--name", name,
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint", "/bin/bash", IMAGE, "-lc", shell,
    ], name


def run_case(command: list[str], name: str, log_path: Path) -> None:
    try:
        with log_path.open("xb") as stream:
            completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False, timeout=1200.0)
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "stop", "--timeout", "30", name], check=False, capture_output=True)
        raise RuntimeError(f"readiness case exceeded 1200 wall seconds: {name}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"readiness case failed with exit {completed.returncode}: {name}")


def verified_edge_count(snapshot: dict[str, Any]) -> int:
    edges = snapshot["runtime"]["graph"]["edges"]
    return sum(
        edge.get("kind") == "verified_traversed"
        and int(edge.get("verified_traversal_count", 0)) >= 1
        for edge in edges
    )


def audit_case(case_dir: Path, expected_case: dict[str, Any]) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    summary = load_json(summary_path)
    if summary.get("status") != "PASS_SINGLE_ROBOT_CASE_V2_PENDING_HOST_ARCHIVE":
        raise RuntimeError("readiness case did not reach host archive boundary")
    contract = summary.get("case", {})
    if contract.get("case_id") != expected_case["case_id"] or contract.get("runtime_sec") != 180.0:
        raise RuntimeError("readiness case identity/runtime drift")
    metrics = summary["metrics"]
    if not metrics.get("recording_audit", {}).get("passed"):
        raise RuntimeError("readiness required-topic audit failed")
    snapshot_path = case_dir / summary["planner_evidence"]["topology_snapshot"]
    trace_path = case_dir / summary["planner_evidence"]["decision_trace"]
    snapshot = load_json(snapshot_path)
    if snapshot.get("failed_cycles") != 0:
        raise RuntimeError("readiness planner cycle failed")
    if snapshot.get("checkpoint", {}).get("mode") != EXPECTED_CHECKPOINT_MODE:
        raise RuntimeError("readiness did not deploy adapted checkpoint mode")
    graph = snapshot["runtime"]["graph"]
    nodes = int(graph["node_count"])
    verified_edges = verified_edge_count(snapshot)
    non_hold = 0
    trace_frames = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        trace_frames += 1
        non_hold += int(record.get("target", {}).get("mode") != "hold")
        if "semantic_fallback" not in record:
            raise RuntimeError("readiness trace lacks semantic fallback audit")
        if EXPECTED_CHECKPOINT_MODE == "M1D_AEE_CORRECTIVE_COMPOSITE_V9":
            composite = record.get("composite_semantics")
            if not isinstance(composite, dict) or composite.get("neural_count_role_ignored") is not True:
                raise RuntimeError("V9 readiness trace lacks composite semantic audit")
            if int(composite.get("b0_branch_count", -1)) not in range(0, 9) or int(composite.get("b0_role_index", -1)) not in range(3):
                raise RuntimeError("V9 readiness composite semantic values are invalid")
    post_cycles = int(snapshot.get("post_warmup_cycles", -1))
    post_fallback = int(snapshot.get("post_warmup_fallback_cycles", -1))
    fallback_rate = post_fallback / max(post_cycles, 1)
    gates = {
        "required_topics": True,
        "zero_failed_planner_cycles": True,
        "travel_at_least_5m": float(metrics["traveling_distance_m"]) >= 5.0,
        "at_least_2_nodes": nodes >= 2,
        "at_least_1_verified_edge": verified_edges >= 1,
        "at_least_1_non_hold_waypoint": non_hold >= 1,
        "post20_fallback_at_most_5pct": post_cycles > 0 and fallback_rate <= 0.05,
    }
    if not all(gates.values()):
        raise RuntimeError(f"readiness scientific gate failed: {expected_case['case_id']}: {gates}")
    return {
        "case_id": expected_case["case_id"],
        "world": expected_case["world"],
        "environment_seed": expected_case["environment_seed"],
        "checkpoint_seed": expected_case["checkpoint_seed"],
        "traveling_distance_m": metrics["traveling_distance_m"],
        "graph_node_count": nodes,
        "verified_edge_count": verified_edges,
        "non_hold_trace_cycles": non_hold,
        "trace_frames": trace_frames,
        "post_warmup_cycles": post_cycles,
        "post_warmup_fallback_cycles": post_fallback,
        "post_warmup_fallback_rate": fallback_rate,
        "gates": gates,
    }


def validate_deployment_source(spec: dict[str, Any]) -> list[dict[str, Any]]:
    source = (PROJECT_ROOT / spec["adapted_deployment_run"]).resolve()
    source.relative_to(PROJECT_ROOT)
    state = load_json(source / "RUN_STATE.json")
    summary = load_json(source / "metrics/summary.json")
    if state.get("state") != "COMPLETED" or state.get("overall_status") != DEPLOYMENT_STATUS_PASS or summary.get("checkpoint_count") != 3:
        raise RuntimeError("adapted ROS deployment source is not exact PASS")
    seal_path = source / "artifacts/evidence_sha256.txt"
    if sha256(seal_path) != spec["adapted_deployment_seal_sha256"]:
        raise RuntimeError("adapted ROS deployment seal identity drift")
    prefix = source.relative_to(PROJECT_ROOT).as_posix() + "/"
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if not relative.startswith(prefix) or sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"adapted ROS deployment seal mismatch: {relative}")
    checkpoints = []
    for seed in (0, 1, 2):
        path = source / f"artifacts/checkpoints/{DEPLOYMENT_CHECKPOINT_PATTERN.format(seed=seed)}"
        checkpoints.append({"seed": seed, "path": str(path.relative_to(PROJECT_ROOT)), "sha256": sha256(path)})
    return checkpoints


def failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {"schema_version": "aee_adapted_readiness_summary_v1", "overall_status": STATUS_FAIL, "failure_reason": message, "completed_cases": completed, "planned_cases": 6, "training_steps": 0, "optimizer_steps": 0, "c09_reads": 0, "c10_reads": 0, "duration_seconds": time.monotonic() - started})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": STATUS_FAIL})
    seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("readiness run identity/state mismatch")
    started = time.monotonic()
    results = []
    try:
        if spec.get("gate") != 6 or spec.get("operation") != "closed_loop_single" or spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("approved Gate-6 readiness scope required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen readiness tool drift: {name}")
        cases = validate_schedule(spec["readiness_cases"])
        topic_contract = str(spec.get("readiness_topic_contract", "configs/v3/gate5/closed_loop_recording_topics_v1.json"))
        contract_path = (PROJECT_ROOT / topic_contract).resolve()
        contract_path.relative_to(PROJECT_ROOT)
        if not contract_path.is_file():
            raise RuntimeError("readiness topic contract is missing")
        checkpoints = validate_deployment_source(spec)
        image_id = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
        if image_id != spec["derived_ros_image_id"]:
            raise RuntimeError("readiness image identity drift")
        planner_path = "/home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node"
        if image_file_hash(planner_path) != spec["planner_binary_sha256"]:
            raise RuntimeError("readiness planner binary identity drift")
        write_json(run_dir / "config/readiness_schedule.json", {"schema_version": "aee_adapted_readiness_schedule_v1", "cases": cases})
        write_json(run_dir / "config/input_integrity.json", {"image_id": image_id, "checkpoints": checkpoints})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        (run_dir / "artifacts/cases").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/cases").mkdir(parents=True, exist_ok=False)
        for case in cases:
            if shutil.disk_usage(run_dir).free < MINIMUM_FREE_BYTES:
                raise RuntimeError("free disk below 20 GiB before readiness case")
            checkpoint = checkpoints[int(case["checkpoint_seed"])]
            command, name = case_command(run_dir, case, checkpoint, topic_contract)
            run_case(command, name, run_dir / f"logs/cases/{case['case_id']}.log")
            case_dir = run_dir / f"artifacts/cases/{case['case_id']}"
            qualification = audit_case(case_dir, case)
            pending = load_json(case_dir / "summary.json")
            storage = finalize_handed_off_bag(case_dir, expected_uid=HOST_UID, expected_gid=HOST_GID, expected_raw_sha256=pending["storage"]["original_bag_sha256"])
            pending["storage"] = storage
            pending["status"] = "PASS_SINGLE_ROBOT_CASE_V2"
            pending["readiness_qualification"] = qualification
            write_json(case_dir / "summary.json", pending)
            results.append(qualification)
            write_json(run_dir / "metrics/progress.json", {"schema_version": "aee_adapted_readiness_progress_v1", "completed_cases": len(results), "planned_cases": 6, "last_case_id": case["case_id"], "elapsed_seconds": time.monotonic() - started})
        summary = {"schema_version": "aee_adapted_readiness_summary_v1", "overall_status": STATUS_PASS, "completed_cases": 6, "planned_cases": 6, "case_results": results, "all_case_gates_passed": True, "maximum_post_warmup_fallback_rate": max(item["post_warmup_fallback_rate"] for item in results), "minimum_traveling_distance_m": min(item["traveling_distance_m"] for item in results), "training_steps": 0, "optimizer_steps": 0, "c09_reads": 0, "c10_reads": 0, "duration_seconds": time.monotonic() - started, "finished_at_utc": datetime.now(timezone.utc).isoformat()}
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": STATUS_PASS})
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure(run_dir, started, len(results), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
