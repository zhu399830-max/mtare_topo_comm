#!/home/zeng-workstation/anaconda3/bin/python
"""Execute the adapted-checkpoint 90-case stochastic study with host archives."""

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
from mtare_topo.evaluation.stochastic_closed_loop import analyze_stochastic_cases
from mtare_topo.governance import load_json, write_json
import run_mtare_single_robot_stochastic_v1 as base


RUN_ID = "gate6_20260820_mtare_single_robot_stochastic_v2_seed20260820"
IMAGE = base.IMAGE
STATUS_PASS = "PASS_MTARE_SINGLE_ROBOT_STOCHASTIC_V2"
STATUS_FAIL = "FAIL_MTARE_SINGLE_ROBOT_STOCHASTIC_V2"
HOST_UID = 1000
HOST_GID = 1000
MINIMUM_FREE_BYTES = 150 * 1024**3
EXPECTED_READINESS_STATUS = "PASS_AEE_ADAPTED_READINESS_V1"
EXPECTED_CHECKPOINT_MODE = "M1D_AEE_HEAD_ADAPTED_V1"
TOPIC_CONTRACT = "configs/v3/gate5/closed_loop_recording_topics_v1.json"
CONTAINER_PREFIX = "mtare-single-v2"


def verify_readiness_source(spec: dict[str, Any]) -> dict[str, Any]:
    source = (PROJECT_ROOT / spec["readiness_run"]).resolve()
    source.relative_to(PROJECT_ROOT)
    state = load_json(source / "RUN_STATE.json")
    summary = load_json(source / "metrics/summary.json")
    if (
        state.get("state") != "COMPLETED"
        or state.get("overall_status") != EXPECTED_READINESS_STATUS
        or summary.get("overall_status") != EXPECTED_READINESS_STATUS
        or summary.get("completed_cases") != 6
        or summary.get("all_case_gates_passed") is not True
        or float(summary.get("maximum_post_warmup_fallback_rate", 1.0)) > 0.05
    ):
        raise RuntimeError("adapted readiness source is not an exact six-case PASS")
    seal_path = source / "artifacts/evidence_sha256.txt"
    if base.sha256(seal_path) != spec["readiness_seal_sha256"]:
        raise RuntimeError("adapted readiness seal identity drift")
    prefix = source.relative_to(PROJECT_ROOT).as_posix() + "/"
    entries = 0
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if not relative.startswith(prefix) or base.sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"adapted readiness seal mismatch: {relative}")
        entries += 1
    return {
        "run": str(source.relative_to(PROJECT_ROOT)),
        "seal_sha256": spec["readiness_seal_sha256"],
        "seal_entries": entries,
        "maximum_post_warmup_fallback_rate": summary["maximum_post_warmup_fallback_rate"],
        "minimum_traveling_distance_m": summary["minimum_traveling_distance_m"],
    }


def case_command(run_dir: Path, case: Any, matrix: dict[str, Any]) -> tuple[list[str], str]:
    world = next(item for item in matrix["worlds"] if item["name"] == case.world)
    checkpoint = None
    if case.checkpoint_seed is not None:
        checkpoint = next(item for item in matrix["m1d_checkpoints"] if item["seed"] == case.checkpoint_seed)
    name = f"{CONTAINER_PREFIX}-{case.index:03d}"
    inner = [
        "python3", "/workspace/tools/v3/run_mtare_single_robot_case_v1.py",
        "--case-dir", f"/evidence/cases/{case.case_id}",
        "--case-id", case.case_id,
        "--world", case.world,
        "--environment-seed", str(case.environment_seed),
        "--method-id", case.method_id,
        "--method-family", case.method_family,
        "--execution-repeat", str(case.execution_repeat),
        "--block-id", case.block_id,
        "--runtime-sec", str(case.runtime_sec),
        "--topic-contract", f"/workspace/{TOPIC_CONTRACT}",
        "--complete-map", world["complete_map_path_in_image"],
        "--complete-map-sha256", world["complete_map_sha256"],
        "--archive-mode", "host",
        "--host-uid", str(HOST_UID),
        "--host-gid", str(HOST_GID),
    ]
    if checkpoint is not None:
        inner.extend(["--checkpoint", checkpoint["path"], "--checkpoint-sha256", checkpoint["sha256"], "--checkpoint-seed", str(checkpoint["seed"])])
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


def method_identity_evidence(case_dir: Path, summary: dict[str, Any]) -> dict[str, Any]:
    family = summary["case"]["method_family"]
    if family != "m1d_topology":
        return {"method_family": family, "semantic_fallback_applicable": False}
    snapshot = load_json(case_dir / summary["planner_evidence"]["topology_snapshot"])
    if snapshot.get("checkpoint", {}).get("mode") != EXPECTED_CHECKPOINT_MODE:
        raise RuntimeError("90-case M1D case did not use the frozen expected checkpoint mode")
    post_cycles = int(snapshot.get("post_warmup_cycles", -1))
    post_fallback = int(snapshot.get("post_warmup_fallback_cycles", -1))
    if post_cycles <= 0 or post_fallback < 0 or post_fallback > post_cycles:
        raise RuntimeError("90-case semantic fallback counters invalid")
    return {
        "method_family": family,
        "semantic_fallback_applicable": True,
        "learned_empty_cycles": int(snapshot["learned_empty_cycles"]),
        "fallback_cycles": int(snapshot["fallback_cycles"]),
        "post_warmup_cycles": post_cycles,
        "post_warmup_fallback_cycles": post_fallback,
        "post_warmup_fallback_rate": post_fallback / post_cycles,
        "checkpoint_mode": snapshot["checkpoint"]["mode"],
        "checkpoint_seed": snapshot["checkpoint"]["source_seed"],
    }


def finalize_case(case_dir: Path, expected_case_id: str) -> dict[str, Any]:
    summary_path = case_dir / "summary.json"
    summary = load_json(summary_path)
    if summary.get("status") != "PASS_SINGLE_ROBOT_CASE_V2_PENDING_HOST_ARCHIVE" or summary.get("case", {}).get("case_id") != expected_case_id:
        raise RuntimeError(f"90-case result did not reach host archive boundary: {expected_case_id}")
    identity = method_identity_evidence(case_dir, summary)
    storage = finalize_handed_off_bag(
        case_dir,
        expected_uid=HOST_UID,
        expected_gid=HOST_GID,
        expected_raw_sha256=summary["storage"]["original_bag_sha256"],
    )
    summary["storage"] = storage
    summary["method_identity"] = identity
    summary["status"] = "PASS_SINGLE_ROBOT_CASE_V2"
    write_json(summary_path, summary)
    return summary


def failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(run_dir / "metrics/summary.json", {"schema_version": "mtare_single_robot_stochastic_run_summary_v2", "overall_status": STATUS_FAIL, "failure_reason": message, "completed_case_count": completed, "planned_case_count": 90, "training_steps": 0, "optimizer_steps": 0, "c09_reads": 0, "c10_reads": 0, "duration_seconds": time.monotonic() - started})
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": STATUS_FAIL})
    base.seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec, run_dir = load_json(args.spec.resolve()), args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("stochastic V2 run identity/state mismatch")
    started = time.monotonic()
    summaries = []
    try:
        matrix, cases, audit = base.validate_frozen_inputs(spec)
        readiness = verify_readiness_source(spec)
        write_json(run_dir / "config/readiness_source.json", readiness)
        write_json(run_dir / "config/matrix_audit.json", audit)
        write_json(run_dir / "config/case_schedule.json", {"schema_version": "mtare_case_schedule_v2", "cases": [case.to_dict() for case in cases]})
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        (run_dir / "artifacts/cases").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/cases").mkdir(parents=True, exist_ok=False)
        for case in cases:
            if shutil.disk_usage(run_dir).free < MINIMUM_FREE_BYTES:
                raise RuntimeError(f"free disk below 150 GiB before case {case.case_id}")
            command, name = case_command(run_dir, case, matrix)
            base.run_case(command, name, run_dir / f"logs/cases/{case.case_id}.log")
            summary = finalize_case(run_dir / f"artifacts/cases/{case.case_id}", case.case_id)
            summaries.append(summary)
            write_json(run_dir / "metrics/progress.json", {"schema_version": "mtare_single_robot_stochastic_progress_v2", "completed_case_count": len(summaries), "planned_case_count": 90, "last_case_id": case.case_id, "elapsed_wall_sec": time.monotonic() - started, "free_disk_bytes": shutil.disk_usage(run_dir).free})
        analysis = analyze_stochastic_cases(summaries)
        write_json(run_dir / "metrics/stochastic_analysis.json", analysis)
        fallback_rates = [item["method_identity"]["post_warmup_fallback_rate"] for item in summaries if item["case"]["method_family"] == "m1d_topology"]
        summary = {
            "schema_version": "mtare_single_robot_stochastic_run_summary_v2",
            "overall_status": STATUS_PASS,
            "completed_case_count": 90,
            "block_count": 10,
            "family_case_counts": audit["family_case_counts"],
            "simulated_runtime_sec": 54000,
            "archive_bytes": sum(int(item["storage"]["archive_bytes"]) for item in summaries),
            "primary_metric": analysis["primary_metric"],
            "primary_m1d_comparison": analysis["m1d_comparisons"][analysis["primary_metric"]],
            "m1d_post_warmup_fallback_rate_mean": sum(fallback_rates) / len(fallback_rates),
            "m1d_post_warmup_fallback_rate_maximum": max(fallback_rates),
            "m1d_case_fallback_rates": fallback_rates,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0,
            "optimizer_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": STATUS_PASS})
        sealed = base.seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure(run_dir, started, len(summaries), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
