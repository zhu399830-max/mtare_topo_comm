#!/usr/bin/env python3
"""Execute, analyze, and seal the approved 90-case stochastic single-robot study."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.evaluation.closed_loop_matrix import (
    enumerate_stochastic_cases,
    load_stochastic_matrix,
    stochastic_case_audit,
)
from mtare_topo.evaluation.stochastic_closed_loop import analyze_stochastic_cases
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate6_20260820_mtare_single_robot_stochastic_v1_seed20260820"
IMAGE = "mtare-semantic-runtime:planner-seed-v1-runnable"
STATUS_PASS = "PASS_MTARE_SINGLE_ROBOT_STOCHASTIC_V1"
STATUS_FAIL = "FAIL_MTARE_SINGLE_ROBOT_STOCHASTIC_V1"
MINIMUM_FREE_BYTES = 150 * 1024**3


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in paths),
        encoding="utf-8",
    )
    return len(paths)


def image_file_hash(path: str) -> str:
    command = ["docker", "run", "--rm", "--entrypoint", "/usr/bin/sha256sum", IMAGE, path]
    completed = subprocess.run(command, text=True, capture_output=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"cannot hash frozen image input: {path}: {completed.stderr}")
    return completed.stdout.split()[0]


def case_command(run_dir: Path, case: Any, matrix: dict[str, Any]) -> tuple[list[str], str]:
    world = next(item for item in matrix["worlds"] if item["name"] == case.world)
    checkpoint = None
    if case.checkpoint_seed is not None:
        checkpoint = next(item for item in matrix["m1d_checkpoints"] if item["seed"] == case.checkpoint_seed)
    container_name = f"mtare-single-v1-{case.index:03d}"
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
        "--complete-map", world["complete_map_path_in_image"],
        "--complete-map-sha256", world["complete_map_sha256"],
    ]
    if checkpoint is not None:
        inner.extend(
            ["--checkpoint", checkpoint["path"], "--checkpoint-sha256", checkpoint["sha256"],
             "--checkpoint-seed", str(checkpoint["seed"])]
        )
    shell_command = (
        "source /opt/ros/noetic/setup.bash && "
        "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && "
        "source /home/docker-user/mtare/tare_system/devel/setup.bash --extend && "
        "export PYTHONPATH=/workspace/src:$PYTHONPATH && "
        + shlex.join(inner)
    )
    return [
        "docker", "run", "--rm", "--name", container_name,
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint", "/bin/bash", IMAGE, "-lc", shell_command,
    ], container_name


def run_case(command: list[str], container_name: str, log_path: Path) -> None:
    try:
        with log_path.open("xb") as stream:
            completed = subprocess.run(
                command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT,
                check=False, timeout=2400.0,
            )
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "stop", "--time", "30", container_name], check=False, capture_output=True)
        raise RuntimeError(f"case container exceeded 2400 wall seconds: {container_name}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"case container failed with exit code {completed.returncode}: {container_name}")


def failure(run_dir: Path, started: float, completed_cases: int, message: str) -> None:
    summary = {
        "schema_version": "mtare_single_robot_stochastic_run_summary_v1",
        "overall_status": STATUS_FAIL,
        "failure_reason": message,
        "completed_case_count": completed_cases,
        "planned_case_count": 90,
        "duration_seconds": time.monotonic() - started,
        "training_steps": 0,
        "optimizer_steps": 0,
        "c09_reads": 0,
        "c10_reads": 0,
    }
    write_json(run_dir / "metrics/summary.json", summary)
    write_json(
        run_dir / "RUN_STATE.json",
        {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": STATUS_FAIL},
    )
    seal(run_dir)


def validate_frozen_inputs(spec: dict[str, Any]) -> tuple[dict[str, Any], tuple[Any, ...], dict[str, Any]]:
    if spec.get("gate") != 6 or spec.get("operation") != "closed_loop_single":
        raise RuntimeError("approved Gate-6 single-robot closed-loop scope required")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("explicit user approval required")
    matrix = load_stochastic_matrix(PROJECT_ROOT / spec["matrix"])
    cases = enumerate_stochastic_cases(matrix)
    audit = stochastic_case_audit(cases)
    if not audit["all_blocks_complete"] or audit["case_count"] != 90:
        raise RuntimeError("stochastic matrix audit failed")
    for name, item in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    for checkpoint in matrix["m1d_checkpoints"]:
        if sha256(PROJECT_ROOT / checkpoint["path"]) != checkpoint["sha256"]:
            raise RuntimeError(f"checkpoint drift: seed {checkpoint['seed']}")
    image_id = subprocess.check_output(
        ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
    ).strip()
    if image_id != spec["derived_ros_image_id"]:
        raise RuntimeError("derived image identity drift")
    planner_path = "/home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node"
    if image_file_hash(planner_path) != spec["planner_binary_sha256"]:
        raise RuntimeError("planner binary identity drift")
    for world in matrix["worlds"]:
        if image_file_hash(world["world_path_in_image"]) != world["world_sha256"]:
            raise RuntimeError(f"world hash drift: {world['name']}")
        if image_file_hash(world["complete_map_path_in_image"]) != world["complete_map_sha256"]:
            raise RuntimeError(f"complete-map hash drift: {world['name']}")
    return matrix, cases, audit


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run identity/state mismatch")
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    try:
        matrix, cases, audit = validate_frozen_inputs(spec)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        (run_dir / "artifacts/cases").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/cases").mkdir(parents=True, exist_ok=False)
        write_json(run_dir / "config/matrix_audit.json", audit)
        write_json(run_dir / "config/case_schedule.json", {"schema_version": "mtare_case_schedule_v1", "cases": [case.to_dict() for case in cases]})
    except Exception as exc:
        failure(run_dir, started, 0, str(exc))
        raise
    try:
        for case in cases:
            free_bytes = shutil.disk_usage(run_dir).free
            if free_bytes < MINIMUM_FREE_BYTES:
                raise RuntimeError(f"free disk below 150 GiB before case {case.case_id}: {free_bytes}")
            command, container_name = case_command(run_dir, case, matrix)
            run_case(command, container_name, run_dir / f"logs/cases/{case.case_id}.log")
            summary_path = run_dir / f"artifacts/cases/{case.case_id}/summary.json"
            if not summary_path.is_file():
                raise RuntimeError(f"case summary missing: {case.case_id}")
            summary = load_json(summary_path)
            if summary.get("status") != "PASS_SINGLE_ROBOT_CASE_V1" or summary.get("case", {}).get("case_id") != case.case_id:
                raise RuntimeError(f"case summary invalid: {case.case_id}")
            summaries.append(summary)
            write_json(
                run_dir / "metrics/progress.json",
                {
                    "schema_version": "mtare_single_robot_stochastic_progress_v1",
                    "completed_case_count": len(summaries),
                    "planned_case_count": 90,
                    "last_case_id": case.case_id,
                    "last_case_index": case.index,
                    "elapsed_wall_sec": time.monotonic() - started,
                    "free_disk_bytes": shutil.disk_usage(run_dir).free,
                },
            )
        analysis = analyze_stochastic_cases(summaries)
        write_json(run_dir / "metrics/stochastic_analysis.json", analysis)
        duration = time.monotonic() - started
        archive_bytes = sum(
            int(summary["storage"]["archive_bytes"]) for summary in summaries
        )
        summary = {
            "schema_version": "mtare_single_robot_stochastic_run_summary_v1",
            "overall_status": STATUS_PASS,
            "completed_case_count": 90,
            "block_count": 10,
            "family_case_counts": audit["family_case_counts"],
            "simulated_runtime_sec": 54000,
            "archive_bytes": archive_bytes,
            "primary_metric": analysis["primary_metric"],
            "primary_m1d_comparison": analysis["m1d_comparisons"][analysis["primary_metric"]],
            "duration_seconds": duration,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "training_steps": 0,
            "optimizer_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(
            run_dir / "RUN_STATE.json",
            {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": STATUS_PASS},
        )
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        failure(run_dir, started, len(summaries), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
