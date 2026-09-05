#!/usr/bin/env python3
"""Execute and seal the one-shot M-TARE planner-seed system qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate5_20260820_mtare_planner_seed_qualification_v1_seed11"
IMAGE = "mtare-semantic-runtime:planner-seed-v1-runnable"


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


def docker_trial(run_dir: Path, trial_id: str, seed: int, log_path: Path) -> None:
    command = [
        "docker", "run", "--rm", "--name", f"mtare-seed-qualification-{trial_id}",
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint", "/bin/bash", IMAGE, "-lc",
        "source /opt/ros/noetic/setup.bash && "
        "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && "
        "export PYTHONPATH=/workspace/src:$PYTHONPATH && "
        "python3 /workspace/tools/v3/mtare_planner_seed_trial_v1.py "
        f"--trial-dir /evidence/trials/{trial_id} --seed {seed} "
        "--duration-sec 60 --audit-frames 100 --audit-waypoints 20",
    ]
    with log_path.open("xb") as stream:
        completed = subprocess.run(command, cwd=PROJECT_ROOT, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"trial {trial_id} failed with exit code {completed.returncode}")


def finish_failure(run_dir: Path, message: str, started: float) -> None:
    payload: dict[str, Any] = {
        "schema_version": "mtare_planner_seed_qualification_summary_v1",
        "overall_status": "FAIL_MTARE_PLANNER_SEED_QUALIFICATION_V1",
        "failure_reason": message,
        "duration_seconds": time.monotonic() - started,
        "training_steps": 0,
        "optimizer_steps": 0,
    }
    write_json(run_dir / "metrics/summary.json", payload)
    write_json(
        run_dir / "RUN_STATE.json",
        {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": payload["overall_status"]},
    )
    seal(run_dir)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", required=True, type=Path)
    parser.add_argument("--run-dir", required=True, type=Path)
    args = parser.parse_args()
    spec = load_json(args.spec.resolve())
    run_dir = args.run_dir.resolve()
    if run_dir.name != RUN_ID or load_json(run_dir / "RUN_STATE.json").get("state") != "CREATED_NOT_EXECUTED":
        raise RuntimeError("run identity/state mismatch")
    if spec.get("gate") != 5 or spec.get("operation") != "infrastructure":
        raise RuntimeError("Gate-5 infrastructure scope required")
    if spec.get("user_authorization", {}).get("status") != "APPROVED":
        raise RuntimeError("explicit user approval is required")
    for name, item in spec["frozen_tools"].items():
        if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
            raise RuntimeError(f"frozen tool drift: {name}")
    image_id = subprocess.check_output(["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True).strip()
    if image_id != spec["derived_ros_image_id"]:
        raise RuntimeError(f"derived ROS image drift: {image_id}")
    write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
    started = time.monotonic()
    (run_dir / "artifacts/trials").mkdir(parents=True, exist_ok=False)
    try:
        trials = (("seed11_a", 11), ("seed11_b", 11), ("seed23", 23))
        for trial_id, seed in trials:
            docker_trial(run_dir, trial_id, seed, run_dir / f"logs/{trial_id}.log")
        comparison_path = run_dir / "metrics/comparison.json"
        comparison_command = [
            "python3", "tools/v3/compare_mtare_planner_seed_trials_v1.py",
            "--seed11-a", str(run_dir / "artifacts/trials/seed11_a/summary.json"),
            "--seed11-b", str(run_dir / "artifacts/trials/seed11_b/summary.json"),
            "--seed23", str(run_dir / "artifacts/trials/seed23/summary.json"),
            "--output", str(comparison_path),
        ]
        completed = subprocess.run(comparison_command, cwd=PROJECT_ROOT, text=True, capture_output=True, check=False)
        (run_dir / "logs/compare.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError("same-seed/different-seed comparison failed")
        comparison = load_json(comparison_path)
        summary = {
            "schema_version": "mtare_planner_seed_qualification_summary_v1",
            "overall_status": "PASS_MTARE_PLANNER_SEED_QUALIFICATION_V1",
            "world": "tunnel",
            "trials": 3,
            "sim_duration_seconds": 180,
            "audited_synchronized_frames": 300,
            "audited_waypoints": 60,
            "same_seed_full_frame_and_waypoint_identity": comparison["same_seed_full_frame_and_waypoint_identity"],
            "different_seed_changes_observed_sensor_trajectory_sequence": comparison["different_seed_changes_observed_sensor_trajectory_sequence"],
            "derived_ros_image_id": image_id,
            "training_steps": 0,
            "optimizer_steps": 0,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": summary["overall_status"]})
        sealed = seal(run_dir)
        print(json.dumps({**summary, "sealed_files": sealed}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        finish_failure(run_dir, str(exc), started)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
