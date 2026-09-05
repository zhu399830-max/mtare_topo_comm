#!/usr/bin/env python3
"""Execute and seal the approved ten-trajectory AEE sensor export."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_domain_adaptation import (
    EFFECTIVE_FRAMES_PER_TRAJECTORY,
    RAW_FRAMES_PER_TRAJECTORY,
    enumerate_aee_domain_trajectories,
    sha256,
)
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260820_aee_domain_sensor_export_v1_seed20260820"
IMAGE = "mtare-semantic-runtime:planner-seed-v1-runnable"
STATUS_PASS = "PASS_AEE_DOMAIN_SENSOR_EXPORT_V1"
STATUS_FAIL = "FAIL_AEE_DOMAIN_SENSOR_EXPORT_V1"
MINIMUM_FREE_BYTES = 50 * 1024**3


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in paths),
        encoding="utf-8",
    )
    return len(paths)


def image_file_hash(path: str) -> str:
    completed = subprocess.run(
        ["docker", "run", "--rm", "--entrypoint", "/usr/bin/sha256sum", IMAGE, path],
        text=True,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"cannot hash frozen image input {path}: {completed.stderr}")
    return completed.stdout.split()[0]


def verify_zstd_archive(archive: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(block)
    returncode = process.wait()
    actual = digest.hexdigest()
    if returncode != 0 or actual != expected_sha256:
        raise RuntimeError("host zstd decompression hash verification failed")
    return actual


def host_archive(case_dir: Path, expected_sha256: str) -> dict[str, Any]:
    raw = case_dir / "raw.bag"
    if not raw.is_file() or sha256(raw) != expected_sha256:
        raise RuntimeError("raw bag identity missing or drifted before host archive")
    archive = case_dir / "raw.bag.zst"
    completed = subprocess.run(
        ["zstd", "-10", "-T0", "-q", "-f", str(raw), "-o", str(archive)],
        check=False,
    )
    if completed.returncode != 0 or not archive.is_file():
        raise RuntimeError("host zstd compression failed")
    verified = verify_zstd_archive(archive, expected_sha256)
    storage = {
        "schema_version": "aee_domain_host_archive_v1",
        "host_zstd_sha256": sha256(Path("/usr/bin/zstd")),
        "zstd_level": 10,
        "original_bag_sha256": expected_sha256,
        "original_bag_bytes": raw.stat().st_size,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "decompressed_sha256": verified,
    }
    write_json(case_dir / "storage.json", storage)
    raw.unlink()
    return storage


def case_command(run_dir: Path, trajectory: Any) -> tuple[list[str], str]:
    name = f"aee-domain-export-{trajectory.index:02d}"
    inner = [
        "python3",
        "/workspace/tools/v3/collect_aee_domain_trajectory_v1.py",
        "--output-dir",
        f"/evidence/trajectories/{trajectory.trajectory_id}",
        "--trajectory-id",
        trajectory.trajectory_id,
        "--world",
        trajectory.world,
        "--split",
        trajectory.split,
        "--environment-seed",
        str(trajectory.environment_seed),
    ]
    import shlex

    shell = (
        "source /opt/ros/noetic/setup.bash && "
        "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && "
        "source /home/docker-user/mtare/tare_system/devel/setup.bash --extend && "
        "export PYTHONPATH=/workspace/src:$PYTHONPATH && "
        + shlex.join(inner)
    )
    return [
        "docker",
        "run",
        "--rm",
        "--name",
        name,
        "-v",
        f"{PROJECT_ROOT}:/workspace:ro",
        "-v",
        f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint",
        "/bin/bash",
        IMAGE,
        "-lc",
        shell,
    ], name


def run_case(command: list[str], container_name: str, log_path: Path) -> None:
    try:
        with log_path.open("xb") as stream:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=2700.0,
            )
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "stop", "--time", "30", container_name], check=False, capture_output=True)
        raise RuntimeError(f"trajectory container exceeded 2700 wall seconds: {container_name}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"trajectory container failed with exit code {completed.returncode}: {container_name}")


def failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "aee_domain_sensor_export_summary_v1",
            "overall_status": STATUS_FAIL,
            "failure_reason": message,
            "completed_trajectories": completed,
            "planned_trajectories": 10,
            "duration_seconds": time.monotonic() - started,
            "teacher_queries": 0,
            "training_steps": 0,
            "c09_reads": 0,
            "c10_reads": 0,
        },
    )
    write_json(
        run_dir / "RUN_STATE.json",
        {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "FAILED", "overall_status": STATUS_FAIL},
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
    started = time.monotonic()
    summaries: list[dict[str, Any]] = []
    try:
        if spec.get("gate") != 2 or spec.get("operation") != "data_export":
            raise RuntimeError("approved Gate-2 data-export scope required")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("explicit user approval required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen tool drift: {name}")
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
        ).strip()
        if image_id != spec["derived_ros_image_id"]:
            raise RuntimeError("derived image identity drift")
        planner_path = "/home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node"
        if image_file_hash(planner_path) != spec["planner_binary_sha256"]:
            raise RuntimeError("planner binary identity drift")
        for world in spec["worlds"]:
            if image_file_hash(world["world_path_in_image"]) != world["world_sha256"]:
                raise RuntimeError(f"world identity drift: {world['name']}")
        if sha256(Path("/usr/bin/zstd")) != spec["host_zstd_sha256"]:
            raise RuntimeError("host zstd identity drift")
        trajectories = enumerate_aee_domain_trajectories()
        write_json(run_dir / "RUN_STATE.json", {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"})
        (run_dir / "artifacts/trajectories").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/trajectories").mkdir(parents=True, exist_ok=False)
        write_json(
            run_dir / "config/trajectory_schedule.json",
            {"schema_version": "aee_domain_trajectory_schedule_v1", "trajectories": [item.to_dict() for item in trajectories]},
        )
        for trajectory in trajectories:
            if shutil.disk_usage(run_dir).free < MINIMUM_FREE_BYTES:
                raise RuntimeError("free disk below 50 GiB before trajectory")
            command, container_name = case_command(run_dir, trajectory)
            run_case(command, container_name, run_dir / f"logs/trajectories/{trajectory.trajectory_id}.log")
            case_dir = run_dir / f"artifacts/trajectories/{trajectory.trajectory_id}"
            summary = load_json(case_dir / "summary.json")
            if summary.get("status") != "PASS_AEE_DOMAIN_TRAJECTORY_V1":
                raise RuntimeError(f"invalid trajectory summary: {trajectory.trajectory_id}")
            storage = host_archive(case_dir, summary["raw_bag_sha256"])
            summary["storage"] = storage
            summaries.append(summary)
            write_json(
                run_dir / "metrics/progress.json",
                {
                    "schema_version": "aee_domain_sensor_export_progress_v1",
                    "completed_trajectories": len(summaries),
                    "planned_trajectories": 10,
                    "last_trajectory_id": trajectory.trajectory_id,
                    "elapsed_seconds": time.monotonic() - started,
                    "free_disk_bytes": shutil.disk_usage(run_dir).free,
                },
            )
        raw_frames = sum(int(item["evidence"]["raw_frames"]) for item in summaries)
        effective_frames = sum(int(item["evidence"]["effective_frames"]) for item in summaries)
        split_counts = {
            split: sum(int(item["evidence"]["effective_frames"]) for item in summaries if item["contract"]["split"] == split)
            for split in ("train", "validation")
        }
        if raw_frames != 30000 or effective_frames != 6000 or split_counts != {"train": 3000, "validation": 3000}:
            raise RuntimeError("aggregate data count or split drift")
        manifest = [
            {
                "trajectory_id": item["contract"]["trajectory_id"],
                "world": item["contract"]["world"],
                "split": item["contract"]["split"],
                "environment_seed": item["contract"]["environment_seed"],
                "raw_frames": item["evidence"]["raw_frames"],
                "effective_frames": item["evidence"]["effective_frames"],
                "trajectory_distance_m": item["evidence"]["trajectory_distance_m"],
                "sensor_shard": f"artifacts/trajectories/{item['contract']['trajectory_id']}/sensor_shard.npz",
                "sensor_shard_sha256": item["evidence"]["sensor_shard_sha256"],
                "bag_archive": f"artifacts/trajectories/{item['contract']['trajectory_id']}/raw.bag.zst",
                "bag_archive_sha256": item["storage"]["archive_sha256"],
                "raw_bag_sha256": item["storage"]["original_bag_sha256"],
            }
            for item in summaries
        ]
        write_json(run_dir / "artifacts/data_manifest.json", {"schema_version": "aee_domain_data_manifest_v1", "records": manifest})
        summary = {
            "schema_version": "aee_domain_sensor_export_summary_v1",
            "overall_status": STATUS_PASS,
            "trajectory_count": 10,
            "raw_frames": raw_frames,
            "effective_frames": effective_frames,
            "split_effective_frames": split_counts,
            "total_trajectory_distance_m": sum(float(item["evidence"]["trajectory_distance_m"]) for item in summaries),
            "archive_bytes": sum(int(item["storage"]["archive_bytes"]) for item in summaries),
            "sensor_shard_bytes": sum(int(item["evidence"]["sensor_shard_bytes"]) for item in summaries),
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
            "teacher_queries": 0,
            "training_steps": 0,
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
