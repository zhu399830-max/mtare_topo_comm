#!/home/zeng-workstation/anaconda3/bin/python
"""Execute and seal the approved 11,000-frame corrective sensor/teacher run."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_corrective_dataset import (
    AEE_FRAMES_PER_TRAJECTORY,
    TOTAL_CORRECTIVE_FRAMES,
    aee_corrective_frame_indices,
    audit_corrective_sensor_shard,
    enumerate_aee_corrective_trajectories,
)
from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260822_aee_corrective_sensor_teacher_dataset_v1_seed20260822"
IMAGE = "mtare-semantic-runtime:planner-seed-v1-runnable"
STATUS_PASS = "PASS_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1"
STATUS_FAIL = "FAIL_AEE_CORRECTIVE_SENSOR_TEACHER_DATASET_V1"
CANO_PYTHON = Path("/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python3")
CANO_EXECUTOR = "tools/v3/execute_aee_corrective_cano_dataset_v1.py"
CANO_STATUS_PASS = "PASS_AEE_CORRECTIVE_CANO_DATASET_V1"
CANO_EXPECTED_METRICS: dict[str, int] = {}
SENSOR_STATUS_PASS = "PASS_AEE_CORRECTIVE_TRAJECTORY_V1"
SENSOR_REQUIRED_PAIRING_METHOD: str | None = None
HOST_UID = 1000
HOST_GID = 1000


def seal(run_dir: Path) -> tuple[int, str]:
    target = run_dir / "artifacts/evidence_sha256.txt"
    paths = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in paths),
        encoding="utf-8",
    )
    return len(paths), sha256(target)


def verify_source_seal(source_run: Path, expected_seal_sha256: str) -> int:
    seal_path = source_run / "artifacts/evidence_sha256.txt"
    if sha256(seal_path) != expected_seal_sha256:
        raise RuntimeError("corrective mesh source seal identity drift")
    prefix = source_run.relative_to(PROJECT_ROOT).as_posix() + "/"
    entries = 0
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        lowered = relative.lower()
        if "_c09" in lowered or "_c10" in lowered:
            raise RuntimeError(f"forbidden strict-test entry in source seal: {relative}")
        if not relative.startswith(prefix) or sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"corrective mesh source seal mismatch: {relative}")
        entries += 1
    if entries != 287:
        raise RuntimeError(f"corrective mesh source seal must contain 287 entries, observed {entries}")
    return entries


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


def verify_archive(archive: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(block)
    returncode = process.wait()
    actual = digest.hexdigest()
    if returncode != 0 or actual != expected_sha256:
        raise RuntimeError("zstd decompression identity verification failed")
    return actual


def archive_raw_bag(case_dir: Path, expected_sha256: str) -> dict[str, Any]:
    raw = case_dir / "raw.bag"
    handoff = load_json(case_dir / "host_ownership_handoff.json")
    if (
        handoff.get("status") != "PASS_AEE_CORRECTIVE_HOST_OWNERSHIP_HANDOFF_V1"
        or handoff.get("host_uid") != HOST_UID
        or handoff.get("host_gid") != HOST_GID
    ):
        raise RuntimeError("AEE corrective ownership handoff failed")
    if os.getuid() != HOST_UID or os.getgid() != HOST_GID:
        raise RuntimeError("formal host UID/GID identity drift")
    for path in (case_dir, raw):
        stat = path.stat()
        if stat.st_uid != HOST_UID or stat.st_gid != HOST_GID:
            raise RuntimeError(f"host ownership drift: {path}")
    if not raw.is_file() or sha256(raw) != expected_sha256:
        raise RuntimeError("raw bag identity missing before archive")
    archive = case_dir / "raw.bag.zst"
    storage_path = case_dir / "storage.json"
    if archive.exists() or storage_path.exists():
        raise RuntimeError("archive output exists; overwrite forbidden")
    completed = subprocess.run(
        ["zstd", "-10", "-T0", "-q", str(raw), "-o", str(archive)],
        check=False,
    )
    if completed.returncode != 0 or not archive.is_file():
        raise RuntimeError("host zstd compression failed")
    decompressed = verify_archive(archive, expected_sha256)
    storage = {
        "schema_version": "aee_corrective_host_archive_v1",
        "host_zstd_sha256": sha256(Path("/usr/bin/zstd")),
        "zstd_level": 10,
        "overwrite_permitted": False,
        "original_bag_sha256": expected_sha256,
        "original_bag_bytes": raw.stat().st_size,
        "archive_sha256": sha256(archive),
        "archive_bytes": archive.stat().st_size,
        "decompressed_sha256": decompressed,
        "ownership_handoff": handoff,
    }
    write_json(storage_path, storage)
    raw.unlink()
    return storage


def collector_command(run_dir: Path, trajectory: Any) -> tuple[list[str], str]:
    name = f"aee-corrective-{trajectory.index:02d}"
    inner = [
        "python3",
        "/workspace/tools/v3/collect_aee_corrective_trajectory_v1r.py",
        "--output-dir",
        f"/evidence/aee_trajectories/{trajectory.trajectory_id}",
        "--trajectory-id",
        trajectory.trajectory_id,
        "--world",
        trajectory.world,
        "--split",
        trajectory.split,
        "--environment-seed",
        str(trajectory.environment_seed),
        "--host-uid",
        str(HOST_UID),
        "--host-gid",
        str(HOST_GID),
    ]
    shell = (
        "source /opt/ros/noetic/setup.bash && "
        "source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && "
        "source /home/docker-user/mtare/tare_system/devel/setup.bash --extend && "
        "export PYTHONPATH=/workspace/src:$PYTHONPATH && "
        + shlex.join(inner)
    )
    return [
        "docker", "run", "--rm", "--name", name,
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint", "/bin/bash", IMAGE, "-lc", shell,
    ], name


def teacher_command(run_dir: Path, trajectory: Any, complete_map: dict[str, Any]) -> tuple[list[str], str]:
    name = f"aee-corrective-teacher-{trajectory.index:02d}"
    sensor = run_dir / f"artifacts/aee_trajectories/{trajectory.trajectory_id}/sensor_shard.npz"
    output = f"/evidence/aee_teacher/{trajectory.trajectory_id}.npz"
    inner = [
        "python3", "/workspace/tools/v3/generate_aee_corrective_teacher_shard_v1.py",
        "--sensor-shard", f"/evidence/aee_trajectories/{trajectory.trajectory_id}/sensor_shard.npz",
        "--sensor-shard-sha256", sha256(sensor),
        "--obstacle-map", complete_map["path_in_image"],
        "--obstacle-map-sha256", complete_map["sha256"],
        "--support-mesh", complete_map["support_mesh_path_in_image"],
        "--support-mesh-sha256", complete_map["support_mesh_sha256"],
        "--world-file", complete_map["world_path_in_image"],
        "--world-file-sha256", complete_map["world_sha256"],
        "--model-sdf", complete_map["model_sdf_path_in_image"],
        "--model-sdf-sha256", complete_map["model_sdf_sha256"],
        "--include-uri", complete_map["include_uri"],
        "--mesh-uri", complete_map["mesh_uri"],
        "--trajectory-id", trajectory.trajectory_id,
        "--output", output,
    ]
    shell = "export PYTHONPATH=/workspace/src:$PYTHONPATH && " + shlex.join(inner)
    return [
        "docker", "run", "--rm", "--network", "none", "--name", name,
        "-v", f"{PROJECT_ROOT}:/workspace:ro",
        "-v", f"{run_dir / 'artifacts'}:/evidence:rw",
        "--entrypoint", "/bin/bash", IMAGE, "-lc", shell,
    ], name


def run_logged(command: list[str], name: str, log_path: Path, timeout_sec: float) -> None:
    try:
        with log_path.open("xb") as stream:
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
                timeout=timeout_sec,
            )
    except subprocess.TimeoutExpired as exc:
        subprocess.run(["docker", "stop", "--time", "30", name], check=False, capture_output=True)
        raise RuntimeError(f"case exceeded {timeout_sec}s: {name}") from exc
    if completed.returncode != 0:
        raise RuntimeError(f"case failed with exit code {completed.returncode}: {name}")


def audit_teacher(sensor_path: Path, teacher_path: Path, trajectory_id: str) -> dict[str, Any]:
    with np.load(sensor_path, allow_pickle=False) as source:
        sensor = {key: np.asarray(source[key]) for key in source.files}
    if not audit_corrective_sensor_shard(sensor, AEE_FRAMES_PER_TRAJECTORY)["passed"]:
        raise RuntimeError(f"source sensor shard failed final audit: {trajectory_id}")
    if not np.array_equal(sensor["raw_frame_index"], aee_corrective_frame_indices()):
        raise RuntimeError(f"source raw indices drift: {trajectory_id}")
    with np.load(teacher_path, allow_pickle=False) as source:
        teacher = {key: np.asarray(source[key]) for key in source.files}
    expected = {
        "direction_target", "count_target", "role_target", "exit_count", "heading_count",
        "support_z_m", "frame_id", "raw_frame_index",
    }
    if set(teacher) != expected or teacher["direction_target"].shape != (100, 720):
        raise RuntimeError(f"teacher field/shape drift: {trajectory_id}")
    if not np.array_equal(teacher["frame_id"].astype(str), sensor["frame_id"].astype(str)):
        raise RuntimeError(f"teacher frame identity drift: {trajectory_id}")
    if not np.array_equal(teacher["raw_frame_index"], sensor["raw_frame_index"]):
        raise RuntimeError(f"teacher raw-index identity drift: {trajectory_id}")
    if not np.array_equal(teacher["exit_count"], teacher["heading_count"]):
        raise RuntimeError(f"teacher direction-count identity drift: {trajectory_id}")
    if not np.array_equal(teacher["count_target"] + 1, teacher["exit_count"]):
        raise RuntimeError(f"teacher count target drift: {trajectory_id}")
    expected_role = np.where(teacher["exit_count"] <= 1, 2, np.where(teacher["exit_count"] == 2, 0, 1))
    if not np.array_equal(teacher["role_target"], expected_role):
        raise RuntimeError(f"teacher role semantics drift: {trajectory_id}")
    if np.any((teacher["exit_count"] < 1) | (teacher["exit_count"] > 6)):
        raise RuntimeError(f"teacher exit count out of range: {trajectory_id}")
    return {
        "samples": 100,
        "frame_ids": teacher["frame_id"].astype(str).tolist(),
        "exit_histogram": dict(Counter(int(value) for value in teacher["exit_count"])),
        "role_histogram": dict(Counter(int(value) for value in teacher["role_target"])),
    }


def fail(run_dir: Path, started: float, message: str, stage: str) -> None:
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "aee_corrective_sensor_teacher_dataset_summary_v1",
            "overall_status": STATUS_FAIL,
            "failure_stage": stage,
            "failure_reason": message,
            "duration_seconds": time.monotonic() - started,
            "training_steps": 0,
            "models": 0,
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
        raise RuntimeError("formal run identity/state mismatch")
    started = time.monotonic()
    stage = "integrity"
    try:
        if spec.get("gate") != 2 or spec.get("operation") != "data_export":
            raise RuntimeError("approved Gate-2 corrective data/teacher scope required")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("explicit user approval required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen tool drift: {name}")
        source_run = (PROJECT_ROOT / spec["source_mesh_run"]).resolve()
        source_entries = verify_source_seal(source_run, spec["source_mesh_seal_sha256"])
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
        ).strip()
        if image_id != spec["derived_ros_image_id"]:
            raise RuntimeError("derived ROS image identity drift")
        if sha256(Path("/usr/bin/zstd")) != spec["host_zstd_sha256"]:
            raise RuntimeError("host zstd identity drift")
        for complete_map in spec["complete_maps"]:
            for path_key, hash_key in (
                ("path_in_image", "sha256"),
                ("support_mesh_path_in_image", "support_mesh_sha256"),
                ("world_path_in_image", "world_sha256"),
                ("model_sdf_path_in_image", "model_sdf_sha256"),
            ):
                if image_file_hash(complete_map[path_key]) != complete_map[hash_key]:
                    raise RuntimeError(f"AEE complete-map input drift: {complete_map['world']}:{path_key}")
        if shutil_disk_free(run_dir) < int(spec["resource_contract"]["minimum_free_bytes"]):
            raise RuntimeError("insufficient free disk before formal run")

        stage = "cano_sensor_teacher"
        cano_log = run_dir / "logs/cano_dataset.log"
        cano_command = [str(CANO_PYTHON), CANO_EXECUTOR, "--run-dir", str(run_dir)]
        run_logged(cano_command, "cano-corrective-dataset", cano_log, 7200.0)
        cano_summary = load_json(run_dir / "metrics/cano_summary.json")
        if cano_summary.get("overall_status") != CANO_STATUS_PASS or cano_summary.get("frames") != 10_000:
            raise RuntimeError("corrective Cano aggregate failed")
        for key, expected in CANO_EXPECTED_METRICS.items():
            if cano_summary.get(key) != expected:
                raise RuntimeError(f"corrective Cano aggregate {key} drift")

        stage = "aee_sensor_teacher"
        (run_dir / "artifacts/aee_trajectories").mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts/aee_teacher").mkdir(parents=True, exist_ok=True)
        map_by_world = {item["world"]: item for item in spec["complete_maps"]}
        aee_records = []
        world_roles: dict[str, set[int]] = defaultdict(set)
        frame_ids: set[str] = set()
        archive_bytes = 0
        for trajectory in enumerate_aee_corrective_trajectories():
            command, name = collector_command(run_dir, trajectory)
            run_logged(command, name, run_dir / f"logs/aee_sensor_{trajectory.index:02d}.log", 2700.0)
            case_dir = run_dir / f"artifacts/aee_trajectories/{trajectory.trajectory_id}"
            sensor_summary = load_json(case_dir / "summary.json")
            if sensor_summary.get("status") != SENSOR_STATUS_PASS:
                raise RuntimeError(f"AEE corrective sensor failed: {trajectory.trajectory_id}")
            sensor_evidence = sensor_summary.get("evidence", {})
            if SENSOR_REQUIRED_PAIRING_METHOD is not None:
                if sensor_evidence.get("pairing_method") != SENSOR_REQUIRED_PAIRING_METHOD:
                    raise RuntimeError(f"AEE pairing method drift: {trajectory.trajectory_id}")
                if (
                    sensor_evidence.get("raw_frames") != 3_000
                    or sensor_evidence.get("registered_frames_with_exact_odometry", 0) < 3_000
                    or not 0.0 <= float(sensor_evidence.get("maximum_raw_registered_stamp_delta_sec", 1.0)) < 0.1
                ):
                    raise RuntimeError(f"AEE pairing evidence failed: {trajectory.trajectory_id}")
            storage = archive_raw_bag(case_dir, sensor_summary["raw_bag_sha256"])
            archive_bytes += int(storage["archive_bytes"])
            teacher_cmd, teacher_name = teacher_command(run_dir, trajectory, map_by_world[trajectory.world])
            run_logged(teacher_cmd, teacher_name, run_dir / f"logs/aee_teacher_{trajectory.index:02d}.log", 1800.0)
            sensor_path = case_dir / "sensor_shard.npz"
            teacher_path = run_dir / f"artifacts/aee_teacher/{trajectory.trajectory_id}.npz"
            audit = audit_teacher(sensor_path, teacher_path, trajectory.trajectory_id)
            if frame_ids.intersection(audit["frame_ids"]):
                raise RuntimeError("duplicate AEE corrective frame identity")
            frame_ids.update(audit["frame_ids"])
            world_roles[trajectory.world].update(int(key) for key in audit["role_histogram"])
            aee_records.append(
                {
                    **trajectory.to_dict(),
                    "sensor_shard": str(sensor_path.relative_to(run_dir)),
                    "sensor_shard_sha256": sha256(sensor_path),
                    "teacher_shard": str(teacher_path.relative_to(run_dir)),
                    "teacher_shard_sha256": sha256(teacher_path),
                    "archive": str((case_dir / "raw.bag.zst").relative_to(run_dir)),
                    "archive_sha256": storage["archive_sha256"],
                    "audit": audit,
                }
            )
            write_json(run_dir / "metrics/progress.json", {"completed_aee_trajectories": len(aee_records), "planned": 10})
        if len(aee_records) != 10 or len(frame_ids) != 1_000:
            raise RuntimeError("AEE corrective aggregate count drift")
        if any(world_roles[world] != {0, 1, 2} for world in ("tunnel", "garage")):
            raise RuntimeError(f"AEE world role coverage failed: {dict(world_roles)}")
        write_json(run_dir / "artifacts/aee_manifest.json", {"trajectories": aee_records})

        stage = "aggregate"
        result_bytes = sum(path.stat().st_size for path in run_dir.rglob("*") if path.is_file())
        if result_bytes > int(spec["resource_contract"]["result_bytes_limit"]):
            raise RuntimeError("formal result size exceeded frozen limit")
        summary = {
            "schema_version": "aee_corrective_sensor_teacher_dataset_summary_v1",
            "overall_status": STATUS_PASS,
            "total_frames": TOTAL_CORRECTIVE_FRAMES,
            "corrective_train_frames": 6_000,
            "corrective_validation_frames": 5_000,
            "cano_frames": 10_000,
            "cano_eligibility": {
                key: cano_summary.get(key) for key in sorted(CANO_EXPECTED_METRICS)
            },
            "aee_frames": 1_000,
            "teacher_labels": 11_000,
            "cano_worlds": 20,
            "aee_worlds": 2,
            "aee_trajectories": 10,
            "aee_pairing_method": SENSOR_REQUIRED_PAIRING_METHOD,
            "source_seal_entries_verified": source_entries,
            "unique_aee_frame_ids": len(frame_ids),
            "aee_archive_bytes": archive_bytes,
            "result_bytes_before_seal": result_bytes,
            "duration_seconds": time.monotonic() - started,
            "training_steps": 0,
            "models": 0,
            "c09_reads": 0,
            "c10_reads": 0,
            "mtare_changes": 0,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        write_json(run_dir / "metrics/summary.json", summary)
        write_json(
            run_dir / "RUN_STATE.json",
            {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "COMPLETED", "overall_status": STATUS_PASS},
        )
        entries, seal_sha = seal(run_dir)
        print(json.dumps({**summary, "seal_entries": entries, "seal_sha256": seal_sha}, indent=2, sort_keys=True))
        return 0
    except Exception as exc:
        fail(run_dir, started, f"{type(exc).__name__}: {exc}", stage)
        raise


def shutil_disk_free(path: Path) -> int:
    import shutil
    return int(shutil.disk_usage(path).free)


if __name__ == "__main__":
    raise SystemExit(main())
