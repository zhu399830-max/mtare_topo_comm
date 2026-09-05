#!/home/zeng-workstation/anaconda3/bin/python
"""Generate, audit, visualize and seal all ten AEE objective-teacher shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import subprocess
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from _bootstrap import PROJECT_ROOT
from mtare_topo.data.aee_domain_adaptation import (
    EFFECTIVE_FRAMES_PER_TRAJECTORY,
    audit_sensor_shard,
    enumerate_aee_domain_trajectories,
    sha256,
)
from mtare_topo.governance import load_json, write_json


RUN_ID = "gate2_20260820_aee_objective_teacher_export_v1_seed20260820"
IMAGE = "mtare-semantic-runtime:planner-seed-v1-runnable"
STATUS_PASS = "PASS_AEE_OBJECTIVE_TEACHER_EXPORT_V1"
STATUS_FAIL = "FAIL_AEE_OBJECTIVE_TEACHER_EXPORT_V1"


def seal(run_dir: Path) -> int:
    target = run_dir / "artifacts/evidence_sha256.txt"
    files = sorted(path for path in run_dir.rglob("*") if path.is_file() and path != target)
    target.write_text(
        "".join(f"{sha256(path)}  {path.relative_to(PROJECT_ROOT)}\n" for path in files),
        encoding="utf-8",
    )
    return len(files)


def verify_source_seal(source_run: Path, expected_seal_sha256: str) -> int:
    seal_path = source_run / "artifacts/evidence_sha256.txt"
    if sha256(seal_path) != expected_seal_sha256:
        raise RuntimeError("source sensor-export seal identity drift")
    entries = 0
    source_prefix = source_run.relative_to(PROJECT_ROOT).as_posix() + "/"
    for line in seal_path.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        lowered = relative.lower()
        if "_c09" in lowered or "_c10" in lowered:
            raise RuntimeError(f"forbidden strict-test source in sensor seal: {relative}")
        if not relative.startswith(source_prefix):
            raise RuntimeError(f"source seal escapes sensor run: {relative}")
        if sha256(PROJECT_ROOT / relative) != expected:
            raise RuntimeError(f"source sensor seal mismatch: {relative}")
        entries += 1
    if entries < 1:
        raise RuntimeError("source sensor seal is empty")
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


def case_command(
    source_run: Path,
    run_dir: Path,
    record: dict[str, Any],
    complete_map: dict[str, Any],
    index: int,
) -> tuple[list[str], str]:
    trajectory_id = record["trajectory_id"]
    sensor_relative = Path(record["sensor_shard"])
    if sensor_relative.is_absolute() or ".." in sensor_relative.parts:
        raise RuntimeError("sensor shard path escapes source run")
    output = f"/evidence/teacher_shards/{trajectory_id}.npz"
    inner = [
        "python3",
        "/workspace/tools/v3/generate_aee_objective_teacher_shard_v1.py",
        "--sensor-shard",
        f"/source/{sensor_relative.as_posix()}",
        "--sensor-shard-sha256",
        record["sensor_shard_sha256"],
        "--complete-map",
        complete_map["path_in_image"],
        "--complete-map-sha256",
        complete_map["sha256"],
        "--trajectory-id",
        trajectory_id,
        "--output",
        output,
    ]
    shell = "export PYTHONPATH=/workspace/src:$PYTHONPATH && " + shlex.join(inner)
    name = f"aee-objective-teacher-{index:02d}"
    return [
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--name",
        name,
        "-v",
        f"{PROJECT_ROOT}:/workspace:ro",
        "-v",
        f"{source_run}:/source:ro",
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
                timeout=3600.0,
            )
    except subprocess.TimeoutExpired as exc:
        subprocess.run(
            ["docker", "stop", "--time", "30", container_name],
            check=False,
            capture_output=True,
        )
        raise RuntimeError(f"teacher container exceeded 3600 wall seconds: {container_name}") from exc
    if completed.returncode != 0:
        raise RuntimeError(
            f"teacher container failed with exit code {completed.returncode}: {container_name}"
        )


def audit_teacher_shard(
    sensor_path: Path,
    teacher_path: Path,
    summary: dict[str, Any],
    trajectory_id: str,
) -> dict[str, Any]:
    with np.load(sensor_path, allow_pickle=False) as source:
        sensor = {key: np.asarray(source[key]) for key in source.files}
    sensor_audit = audit_sensor_shard(sensor)
    if not sensor_audit["passed"]:
        raise RuntimeError(f"source sensor shard audit failed: {trajectory_id}")
    with np.load(teacher_path, allow_pickle=False) as source:
        teacher = {key: np.asarray(source[key]) for key in source.files}
    expected = {
        "direction_target",
        "count_target",
        "role_target",
        "exit_count",
        "heading_count",
        "support_z_m",
        "frame_id",
        "raw_frame_index",
    }
    if set(teacher) != expected:
        raise RuntimeError(f"teacher field drift: {trajectory_id}")
    samples = EFFECTIVE_FRAMES_PER_TRAJECTORY
    if teacher["direction_target"].shape != (samples, 720):
        raise RuntimeError(f"teacher direction shape drift: {trajectory_id}")
    for key in ("count_target", "role_target", "exit_count", "heading_count", "support_z_m", "frame_id", "raw_frame_index"):
        if teacher[key].shape != (samples,):
            raise RuntimeError(f"teacher {key} shape drift: {trajectory_id}")
    if teacher["direction_target"].dtype != np.uint8 or not np.all(
        (teacher["direction_target"] == 0) | (teacher["direction_target"] == 1)
    ):
        raise RuntimeError(f"teacher direction dtype/value drift: {trajectory_id}")
    for key in ("count_target", "role_target", "exit_count", "heading_count"):
        if teacher[key].dtype != np.int64:
            raise RuntimeError(f"teacher {key} dtype drift: {trajectory_id}")
    if not np.all(np.isfinite(teacher["support_z_m"])):
        raise RuntimeError(f"teacher support contains nonfinite values: {trajectory_id}")
    if not np.array_equal(teacher["frame_id"].astype(str), sensor["frame_id"].astype(str)):
        raise RuntimeError(f"teacher frame identity mismatch: {trajectory_id}")
    if not np.array_equal(teacher["raw_frame_index"], sensor["raw_frame_index"]):
        raise RuntimeError(f"teacher raw-index mismatch: {trajectory_id}")
    if not np.all(np.char.startswith(teacher["frame_id"].astype(str), f"{trajectory_id}:")):
        raise RuntimeError(f"teacher trajectory prefix mismatch: {trajectory_id}")
    if not np.array_equal(teacher["exit_count"], teacher["heading_count"]):
        raise RuntimeError(f"teacher exit/heading identity failed: {trajectory_id}")
    if not np.array_equal(teacher["count_target"] + 1, teacher["exit_count"]):
        raise RuntimeError(f"teacher count target mismatch: {trajectory_id}")
    if np.any((teacher["exit_count"] < 1) | (teacher["exit_count"] > 6)):
        raise RuntimeError(f"teacher empty/out-of-range exits: {trajectory_id}")
    if np.any((teacher["role_target"] < 0) | (teacher["role_target"] > 2)):
        raise RuntimeError(f"teacher role target out of range: {trajectory_id}")
    active = teacher["direction_target"].astype(bool)
    component_counts = np.sum(active & ~np.roll(active, 1, axis=1), axis=1)
    component_counts[np.all(active, axis=1)] = 1
    if not np.array_equal(component_counts, teacher["exit_count"]):
        raise RuntimeError(f"teacher direction component/count mismatch: {trajectory_id}")
    expected_roles = np.where(
        teacher["exit_count"] <= 1,
        2,
        np.where(teacher["exit_count"] == 2, 0, 1),
    )
    if not np.array_equal(teacher["role_target"], expected_roles):
        raise RuntimeError(f"teacher role semantics mismatch: {trajectory_id}")
    if summary.get("status") != "PASS_AEE_OBJECTIVE_TEACHER_SHARD_V1":
        raise RuntimeError(f"teacher summary status failed: {trajectory_id}")
    if summary.get("trajectory_id") != trajectory_id or summary.get("samples") != samples:
        raise RuntimeError(f"teacher summary identity/count drift: {trajectory_id}")
    if summary.get("teacher_queries") != samples or summary.get("nonempty_samples") != samples:
        raise RuntimeError(f"teacher query/nonempty count drift: {trajectory_id}")
    if summary.get("direction_heading_identity") is not True:
        raise RuntimeError(f"teacher summary heading identity failed: {trajectory_id}")
    if summary.get("teacher_shard_sha256") != sha256(teacher_path):
        raise RuntimeError(f"teacher shard hash drift: {trajectory_id}")
    return {
        "samples": samples,
        "frame_ids": teacher["frame_id"].astype(str).tolist(),
        "exit_histogram": Counter(int(value) for value in teacher["exit_count"]),
        "role_histogram": Counter(int(value) for value in teacher["role_target"]),
    }


def render_previews(
    run_dir: Path,
    source_run: Path,
    manifest: list[dict[str, Any]],
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outputs: list[str] = []
    for world in ("tunnel", "garage"):
        xy_parts: list[np.ndarray] = []
        role_parts: list[np.ndarray] = []
        exit_parts: list[np.ndarray] = []
        examples: dict[int, tuple[np.ndarray, np.ndarray, str]] = {}
        for record in manifest:
            if record["world"] != world:
                continue
            with np.load(source_run / record["sensor_shard"], allow_pickle=False) as source:
                xyz = np.asarray(source["sensor_xyz_m"])
                ranges = np.asarray(source["range_m"])
            with np.load(run_dir / record["teacher_shard"], allow_pickle=False) as source:
                roles = np.asarray(source["role_target"])
                exits = np.asarray(source["exit_count"])
                directions = np.asarray(source["direction_target"])
            xy_parts.append(xyz[:, :2])
            role_parts.append(roles)
            exit_parts.append(exits)
            for role in (0, 1, 2):
                indices = np.flatnonzero(roles == role)
                if role not in examples and len(indices):
                    index = int(indices[0])
                    examples[role] = (ranges[index], directions[index], record["trajectory_id"])
        xy = np.concatenate(xy_parts)
        roles = np.concatenate(role_parts)
        exits = np.concatenate(exit_parts)
        figure, axis = plt.subplots(figsize=(9, 7))
        scatter = axis.scatter(xy[:, 0], xy[:, 1], c=roles, s=5 + exits * 2, cmap="viridis", alpha=0.75)
        axis.set_aspect("equal", adjustable="box")
        axis.set_title(f"{world}: 3000 objective teacher samples")
        axis.set_xlabel("x (m)")
        axis.set_ylabel("y (m)")
        figure.colorbar(scatter, ax=axis, label="role target (0 corridor, 1 junction, 2 terminal)")
        coverage = run_dir / f"previews/{world}_teacher_coverage.png"
        figure.tight_layout()
        figure.savefig(coverage, dpi=160)
        plt.close(figure)
        outputs.append(str(coverage.relative_to(run_dir)))

        figure, axes = plt.subplots(3, 2, figsize=(12, 9))
        for role in (0, 1, 2):
            ranges, directions, trajectory_id = examples[role]
            axes[role, 0].imshow(ranges, aspect="auto", vmin=0.3, vmax=50.0, cmap="magma")
            axes[role, 0].set_title(f"role {role} range — {trajectory_id}")
            axes[role, 1].plot(np.arange(720) * 0.5, directions, linewidth=1.0)
            axes[role, 1].set_ylim(-0.05, 1.05)
            axes[role, 1].set_title(f"role {role} objective directions")
            axes[role, 1].set_xlabel("relative azimuth (deg)")
        examples_path = run_dir / f"previews/{world}_all_role_examples.png"
        figure.tight_layout()
        figure.savefig(examples_path, dpi=160)
        plt.close(figure)
        outputs.append(str(examples_path.relative_to(run_dir)))
    return outputs


def failure(run_dir: Path, started: float, completed: int, message: str) -> None:
    write_json(
        run_dir / "metrics/summary.json",
        {
            "schema_version": "aee_objective_teacher_export_summary_v1",
            "overall_status": STATUS_FAIL,
            "failure_reason": message,
            "completed_teacher_shards": completed,
            "planned_teacher_shards": 10,
            "duration_seconds": time.monotonic() - started,
            "training_steps": 0,
            "model_inference_frames": 0,
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
    completed: list[dict[str, Any]] = []
    try:
        if spec.get("gate") != 2 or spec.get("operation") != "teacher_generation":
            raise RuntimeError("approved Gate-2 teacher-generation scope required")
        if spec.get("user_authorization", {}).get("status") != "APPROVED":
            raise RuntimeError("explicit user approval required")
        for name, item in spec["frozen_tools"].items():
            if sha256(PROJECT_ROOT / item["path"]) != item["sha256"]:
                raise RuntimeError(f"frozen tool drift: {name}")
        source_run = (PROJECT_ROOT / spec["source_sensor_run"]).resolve()
        source_run.relative_to(PROJECT_ROOT)
        source_state = load_json(source_run / "RUN_STATE.json")
        if source_state.get("state") != "COMPLETED" or source_state.get("overall_status") != "PASS_AEE_DOMAIN_SENSOR_EXPORT_V1":
            raise RuntimeError("source sensor export is not a completed PASS")
        source_entries = verify_source_seal(source_run, spec["source_sensor_seal_sha256"])
        source_summary = load_json(source_run / "metrics/summary.json")
        if source_summary.get("raw_frames") != 30000 or source_summary.get("effective_frames") != 6000:
            raise RuntimeError("source sensor aggregate count drift")
        if any(source_summary.get(key) != 0 for key in ("teacher_queries", "training_steps", "c09_reads", "c10_reads")):
            raise RuntimeError("source sensor run contains forbidden operations")
        manifest_path = source_run / "artifacts/data_manifest.json"
        if sha256(manifest_path) != spec["source_sensor_manifest_sha256"]:
            raise RuntimeError("source sensor manifest identity drift")
        records = load_json(manifest_path).get("records")
        trajectories = enumerate_aee_domain_trajectories()
        if not isinstance(records, list) or len(records) != len(trajectories):
            raise RuntimeError("source sensor manifest must contain exactly ten records")
        for record, trajectory in zip(records, trajectories):
            expected = (trajectory.trajectory_id, trajectory.world, trajectory.split, trajectory.environment_seed)
            observed = (record.get("trajectory_id"), record.get("world"), record.get("split"), record.get("environment_seed"))
            if observed != expected:
                raise RuntimeError(f"source trajectory schedule drift: {observed}")
            sensor_path = source_run / record["sensor_shard"]
            if sha256(sensor_path) != record["sensor_shard_sha256"]:
                raise RuntimeError(f"source sensor shard drift: {trajectory.trajectory_id}")
        image_id = subprocess.check_output(
            ["docker", "image", "inspect", IMAGE, "--format", "{{.Id}}"], text=True
        ).strip()
        if image_id != spec["derived_ros_image_id"]:
            raise RuntimeError("teacher image identity drift")
        maps = {item["world"]: item for item in spec["complete_maps"]}
        if set(maps) != {"tunnel", "garage"}:
            raise RuntimeError("teacher map binding must contain tunnel and garage exactly")
        for world, item in maps.items():
            if image_file_hash(item["path_in_image"]) != item["sha256"]:
                raise RuntimeError(f"complete-map identity drift: {world}")

        write_json(
            run_dir / "config/input_integrity.json",
            {
                "source_run": str(source_run.relative_to(PROJECT_ROOT)),
                "source_seal_entries": source_entries,
                "source_sensor_manifest_sha256": sha256(manifest_path),
                "image_id": image_id,
                "complete_maps": maps,
            },
        )
        write_json(
            run_dir / "RUN_STATE.json",
            {"schema_version": "v3_run_state_v1", "run_id": RUN_ID, "state": "RUNNING"},
        )
        (run_dir / "artifacts/teacher_shards").mkdir(parents=True, exist_ok=False)
        (run_dir / "logs/teacher_shards").mkdir(parents=True, exist_ok=False)
        all_frame_ids: set[str] = set()
        split_samples = Counter()
        split_roles: dict[str, Counter[int]] = {"train": Counter(), "validation": Counter()}
        split_exits: dict[str, Counter[int]] = {"train": Counter(), "validation": Counter()}
        teacher_manifest: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            command, name = case_command(source_run, run_dir, record, maps[record["world"]], index)
            run_case(command, name, run_dir / f"logs/teacher_shards/{record['trajectory_id']}.log")
            output = run_dir / f"artifacts/teacher_shards/{record['trajectory_id']}.npz"
            summary_path = output.with_suffix(".summary.json")
            summary = load_json(summary_path)
            if summary.get("sensor_shard_sha256") != record["sensor_shard_sha256"]:
                raise RuntimeError(f"teacher source sensor hash mismatch: {record['trajectory_id']}")
            if summary.get("complete_map_sha256") != maps[record["world"]]["sha256"]:
                raise RuntimeError(f"teacher complete-map hash mismatch: {record['trajectory_id']}")
            audit = audit_teacher_shard(
                source_run / record["sensor_shard"], output, summary, record["trajectory_id"]
            )
            duplicate_ids = all_frame_ids.intersection(audit["frame_ids"])
            if duplicate_ids:
                raise RuntimeError(f"duplicate global teacher frame IDs: {record['trajectory_id']}")
            all_frame_ids.update(audit["frame_ids"])
            split = record["split"]
            split_samples[split] += audit["samples"]
            split_roles[split].update(audit["role_histogram"])
            split_exits[split].update(audit["exit_histogram"])
            expected_exit_histogram = {
                str(key): value for key, value in sorted(audit["exit_histogram"].items())
            }
            expected_role_histogram = {
                str(key): value for key, value in sorted(audit["role_histogram"].items())
            }
            if summary.get("exit_count_histogram") != expected_exit_histogram:
                raise RuntimeError(f"teacher exit histogram mismatch: {record['trajectory_id']}")
            if summary.get("role_histogram") != expected_role_histogram:
                raise RuntimeError(f"teacher role histogram mismatch: {record['trajectory_id']}")
            teacher_record = {
                **record,
                "teacher_shard": str(output.relative_to(run_dir)),
                "teacher_shard_sha256": sha256(output),
                "teacher_summary": str(summary_path.relative_to(run_dir)),
                "teacher_summary_sha256": sha256(summary_path),
                "samples": audit["samples"],
                "exit_count_histogram": expected_exit_histogram,
                "role_histogram": expected_role_histogram,
                "complete_map_sha256": maps[record["world"]]["sha256"],
            }
            teacher_manifest.append(teacher_record)
            completed.append(teacher_record)
            write_json(
                run_dir / "metrics/progress.json",
                {
                    "schema_version": "aee_objective_teacher_progress_v1",
                    "completed_teacher_shards": len(completed),
                    "planned_teacher_shards": 10,
                    "teacher_queries": sum(item["samples"] for item in completed),
                    "last_trajectory_id": record["trajectory_id"],
                    "duration_seconds": time.monotonic() - started,
                },
            )
        if split_samples != Counter({"train": 3000, "validation": 3000}) or len(all_frame_ids) != 6000:
            raise RuntimeError("teacher aggregate sample/split/global-identity drift")
        for split in ("train", "validation"):
            if set(split_roles[split]) != {0, 1, 2}:
                raise RuntimeError(f"teacher split lacks one or more structural roles: {split}")
        write_json(
            run_dir / "artifacts/teacher_manifest.json",
            {"schema_version": "aee_objective_teacher_manifest_v1", "records": teacher_manifest},
        )
        previews = render_previews(run_dir, source_run, teacher_manifest)
        summary = {
            "schema_version": "aee_objective_teacher_export_summary_v1",
            "overall_status": STATUS_PASS,
            "teacher_shards": 10,
            "samples": 6000,
            "teacher_queries": 6000,
            "split_samples": dict(split_samples),
            "unique_frame_ids": len(all_frame_ids),
            "split_role_histograms": {
                split: {str(key): value for key, value in sorted(split_roles[split].items())}
                for split in ("train", "validation")
            },
            "split_exit_count_histograms": {
                split: {str(key): value for key, value in sorted(split_exits[split].items())}
                for split in ("train", "validation")
            },
            "preview_files": previews,
            "training_steps": 0,
            "model_inference_frames": 0,
            "c09_reads": 0,
            "c10_reads": 0,
            "duration_seconds": time.monotonic() - started,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
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
        failure(run_dir, started, len(completed), str(exc))
        raise


if __name__ == "__main__":
    raise SystemExit(main())
