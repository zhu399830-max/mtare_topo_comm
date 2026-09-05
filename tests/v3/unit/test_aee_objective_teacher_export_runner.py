"""Unit contracts for the AEE objective-teacher export runner.

These tests exercise only the runner's pure command/audit/seal contracts; they
never launch Docker or generate a teacher shard.
"""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest

from mtare_topo.data.aee_domain_adaptation import EFFECTIVE_FRAMES_PER_TRAJECTORY


TRAJECTORY_ID = "00_tunnel_seed11"
ROOT = Path(__file__).resolve().parents[3]


def _load_runner():
    import sys

    tools = ROOT / "tools/v3"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "run_aee_objective_teacher_export_v1", tools / "run_aee_objective_teacher_export_v1.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner = _load_runner()


def _sensor_payload(trajectory_id: str = TRAJECTORY_ID) -> dict[str, np.ndarray]:
    n = EFFECTIVE_FRAMES_PER_TRAJECTORY
    return {
        "range_m": np.full((n, 16, 720), 10.0, dtype=np.float32),
        "valid_mask": np.ones((n, 16, 720), dtype=np.uint8),
        "sensor_xyz_m": np.zeros((n, 3), dtype=np.float64),
        "sensor_orientation_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (n, 1)),
        "yaw_deg": np.zeros(n, dtype=np.float64),
        "stamp_sec": np.arange(n, dtype=np.float64),
        "raw_frame_index": np.arange(0, 3000, 5, dtype=np.int64),
        "frame_id": np.asarray([f"{trajectory_id}:{i}" for i in range(n)], dtype="U64"),
    }


def _teacher_payload(sensor: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    n = len(sensor["frame_id"])
    directions = np.zeros((n, 720), dtype=np.uint8)
    directions[:, 0] = 1
    return {
        "direction_target": directions,
        "count_target": np.zeros(n, dtype=np.int64),
        "role_target": np.full(n, 2, dtype=np.int64),
        "exit_count": np.ones(n, dtype=np.int64),
        "heading_count": np.ones(n, dtype=np.int64),
        "support_z_m": np.zeros(n, dtype=np.float64),
        "frame_id": sensor["frame_id"].copy(),
        "raw_frame_index": sensor["raw_frame_index"].copy(),
    }


def _write_shard(path: Path, payload: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **payload)


def _summary(teacher_path: Path, trajectory_id: str = TRAJECTORY_ID) -> dict[str, object]:
    return {
        "status": "PASS_AEE_OBJECTIVE_TEACHER_SHARD_V1",
        "trajectory_id": trajectory_id,
        "samples": 600,
        "teacher_queries": 600,
        "nonempty_samples": 600,
        "direction_heading_identity": True,
        "teacher_shard_sha256": runner.sha256(teacher_path),
    }


def test_case_command_binds_each_case_and_disables_network(tmp_path: Path) -> None:
    source = tmp_path / "sensor-export"
    output = tmp_path / "teacher-export"
    record = {
        "trajectory_id": TRAJECTORY_ID,
        "sensor_shard": "artifacts/trajectories/00_tunnel_seed11/sensor_shard.npz",
        "sensor_shard_sha256": "sensor-hash",
        "world": "tunnel",
        "split": "train",
    }
    complete_map = {"path_in_image": "/maps/tunnel.ply", "sha256": "map-hash"}
    command, name = runner.case_command(source, output, record, complete_map, 0)
    assert name == "aee-objective-teacher-00"
    assert command[0:4] == ["docker", "run", "--rm", "--network"]
    assert command[4] == "none"
    joined = " ".join(command)
    assert "/source/artifacts/trajectories/00_tunnel_seed11/sensor_shard.npz" in joined
    assert "--sensor-shard-sha256 sensor-hash" in joined
    assert "--complete-map /maps/tunnel.ply" in joined
    assert "--complete-map-sha256 map-hash" in joined
    assert f"--trajectory-id {TRAJECTORY_ID}" in joined
    assert "/evidence/teacher_shards/00_tunnel_seed11.npz" in joined


def test_audit_teacher_shard_accepts_valid_600_sample_fixture(tmp_path: Path) -> None:
    sensor_path = tmp_path / "sensor.npz"
    teacher_path = tmp_path / "teacher.npz"
    sensor = _sensor_payload()
    _write_shard(sensor_path, sensor)
    _write_shard(teacher_path, _teacher_payload(sensor))
    result = runner.audit_teacher_shard(
        sensor_path, teacher_path, _summary(teacher_path), TRAJECTORY_ID
    )
    assert result["samples"] == 600
    assert sum(result["exit_histogram"].values()) == 600
    assert set(result["role_histogram"]) == {2}


@pytest.mark.parametrize(
    "mutation,match",
    [
        (lambda t: t["frame_id"].__setitem__(0, "wrong:0"), "frame identity"),
        (lambda t: t["exit_count"].__setitem__(0, 2), "exit/heading"),
        (lambda t: t["count_target"].__setitem__(0, 1), "count target"),
        (lambda t: t["role_target"].__setitem__(0, 3), "role target"),
        (lambda t: t.__setitem__("direction_target", t["direction_target"].astype(np.float32)), "direction dtype"),
    ],
)
def test_audit_teacher_shard_rejects_field_drift(tmp_path: Path, mutation, match: str) -> None:
    sensor_path = tmp_path / "sensor.npz"
    teacher_path = tmp_path / "teacher.npz"
    sensor = _sensor_payload()
    teacher = _teacher_payload(sensor)
    _write_shard(sensor_path, sensor)
    mutation(teacher)
    _write_shard(teacher_path, teacher)
    with pytest.raises(RuntimeError, match=match):
        runner.audit_teacher_shard(
            sensor_path, teacher_path, _summary(teacher_path), TRAJECTORY_ID
        )


def test_verify_source_seal_rejects_escape_and_strict_world_entries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    project = tmp_path / "project"
    source = project / "results" / "gate2" / "sensor"
    source.mkdir(parents=True)
    (source / "ok.txt").write_text("ok\n", encoding="utf-8")
    monkeypatch.setattr(runner, "PROJECT_ROOT", project)

    seal = source / "artifacts" / "evidence_sha256.txt"
    seal.parent.mkdir()
    expected = runner.sha256(source / "ok.txt")
    seal.write_text(f"{expected}  {source.relative_to(project)}/ok.txt\n", encoding="utf-8")
    assert runner.verify_source_seal(source, runner.sha256(seal)) == 1

    seal.write_text(f"{expected}  ../outside.txt\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="escapes"):
        runner.verify_source_seal(source, runner.sha256(seal))

    seal.write_text(f"{expected}  {source.relative_to(project)}/foo_C09.npz\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden strict-test"):
        runner.verify_source_seal(source, runner.sha256(seal))

    seal.write_text(f"{expected}  {source.relative_to(project)}/foo_C10.npz\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="forbidden strict-test"):
        runner.verify_source_seal(source, runner.sha256(seal))


def test_runner_has_no_retry_and_has_frozen_aggregate_and_role_gates() -> None:
    source = inspect.getsource(runner)
    assert "retry" not in source.lower()
    assert "split_samples != Counter({\"train\": 3000, \"validation\": 3000})" in source
    assert "len(all_frame_ids) != 6000" in source
    assert "split_roles[split]) != {0, 1, 2}" in source
    assert '"teacher_queries": 6000' in source
    assert '"c09_reads": 0' in source
    assert '"c10_reads": 0' in source
