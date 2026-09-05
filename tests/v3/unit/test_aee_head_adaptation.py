"""Contracts for the AEE head-adaptation dataset and balanced sampler."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from mtare_topo.data.aee_domain_adaptation import EFFECTIVE_FRAMES_PER_TRAJECTORY, sha256


ROOT = Path(__file__).resolve().parents[3]


def _load_module():
    import sys

    source_root = ROOT / "src"
    tools = ROOT / "tools/v3"
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))
    spec = importlib.util.spec_from_file_location(
        "aee_head_adaptation", source_root / "mtare_topo/data/aee_head_adaptation.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


aee = _load_module()


def _sensor_payload(trajectory_id: str = "tiny_tunnel_seed11") -> dict[str, np.ndarray]:
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
    direction = np.zeros((n, 720), dtype=np.uint8)
    direction[:, 0] = 1
    return {
        "direction_target": direction,
        "count_target": np.zeros(n, dtype=np.int64),
        "role_target": np.full(n, 2, dtype=np.int64),
        "exit_count": np.ones(n, dtype=np.int64),
        "heading_count": np.ones(n, dtype=np.int64),
        "support_z_m": np.zeros(n, dtype=np.float64),
        "frame_id": sensor["frame_id"].copy(),
        "raw_frame_index": sensor["raw_frame_index"].copy(),
    }


def _write_fixture(tmp_path: Path, *, sensor=None, teacher=None, record_overrides=None):
    sensor_run = tmp_path / "sensor_run"
    teacher_run = tmp_path / "teacher_run"
    sensor_path = sensor_run / "artifacts/trajectories/tiny/sensor.npz"
    teacher_path = teacher_run / "artifacts/teacher_shards/tiny.npz"
    sensor_path.parent.mkdir(parents=True)
    teacher_path.parent.mkdir(parents=True)
    sensor = _sensor_payload() if sensor is None else sensor
    teacher = _teacher_payload(sensor) if teacher is None else teacher
    np.savez_compressed(sensor_path, **sensor)
    np.savez_compressed(teacher_path, **teacher)
    record = {
        "trajectory_id": "tiny_tunnel_seed11",
        "world": "tunnel",
        "split": "train",
        "environment_seed": 11,
        "samples": 600,
        "sensor_shard": "artifacts/trajectories/tiny/sensor.npz",
        "sensor_shard_sha256": sha256(sensor_path),
        "teacher_shard": "artifacts/teacher_shards/tiny.npz",
        "teacher_shard_sha256": sha256(teacher_path),
    }
    if record_overrides:
        record.update(record_overrides)
    (teacher_run / "artifacts").mkdir(exist_ok=True)
    (teacher_run / "artifacts/teacher_manifest.json").write_text(
        json.dumps({"records": [record]}), encoding="utf-8"
    )
    return sensor_run, teacher_run, sensor_path, teacher_path


def test_dataset_reads_normalizes_and_preserves_identity(tmp_path: Path) -> None:
    sensor_run, teacher_run, _, _ = _write_fixture(tmp_path)
    dataset = aee.AEETeacherMultitaskDataset(
        sensor_run, teacher_run, "train", enforce_formal_contract=False
    )
    sample = dataset[0]
    assert len(dataset) == 600
    assert sample["student"].shape == (2, 16, 720)
    assert sample["student"].dtype == np.float32
    assert np.allclose(sample["student"][0], 10.0 / 50.0)
    assert np.all(sample["student"][1] == 1.0)
    assert sample["direction_target"].shape == (720,)
    assert sample["count_target"] == 0
    assert sample["role_target"] == 2
    assert sample["frame_id"] == "tiny_tunnel_seed11:0"
    assert sample["cluster_id"] == "aee:tiny_tunnel_seed11:0"
    assert sample["domain"] == "aee"
    assert sample["headings_robot_deg"] == (0.0,)


@pytest.mark.parametrize("kind", ["sensor_hash", "teacher_hash", "frame", "raw", "count"])
def test_dataset_rejects_identity_and_count_drift(tmp_path: Path, kind: str) -> None:
    sensor = _sensor_payload()
    teacher = _teacher_payload(sensor)
    sensor_run, teacher_run, sensor_path, teacher_path = _write_fixture(
        tmp_path, sensor=sensor, teacher=teacher
    )
    manifest_path = teacher_run / "artifacts/teacher_manifest.json"
    record = json.loads(manifest_path.read_text())["records"][0]
    if kind == "sensor_hash":
        sensor_path.write_bytes(sensor_path.read_bytes() + b"drift")
    elif kind == "teacher_hash":
        teacher_path.write_bytes(teacher_path.read_bytes() + b"drift")
    elif kind == "frame":
        teacher["frame_id"][0] = "other:0"
        np.savez_compressed(teacher_path, **teacher)
        record["teacher_shard_sha256"] = sha256(teacher_path)
    elif kind == "raw":
        teacher["raw_frame_index"][0] = 1
        np.savez_compressed(teacher_path, **teacher)
        record["teacher_shard_sha256"] = sha256(teacher_path)
    else:
        teacher["count_target"][0] = 1
        np.savez_compressed(teacher_path, **teacher)
        record["teacher_shard_sha256"] = sha256(teacher_path)
    manifest_path.write_text(json.dumps({"records": [record]}))
    dataset = aee.AEETeacherMultitaskDataset(
        sensor_run, teacher_run, "train", enforce_formal_contract=False
    )
    with pytest.raises(RuntimeError):
        dataset[0]


def test_dataset_rejects_manifest_path_escape(tmp_path: Path) -> None:
    sensor_run, teacher_run, _, _ = _write_fixture(
        tmp_path, record_overrides={"sensor_shard": "../outside.npz"}
    )
    dataset = aee.AEETeacherMultitaskDataset(
        sensor_run, teacher_run, "train", enforce_formal_contract=False
    )
    with pytest.raises(ValueError, match="escapes"):
        dataset[0]


def test_binary_direction_headings_merges_component_across_zero() -> None:
    direction = np.zeros(720, dtype=np.uint8)
    direction[[718, 719, 0, 1]] = 1
    direction[180:183] = 1
    headings = aee.binary_direction_headings(direction)
    assert len(headings) == 2
    assert min(min(abs(value), abs(360.0 - value)) for value in headings) == pytest.approx(0.0, abs=0.6)
    assert min(abs(value - 90.5) for value in headings) == pytest.approx(0.0, abs=0.6)
    assert aee.binary_direction_headings(np.zeros(720, dtype=np.uint8)) == ()
    assert aee.binary_direction_headings(np.ones(720, dtype=np.uint8)) == (0.0,)


def test_cano_compatible_direction_target_uses_component_centers_and_three_degree_sigma() -> None:
    direction = np.zeros(720, dtype=np.uint8)
    direction[710:] = 1
    direction[:11] = 1
    direction[176:185] = 1
    target = aee.cano_compatible_direction_target(direction)
    headings = aee.binary_direction_headings(direction)
    from mtare_topo.data.cano_sensor_smoke import circular_gaussian_label

    assert np.array_equal(target, circular_gaussian_label(headings))
    assert target.dtype == np.float32
    assert target.shape == (720,)
    assert target[0] == pytest.approx(1.0)
    assert target[180] == pytest.approx(1.0)
    assert float(target.mean()) == pytest.approx(0.04177714, abs=1e-8)


def test_dataset_can_emit_cano_compatible_direction_encoding(tmp_path: Path) -> None:
    sensor = _sensor_payload()
    teacher = _teacher_payload(sensor)
    teacher["direction_target"][:, :21] = 1
    sensor_run, teacher_run, _, _ = _write_fixture(
        tmp_path, sensor=sensor, teacher=teacher
    )
    dataset = aee.AEETeacherMultitaskDataset(
        sensor_run,
        teacher_run,
        "train",
        enforce_formal_contract=False,
        direction_encoding="cano_gaussian_component_centers",
    )
    sample = dataset[0]
    assert sample["direction_encoding"] == "cano_gaussian_component_centers"
    assert sample["headings_robot_deg"] == pytest.approx((5.0,), abs=1e-12)
    assert sample["direction_target"][10] == pytest.approx(1.0)
    assert 0.0 < float(sample["direction_target"].mean()) < 0.1


def test_balanced_sampler_is_exact_deterministic_and_one_pass() -> None:
    sampler = aee.BalancedDomainBatchSampler(cano_size=20, aee_size=6, batch_size=4, seed=7)
    sampler.set_epoch(1)
    batches = list(sampler)
    assert len(batches) == 3
    flat = [index for batch in batches for index in batch]
    aee_indices = [index - 20 for index in flat if index >= 20]
    cano_indices = [index for index in flat if index < 20]
    assert all(sum(index < 20 for index in batch) == 2 for batch in batches)
    assert sorted(aee_indices) == list(range(6))
    assert len(cano_indices) == len(set(cano_indices)) == 6
    repeat = aee.BalancedDomainBatchSampler(20, 6, 4, 7)
    repeat.set_epoch(1)
    assert batches == list(repeat)
    sampler.set_epoch(2)
    assert batches != list(sampler)


def test_balanced_sampler_rejects_invalid_parameters_and_unset_epoch() -> None:
    with pytest.raises(ValueError):
        aee.BalancedDomainBatchSampler(2, 3, 4, 1)
    with pytest.raises(ValueError):
        aee.BalancedDomainBatchSampler(3, 0, 4, 1)
    with pytest.raises(ValueError):
        aee.BalancedDomainBatchSampler(3, 2, 3, 1)
    with pytest.raises(ValueError):
        aee.BalancedDomainBatchSampler(3, 2, 1, 1)
    sampler = aee.BalancedDomainBatchSampler(3, 2, 2, 1)
    with pytest.raises(RuntimeError):
        list(sampler)
    with pytest.raises(ValueError):
        sampler.set_epoch(0)
