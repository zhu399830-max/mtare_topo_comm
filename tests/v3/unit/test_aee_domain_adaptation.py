from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from mtare_topo.data.aee_domain_adaptation import (
    EFFECTIVE_FRAMES_PER_TRAJECTORY,
    RAW_FRAMES_PER_TRAJECTORY,
    audit_sensor_shard,
    effective_frame_indices,
    enumerate_aee_domain_trajectories,
    quaternion_yaw_deg,
    trajectory_distance_m,
)


ROOT = Path(__file__).resolve().parents[3]


def valid_payload() -> dict[str, np.ndarray]:
    ranges = np.full((600, 16, 720), 50.0, dtype=np.float32)
    valid = np.zeros((600, 16, 720), dtype=np.uint8)
    valid[:, 8, 0] = 1
    return {
        "range_m": ranges,
        "valid_mask": valid,
        "sensor_xyz_m": np.zeros((600, 3), dtype=np.float64),
        "sensor_orientation_xyzw": np.tile([0.0, 0.0, 0.0, 1.0], (600, 1)),
        "yaw_deg": np.zeros(600, dtype=np.float64),
        "stamp_sec": np.arange(600, dtype=np.float64),
        "raw_frame_index": effective_frame_indices(),
        "frame_id": np.asarray([str(index) for index in range(600)]),
    }


def test_exact_ten_trajectory_world_split_and_seed_contract() -> None:
    values = enumerate_aee_domain_trajectories()
    assert len(values) == 10
    assert [item.world for item in values[:5]] == ["tunnel"] * 5
    assert [item.world for item in values[5:]] == ["garage"] * 5
    assert {item.environment_seed for item in values} == {11, 23, 37, 53, 71}
    assert all(item.split == "train" for item in values[:5])
    assert all(item.split == "validation" for item in values[5:])


def test_effective_sampling_is_exact_and_not_outcome_selective() -> None:
    indices = effective_frame_indices()
    assert RAW_FRAMES_PER_TRAJECTORY == 3000
    assert EFFECTIVE_FRAMES_PER_TRAJECTORY == 600
    assert np.array_equal(indices, np.arange(0, 3000, 5))
    with pytest.raises(ValueError):
        effective_frame_indices(2999)


def test_sensor_shard_audit_rejects_shape_index_and_range_drift() -> None:
    payload = valid_payload()
    assert audit_sensor_shard(payload)["passed"]
    bad = dict(payload)
    bad["raw_frame_index"] = np.arange(600)
    assert not audit_sensor_shard(bad)["passed"]
    bad = dict(payload)
    bad["range_m"] = payload["range_m"].copy()
    bad["range_m"][0, 0, 0] = 51.0
    assert not audit_sensor_shard(bad)["passed"]


def test_distance_and_quaternion_contracts() -> None:
    xyz = np.zeros((3000, 3), dtype=np.float64)
    xyz[:, 0] = np.linspace(0.0, 100.0, 3000)
    assert trajectory_distance_m(xyz) == pytest.approx(100.0)
    assert quaternion_yaw_deg([0.0, 0.0, 0.0, 1.0]) == pytest.approx(0.0)
    with pytest.raises(ValueError):
        trajectory_distance_m(xyz[:-1])


def test_collector_uses_original_mtare_and_has_no_teacher_or_retry() -> None:
    source = (ROOT / "tools/v3/collect_aee_domain_trajectory_v1.py").read_text(encoding="utf-8")
    assert "explore_seeded.launch" in source
    assert "test_id:=0001" in source
    assert "planner_seed:={args.environment_seed}" in source
    assert "teacher_queries\": 0" in source
    assert "retry" not in source.lower()
    assert "c09_reads\": 0" in source and "c10_reads\": 0" in source
