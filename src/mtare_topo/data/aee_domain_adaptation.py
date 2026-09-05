"""Frozen sampling and integrity contracts for AEE semantic-domain adaptation."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np


WORLDS = ("tunnel", "garage")
ENVIRONMENT_SEEDS = (11, 23, 37, 53, 71)
RAW_FRAMES_PER_TRAJECTORY = 3000
EFFECTIVE_STRIDE = 5
EFFECTIVE_FRAMES_PER_TRAJECTORY = 600
MINIMUM_TRAJECTORY_DISTANCE_M = 50.0
MAXIMUM_COLLECTION_SIM_SECONDS = 660.0


@dataclass(frozen=True)
class AEEDomainTrajectory:
    index: int
    world: str
    split: str
    environment_seed: int

    @property
    def trajectory_id(self) -> str:
        return f"{self.index:02d}_{self.world}_seed{self.environment_seed}"

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "trajectory_id": self.trajectory_id}


def enumerate_aee_domain_trajectories() -> tuple[AEEDomainTrajectory, ...]:
    values = tuple(
        AEEDomainTrajectory(
            index=index,
            world=world,
            split="train" if world == "tunnel" else "validation",
            environment_seed=seed,
        )
        for index, (world, seed) in enumerate(
            (world, seed) for world in WORLDS for seed in ENVIRONMENT_SEEDS
        )
    )
    if len(values) != 10 or len({item.trajectory_id for item in values}) != 10:
        raise RuntimeError("AEE adaptation trajectory identity drift")
    return values


def effective_frame_indices(raw_count: int = RAW_FRAMES_PER_TRAJECTORY) -> np.ndarray:
    if raw_count != RAW_FRAMES_PER_TRAJECTORY:
        raise ValueError(f"raw trajectory must contain exactly {RAW_FRAMES_PER_TRAJECTORY} frames")
    indices = np.arange(0, raw_count, EFFECTIVE_STRIDE, dtype=np.int64)
    if len(indices) != EFFECTIVE_FRAMES_PER_TRAJECTORY or int(indices[-1]) != 2995:
        raise RuntimeError("effective-frame selection contract drift")
    return indices


def trajectory_distance_m(xyz_m: Sequence[Sequence[float]]) -> float:
    values = np.asarray(xyz_m, dtype=np.float64)
    if values.shape != (RAW_FRAMES_PER_TRAJECTORY, 3) or not np.all(np.isfinite(values)):
        raise ValueError("trajectory positions must be finite [3000,3]")
    return float(np.linalg.norm(np.diff(values, axis=0), axis=1).sum())


def audit_sensor_shard(payload: dict[str, np.ndarray]) -> dict[str, Any]:
    expected = {
        "range_m": (EFFECTIVE_FRAMES_PER_TRAJECTORY, 16, 720),
        "valid_mask": (EFFECTIVE_FRAMES_PER_TRAJECTORY, 16, 720),
        "sensor_xyz_m": (EFFECTIVE_FRAMES_PER_TRAJECTORY, 3),
        "sensor_orientation_xyzw": (EFFECTIVE_FRAMES_PER_TRAJECTORY, 4),
        "yaw_deg": (EFFECTIVE_FRAMES_PER_TRAJECTORY,),
        "stamp_sec": (EFFECTIVE_FRAMES_PER_TRAJECTORY,),
        "raw_frame_index": (EFFECTIVE_FRAMES_PER_TRAJECTORY,),
        "frame_id": (EFFECTIVE_FRAMES_PER_TRAJECTORY,),
    }
    missing = sorted(set(expected) - set(payload))
    mismatches = {
        key: {"actual": list(np.asarray(payload[key]).shape), "expected": list(shape)}
        for key, shape in expected.items()
        if key in payload and np.asarray(payload[key]).shape != shape
    }
    finite_keys = ("range_m", "sensor_xyz_m", "sensor_orientation_xyzw", "yaw_deg", "stamp_sec")
    nonfinite = [key for key in finite_keys if key in payload and not np.all(np.isfinite(payload[key]))]
    index_ok = (
        "raw_frame_index" in payload
        and np.array_equal(np.asarray(payload["raw_frame_index"], dtype=np.int64), effective_frame_indices())
    )
    range_values = np.asarray(payload.get("range_m", []), dtype=np.float32)
    valid = np.asarray(payload.get("valid_mask", []), dtype=bool)
    range_ok = bool(
        range_values.shape == expected["range_m"]
        and valid.shape == expected["valid_mask"]
        and np.all((range_values >= 0.3) & (range_values <= 50.0))
        and np.all(valid.sum(axis=(1, 2)) > 0)
    )
    passed = not missing and not mismatches and not nonfinite and index_ok and range_ok
    return {
        "schema_version": "aee_domain_sensor_shard_audit_v1",
        "passed": passed,
        "missing_arrays": missing,
        "shape_mismatches": mismatches,
        "nonfinite_arrays": nonfinite,
        "raw_frame_indices_exact": bool(index_ok),
        "range_and_valid_contract_passed": range_ok,
        "effective_frames": EFFECTIVE_FRAMES_PER_TRAJECTORY,
    }


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion_yaw_deg(values: Sequence[float]) -> float:
    q = np.asarray(values, dtype=np.float64)
    if q.shape != (4,) or not np.all(np.isfinite(q)):
        raise ValueError("orientation must be finite xyzw")
    norm = float(np.linalg.norm(q))
    if norm <= np.finfo(np.float64).tiny:
        raise ValueError("orientation quaternion must be nonzero")
    x, y, z, w = q / norm
    return math.degrees(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


__all__ = [
    "AEEDomainTrajectory",
    "EFFECTIVE_FRAMES_PER_TRAJECTORY",
    "EFFECTIVE_STRIDE",
    "ENVIRONMENT_SEEDS",
    "MAXIMUM_COLLECTION_SIM_SECONDS",
    "MINIMUM_TRAJECTORY_DISTANCE_M",
    "RAW_FRAMES_PER_TRAJECTORY",
    "WORLDS",
    "audit_sensor_shard",
    "effective_frame_indices",
    "enumerate_aee_domain_trajectories",
    "quaternion_yaw_deg",
    "sha256",
    "trajectory_distance_m",
]
