"""Dataset and deterministic domain-balanced sampling for AEE head adaptation."""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from mtare_topo.data.aee_domain_adaptation import (
    EFFECTIVE_FRAMES_PER_TRAJECTORY,
    audit_sensor_shard,
    sha256,
)
from mtare_topo.data.cano_sensor_smoke import LABEL_SIGMA_DEG, circular_gaussian_label
from mtare_topo.data.phase3_multitask_dataset import RANGE_NORMALIZER_M


def _safe_child(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"manifest path escapes its run: {relative}")
    resolved = (root / value).resolve()
    resolved.relative_to(root)
    return resolved


def binary_direction_headings(direction_target: np.ndarray) -> tuple[float, ...]:
    active = np.asarray(direction_target, dtype=bool)
    if active.shape != (720,):
        raise ValueError("direction target must have shape [720]")
    if not np.any(active):
        return ()
    if np.all(active):
        return (0.0,)
    first_inactive = int(np.flatnonzero(~active)[0])
    rolled = np.roll(active, -first_inactive)
    headings: list[float] = []
    index = 0
    while index < 720:
        if not rolled[index]:
            index += 1
            continue
        end = index
        while end < 720 and rolled[end]:
            end += 1
        columns = (np.arange(index, end) + first_inactive) % 720
        radians = columns * (2.0 * np.pi / 720.0)
        angle = float(np.arctan2(np.sin(radians).sum(), np.cos(radians).sum()) % (2.0 * np.pi))
        headings.append(angle * 180.0 / np.pi)
        index = end
    return tuple(sorted(0.0 if abs(value - 360.0) < 1e-10 else value for value in headings))


def cano_compatible_direction_target(
    traversable_direction_mask: np.ndarray,
) -> np.ndarray:
    """Encode AEE exit-component centers with the sealed Cano 3-degree target."""

    headings = binary_direction_headings(traversable_direction_mask)
    if not headings:
        raise ValueError("AEE traversable mask must contain at least one exit")
    return circular_gaussian_label(headings, columns=720, sigma_deg=LABEL_SIGMA_DEG)


class AEETeacherMultitaskDataset:
    """Read AEE current-scan inputs paired with objective teacher targets."""

    def __init__(
        self,
        sensor_run: str | Path,
        teacher_run: str | Path,
        split: str,
        *,
        enforce_formal_contract: bool = True,
        direction_encoding: str = "raw_traversable_mask",
    ) -> None:
        if split not in {"train", "validation"}:
            raise ValueError("split must be train or validation")
        self.sensor_run = Path(sensor_run).resolve()
        self.teacher_run = Path(teacher_run).resolve()
        self.split = split
        if direction_encoding not in {
            "raw_traversable_mask",
            "cano_gaussian_component_centers",
        }:
            raise ValueError("unsupported AEE direction encoding")
        self.direction_encoding = direction_encoding
        manifest = json.loads(
            (self.teacher_run / "artifacts/teacher_manifest.json").read_text(encoding="utf-8")
        )
        records = manifest.get("records")
        if not isinstance(records, list):
            raise ValueError("teacher manifest records must be a list")
        self.records = [item for item in records if item.get("split") == split]
        if not self.records:
            raise ValueError(f"teacher manifest has no {split} records")
        trajectory_ids = [str(item.get("trajectory_id")) for item in self.records]
        if len(set(trajectory_ids)) != len(trajectory_ids):
            raise ValueError("duplicate trajectory ID in teacher manifest")
        if enforce_formal_contract:
            expected_world = "tunnel" if split == "train" else "garage"
            if len(self.records) != 5 or any(item.get("world") != expected_world for item in self.records):
                raise ValueError(f"formal {split} split must contain five {expected_world} trajectories")
            if {int(item.get("environment_seed", -1)) for item in self.records} != {11, 23, 37, 53, 71}:
                raise ValueError(f"formal {split} environment-seed set drift")
            if any(int(item.get("samples", -1)) != EFFECTIVE_FRAMES_PER_TRAJECTORY for item in self.records):
                raise ValueError("formal teacher shard sample count drift")
        self._locations: list[tuple[int, int]] = []
        for record_index, item in enumerate(self.records):
            samples = int(item.get("samples", EFFECTIVE_FRAMES_PER_TRAJECTORY))
            self._locations.extend((record_index, row) for row in range(samples))
        if enforce_formal_contract and len(self._locations) != 3000:
            raise ValueError(f"formal {split} sample count must be 3000")
        self._cache: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    def __len__(self) -> int:
        return len(self._locations)

    def _load(self, record_index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if record_index in self._cache:
            return self._cache[record_index]
        record = self.records[record_index]
        sensor_path = _safe_child(self.sensor_run, str(record["sensor_shard"]))
        teacher_path = _safe_child(self.teacher_run, str(record["teacher_shard"]))
        if sha256(sensor_path) != record["sensor_shard_sha256"]:
            raise RuntimeError(f"sensor shard identity drift: {record['trajectory_id']}")
        if sha256(teacher_path) != record["teacher_shard_sha256"]:
            raise RuntimeError(f"teacher shard identity drift: {record['trajectory_id']}")
        with np.load(sensor_path, allow_pickle=False) as source:
            sensor = {key: np.asarray(source[key]) for key in source.files}
        with np.load(teacher_path, allow_pickle=False) as source:
            teacher = {key: np.asarray(source[key]) for key in source.files}
        audit = audit_sensor_shard(sensor)
        if not audit["passed"]:
            raise RuntimeError(f"sensor shard contract failed: {record['trajectory_id']}")
        samples = int(record["samples"])
        if teacher.get("direction_target", np.empty(0)).shape != (samples, 720):
            raise RuntimeError(f"teacher direction shape failed: {record['trajectory_id']}")
        for key in ("count_target", "role_target", "exit_count", "frame_id", "raw_frame_index"):
            if teacher.get(key, np.empty(0)).shape != (samples,):
                raise RuntimeError(f"teacher {key} shape failed: {record['trajectory_id']}")
        if not np.array_equal(sensor["frame_id"].astype(str), teacher["frame_id"].astype(str)):
            raise RuntimeError(f"sensor/teacher frame identity failed: {record['trajectory_id']}")
        if not np.array_equal(sensor["raw_frame_index"], teacher["raw_frame_index"]):
            raise RuntimeError(f"sensor/teacher raw index failed: {record['trajectory_id']}")
        if not np.array_equal(teacher["count_target"] + 1, teacher["exit_count"]):
            raise RuntimeError(f"teacher count identity failed: {record['trajectory_id']}")
        self._cache[record_index] = (sensor, teacher)
        return sensor, teacher

    def __getitem__(self, index: int) -> dict[str, Any]:
        record_index, row = self._locations[index]
        record = self.records[record_index]
        sensor, teacher = self._load(record_index)
        range_m = np.asarray(sensor["range_m"][row], dtype=np.float32)
        valid = np.asarray(sensor["valid_mask"][row], dtype=np.float32)
        traversable_mask = np.asarray(teacher["direction_target"][row], dtype=np.float32)
        direction = (
            cano_compatible_direction_target(traversable_mask)
            if self.direction_encoding == "cano_gaussian_component_centers"
            else traversable_mask
        )
        student = np.stack(
            (np.clip(range_m, 0.3, RANGE_NORMALIZER_M) / RANGE_NORMALIZER_M, valid)
        ).astype(np.float32, copy=False)
        headings = binary_direction_headings(traversable_mask)
        if len(headings) != int(teacher["exit_count"][row]):
            raise RuntimeError(f"direction component drift: {record['trajectory_id']} row {row}")
        return {
            "student": student,
            "direction_target": direction,
            "count_target": np.int64(teacher["count_target"][row]),
            "role_target": np.int64(teacher["role_target"][row]),
            "frame_id": str(teacher["frame_id"][row]),
            "cluster_id": f"aee:{record['trajectory_id']}:{row}",
            "parent_id": f"AEE_{record['world']}",
            "headings_robot_deg": headings,
            "domain": "aee",
            "direction_encoding": self.direction_encoding,
        }


class BalancedDomainBatchSampler:
    """Yield exact half-Cano/half-AEE batches with one AEE pass per epoch."""

    def __init__(self, cano_size: int, aee_size: int, batch_size: int, seed: int) -> None:
        if cano_size < aee_size or aee_size < 1:
            raise ValueError("balanced adaptation requires Cano size >= positive AEE size")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("balanced batch size must be a positive even integer")
        self.cano_size = int(cano_size)
        self.aee_size = int(aee_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return math.ceil(self.aee_size / (self.batch_size // 2))

    def __iter__(self) -> Iterator[list[int]]:
        if self.epoch < 1:
            raise RuntimeError("set_epoch must be called before iteration")
        rng = random.Random(self.seed + self.epoch)
        cano = list(range(self.cano_size))
        aee = list(range(self.aee_size))
        rng.shuffle(cano)
        rng.shuffle(aee)
        cano = cano[: self.aee_size]
        half = self.batch_size // 2
        for start in range(0, self.aee_size, half):
            left = cano[start : start + half]
            right = [self.cano_size + index for index in aee[start : start + half]]
            batch = left + right
            rng.shuffle(batch)
            yield batch
