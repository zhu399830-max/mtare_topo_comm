"""Training adapters for the sealed V1R4 combined Cano/AEE dataset."""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np

from mtare_topo.data.aee_domain_adaptation import sha256
from mtare_topo.data.aee_head_adaptation import (
    binary_direction_headings,
    cano_compatible_direction_target,
)
from mtare_topo.data.phase3_multitask_dataset import RANGE_NORMALIZER_M, ROLE_TO_INDEX


def _safe_child(root: Path, relative: str) -> Path:
    value = Path(relative)
    if value.is_absolute() or ".." in value.parts:
        raise ValueError(f"manifest path escapes its run: {relative}")
    resolved = (root / value).resolve()
    resolved.relative_to(root)
    return resolved


class CorrectiveCanoMultitaskDataset:
    """Read the exact 5,000-frame corrective Cano train or validation split."""

    SPLITS = {"train": "corrective_train", "validation": "corrective_validation"}

    def __init__(self, run_dir: str | Path, split: str, *, enforce_formal_contract: bool = True):
        if split not in self.SPLITS:
            raise ValueError("split must be train or validation")
        self.run_dir = Path(run_dir).resolve()
        self.split = split
        manifest = self.run_dir / "artifacts/cano_manifest.jsonl"
        with manifest.open("r", encoding="utf-8") as stream:
            all_records = [json.loads(line) for line in stream if line.strip()]
        sealed_split = self.SPLITS[split]
        self.records = [item for item in all_records if item.get("split") == sealed_split]
        frame_ids = [str(item.get("frame_id")) for item in self.records]
        parents = {str(item.get("parent_id")) for item in self.records}
        if len(set(frame_ids)) != len(frame_ids):
            raise ValueError(f"duplicate corrective Cano frame ID in {split}")
        if enforce_formal_contract and (len(self.records) != 5000 or len(parents) != 10):
            raise ValueError(f"formal corrective Cano {split} must contain 5000 frames/10 parents")
        forbidden = ("C09", "C10", "unseen_mine", "external_cave")
        if any(any(token in parent for token in forbidden) for parent in parents):
            raise ValueError("forbidden world entered corrective Cano dataset")
        self._groups: dict[str, Any] = {}

    def __len__(self) -> int:
        return len(self.records)

    def _group(self, parent_id: str):
        import zarr

        if parent_id not in self._groups:
            path = self.run_dir / "artifacts/cano_dataset" / self.SPLITS[self.split] / f"{parent_id}.zarr"
            self._groups[parent_id] = zarr.open_group(str(path), mode="r")
        return self._groups[parent_id]

    def __getitem__(self, index: int) -> dict[str, Any]:
        record = self.records[index]
        row = int(record["zarr_row"])
        group = self._group(str(record["parent_id"]))
        range_m = np.asarray(group["range_m"][row], dtype=np.float32)
        valid = np.asarray(group["valid_mask"][row], dtype=np.float32)
        direction = np.asarray(group["exit_target"][row], dtype=np.float32)
        if range_m.shape != (16, 720) or valid.shape != (16, 720) or direction.shape != (720,):
            raise RuntimeError(f"{record['frame_id']}: array shape contract failed")
        if not np.isfinite(range_m).all() or not np.isfinite(direction).all():
            raise RuntimeError(f"{record['frame_id']}: non-finite array")
        student = np.stack((np.clip(range_m, 0.3, RANGE_NORMALIZER_M) / RANGE_NORMALIZER_M, valid))
        return {
            "student": student.astype(np.float32, copy=False),
            "direction_target": direction,
            "count_target": np.int64(int(record["branch_count"]) - 1),
            "role_target": np.int64(ROLE_TO_INDEX[str(record["primary_role"])]),
            "frame_id": str(record["frame_id"]),
            "cluster_id": str(record["cluster_id"]),
            "parent_id": str(record["parent_id"]),
            "headings_robot_deg": tuple(float(value) for value in record["headings_robot_deg"]),
            "domain": "cano",
        }


class CorrectiveAEEMultitaskDataset:
    """Read all 1,000 train-only AEE samples from the combined V1R4 run."""

    def __init__(self, run_dir: str | Path, *, enforce_formal_contract: bool = True):
        self.run_dir = Path(run_dir).resolve()
        manifest = json.loads((self.run_dir / "artifacts/aee_manifest.json").read_text(encoding="utf-8"))
        self.records = list(manifest.get("trajectories", []))
        ids = [str(item.get("trajectory_id")) for item in self.records]
        if len(set(ids)) != len(ids):
            raise ValueError("duplicate corrective AEE trajectory ID")
        if enforce_formal_contract:
            worlds = [str(item.get("world")) for item in self.records]
            seeds = [int(item.get("environment_seed", -1)) for item in self.records]
            if len(self.records) != 10 or worlds.count("tunnel") != 5 or worlds.count("garage") != 5:
                raise ValueError("formal corrective AEE dataset must contain five tunnel and five garage trajectories")
            if seeds[:5] != [11, 23, 37, 53, 71] or seeds[5:] != [11, 23, 37, 53, 71]:
                raise ValueError("formal corrective AEE seed order drift")
            if any(item.get("split") != "corrective_train" for item in self.records):
                raise ValueError("all inspected AEE worlds must remain train-only")
        self._locations = [(record_index, row) for record_index in range(len(self.records)) for row in range(100)]
        if enforce_formal_contract and len(self._locations) != 1000:
            raise ValueError("formal corrective AEE sample count must be 1000")
        self._cache: dict[int, tuple[dict[str, np.ndarray], dict[str, np.ndarray]]] = {}

    def __len__(self) -> int:
        return len(self._locations)

    def _load(self, record_index: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
        if record_index in self._cache:
            return self._cache[record_index]
        record = self.records[record_index]
        sensor_path = _safe_child(self.run_dir, str(record["sensor_shard"]))
        teacher_path = _safe_child(self.run_dir, str(record["teacher_shard"]))
        if sha256(sensor_path) != record["sensor_shard_sha256"]:
            raise RuntimeError(f"sensor shard identity drift: {record['trajectory_id']}")
        if sha256(teacher_path) != record["teacher_shard_sha256"]:
            raise RuntimeError(f"teacher shard identity drift: {record['trajectory_id']}")
        with np.load(sensor_path, allow_pickle=False) as source:
            sensor = {key: np.asarray(source[key]) for key in source.files}
        with np.load(teacher_path, allow_pickle=False) as source:
            teacher = {key: np.asarray(source[key]) for key in source.files}
        required_sensor_shapes = {
            "range_m": (100, 16, 720), "valid_mask": (100, 16, 720),
            "sensor_xyz_m": (100, 3), "sensor_orientation_xyzw": (100, 4),
            "yaw_deg": (100,), "raw_frame_index": (100,), "frame_id": (100,),
            "source_raw_message_index": (100,), "source_registered_message_index": (100,),
            "pair_delta_sec": (100,),
        }
        if any(key not in sensor or np.asarray(sensor[key]).shape != shape for key, shape in required_sensor_shapes.items()):
            raise RuntimeError(f"sensor shard shape contract failed: {record['trajectory_id']}")
        if teacher["direction_target"].shape != (100, 720):
            raise RuntimeError(f"corrective AEE shape failed: {record['trajectory_id']}")
        if not all(np.isfinite(sensor[key]).all() for key in ("range_m", "sensor_xyz_m", "sensor_orientation_xyzw", "yaw_deg", "pair_delta_sec")):
            raise RuntimeError(f"sensor shard finite contract failed: {record['trajectory_id']}")
        ranges = np.asarray(sensor["range_m"])
        valid = np.asarray(sensor["valid_mask"], dtype=bool)
        if np.any((ranges < 0.3) | (ranges > 50.0)) or np.any(valid.sum(axis=(1, 2)) == 0):
            raise RuntimeError(f"sensor range/valid contract failed: {record['trajectory_id']}")
        if np.any(np.asarray(sensor["pair_delta_sec"]) >= 0.1):
            raise RuntimeError(f"sensor pairing threshold failed: {record['trajectory_id']}")
        for key in ("source_raw_message_index", "source_registered_message_index"):
            if np.any(np.diff(np.asarray(sensor[key], dtype=np.int64)) <= 0):
                raise RuntimeError(f"sensor monotonic pairing failed: {record['trajectory_id']}:{key}")
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
        raw_direction = np.asarray(teacher["direction_target"][row], dtype=np.float32)
        headings = binary_direction_headings(raw_direction)
        if len(headings) != int(teacher["exit_count"][row]):
            raise RuntimeError(f"direction component drift: {record['trajectory_id']} row {row}")
        return {
            "student": np.stack((np.clip(range_m, 0.3, RANGE_NORMALIZER_M) / RANGE_NORMALIZER_M, valid)).astype(np.float32, copy=False),
            "direction_target": cano_compatible_direction_target(raw_direction),
            "count_target": np.int64(teacher["count_target"][row]),
            "role_target": np.int64(teacher["role_target"][row]),
            "frame_id": str(teacher["frame_id"][row]),
            "cluster_id": f"aee:{record['trajectory_id']}:{row}",
            "parent_id": f"AEE_{record['world']}",
            "headings_robot_deg": headings,
            "domain": "aee",
        }


class AllSamplesBatchSampler:
    """Shuffle each declared independent sample exactly once per epoch."""

    def __init__(self, size: int, batch_size: int, seed: int):
        if size < 1 or batch_size < 1:
            raise ValueError("positive size and batch size required")
        self.size, self.batch_size, self.seed, self.epoch = int(size), int(batch_size), int(seed), 0

    def set_epoch(self, epoch: int) -> None:
        if epoch < 1:
            raise ValueError("epoch must be positive")
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return (self.size + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[list[int]]:
        if self.epoch < 1:
            raise RuntimeError("set_epoch must be called before iteration")
        order = list(range(self.size))
        random.Random(self.seed + self.epoch).shuffle(order)
        for start in range(0, self.size, self.batch_size):
            yield order[start : start + self.batch_size]


def _mask_index(frame_id: str, seed: int, mask_count: int) -> int:
    digest = hashlib.sha256(f"{int(seed)}\0{frame_id}".encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % mask_count


def mask_matched_cano_sample(cano: dict[str, Any], aee_mask: dict[str, Any]) -> dict[str, Any]:
    student = np.asarray(cano["student"], dtype=np.float32).copy()
    aee_student = np.asarray(aee_mask["student"], dtype=np.float32)
    if student.shape != (2, 16, 720) or aee_student.shape != (2, 16, 720):
        raise ValueError("student view must have shape [2,16,720]")
    matched = (student[1] > 0.5) & (aee_student[1] > 0.5)
    student[0, ~matched] = 1.0
    student[1] = matched.astype(np.float32)
    result = dict(cano)
    result.update(student=student, domain="cano_mask_matched")
    return result


class MaskMatchedCanoValidationDataset:
    """Fixed sparse validation view; AEE labels never enter validation."""

    def __init__(self, cano: CorrectiveCanoMultitaskDataset, masks: CorrectiveAEEMultitaskDataset, seed: int):
        if cano.split != "validation" or len(cano) != 5000 or len(masks) != 1000:
            raise ValueError("formal mask-matched validation requires 5000 Cano and 1000 AEE masks")
        self.cano, self.masks, self.seed = cano, masks, int(seed)

    def __len__(self) -> int:
        return len(self.cano)

    def __getitem__(self, index: int) -> dict[str, Any]:
        cano = self.cano[index]
        mask = self.masks[_mask_index(str(cano["frame_id"]), self.seed, len(self.masks))]
        result = mask_matched_cano_sample(cano, mask)
        result["mask_source_frame_id"] = str(mask["frame_id"])
        return result


def corrective_role_counts(cano: CorrectiveCanoMultitaskDataset, aee: CorrectiveAEEMultitaskDataset) -> np.ndarray:
    counts = np.bincount([ROLE_TO_INDEX[str(item["primary_role"])] for item in cano.records], minlength=3).astype(np.int64)
    for record_index, row in aee._locations:
        _, teacher = aee._load(record_index)
        counts[int(teacher["role_target"][row])] += 1
    if np.any(counts == 0):
        raise RuntimeError(f"corrective training data lacks a role class: {counts.tolist()}")
    return counts


__all__ = [
    "AllSamplesBatchSampler", "CorrectiveAEEMultitaskDataset",
    "CorrectiveCanoMultitaskDataset", "MaskMatchedCanoValidationDataset",
    "corrective_role_counts", "mask_matched_cano_sample",
]
