from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import numpy as np


class StructuralMapDataset:
    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.files: List[Path] = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz samples found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            sample: Dict[str, object] = {
                "partial_map": data["partial_map"].astype(np.float32),
                "teacher_map": data["teacher_map"].astype(np.float32),
                "direction_reachability": data["direction_reachability"].astype(np.float32),
                "direction_distance": data["direction_distance"].astype(np.float32),
                "direction_clearance": data["direction_clearance"].astype(np.float32),
                "center_pose": data["center_pose"].astype(np.float32),
                "stamp_ns": np.int64(data["stamp_ns"]),
                "path": str(path),
            }
        return sample


def collate_numpy(samples: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    keys = [
        "partial_map",
        "teacher_map",
        "direction_reachability",
        "direction_distance",
        "direction_clearance",
    ]
    return {key: np.stack([s[key] for s in samples], axis=0) for key in keys}


def collate_torch(samples: List[Dict[str, object]]):
    import torch

    return {key: torch.from_numpy(value) for key, value in collate_numpy(samples).items()}


class StructuralSurfaceDataset:
    """Dataset reader for the ray-free surface_evidence_v1 contract."""

    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.files: List[Path] = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz samples found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            robot_or_trajectory = str(data["robot_or_trajectory"]) if "robot_or_trajectory" in data.files else str(data["robot"])
            return {
                "input_surface": data["input_surface"].astype(np.float32),
                "teacher_surface": data["teacher_surface"].astype(np.float32),
                "center_pose": data["center_pose"].astype(np.float32),
                "stamp_ns": np.int64(data["stamp_ns"] if "stamp_ns" in data.files else data["timestamp"]),
                "dataset_source": str(data["dataset_source"]) if "dataset_source" in data.files else "",
                "environment": str(data["environment"]) if "environment" in data.files else "",
                "robot": str(data["robot"]) if "robot" in data.files else robot_or_trajectory,
                "robot_or_trajectory": robot_or_trajectory,
                "key": np.int64(data["key"]) if "key" in data.files else np.int64(index),
                "split": str(data["split"]),
                "region_id": str(data["region_id"]) if "region_id" in data.files else str(data["region"]),
                "source_scan_identifier": str(data["source_scan_identifier"]) if "source_scan_identifier" in data.files else "",
                "path": str(path),
            }


def collate_surface_numpy(samples: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    return {key: np.stack([sample[key] for sample in samples], axis=0) for key in ("input_surface", "teacher_surface", "center_pose")}


def collate_surface_torch(samples: List[Dict[str, object]]):
    import torch
    return {key: torch.from_numpy(value) for key, value in collate_surface_numpy(samples).items()}


class TopologicalSemanticDatasetV4:
    """Reader for the fixed, causal topological supervision contract."""

    def __init__(self, root: str | Path, split: str = "train") -> None:
        self.root = Path(root)
        self.split = split
        self.files: List[Path] = sorted((self.root / split).glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"no .npz samples found in {self.root / split}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> Dict[str, object]:
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            required = [
                "student_input_current", "student_input_history", "history_valid_mask",
                "history_time_deltas", "relative_poses", "direction_traversable",
                "direction_reachable_distance", "direction_reachable_area", "exit_sector_soft",
                "exit_sector_binary", "exit_centers", "exit_widths", "exit_lengths",
                "exit_valid_mask", "exit_count", "canonical_topological_role", "canonical_role_valid",
                "continuous_static_scores", "continuous_static_score_mask",
                "continuous_dynamic_scores", "continuous_dynamic_score_mask",
            ]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{path}: missing v4 fields {missing}")
            sample: Dict[str, object] = {
                "student_input_current": data["student_input_current"].astype(np.float32),
                "student_input_history": data["student_input_history"].astype(np.float32),
                "history_valid_mask": data["history_valid_mask"].astype(np.float32),
                "history_time_deltas": data["history_time_deltas"].astype(np.float32),
                "relative_poses": data["relative_poses"].astype(np.float32),
                "direction_traversable": data["direction_traversable"].astype(np.float32),
                "direction_reachable_distance": data["direction_reachable_distance"].astype(np.float32),
                "direction_reachable_area": data["direction_reachable_area"].astype(np.float32),
                "exit_sector_soft": data["exit_sector_soft"].astype(np.float32),
                "exit_sector_binary": data["exit_sector_binary"].astype(np.float32),
                "exit_centers": data["exit_centers"].astype(np.float32),
                "exit_widths": data["exit_widths"].astype(np.float32),
                "exit_lengths": data["exit_lengths"].astype(np.float32),
                "exit_valid_mask": data["exit_valid_mask"].astype(np.float32),
                "exit_count": np.int64(data["exit_count"]),
                "canonical_topological_role": data["canonical_topological_role"].astype(np.float32),
                "canonical_role_valid": data["canonical_role_valid"].astype(np.float32),
                "continuous_static_scores": data["continuous_static_scores"].astype(np.float32),
                "continuous_static_score_mask": data["continuous_static_score_mask"].astype(np.float32),
                "continuous_dynamic_scores": data["continuous_dynamic_scores"].astype(np.float32),
                "continuous_dynamic_score_mask": data["continuous_dynamic_score_mask"].astype(np.float32),
                "world": str(data["world"]), "trajectory": str(data["trajectory"]),
                "split": str(data["split"]), "timestamp": np.int64(data["timestamp"]),
                "center_pose": data["center_pose"].astype(np.float32), "path": str(path),
            }
        return sample


def collate_topological_semantic_v4_numpy(samples: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    keys = [
        "student_input_current", "student_input_history", "history_valid_mask", "history_time_deltas",
        "relative_poses", "direction_traversable", "direction_reachable_distance",
        "direction_reachable_area", "exit_sector_soft", "exit_sector_binary", "exit_centers",
        "exit_widths", "exit_lengths", "exit_valid_mask", "exit_count", "canonical_topological_role",
        "canonical_role_valid", "continuous_static_scores", "continuous_static_score_mask",
        "continuous_dynamic_scores", "continuous_dynamic_score_mask", "center_pose",
    ]
    return {key: np.stack([np.asarray(sample[key]) for sample in samples], axis=0) for key in keys}


def collate_topological_semantic_v4_torch(samples: List[Dict[str, object]]):
    import torch
    return {key: torch.from_numpy(value) for key, value in collate_topological_semantic_v4_numpy(samples).items()}


class TopologicalSemanticDatasetV5(TopologicalSemanticDatasetV4):
    """Reader for v5 role/event teachers; keeps the v4 student contract."""

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = super().__getitem__(index)
        path = self.files[index]
        with np.load(path, allow_pickle=False) as data:
            required = [
                "canonical_topological_role_v2", "canonical_topological_role_v2_valid",
                "continuous_dynamic_scores_v2", "continuous_dynamic_score_mask_v2",
                "dynamic_event_provenance_v2",
            ]
            missing = [key for key in required if key not in data.files]
            if missing:
                raise KeyError(f"{path}: missing v5 fields {missing}")
            sample.update({
                "canonical_topological_role_v2": data["canonical_topological_role_v2"].astype(np.float32),
                "canonical_topological_role_v2_valid": data["canonical_topological_role_v2_valid"].astype(np.float32),
                "continuous_dynamic_scores_v2": data["continuous_dynamic_scores_v2"].astype(np.float32),
                "continuous_dynamic_score_mask_v2": data["continuous_dynamic_score_mask_v2"].astype(np.float32),
                "dynamic_event_provenance_v2": data["dynamic_event_provenance_v2"].astype(np.float32),
            })
        return sample


def collate_topological_semantic_v5_numpy(samples: List[Dict[str, object]]) -> Dict[str, np.ndarray]:
    result = collate_topological_semantic_v4_numpy(samples)
    for key in (
        "canonical_topological_role_v2", "canonical_topological_role_v2_valid",
        "continuous_dynamic_scores_v2", "continuous_dynamic_score_mask_v2",
        "dynamic_event_provenance_v2",
    ):
        result[key] = np.stack([np.asarray(sample[key]) for sample in samples], axis=0)
    return result


def collate_topological_semantic_v5_torch(samples: List[Dict[str, object]]):
    import torch

    return {key: torch.from_numpy(value) for key, value in collate_topological_semantic_v5_numpy(samples).items()}
