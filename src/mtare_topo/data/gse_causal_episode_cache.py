"""On-disk frozen spatial cache for twelve-frame GSE episode training."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from mtare_topo.representation.gse_causal_episode_detector import (
    DIRECTIONAL_BINS,
    ENCODER_DIM,
    HISTORY_FRAMES,
)


def compact_global_references(
    global_references: np.ndarray,
    valid_mask: np.ndarray,
    global_frame_index: np.ndarray,
) -> np.ndarray:
    """Map sparse stable global frame ids to one compact frozen cache."""

    references = np.asarray(global_references, dtype=np.int64)
    mask = np.asarray(valid_mask, dtype=np.bool_)
    frames = np.asarray(global_frame_index, dtype=np.int64)
    if references.ndim != 2 or references.shape != mask.shape:
        raise ValueError("global references and mask must be aligned matrices")
    if references.shape[1] != HISTORY_FRAMES:
        raise ValueError("global references must contain twelve columns")
    if frames.ndim != 1 or len(frames) == 0 or np.any(np.diff(frames) <= 0):
        raise ValueError("global frame cache ids must be strictly increasing")
    if np.any(mask & (references < 0)) or np.any(~mask & (references != -1)):
        raise ValueError("global reference padding contract drift")
    result = np.full(references.shape, -1, dtype=np.int32)
    values = references[mask]
    locations = np.searchsorted(frames, values)
    if np.any(locations >= len(frames)) or not np.array_equal(frames[locations], values):
        raise RuntimeError("one or more global references are absent from the frozen cache")
    if len(frames) > np.iinfo(np.int32).max:
        raise ValueError("frozen cache exceeds compact int32 indexing")
    result[mask] = locations.astype(np.int32)
    return result


def materialize_boundary_targets(
    teacher_rows: Sequence[Mapping[str, Any]],
    transition_timing_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    """Align causal transition boundaries without altering pre-boundary labels.

    A transition-labelled observation can precede the geometric boundary by a
    fraction of a metre.  Such a row remains part of the episode-level event
    loss, but it cannot supervise a non-negative backprojection into history.
    """

    index_by_key: dict[tuple[str, int], int] = {}
    for index, row in enumerate(teacher_rows):
        key = (str(row["traversal_id"]), int(row["sequence_index"]))
        if key in index_by_key:
            raise ValueError("duplicate Teacher traversal/sequence key")
        index_by_key[key] = index
    target = np.zeros(len(teacher_rows), dtype=np.float32)
    valid = np.zeros(len(teacher_rows), dtype=np.bool_)
    seen: set[tuple[str, int]] = set()
    pre_boundary = 0
    for timing in transition_timing_rows:
        key = (str(timing["traversal_id"]), int(timing["sequence_index"]))
        if key in seen:
            raise ValueError("duplicate transition timing key")
        seen.add(key)
        if key not in index_by_key:
            raise RuntimeError("transition timing row is absent from the frozen Teacher")
        index = index_by_key[key]
        teacher = teacher_rows[index]
        if str(teacher["event"]) != "geometry_transition":
            raise RuntimeError("transition timing maps to a non-transition Teacher row")
        offset = float(timing["metres_after_boundary"])
        if not np.isfinite(offset) or offset > HISTORY_FRAMES - 1:
            raise RuntimeError("transition boundary lies outside the twelve-frame history")
        if offset < 0.0:
            pre_boundary += 1
            continue
        target[index] = np.float32(offset)
        valid[index] = True
    return target, valid, {
        "timing_rows": len(transition_timing_rows),
        "valid_backprojection_rows": int(valid.sum()),
        "pre_boundary_event_rows": pre_boundary,
    }


class CausalEpisodeEmbeddingDataset:
    """Read one partition from a temporary frozen per-scan embedding cache."""

    def __init__(
        self,
        cache_dir: str | Path,
        baseline_feature_path: str | Path,
        observation_indices: np.ndarray,
    ) -> None:
        self.cache_dir = Path(cache_dir).resolve()
        manifest = json.loads((self.cache_dir / "manifest.json").read_text(encoding="utf-8"))
        if (
            manifest.get("schema_version") != "gse_causal_episode_spatial_cache_v1"
            or manifest.get("history_frames") != HISTORY_FRAMES
            or manifest.get("encoder_dim") != ENCODER_DIM
            or manifest.get("directional_bins") != DIRECTIONAL_BINS
        ):
            raise RuntimeError("causal episode spatial cache schema drift")
        self.pooled = np.load(self.cache_dir / "pooled.npy", mmap_mode="r")
        self.directional = np.load(self.cache_dir / "directional.npy", mmap_mode="r")
        self.references = np.load(self.cache_dir / "compact_reference_index.npy", mmap_mode="r")
        self.history_mask = np.load(self.cache_dir / "valid_history_mask.npy", mmap_mode="r")
        self.episode_id_all = np.load(self.cache_dir / "episode_id.npy", mmap_mode="r")
        self.event_all = np.load(self.cache_dir / "event_index.npy", mmap_mode="r")
        self.boundary_target = np.load(self.cache_dir / "boundary_offset_m.npy", mmap_mode="r")
        self.boundary_valid = np.load(self.cache_dir / "boundary_valid.npy", mmap_mode="r")
        self.baseline = np.load(Path(baseline_feature_path).resolve(), mmap_mode="r")
        self.indices = np.asarray(observation_indices, dtype=np.int64)
        frame_count = int(manifest["unique_frames"])
        observation_count = int(manifest["causal_observations"])
        if (
            self.pooled.shape != (frame_count, ENCODER_DIM)
            or self.pooled.dtype != np.float16
            or self.directional.shape != (frame_count, ENCODER_DIM, DIRECTIONAL_BINS)
            or self.directional.dtype != np.float16
            or self.references.shape != (observation_count, HISTORY_FRAMES)
            or self.history_mask.shape != self.references.shape
            or self.episode_id_all.shape != (observation_count,)
            or self.event_all.shape != (observation_count,)
            or self.boundary_target.shape != (observation_count,)
            or self.boundary_valid.shape != (observation_count,)
            or self.baseline.shape != (observation_count, 146)
            or np.any(self.indices < 0)
            or np.any(self.indices >= observation_count)
            or len(np.unique(self.indices)) != len(self.indices)
        ):
            raise RuntimeError("causal episode cache array contract drift")
        original_episode = np.asarray(self.episode_id_all[self.indices], dtype=np.int64)
        positive = np.unique(original_episode[original_episode >= 0])
        mapping = {int(value): index for index, value in enumerate(positive)}
        self.episode_id = np.asarray([mapping.get(int(value), -1) for value in original_episode], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int) -> dict[str, Any]:
        row = int(self.indices[item])
        references = np.asarray(self.references[row], dtype=np.int64)
        mask = np.asarray(self.history_mask[row], dtype=np.bool_)
        pooled = np.zeros((HISTORY_FRAMES, ENCODER_DIM), dtype=np.float16)
        directional = np.zeros((HISTORY_FRAMES, ENCODER_DIM, DIRECTIONAL_BINS), dtype=np.float16)
        pooled[mask] = self.pooled[references[mask]]
        directional[mask] = self.directional[references[mask]]
        probability = np.array(self.baseline[row, :5], dtype=np.float32, copy=True)
        probability /= probability.sum()
        if np.any(probability <= 0.0) or not np.all(np.isfinite(probability)):
            raise RuntimeError("frozen baseline event probability is invalid")
        return {
            "pooled": pooled,
            "directional": directional,
            "history_mask": mask,
            "baseline_event_logits": np.log(probability),
            "event_index": np.int64(self.event_all[row]),
            "episode_id": np.int64(self.episode_id[item]),
            "boundary_offset_m": np.float32(self.boundary_target[row]),
            "boundary_valid": np.bool_(self.boundary_valid[row]),
            "observation_row": np.int64(row),
        }


__all__ = [
    "CausalEpisodeEmbeddingDataset",
    "compact_global_references",
    "materialize_boundary_targets",
]
