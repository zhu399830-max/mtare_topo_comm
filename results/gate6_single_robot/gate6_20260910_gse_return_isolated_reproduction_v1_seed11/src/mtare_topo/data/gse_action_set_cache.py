"""Disk-backed five-observation action-set training cache."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset

from mtare_topo.representation.gse_action_set_node import ACTION_HISTORY, RAW_TOKEN_DIM


class ActionSetNodeDataset(Dataset):
    """Gather normalized past-only token histories from one immutable cache."""

    def __init__(self, cache_dir: Path, row_indices: Sequence[int]) -> None:
        self.cache_dir = Path(cache_dir)
        self.tokens = np.load(self.cache_dir / "raw_tokens.npy", mmap_mode="r")
        self.references = np.load(self.cache_dir / "history_references.npy", mmap_mode="r")
        self.history_mask = np.load(self.cache_dir / "history_mask.npy", mmap_mode="r")
        self.target_all = np.load(self.cache_dir / "decision_target.npy", mmap_mode="r")
        self.episode_all = np.load(self.cache_dir / "decision_episode_id.npy", mmap_mode="r")
        self.mean = np.load(self.cache_dir / "normalization_mean.npy")
        self.scale = np.load(self.cache_dir / "normalization_scale.npy")
        self.rows = np.asarray(row_indices, dtype=np.int64)
        if (
            self.tokens.ndim != 4 or self.tokens.shape[1:] != (3, 6, RAW_TOKEN_DIM)
            or self.references.shape != (len(self.tokens), ACTION_HISTORY)
            or self.history_mask.shape != self.references.shape
            or self.target_all.shape != (len(self.tokens),)
            or self.episode_all.shape != (len(self.tokens),)
            or self.mean.shape != (RAW_TOKEN_DIM,) or self.scale.shape != (RAW_TOKEN_DIM,)
            or self.rows.ndim != 1 or len(self.rows) == 0
            or np.any(self.rows < 0) or np.any(self.rows >= len(self.tokens))
        ):
            raise ValueError("action-set cache contract drift")
        source_episode = np.asarray(self.episode_all[self.rows], dtype=np.int64)
        positive = np.unique(source_episode[source_episode >= 0])
        remap = {int(value): index for index, value in enumerate(positive.tolist())}
        self.episode_id = np.asarray(
            [remap.get(int(value), -1) for value in source_episode], dtype=np.int64
        )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = int(self.rows[index])
        mask = np.asarray(self.history_mask[row], dtype=np.bool_)
        references = np.asarray(self.references[row], dtype=np.int64)
        safe = references.copy()
        safe[~mask] = row
        tokens = np.asarray(self.tokens[safe], dtype=np.float32)
        tokens = (tokens - self.mean) / self.scale
        tokens[~mask] = 0.0
        return {
            "tokens": tokens,
            "history_mask": mask.copy(),
            "decision_target": np.int64(self.target_all[row]),
            "episode_id": np.int64(self.episode_id[index]),
            "observation_row": np.int64(row),
        }


__all__ = ["ActionSetNodeDataset"]
