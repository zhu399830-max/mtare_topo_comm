"""Disk-backed dataset for relational exit-token transport training."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
from torch.utils.data import Dataset


class RelationalExitTransportDataset(Dataset):
    """Gather raw token histories and final learned geometry by row identity."""

    def __init__(
        self,
        cache_dir: Path,
        observation_paths: Sequence[Path],
        row_indices: Sequence[int],
    ) -> None:
        self.cache_dir = Path(cache_dir)
        self.tokens = np.load(self.cache_dir / "raw_tokens.npy", mmap_mode="r")
        self.references = np.load(self.cache_dir / "history_references.npy", mmap_mode="r")
        self.history_mask = np.load(self.cache_dir / "history_mask.npy", mmap_mode="r")
        self.target_all = np.load(self.cache_dir / "decision_target.npy", mmap_mode="r")
        self.episode_all = np.load(self.cache_dir / "decision_episode_id.npy", mmap_mode="r")
        self.observation = [np.load(Path(path), mmap_mode="r") for path in observation_paths]
        self.rows = np.asarray(row_indices, dtype=np.int64)
        count = len(self.tokens)
        if (
            self.tokens.shape != (count, 3, 6, 40)
            or self.references.shape != (count, 5)
            or self.history_mask.shape != (count, 5)
            or self.target_all.shape != (count,)
            or self.episode_all.shape != (count,)
            or len(self.observation) != 3
            or any(value.shape != (count, 146) or value.dtype != np.float32 for value in self.observation)
            or self.rows.ndim != 1
            or len(self.rows) == 0
            or np.any((self.rows < 0) | (self.rows >= count))
        ):
            raise ValueError("relational transport dataset contract drift")
        source_episode = np.asarray(self.episode_all[self.rows], dtype=np.int64)
        positive = np.unique(source_episode[source_episode >= 0])
        remap = {int(value): index for index, value in enumerate(positive.tolist())}
        self.episode_id = np.asarray([remap.get(int(value), -1) for value in source_episode], dtype=np.int64)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict:
        row = int(self.rows[index])
        mask = np.asarray(self.history_mask[row], dtype=np.bool_)
        references = np.asarray(self.references[row], dtype=np.int64)
        safe = references.copy()
        safe[~mask] = row
        tokens = np.asarray(self.tokens[safe], dtype=np.float32)
        geometry = np.empty((5, 3, 8), dtype=np.float32)
        for seed, values in enumerate(self.observation):
            selected = np.asarray(values[safe], dtype=np.float32)
            geometry[:, seed] = np.concatenate((selected[:, 5:12], selected[:, 140:141]), axis=1)
        tokens[~mask] = 0.0
        tokens[~mask, :, :, 0] = 0.0
        tokens[~mask, :, :, 2] = 1.0
        tokens[~mask, :, :, 3] = 1.0
        geometry[~mask] = 0.0
        return {
            "tokens": tokens,
            "geometry_context": geometry,
            "history_mask": mask.copy(),
            "decision_target": np.int64(self.target_all[row]),
            "episode_id": np.int64(self.episode_id[index]),
            "observation_row": np.int64(row),
        }


__all__ = ["RelationalExitTransportDataset"]
