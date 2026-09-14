"""Deterministic batches that never split a structural Teacher episode."""

from __future__ import annotations

import heapq
import math
from typing import Iterator

import numpy as np
from torch.utils.data import Sampler


class EpisodePreservingBatchSampler(Sampler[list[int]]):
    """Visit every row once while preserving complete positive bags.

    Structural episodes are first balanced across the minimum number of
    batches needed for the full population, then corridor negatives fill the
    remaining capacity.  With the formal population every batch contains both
    positive and negative evidence, so the multiple-instance loss is defined.
    """

    def __init__(self, episode_id: np.ndarray, *, batch_size: int, seed: int) -> None:
        values = np.asarray(episode_id, dtype=np.int64)
        if values.ndim != 1 or len(values) == 0:
            raise ValueError("episode ids must be a nonempty vector")
        if not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 2:
            raise ValueError("batch_size must be an integer >=2")
        positive = values[values >= 0]
        if len(positive) == 0 or not np.any(values < 0):
            raise ValueError("sampler requires structural episodes and corridor negatives")
        unique = np.unique(positive)
        if not np.array_equal(unique, np.arange(len(unique), dtype=np.int64)):
            raise ValueError("positive episode ids must be compact 0..K-1")
        counts = np.bincount(positive)
        if np.any(counts <= 0) or np.any(counts >= batch_size):
            raise ValueError("every episode must be nonempty and smaller than one batch")
        self.episode_id = values
        self.batch_size = batch_size
        self.seed = int(seed)
        self.epoch = 0
        self._batch_count = math.ceil(len(values) / batch_size)
        if len(unique) < self._batch_count:
            raise ValueError("formal batching requires at least one structural episode per batch")

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def __len__(self) -> int:
        return self._batch_count

    def __iter__(self) -> Iterator[list[int]]:
        generator = np.random.default_rng(self.seed + 1_000_003 * self.epoch)
        episode_groups = [
            np.flatnonzero(self.episode_id == episode).astype(np.int64).tolist()
            for episode in range(int(self.episode_id.max()) + 1)
        ]
        generator.shuffle(episode_groups)
        # Minimum-load placement preserves every bag and spreads structural
        # mass before any corridor row is assigned.
        bins: list[list[int]] = [[] for _ in range(self._batch_count)]
        heap = [(0, index) for index in range(self._batch_count)]
        heapq.heapify(heap)
        for group in episode_groups:
            load, index = heapq.heappop(heap)
            if load + len(group) > self.batch_size:
                raise RuntimeError("episode packing exceeded the frozen batch capacity")
            bins[index].extend(group)
            heapq.heappush(heap, (load + len(group), index))
        if any(not batch for batch in bins):
            raise RuntimeError("episode packing produced a positive-free batch")

        corridor = np.flatnonzero(self.episode_id < 0).astype(np.int64)
        generator.shuffle(corridor)
        offset = 0
        order = np.arange(self._batch_count)
        generator.shuffle(order)
        for index in order:
            capacity = self.batch_size - len(bins[int(index)])
            take = min(capacity, len(corridor) - offset)
            if take > 0:
                bins[int(index)].extend(corridor[offset : offset + take].tolist())
                offset += take
        if offset != len(corridor):
            raise RuntimeError("episode batches cannot hold the complete corridor population")
        if any(not any(self.episode_id[row] < 0 for row in batch) for batch in bins):
            raise RuntimeError("episode packing produced a corridor-free batch")
        for batch in bins:
            generator.shuffle(batch)
        generator.shuffle(bins)
        flat = [row for batch in bins for row in batch]
        if len(flat) != len(self.episode_id) or len(set(flat)) != len(flat):
            raise RuntimeError("episode batches do not visit every row exactly once")
        yield from bins


__all__ = ["EpisodePreservingBatchSampler"]
