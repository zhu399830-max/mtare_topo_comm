"""Deterministic full-coverage batches that retain place-positive pairs."""

from __future__ import annotations

from collections import defaultdict
import random
from typing import Iterator, Sequence

import numpy as np
from torch.utils.data import Sampler


class PairAwareBatchSampler(Sampler[list[int]]):
    """Visit every sample once while keeping same-place pairs in one batch.

    Valid association identities are partitioned into deterministic two-sample
    units. Unpaired structural samples and samples without a place identity are
    singleton units. Units are shuffled, never their members across a boundary.
    """

    def __init__(self, identity_labels: Sequence[int], *, batch_size: int, seed: int) -> None:
        labels = np.asarray(identity_labels, dtype=np.int64)
        if labels.ndim != 1 or len(labels) == 0:
            raise ValueError("identity labels must be a nonempty vector")
        if batch_size < 2 or batch_size % 2:
            raise ValueError("pair-aware batch size must be an even integer at least two")
        self.labels = labels
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        if not isinstance(epoch, int) or isinstance(epoch, bool) or epoch < 0:
            raise ValueError("epoch must be a nonnegative integer")
        self.epoch = epoch

    def _batches(self) -> list[list[int]]:
        rng = random.Random(self.seed + 1_000_003 * self.epoch)
        grouped: dict[int, list[int]] = defaultdict(list)
        singleton_units: list[list[int]] = []
        for index, identity in enumerate(self.labels):
            if int(identity) < 0:
                singleton_units.append([index])
            else:
                grouped[int(identity)].append(index)
        units: list[list[int]] = singleton_units
        for identity in sorted(grouped):
            indices = grouped[identity]
            rng.shuffle(indices)
            units.extend(indices[start : start + 2] for start in range(0, len(indices), 2))
        rng.shuffle(units)
        batches: list[list[int]] = []
        current: list[int] = []
        for unit in units:
            if current and len(current) + len(unit) > self.batch_size:
                batches.append(current)
                current = []
            current.extend(unit)
        if current:
            batches.append(current)
        flattened = [index for batch in batches for index in batch]
        if len(flattened) != len(self.labels) or len(set(flattened)) != len(self.labels):
            raise RuntimeError("pair-aware batching lost or duplicated a sample")
        return batches

    def __iter__(self) -> Iterator[list[int]]:
        yield from self._batches()

    def __len__(self) -> int:
        return len(self._batches())


__all__ = ["PairAwareBatchSampler"]
