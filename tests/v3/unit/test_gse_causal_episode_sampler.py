from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.data.gse_causal_episode_sampler import EpisodePreservingBatchSampler


def test_sampler_visits_every_row_and_never_splits_episode() -> None:
    # 8 structural bags / 4 batches, plus enough corridor for every batch.
    episode = np.asarray([-1] * 18 + [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7], dtype=np.int64)
    sampler = EpisodePreservingBatchSampler(episode, batch_size=8, seed=5)
    batches = list(sampler)
    assert len(batches) == 4
    assert sorted(row for batch in batches for row in batch) == list(range(len(episode)))
    membership = {}
    for batch_index, batch in enumerate(batches):
        assert len(batch) <= 8
        assert any(episode[row] < 0 for row in batch)
        assert any(episode[row] >= 0 for row in batch)
        for row in batch:
            if episode[row] >= 0:
                membership.setdefault(int(episode[row]), set()).add(batch_index)
    assert all(len(value) == 1 for value in membership.values())


def test_sampler_is_deterministic_per_epoch_and_changes_order() -> None:
    episode = np.asarray([-1] * 18 + [0, 0, 1, 2, 2, 3, 4, 4, 5, 6, 6, 7], dtype=np.int64)
    left = EpisodePreservingBatchSampler(episode, batch_size=8, seed=7)
    right = EpisodePreservingBatchSampler(episode, batch_size=8, seed=7)
    assert list(left) == list(right)
    left.set_epoch(1)
    assert list(left) != list(right)


def test_sampler_rejects_episode_larger_than_batch() -> None:
    with pytest.raises(ValueError, match="smaller"):
        EpisodePreservingBatchSampler(np.asarray([-1, 0, 0, 0]), batch_size=3, seed=0)

