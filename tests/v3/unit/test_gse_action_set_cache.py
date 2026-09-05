from __future__ import annotations

import numpy as np

from mtare_topo.data.gse_action_set_cache import ActionSetNodeDataset


def test_action_set_cache_gathers_only_past_rows(tmp_path) -> None:
    tokens = np.zeros((4, 3, 6, 40), dtype=np.float16)
    for row in range(4):
        tokens[row, ..., 0] = row / 4.0
        tokens[row, ..., 3:] = row + 1
    references = np.asarray([
        [-1, -1, -1, -1, 0], [-1, -1, -1, 0, 1],
        [-1, -1, 0, 1, 2], [-1, 0, 1, 2, 3],
    ], dtype=np.int32)
    mask = references >= 0
    np.save(tmp_path / "raw_tokens.npy", tokens)
    np.save(tmp_path / "history_references.npy", references)
    np.save(tmp_path / "history_mask.npy", mask)
    np.save(tmp_path / "decision_target.npy", np.asarray([1, 1, 0, 2], dtype=np.int8))
    np.save(tmp_path / "decision_episode_id.npy", np.asarray([4, 4, -1, 9], dtype=np.int64))
    np.save(tmp_path / "normalization_mean.npy", np.zeros(40, dtype=np.float32))
    np.save(tmp_path / "normalization_scale.npy", np.ones(40, dtype=np.float32))
    dataset = ActionSetNodeDataset(tmp_path, [0, 1, 2, 3])
    assert dataset.episode_id.tolist() == [0, 0, -1, 1]
    sample = dataset[2]
    assert sample["history_mask"].tolist() == [False, False, True, True, True]
    assert np.all(sample["tokens"][:2] == 0.0)
    assert np.allclose(sample["tokens"][-1, ..., 0], .5)
