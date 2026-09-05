from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.representation.gse_causal_episode_runtime import (
    align_baseline_event_logits,
)


def test_frozen_logits_are_aligned_by_identity_not_archive_row() -> None:
    logits = np.asarray([
        [3, 3, 3, 3, 3],
        [1, 1, 1, 1, 1],
        [2, 2, 2, 2, 2],
    ], dtype=np.float32)
    aligned = align_baseline_event_logits(
        [30, 10, 20], ["c", "a", "b"], logits,
        [10, 20, 30], ["a", "b", "c"],
    )
    np.testing.assert_array_equal(aligned[:, 0], [1, 2, 3])


def test_frozen_logit_alignment_rejects_parent_mismatch() -> None:
    with pytest.raises(RuntimeError, match="bijective"):
        align_baseline_event_logits(
            [1, 2], ["a", "wrong"], np.zeros((2, 5), dtype=np.float32),
            [1, 2], ["a", "b"],
        )
