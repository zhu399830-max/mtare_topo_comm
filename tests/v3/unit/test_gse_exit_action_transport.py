from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_exit_action_transport import (
    attach_teacher_exit_identities,
    decision_from_visible_exit_count,
    descriptor_transport_pair,
    multiclass_metrics,
)


def test_visible_count_is_conservative_executable_event() -> None:
    assert decision_from_visible_exit_count(np.asarray([1, 2, 3, 4])).tolist() == [2, 0, 1, 1]
    metrics = multiclass_metrics(np.asarray([2, 0, 1, 1]), np.asarray([2, 0, 1, 1]))
    assert metrics["macro_f1"] == 1.0


def test_teacher_identity_is_attached_after_heading_width_assignment() -> None:
    token = np.zeros((1, 6, 40), dtype=np.float32)
    token[:, :, 0] = 0.5
    token[:, :, 2] = 1.0
    token[:, :, 3] = 2.0
    token[0, 0, 1:3] = (1.0, 0.0)
    token[0, 1, 1:3] = (-1.0, 0.0)
    target = {
        "mask": np.asarray([[1, 1, 0, 0, 0, 0]], dtype=bool),
        "heading": np.asarray([[[1, 0], [-1, 0], [0, 0], [0, 0], [0, 0], [0, 0]]], dtype=np.float32),
        "width": np.asarray([[2, 2, 0, 0, 0, 0]], dtype=np.float32),
        "width_mask": np.asarray([[1, 1, 0, 0, 0, 0]], dtype=bool),
        "identity": np.asarray([[11, 22, -1, -1, -1, -1]], dtype=np.int64),
    }
    attached = attach_teacher_exit_identities(token, target)
    assert attached[0, :2].tolist() == [11, 22]


def test_descriptor_transport_preserves_permutation() -> None:
    previous = np.zeros((2, 32), dtype=np.float32)
    previous[0, 0] = 1.0
    previous[1, 1] = 1.0
    current = previous[::-1].copy()
    left, right = descriptor_transport_pair(previous, current)
    assert left.tolist() == [0, 1]
    assert right.tolist() == [1, 0]
