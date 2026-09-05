from __future__ import annotations

import numpy as np

from evaluate_gse_circular_exit_set_process_selection_v1 import (
    _best_pairs,
    _decode,
    _select_threshold,
)


def test_continuous_matching_is_circular_one_to_one_and_strict() -> None:
    pairs = _best_pairs(np.asarray([359.5, 4.0]), np.asarray([0.5, 5.5]))
    assert len(pairs) == 2
    assert sorted(round(item[2], 6) for item in pairs) == [1.0, 1.5]
    assert _best_pairs(np.asarray([10.0]), np.asarray([12.1])) == []


def test_count_conditioned_decode_has_no_threshold_or_duplicate_neighbor() -> None:
    mass = np.full((1, 180), 1e-9)
    mass[0, [0, 1, 2]] = [0.45, 0.35, 0.20]
    mass /= mass.sum(axis=1, keepdims=True)
    count = np.asarray([[0.0, 0.0, 1.0, 0.0]])
    decoded = _decode(mass, count, np.zeros((1, 180)))
    assert decoded["count"].tolist() == [3]
    assert decoded["bins"][0, :3].tolist() == [0, 2, 4]


def test_refusal_threshold_uses_grouped_score_and_exit_precision() -> None:
    score = np.asarray([0.9, 0.8, 0.7])
    match = {"row_tp": np.asarray([2, 2, 0])}
    count = np.asarray([2, 2, 2])
    selected = _select_threshold(score, match, count, total_targets=6)
    assert selected is not None
    assert selected["threshold"] == 0.8
    assert selected["precision"] == 1.0
    assert np.isclose(selected["recall"], 4 / 6)
