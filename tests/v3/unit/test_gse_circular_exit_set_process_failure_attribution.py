from __future__ import annotations

import numpy as np

from execute_gse_circular_exit_set_process_failure_attribution_v1 import (
    _average_precision,
    _optimal_pairs,
    _safe_threshold,
)


def test_optimal_pairs_are_circular_permutation_invariant() -> None:
    first = _optimal_pairs(np.asarray([359.0, 91.0]), np.asarray([90.0, 1.0]))
    second = _optimal_pairs(np.asarray([91.0, 359.0]), np.asarray([1.0, 90.0]))
    assert sorted(round(item[2], 8) for item in first) == [1.0, 2.0]
    assert sorted(round(item[2], 8) for item in second) == [1.0, 2.0]


def test_safe_threshold_groups_ties_and_maximizes_recall() -> None:
    score = np.asarray([0.9, 0.8, 0.8, 0.2])
    label = np.asarray([True, True, True, False])
    result = _safe_threshold(score, label)
    assert result is not None
    assert result["threshold"] == 0.8
    assert result["accepted"] == 3
    assert result["precision"] == 1.0
    assert result["recall"] == 1.0


def test_average_precision_is_one_for_perfect_order() -> None:
    assert _average_precision(np.asarray([3.0, 2.0, 1.0]), np.asarray([True, True, False])) == 1.0
