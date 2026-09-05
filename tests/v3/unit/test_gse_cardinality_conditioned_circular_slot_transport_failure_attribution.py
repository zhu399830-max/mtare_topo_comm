import numpy as np

from execute_gse_cardinality_conditioned_circular_slot_transport_failure_attribution_v1 import (
    _align_order,
    _select_safe_threshold,
    angular_error,
    binary_auc,
    circular_bearing,
    local_mode_bearing,
)


def test_circular_and_local_mode_bearings_handle_wrap() -> None:
    mass = np.zeros(180, dtype=np.float64)
    mass[179] = 0.45
    mass[0] = 0.55
    assert angular_error(circular_bearing(mass), 359.1) < 0.02
    assert angular_error(local_mode_bearing(mass), 359.1) < 0.02


def test_alignment_is_permutation_invariant() -> None:
    reference = np.asarray([350.0, 45.0, 170.0])
    candidate = np.asarray([170.5, 349.5, 44.0])
    assert _align_order(reference, candidate) == (1, 2, 0)


def test_binary_auc_handles_ties_and_perfect_order() -> None:
    assert binary_auc(np.asarray([0.0, 0.0, 1.0, 1.0]), np.asarray([False, False, True, True])) == 1.0
    assert binary_auc(np.ones(4), np.asarray([False, False, True, True])) == 0.5


def test_safe_threshold_uses_exit_level_precision() -> None:
    score = np.asarray([0.9, 0.8, 0.1])
    row_tp = np.asarray([2, 2, 0])
    predicted_count = np.asarray([2, 2, 2])
    choice = _select_safe_threshold(score, row_tp, predicted_count, total_targets=6)
    assert choice is not None
    assert choice["accepted_observations"] == 2
    assert choice["precision"] == 1.0
    assert choice["recall"] == 4 / 6
