import numpy as np

from execute_gse_axis_anchored_v2_failure_attribution_v1 import (
    binary_auc,
    binary_average_precision,
    circular_assignment_distances,
    oracle_count_localization,
    precision_recall_envelope,
)


def test_binary_ranking_metrics_are_exact_for_perfect_order() -> None:
    score = np.asarray([0.9, 0.8, 0.2, 0.1])
    truth = np.asarray([True, True, False, False])
    assert binary_average_precision(score, truth) == 1.0
    assert binary_auc(score, truth) == 1.0
    metrics = precision_recall_envelope(score, truth)
    assert metrics["maximum_recall_at_precision_0p98"] == 1.0


def test_circular_assignment_handles_wrap_and_permutation() -> None:
    distance = circular_assignment_distances(np.asarray([179, 20]), np.asarray([21, 0]))
    assert sorted(distance.tolist()) == [1, 1]


def test_oracle_count_localization_reports_nearby_peaks() -> None:
    score = np.zeros((2, 180), dtype=np.float64)
    truth = np.zeros((2, 180), dtype=bool)
    truth[0, 0] = True; score[0, 179] = 1.0
    score[1, 30] = 0.5
    result = oracle_count_localization(score, truth)
    assert result["positive_slices"] == 1
    assert result["empty_slices"] == 1
    assert result["matched_fraction_within_0_bins"] == 0.0
    assert result["matched_fraction_within_1_bins"] == 1.0
