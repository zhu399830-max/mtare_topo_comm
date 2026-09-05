from __future__ import annotations

import numpy as np
import pytest

from mtare_topo.evaluation.gse_causal_geometry_risk_conflict import (
    causal_lag_pairs,
    change_confidence_decomposition,
    fixed_risk_summary,
    hard_negative_tail_summary,
    width_height_change_score,
)


def test_causal_lag_pairs_preserve_traversal_partition_and_validity() -> None:
    current, past = causal_lag_pairs(
        ("a", "a", "a", "b", "b", "b"),
        (0, 1, 2, 0, 1, 2),
        (0, 0, 0, 1, 1, 1),
        (True, True, True, False, True, True),
        lag_steps=2,
    )
    assert current.tolist() == [2]
    assert past.tolist() == [0]


def test_causal_lag_pairs_reject_duplicate_keys() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        causal_lag_pairs(("a", "a"), (0, 0), (0, 0), (True, True), lag_steps=1)


def test_width_height_score_ignores_slope_and_curvature() -> None:
    geometry = np.asarray([[1.0, 2.0, 100.0, 100.0], [4.0, 3.0, 0.0, 0.0]])
    score = width_height_change_score(geometry, np.asarray([1]), np.asarray([0]))
    assert score.tolist() == [3.0]


def test_fixed_risk_summary_freezes_fit_threshold_and_counts_identities() -> None:
    # Five fit corridors fix the higher 99th-percentile threshold at 0.5.
    score = (0.1, 0.2, 0.3, 0.4, 0.5, 0.1, 0.6, 0.7, 0.8)
    current = tuple(range(9))
    event = (0, 0, 0, 0, 0, 0, 0, 4, 4)
    identity = (-1, -1, -1, -1, -1, -1, -1, 10, 11)
    partition = (0, 0, 0, 0, 0, 1, 1, 1, 1)
    summary = fixed_risk_summary(
        score, current, event, identity, partition, lag_steps=4
    )
    assert summary.fit_corridor_threshold == 0.5
    assert summary.selection_false_positive_rate == 0.5
    assert summary.accepted_change_frames == 2
    assert summary.covered_change_identities == 2
    assert summary.total_change_identities == 2


def test_change_confidence_separates_structural_and_conditional_terms() -> None:
    probability = np.asarray(
        [[0.2, 0.0, 0.0, 0.0, 0.8], [0.6, 0.0, 0.0, 0.1, 0.3]]
    )
    result = change_confidence_decomposition(probability, (4, 4))
    assert result["change_frames"] == 2
    assert result["argmax_counts"] == [1, 0, 0, 0, 1]
    assert np.isclose(result["mean_structural_probability"], 0.6)
    assert np.isclose(result["mean_conditional_change_probability"], 0.875)


def test_hard_negative_tail_attributes_removed_old_transition_rows() -> None:
    result = hard_negative_tail_summary(
        (0.9, 0.8, 0.7, 0.95),
        (0, 0, 0, 4),
        (4, 0, 4, 4),
        (1, 1, 1, 1),
        threshold=0.75,
    )
    assert result["high_score_corridor_rows"] == 2
    assert result["removed_old_transition_rows"] == 1
    assert result["ordinary_corridor_rows"] == 1
    assert result["removed_old_transition_fraction"] == 0.5
