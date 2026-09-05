import numpy as np
import pytest

from mtare_topo.evaluation.gse_action_conditioned_state import (
    RobustDeltaScale,
    action_conditioned_state_scores,
    causal_two_block_delta,
    evaluate_state_triggers,
    extract_state_trigger_rows,
    fit_corridor_false_alarm_threshold,
    normalized_delta_score,
)


def test_causal_delta_uses_two_past_blocks_and_never_crosses_traversal():
    values = np.arange(24, dtype=np.float64).reshape(24, 1)
    traversal = np.asarray(["a"] * 12 + ["b"] * 12)
    sequence = np.asarray(list(range(12)) * 2)
    delta, eligible = causal_two_block_delta(values, traversal, sequence, np.asarray([0]))
    assert np.flatnonzero(eligible).tolist() == [11, 23]
    assert delta[11, 0] == pytest.approx(6.0)
    assert delta[23, 0] == pytest.approx(6.0)
    assert np.all(np.isnan(delta[~eligible]))


def test_causal_delta_rejects_noncontiguous_repeated_traversal():
    values = np.zeros((36, 1), dtype=np.float64)
    traversal = np.asarray(["a"] * 12 + ["b"] * 12 + ["a"] * 12)
    sequence = np.asarray(list(range(12)) * 3)
    with pytest.raises(ValueError, match="multiple row blocks"):
        causal_two_block_delta(values, traversal, sequence, np.asarray([0]))


def test_action_conditioned_score_equal_weights_geometry_and_action_groups():
    features = np.zeros((48, 146), dtype=np.float64)
    traversal = np.asarray(["fit"] * 24 + ["selection"] * 24)
    sequence = np.asarray(list(range(24)) * 2)
    fit = np.asarray([True] * 24 + [False] * 24)
    # Nonzero fit variation makes every robust scale data-derived.
    for row in range(48):
        features[row, 5:12] = np.sin(row / 4.0)
        features[row, 141:146] = np.cos(row / 5.0)
    scores, scales = action_conditioned_state_scores(features, traversal, sequence, fit)
    eligible = scores.eligible
    assert set(scales) == {"geometry", "action"}
    np.testing.assert_allclose(
        scores.combined[eligible] ** 2,
        (scores.geometry[eligible] ** 2 + scores.action[eligible] ** 2) / 2.0,
    )


def test_constant_delta_dimension_uses_deterministic_finite_fallback():
    delta = np.zeros((4, 2), dtype=np.float64)
    scale = RobustDeltaScale(np.zeros(2), np.ones(2))
    score = normalized_delta_score(delta, np.ones(4, dtype=np.bool_), scale)
    np.testing.assert_array_equal(score, np.zeros(4))


def test_threshold_is_higher_fit_corridor_quantile_only():
    score = np.asarray([0.0, 1.0, 2.0, 100.0, 999.0])
    event = np.asarray([0, 0, 0, 1, 0])
    fit = np.asarray([True, True, True, True, False])
    eligible = np.ones(5, dtype=np.bool_)
    threshold = fit_corridor_false_alarm_threshold(
        score, event, fit, eligible, corridor_quantile=0.5
    )
    assert threshold == 1.0


def test_trigger_collapse_selects_peak_and_evaluation_matches_unique_episode():
    score = np.asarray([np.nan, 0.0, 2.0, 3.0, 2.5, 0.0, 4.0, 0.0])
    eligible = np.asarray([False, True, True, True, True, True, True, True])
    traversal = np.asarray(["t"] * 8)
    sequence = np.arange(8)
    trigger = extract_state_trigger_rows(score, traversal, sequence, eligible, threshold=2.0)
    assert trigger.tolist() == [3, 6]
    event = np.asarray([0, 0, 1, 1, 1, 0, 0, 0])
    episode = np.asarray([-1, -1, 7, 7, 7, -1, -1, -1])
    identity = np.asarray(["corridor"] * 2 + ["node"] * 3 + ["corridor"] * 3)
    metrics, rows = evaluate_state_triggers(
        score, event, episode, identity, traversal, sequence,
        np.ones(8, dtype=np.bool_), eligible, threshold=2.0,
    )
    assert rows.tolist() == [3, 6]
    assert metrics["structural_trigger_precision"] == 0.5
    assert metrics["structural_episode_recall"] == 1.0
    assert metrics["per_event"]["junction"]["matched_unique_identities"] == 1
