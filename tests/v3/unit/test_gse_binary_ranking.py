"""Synthetic binary ranking and explicit non-calibration boundaries."""
import json

import numpy as np
import pytest

from mtare_topo.evaluation.gse_binary_ranking import binary_ranking


def test_perfect_ranking_can_have_every_prediction_below_fixed_threshold():
    result = binary_ranking([.49, .49, .01, .01], [1, 1, 0, 0])
    assert result["average_precision"] == result["auroc"] == 1
    assert result["confusion"] == dict(tp=0, fp=0, fn=2, tn=2)
    assert result["positive_scores"]["minimum"] == .49
    assert result["negative_scores"]["maximum"] == .01
    assert result["fixed_threshold"] == .5 and not result["threshold_search"]


def test_perfect_distinct_ranking():
    result = binary_ranking([.9, .8, .2, .1], [1, 1, 0, 0])
    assert result["average_precision"] == result["auroc"] == 1
    assert result["score_group_count"] == 4
    assert result["confusion"] == dict(tp=2, fp=0, fn=0, tn=2)


def test_reverse_ranking():
    result = binary_ranking([.1, .2, .8, .9], [1, 1, 0, 0])
    assert result["auroc"] == 0
    assert result["average_precision"] == pytest.approx(5 / 12)


def test_all_tied_score_ap_is_prevalence_auc_half_not_lucky_index_order():
    result = binary_ranking([.5] * 5, [1, 1, 0, 0, 0])
    assert result["average_precision"] == .4
    assert result["auroc"] == .5
    assert result["score_group_count"] == 1
    assert result["confusion"] == dict(tp=2, fp=3, fn=0, tn=0)
    for labels in ([0, 0, 0, 1, 1], [0, 1, 0, 1, 0]):
        assert binary_ranking([.5] * 5, labels) == result


def test_repeated_scores_are_processed_as_whole_groups():
    result = binary_ranking([.9, .9, .8, .2], [1, 0, 1, 0])
    assert result["average_precision"] == pytest.approx(7 / 12)
    assert result["auroc"] == .625
    permuted = binary_ranking([.9, .8, .2, .9], [0, 1, 0, 1])
    assert result == permuted


@pytest.mark.parametrize("labels,prevalence", [([], None), ([0, 0], 0.), ([1, 1], 1.)])
def test_empty_or_single_class_is_explicitly_rank_unidentifiable(labels, prevalence):
    result = binary_ranking([.2] * len(labels), labels)
    assert result["rank_unidentifiable"]
    assert result["average_precision"] is result["auroc"] is None
    assert result["prevalence"] == prevalence
    assert sum(result["confusion"].values()) == len(labels)
    if not labels or labels[0] == 0: assert result["positive_scores"] is None
    if not labels or labels[0] == 1: assert result["negative_scores"] is None
    json.dumps(result, allow_nan=False)


def test_confusion_uses_greater_or_equal_at_exact_point_five():
    result = binary_ranking([.5, .5, .49, .49], [1, 0, 1, 0])
    assert result["confusion"] == dict(tp=1, fp=1, fn=1, tn=1)


def test_exact_zero_one_probabilities_do_not_create_infinite_json_or_bce():
    result = binary_ranking([0., 1.], [1, 0])
    assert result["auroc"] == 0 and "bce" not in result
    json.dumps(result, allow_nan=False)


def test_float32_scores_boolean_labels_supported_without_input_mutation():
    p, y = np.array([.1, .9], np.float32), np.array([False, True])
    before_p, before_y = p.copy(), y.copy()
    result = binary_ranking(p, y)
    np.testing.assert_array_equal(p, before_p); np.testing.assert_array_equal(y, before_y)
    assert result["auroc"] == 1 and result["diagnostic_not_calibration"]


@pytest.mark.parametrize("p,y", [
    ([.1], [0, 1]), ([[.1]], [[0]]), (.1, 0), ([float("nan")], [1]),
    ([float("inf")], [1]), ([-.1], [0]), ([1.1], [1]), ([.2], [2]),
    ([.2], [.3]), ([.2], [float("nan")]), ([".2"], [1]), ([.2], ["1"]),
    ([True], [1]), ([1j], [0]),
])
def test_invalid_input_rejected(p, y):
    with pytest.raises(ValueError): binary_ranking(p, y)


def test_no_near_score_tolerance_or_optimal_threshold_is_introduced():
    p = [.5, np.nextafter(.5, 1.)]
    result = binary_ranking(p, [0, 1])
    assert result["score_group_count"] == 2 and result["auroc"] == 1
    assert "optimal_threshold" not in result and "best_threshold" not in result
