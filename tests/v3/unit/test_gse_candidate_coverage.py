import numpy as np
import pytest

from mtare_topo.evaluation.gse_candidate_coverage import candidate_coverage


def score(x, y, complete=True):
    return candidate_coverage(np.asarray(x, dtype=float).reshape(-1, 3),
        np.asarray(y, dtype=float).reshape(-1, 3), complete_region=complete,
        expected_frame='current_sensor', candidate_frame='current_sensor')


def test_duplicate_proposals_cannot_hide_behind_coverage():
    r = score([[0, 0, 0]], [[0, 0, 0]] * 10)
    s = r['scores']['1.0']
    assert s['coverage_recall'] == 1
    assert s['precision'] == .1 and s['false_positives'] == 9
    assert r['candidate_count'] == 10 and not r['scientific_gate_pass']


def test_one_proposal_cannot_recover_two_structures():
    s = score([[0, 0, 0], [1.5, 0, 0]], [[.75, 0, 0]])['scores']['1.0']
    assert s['coverage_recall'] == 1 and s['one_to_one_recall'] == .5


def test_unknown_background_not_negative():
    s = score([[0, 0, 0]], [[0, 0, 0], [8, 0, 0]], False)['scores']['1.0']
    assert s['unmatched_candidates'] == 1
    assert s['false_positives'] is s['precision'] is s['f1'] is None


def test_empty_straight_counts_every_proposal():
    s = score([], [[0, 0, 0], [100, 0, 0]])['scores']['1.0']
    assert s['false_positives'] == 2 and s['f1'] == 0
    assert s['one_to_one_recall'] is None


def test_empty_predictions_miss_targets_and_no_fabricated_error():
    s = score([[0, 0, 0]], [])['scores']['1.0']
    assert s['missed'] == 1 and s['coverage_recall'] == 0
    assert s['precision'] is s['matched_position_mae_m'] is None


def test_height_and_fixed_sensitivities_preserved():
    r = score([[0, 0, 0]], [[0, 0, 3]])
    assert r['scores']['1.0']['matched'] == r['scores']['2.0']['matched'] == 0
    assert r['scores']['4.0']['matched_position_mae_m'] == 3


def test_common_rigid_transform_and_permutation_preserve_counts():
    x = np.array([[0., 0., 0.], [5., 2., 3.]])
    y = x + [.2, .3, .1]
    rotation = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
    a = score(x, y)['scores']
    b = score(x[::-1] @ rotation.T + 12, y @ rotation.T + 12)['scores']
    for t in a:
        assert a[t]['matched'] == b[t]['matched']
        assert a[t]['matched_position_mae_m'] == pytest.approx(b[t]['matched_position_mae_m'])


def test_invalid_frame_or_values_fail():
    x = np.empty((0, 3))
    with pytest.raises(ValueError):
        candidate_coverage(x, x, complete_region=True, expected_frame='map', candidate_frame='sensor')
    with pytest.raises(ValueError):
        score([[float('nan'), 0, 0]], [])
    with pytest.raises(ValueError):
        score([], [], complete=1)
