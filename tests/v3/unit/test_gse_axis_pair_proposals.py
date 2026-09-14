import numpy as np
import pytest
from mtare_topo.representation.gse_axis_pair_proposals import axis_pair_proposals


def run(axes):
    x = np.asarray(axes, dtype=float)
    return axis_pair_proposals(x, np.ones(len(x), dtype=bool), coordinate_frame='sensor')


X = [[-2., 0., 0.], [0., 0., 0.], [2., 0., 0.]]
Y = [[0., -2., 0.], [0., 0., 0.], [0., 2., 0.]]


def test_crossing_is_only_hypothesis():
    r = run([X, Y])['pairs'][0]
    assert r['position_m'] == pytest.approx([0, 0, 0])
    assert r['separation_m'] == 0 and not r['connection_confirmed']


def test_stacked_intersection_retains_separation():
    r = run([X, np.asarray(Y)+[0, 0, 4]])['pairs'][0]
    assert r['position_m'] == pytest.approx([0, 0, 2])
    assert r['separation_m'] == 4 and not r['connection_confirmed']


def test_parallel_not_robot_position():
    r = run([X, np.asarray(X)+[0, 3, 0]])['pairs'][0]
    assert r['position_m'] is None and r['status'] == 'PARALLEL_POSITION_UNKNOWN'


def test_segment_extrapolation_exposed():
    r = run([X, np.asarray(Y)+[0, 5, 0]])['pairs'][0]
    assert r['extrapolation_m'] == pytest.approx([0, 3])


def test_rotation_translation_reversal_and_permutation():
    angle = .7
    R = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    x = np.array([X, Y])[::-1, ::-1] @ R.T + [8, 4, 2]
    r = run(x)['pairs'][0]
    assert r['position_m'] == pytest.approx([8, 4, 2])
    assert r['separation_m'] < 1e-12


def test_single_corridor_and_degenerate():
    assert run([X])['pair_count'] == 0
    r = run([X, [[0., 0., 0.]]*3])['pairs'][0]
    assert r['status'] == 'DEGENERATE' and r['position_m'] is None


def test_no_pair_truncation_and_invalid_input():
    r = run([X, Y, np.array(Y)+[1, 0, 0]])
    assert r['pair_count'] == 3
    with pytest.raises(ValueError):
        run([X]*33)
    with pytest.raises(ValueError):
        run([np.full((3, 3), np.nan)])
