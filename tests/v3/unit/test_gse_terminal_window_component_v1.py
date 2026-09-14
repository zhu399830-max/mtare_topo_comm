import numpy as np
import pytest

from mtare_topo.teacher.gse_terminal_window_component_v1 import terminal_window_component


def test_straight_component_is_not_a_positive_label():
    r = terminal_window_component([[0, 0, 0], [20, 0, 0]],
                                  endpoint_index=0, center_m=[0, 0, 0])
    assert r['opening_reference_arc_m'] == 10.
    assert r['source_arc_interval_m'] == [0., 10.]
    assert r['membership'] is None and not r['physical_traversability']


def test_reentry_does_not_attach_later_window_opening():
    p = [[0, 0, 0], [12, 0, 0], [12, 5, 0], [0, 5, 0], [-12, 5, 0]]
    r = terminal_window_component(p, endpoint_index=0, center_m=[0, 0, 0])
    assert r['opening_position_m'] == [10., 0., 0.]
    rev = terminal_window_component(p[::-1], endpoint_index=1, center_m=[0, 0, 0])
    np.testing.assert_allclose(rev['opening_position_m'], r['opening_position_m'])


def test_rigid_transform_keeps_component_and_preserves_height():
    p = np.array([[0., 0., 0.], [20., 0., 0.]])
    rotation = np.array([[0., 0., 1.], [1., 0., 0.], [0., 1., 0.]])
    shift = np.array([2., -3., 7.])
    r = terminal_window_component(p @ rotation + shift,
                                  endpoint_index=0, center_m=shift)
    np.testing.assert_allclose(r['opening_position_m'], [2., -3., 17.])


@pytest.mark.parametrize('points,reason', [
    ([[0, 0, 0], [10, 0, 0], [20, 0, 0]], 'AMBIGUOUS_SOURCE_ROI_CROSSING'),
    ([[0, 0, 0], [5, 0, 0]], 'NO_WINDOW_CROSSING_FROM_TERMINAL_COMPONENT'),
    ([[12, 0, 0], [0, 0, 0]], 'SOURCE_ENDPOINT_NOT_INSIDE_ROI'),
])
def test_unknown_cases(points, reason):
    r = terminal_window_component(points, endpoint_index=0, center_m=[0, 0, 0])
    assert r['status'] == 'UNKNOWN' and r['reason'] == reason
    assert r['opening_reference_arc_m'] is None


def test_invalid_source_is_not_repaired():
    with pytest.raises(ValueError):
        terminal_window_component([[0, 0, 0], [0, 0, 0]], endpoint_index=0, center_m=[0, 0, 0])
