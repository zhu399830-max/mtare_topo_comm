import numpy as np
import pytest
from mtare_topo.representation.center_response_lattice_v1 import cell_indices, SIZE
from mtare_topo.representation.center_response_peaks_v1 import regional_peaks


def inputs():
    return np.zeros(SIZE**3), np.zeros((SIZE**3, 3))


def ids(points):
    return cell_indices(np.asarray(points, dtype=float))


def test_empty_response_is_not_a_node():
    p, o = inputs()
    r = regional_peaks(p, o)
    assert r['positions_m'].shape == (0, 3)
    assert not r['capacity_exceeded']


def test_same_xy_separate_levels_are_preserved():
    p, o = inputs()
    selected = ids([[0, 0, -2], [0, 0, 2]])
    p[selected] = .8
    assert np.array_equal(regional_peaks(p, o)['cell_indices'], selected)


def test_plateau_tie_uses_lowest_index_deterministically():
    p, o = inputs()
    selected = ids([[0, 0, 0], [.5, 0, 0], [1, 0, 0]])
    p[selected] = .7
    r = regional_peaks(p, o)
    assert r['cell_indices'].tolist() == [min(selected)]
    assert r['plateau_sizes'].tolist() == [3]
    assert np.array_equal(r['cell_indices'], regional_peaks(p.copy(), o)['cell_indices'])


def test_higher_neighbour_suppresses_entire_long_plateau():
    p, o = inputs()
    plateau = ids([[x, 0, 0] for x in np.arange(0, 3, .5)])
    higher = ids([[3, 0, 0]])
    p[plateau] = .7
    p[higher] = .8
    assert regional_peaks(p, o)['cell_indices'].tolist() == higher.tolist()


def test_overflow_keeps_all_peaks_not_top32():
    p, o = inputs()
    selected = ids([[x, y, z] for x in (-3, -1, 1, 3)
                    for y in (-3, -1, 1, 3) for z in (-3, -1, 1, 3)])
    p[selected] = .5
    r = regional_peaks(p, o)
    assert r['capacity_exceeded'] and not r['truncated']
    assert len(r['cell_indices']) == 64


@pytest.mark.parametrize('value', [np.nan, np.inf, -.1, 1.1])
def test_invalid_response_fails(value):
    p, o = inputs()
    p[0] = value
    with pytest.raises(ValueError):
        regional_peaks(p, o)
