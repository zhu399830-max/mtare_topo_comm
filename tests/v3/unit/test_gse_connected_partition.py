import numpy as np
import pytest
from mtare_topo.representation.gse_connected_partition import split_connected_partition


def run(labels, edges):
    e = np.array(edges, dtype=np.int64).reshape(-1, 2)
    return split_connected_partition(np.array(labels, dtype=np.int64), e[:,0], e[:,1])


def test_disconnected_same_label_split():
    r = run([9]*4, [(0,1),(2,3)])
    assert r.point_to_component.tolist() == [0,0,1,1]
    assert r.component_to_raw.tolist() == [9,9]


def test_cross_label_path_cannot_connect_same_label():
    r = run([0,1,0], [(0,1),(1,2)])
    assert len(r.components) == 3


def test_duplicate_reverse_self_edges_do_not_change_partition():
    a = run([3]*3, [(0,1),(1,2)])
    b = run([3]*3, [(2,1),(1,0),(0,0),(1,2)])
    assert np.array_equal(a.point_to_component, b.point_to_component)


def test_empty_and_isolated():
    assert run([], []).components == ()
    assert run([2,2], []).point_to_component.tolist() == [0,1]


def test_original_mapping_preserved_and_copied():
    labels = np.array([2,2,7]); edges = np.array([0])
    r = split_connected_partition(labels, edges, np.array([1]))
    labels[:] = 0
    assert r.raw_assignment.tolist() == [2,2,7]
    assert np.array_equal(r.component_to_raw[r.point_to_component], r.raw_assignment)
    assert np.array_equal(np.sort(np.concatenate(r.components)), np.arange(3))


@pytest.mark.parametrize('labels,edges', [([-1],[]), ([0],[(0,1)]), ([0],[(-1,0)])])
def test_invalid_rejected(labels, edges):
    with pytest.raises(ValueError):
        run(labels, edges)


def test_float_indices_rejected():
    with pytest.raises(ValueError):
        split_connected_partition(np.array([0]), np.array([0.]), np.array([0.]))
