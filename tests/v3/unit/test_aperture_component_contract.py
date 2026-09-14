import numpy as np
import pytest
from mtare_topo.teacher.aperture_component_contract import partition_boundary,map_reference_sections

# These are whole-cell state/interface counterexamples, not an assertion that
# the actual spherical mesh/scan adapter is already implemented or qualified.
CHAIN=((1,),(0,2),(1,3),(2,4),(3,))


def test_overlapping_reference_sections_do_not_create_two_instances():
    p=partition_boundary(CHAIN,[1,1,1,0,0],[1,1,1,0,0])
    assert p.geometry_components==((0,1,2),)
    assert map_reference_sections(p,[(0,1),(1,2)])==(0,0)
    assert map_reference_sections(p,[(1,2),(0,),(1,)])==(0,0,0)
    assert not p.training_labels_qualified


def test_separate_parallel_or_stacked_regions_do_not_merge_by_heading():
    p=partition_boundary(CHAIN,[1,1,0,1,1],[1,1,0,1,1])
    assert p.geometry_components==((0,1),(3,4))
    assert p.unresolved_component_pairs==()


def test_three_separated_boundary_openings():
    p=partition_boundary(CHAIN,[1,0,1,0,1],[1,0,1,0,1])
    assert len(p.geometry_components)==3


def test_unknown_bridge_is_neither_merge_nor_confirmed_separation():
    p=partition_boundary(CHAIN,[1,1,-1,1,1],[1,1,-1,1,1])
    assert len(p.geometry_components)==2
    assert p.unresolved_component_pairs==((0,1),)
    assert not p.complete_geometry_partition
    assert map_reference_sections(p,[(1,2)])==(None,)


def test_hidden_geometry_connectivity_not_observation_connectivity():
    p=partition_boundary(CHAIN,[1,1,1,1,1],[1,1,-1,1,1])
    assert len(p.geometry_components)==1 and len(p.observed_components)==2
    assert not p.complete_observed_partition


def test_conflicts_are_exposed_not_repaired():
    p=partition_boundary(CHAIN,[1,0,0,0,1],[1,1,0,0,1])
    assert p.conflicting_cells==(1,)
    assert not p.complete_geometry_partition and not p.training_labels_qualified


def test_cell_permutation_preserves_components():
    g=np.array([1,1,0,1,1]);permutation=[3,0,4,2,1]
    old_to_new={old:new for new,old in enumerate(permutation)}
    adjacency=[tuple(old_to_new[x] for x in CHAIN[old]) for old in permutation]
    p=partition_boundary(adjacency,g[permutation],g[permutation])
    actual={frozenset(permutation[x] for x in group) for group in p.geometry_components}
    assert actual=={frozenset((0,1)),frozenset((3,4))}


def test_invalid_adjacency_rejects():
    with pytest.raises(ValueError):partition_boundary([(1,),()], [1,1],[1,1])
