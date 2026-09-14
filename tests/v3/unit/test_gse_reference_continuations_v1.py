"""Synthetic teacher bookkeeping only; no real labels or physical proof."""
import pytest
from mtare_topo.teacher.gse_reference_continuations_v1 import ReferenceSource,reference_continuations


def edge(name,a,b,p,q):
    return ReferenceSource(name,(a,b),(p,q))


def test_degree_two_split_keeps_terminal_to_opening_continuation():
    one=reference_continuations([edge('all','cap','end',(-2,0,0),(10,0,0))])[0]
    split=reference_continuations([edge('near','cap','split',(-2,0,0),(0,0,0)),
        edge('far','split','end',(0,0,0),(10,0,0))])[0]
    assert split.source_ids==('far','near')
    assert {x[0] for x in one.structural_boundaries}=={x[0] for x in split.structural_boundaries}=={'cap','end'}
    assert not split.unresolved_degree_two_nodes and not split.closed_reference_cycle


def test_adjacent_junctions_not_collapsed_by_global_reachability():
    edges=[edge('middle','j1','j2',(0,0,0),(4,0,0)),
        edge('a','a','j1',(-2,0,0),(0,0,0)),edge('b','b','j1',(0,2,0),(0,0,0)),
        edge('c','j2','c',(4,0,0),(6,0,0)),edge('d','j2','d',(4,0,0),(4,2,0))]
    r=reference_continuations(edges)
    assert len(r)==5
    middle=next(x for x in r if x.source_ids==('middle',))
    assert {x[0] for x in middle.structural_boundaries}=={'j1','j2'}
    assert r==reference_continuations(reversed(edges))


@pytest.mark.parametrize('delta',[1e-14,1.])
def test_displaced_connector_is_unresolved_not_free_geometry(delta):
    r=reference_continuations([edge('a','start','join',(-1,0,0),(0,0,0)),
        edge('b','join','end',(delta,0,0),(1,0,0))])
    assert len(r)==2
    assert all(x.unresolved_degree_two_nodes==('join',) for x in r)


def test_stacked_or_coincident_different_nodes_not_spatially_snapped():
    for z in (0.,3.):
        r=reference_continuations([edge('a','a0','a1',(-1,0,0),(0,0,0)),
            edge('b','b0','b1',(0,0,z),(1,0,z))])
        assert len(r)==2  # reference components, NOT a physical separation claim


def test_cycle_without_structure_is_retained_not_made_into_node():
    r=reference_continuations([edge('a','n0','n1',(0,0,0),(1,0,0)),
        edge('b','n1','n0',(1,0,0),(0,0,0))])
    assert len(r)==1 and r[0].closed_reference_cycle
    assert not r[0].structural_boundaries


def test_self_loop_at_junction_retains_both_endpoint_incidences():
    r=reference_continuations([edge('loop','j','j',(0,0,0),(0,0,0)),
        edge('tail','j','end',(0,0,0),(1,0,0))])
    loop=next(x for x in r if x.source_ids==('loop',))
    assert loop.structural_boundaries==(('j','loop',0),('j','loop',1))


def test_duplicate_source_or_nonfinite_geometry_rejected():
    a=edge('a','n0','n1',(0,0,0),(1,0,0))
    with pytest.raises(ValueError):reference_continuations([a,a])
    with pytest.raises(ValueError):reference_continuations([edge('a','x','y',(float('nan'),0,0),(1,0,0))])
