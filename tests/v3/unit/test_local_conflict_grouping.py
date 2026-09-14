import pytest
from mtare_topo.topology.local_conflict_grouping import group_local_conflicts

def groups(ids,edges):return group_local_conflicts(ids,edges).hypotheses

def test_stronger_groups_survive_weak_conflicting_bridge():
    edges=[(0,1,.99),(2,3,.98),(1,2,.92),(0,3,-.91)]
    h=groups([0,1,2,3,4],edges)
    assert h.groups==((0,1),(2,3)) and h.unresolved_ray_ids==(4,)
    assert h.blocked_positive_pairs==((1,2),)

def test_exact_tie_does_not_pick_winner_by_id_or_order():
    edges=[(0,1,.95),(1,2,.95),(0,2,-.92),(3,4,.95)]
    h=groups(list(range(5)),edges)
    assert h.groups==((3,4),) and h.unresolved_ray_ids==(0,1,2)
    assert h==groups(list(range(5))[::-1],edges[::-1])

def test_strong_group_is_not_dismantled_by_later_tied_ambiguity():
    edges=[(0,1,.99),(1,2,.95),(2,3,.95),(0,3,-.99)]
    assert groups([0,1,2,3],edges).groups==((0,1),)

def test_unknown_bridge_does_not_create_connection():
    h=groups([0,1,2,3],[(0,1,.99),(2,3,.99),(1,2,None)])
    assert h.groups==((0,1),(2,3)) and not h.physical_connection_verified

def test_relabeling_changes_only_ids_not_tie_decision():
    edges=[(0,1,.99),(2,3,.98),(1,2,.92),(0,3,-.91)];mapping={0:40,1:7,2:2,3:99}
    h=groups(list(mapping.values()),[(mapping[a],mapping[b],s) for a,b,s in edges])
    assert set(map(frozenset,h.groups))=={frozenset((40,7)),frozenset((2,99))}

def test_no_accepted_group_violates_any_retained_repulsion():
    edges=[(0,1,1.),(1,2,1.),(0,2,-.9),(2,3,.95),(3,4,.94),(0,4,-.91)]
    h=groups(list(range(5)),edges)
    for a,b in h.repulsive_pairs:assert not any(a in g and b in g for g in h.groups)

def test_invalid_inputs_rejected():
    for edges in [[(0,1,.9),(1,0,-.9)],[(0,2,.9)],[(0,1,float('nan'))]]:
        with pytest.raises(ValueError):groups([0,1],edges)
