import numpy as np
import pytest
from mtare_topo.representation.center_response_lattice_v1 import lattice,cell_indices,decode_positions,SIZE
from mtare_topo.teacher.center_response_partial_targets_v1 import partial_lattice_targets


def test_lattice_is_fixed_readonly_and_keeps_spherical_boundary_cells():
    a=lattice();b=lattice()
    assert a.centers_m.shape==(64000,3)
    assert np.array_equal(a.centers_m,b.centers_m)
    assert not a.centers_m.flags.writeable and not a.intersects_domain.flags.writeable
    assert np.any(a.intersects_domain & (np.linalg.norm(a.centers_m,axis=1)>10))
    assert not a.intersects_domain[0]


@pytest.mark.parametrize('axis',[0,1,2])
@pytest.mark.parametrize('sign',[-1,1])
def test_exact_ball_boundary_reconstructs_with_quarter_meter_offsets(axis,sign):
    points=np.zeros((1,3));points[0,axis]=sign*10
    ids=cell_indices(points);offsets=points-lattice().centers_m[ids]
    assert np.max(np.abs(offsets))<=.25
    assert np.allclose(decode_positions(ids,offsets),points,atol=1e-12)


def test_random_domain_positions_need_only_local_offsets():
    rng=np.random.default_rng(0);p=rng.normal(size=(10000,3));p=p/np.linalg.norm(p,axis=1,keepdims=True)*rng.uniform(0,10,(10000,1))
    ids=cell_indices(p);delta=p-lattice().centers_m[ids]
    assert np.max(np.abs(delta))<=.25+1e-12
    assert np.allclose(decode_positions(ids,delta),p,atol=1e-12)


def test_stacked_same_xy_anchors_remain_separate_and_unknown_is_not_negative():
    p=np.asarray([[1.,2.,-2.],[1.,2.,2.]])
    t=partial_lattice_targets(p,confirmed_negative_mask=np.zeros(SIZE**3,bool))
    assert len(set(t['positive_indices']))==2 and t['negative_count']==0
    assert t['unknown_domain_count']==int(lattice().intersects_domain.sum())-2
    assert not t['source_authenticated'] and not t['full_annotation']


def test_positive_wins_without_inventing_additional_negatives():
    p=np.zeros((1,3));ids=cell_indices(p);n=np.zeros(SIZE**3,bool);n[ids]=True
    t=partial_lattice_targets(p,confirmed_negative_mask=n)
    assert t['positive_count']==1 and t['negative_count']==0


def test_same_cell_capacity_conflict_and_outside_input_fail():
    with pytest.raises(ValueError,match='share one cell'):
        partial_lattice_targets(np.asarray([[.1,.1,.1],[.2,.2,.2]]),confirmed_negative_mask=np.zeros(SIZE**3,bool))
    with pytest.raises(ValueError):cell_indices(np.asarray([[10.01,0.,0.]]))
    with pytest.raises(ValueError):decode_positions(np.asarray([0]),np.zeros((1,3)))


def test_no_observation_or_positive_does_not_fabricate_background():
    t=partial_lattice_targets(np.empty((0,3)),confirmed_negative_mask=np.zeros(SIZE**3,bool))
    assert t['positive_count']==t['negative_count']==0 and np.all(t['state']==-1)
