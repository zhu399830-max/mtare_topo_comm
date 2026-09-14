import numpy as np
import pytest
from mtare_topo.evaluation.closed_prism_surface import closed_intervals, declared_surface_candidates
from mtare_topo.evaluation.gse_convex_ray_union import ray_intervals


def cube():
    return np.r_[np.eye(3),-np.eye(3)],np.ones(6)


def test_tangent_retained_while_old_open_interval_is_empty():
    o=np.array([[-2.,0.,0.]]); d=np.array([[1.,1.,0.]])
    result=closed_intervals(o,d,cube())
    np.testing.assert_array_equal(result['intervals'],[[1.,1.]])
    assert result['tangent'][0]
    low,high=ray_intervals(o,d,cube())
    assert np.isposinf(low[0]) and np.isneginf(high[0])


def test_coplanar_and_parallel_outside_are_distinct():
    result=closed_intervals([[-2,1,0],[-2,2,0]],[[1,0,0],[1,0,0]],cube())
    np.testing.assert_array_equal(result['intervals'][0],[1,3])
    assert result['coplanar'].tolist()==[True,False]
    assert result['nonempty'].tolist()==[True,False]


def test_interior_interval_and_plane_order_invariance():
    n,b=cube(); a=closed_intervals([[0,0,0]],[[1,0,0]],(n,b))
    c=closed_intervals([[0,0,0]],[[1,0,0]],(n[::-1],b[::-1]))
    np.testing.assert_array_equal(a['intervals'],[[-1,1]])
    np.testing.assert_array_equal(a['intervals'],c['intervals'])
    assert not a['coplanar'][0]


def test_direction_scaling_changes_parameter_not_geometry():
    result=closed_intervals([[-2,0,0]],[[2,0,0]],cube())
    np.testing.assert_array_equal(result['intervals'],[[.5,1.5]])


def test_declared_reference_has_no_reported_owner_input():
    case=dict(half_axes_m=[2.,2.],shape_exponent=2.,
        program=dict(edges=[dict(points=[[-3,0,0],[3,0,0]])]))
    result=declared_surface_candidates(case,np.array([[0.,0.,0.]]),np.array([[1.,0.,0.]]),np.array([3.],np.float32))
    assert result['possible_exit'].tolist()==[[True]]
    assert not result['surface_uniqueness_certified'][0]


def test_invalid_geometry_and_rays_rejected():
    with pytest.raises(ValueError): closed_intervals([[0,0,0]],[[0,0,0]],cube())
    with pytest.raises(ValueError): closed_intervals([[0,0,0]],[[1,0,0]],(np.zeros((1,3)),[1]))


@pytest.mark.parametrize('distance,possible', [(2.,True),(5.,False)])
def test_coplanar_surface_only_marks_overlapping_observation_cell(distance,possible):
    case=dict(half_axes_m=[2.,2.],shape_exponent=2.,
        program=dict(edges=[dict(points=[[-3,0,0],[3,0,0]])]))
    result=declared_surface_candidates(case,np.array([[0.,2.,0.]]),np.array([[1.,0.,0.]]),
        np.array([distance],np.float32))
    assert result['coplanar'][0,0]
    assert bool(result['possible_surface'][0,0]) == possible
    assert bool(result['uncertain_operands'][0,0]) == possible
