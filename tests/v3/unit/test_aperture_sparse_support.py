import numpy as np
import pytest
from mtare_topo.teacher.gse_portal_ray_evidence import CausalRaySegments
from mtare_topo.teacher.aperture_spherical_boundary import octahedral_sphere
from mtare_topo.teacher.aperture_component_contract import partition_boundary
from mtare_topo.teacher.aperture_sparse_support import window_crossing_support


def rays(ranges,valid=None,frames=None,origin=None):
    n=len(ranges)
    return CausalRaySegments(np.zeros((n,3)) if origin is None else np.tile(origin,(n,1)),
        np.tile([1.,0.,0.],(n,1)),np.asarray(ranges,dtype=np.float32),np.ones(n,bool) if valid is None else np.asarray(valid,bool),
        np.zeros(n,int) if frames is None else np.asarray(frames,int),4,0.)


def support(r,state=None):
    v,f,a=octahedral_sphere(2)
    if state is None:state=np.ones(len(f),dtype=int)
    p=partition_boundary(a,state,np.full(len(f),-1))
    return window_crossing_support(r,v,f,state,p)


def test_positive_occluded_boundary_and_no_return():
    result=support(rays([12.,9.,10.,50.],[True,True,True,False]))
    assert result.crossing_before_return.tolist()==[True,False,False,False]
    assert result.component_ray_counts.tolist()==[1]
    assert not result.whole_cells_observed and not result.complete_background


def test_storage_ulp_is_not_free_margin():
    almost=np.nextafter(np.float32(10),np.float32(11))
    result=support(rays([almost]))
    assert not result.crossing_before_return.any()


def test_unknown_geometry_stays_unassigned():
    v,f,a=octahedral_sphere(2)
    result=support(rays([12]),np.full(len(f),-1))
    assert result.crossing_before_return[0] and result.ray_component[0]==-1
    assert result.ambiguous_crossings==1


def test_past_origin_motion_and_on_boundary():
    assert support(rays([9.5],origin=[1.,0.,0.])).component_ray_counts.tolist()==[1]
    assert not support(rays([12.],origin=[10.,0.,0.])).crossing_before_return.any()


def test_future_evidence_rejected_and_prefix_stable():
    with pytest.raises(ValueError):support(rays([12.],frames=[5]))
    short=support(rays([12.,9.],frames=[0,1]));long=support(rays([12.,9.,12.],frames=[0,1,2]))
    np.testing.assert_array_equal(short.ray_component,long.ray_component[:2])
