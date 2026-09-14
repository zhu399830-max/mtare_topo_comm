import numpy as np
import pytest
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.gse_axis_patch_evidence import axis_patch_evidence


def patches(points):
    xyz=np.asarray(points,dtype=float)
    return extract_surface_patches(xyz,np.ones(len(xyz),dtype=bool),np.arange(len(xyz))%5)


def evaluate(p, axis=None):
    axis = [[0.,.2,.2],[1.,.2,.2],[2.,.2,.2]] if axis is None else axis
    return axis_patch_evidence(np.asarray([axis]),np.array([True]),p,coordinate_frame='sensor')


def test_observed_end_surface_position_not_axis_crop_endpoint():
    p=patches([[3.,y,z] for y in (.1,.2,.3) for z in (.1,.2,.3)])
    e=evaluate(p)
    assert e.unique_intersection[0,0] and e.within_patch_bounds[0,0]
    np.testing.assert_allclose(e.position_m[0,0],[3.,.2,.2])
    assert e.extrapolation_m[0,0] == 1
    assert not e.whole_tunnel_closure_verified


def test_parallel_sidewall_does_not_fabricate_cap():
    p=patches([[x,1.,z] for x in (.1,.2,.3) for z in (.1,.2,.3)])
    e=evaluate(p)
    assert not e.unique_intersection.any() and not e.within_patch_bounds.any()
    assert np.isnan(e.position_m).all()


def test_off_axis_patch_plane_not_enough():
    p=patches([[3.,y,z] for y in (1.1,1.2,1.3) for z in (.1,.2,.3)])
    e=evaluate(p)
    assert e.unique_intersection.all() and not e.within_patch_bounds.any()


def test_small_obstacle_and_hole_not_claimed_full_closure():
    # Four returns surrounding an unobserved center: box contains the axis,
    # but the extractor cannot prove surface occupancy at that center.
    p=patches([[3.,y,z] for y in (.19,.21) for z in (.19,.21)])
    e=evaluate(p)
    assert e.within_patch_bounds.all()
    assert not e.whole_tunnel_closure_verified


def test_empty_and_degenerate_return_unknown():
    e=evaluate(patches([[3.,.2,.2]]))
    assert not e.unique_intersection.any()
    p=extract_surface_patches(np.empty((0,3)),np.empty(0,dtype=bool),np.empty(0,dtype=int))
    assert evaluate(p).position_m.shape == (1,0,3)


def test_axis_reversal_preserves_intersection():
    p=patches([[3.,y,z] for y in (.1,.2,.3) for z in (.1,.2,.3)])
    e=evaluate(p,[[2.,.2,.2],[1.,.2,.2],[0.,.2,.2]])
    np.testing.assert_allclose(e.position_m[0,0],[3.,.2,.2])
    assert e.extrapolation_m[0,0] == 1
    assert not e.position_m.flags.writeable


def test_nonfinite_supported_axes_rejected():
    with pytest.raises(ValueError):
        evaluate(patches([[3.,.2,.2]]),[[float('nan'),0,0]]*3)


def test_sparse_surface_gap_retained_as_continuous_evidence_not_closed():
    p=patches([[3.,y,z] for y in (.1,.2,.3) for z in (.3,.35,.4)])
    e=evaluate(p)
    assert e.unique_intersection.all() and not e.within_patch_bounds.any()
    assert e.distance_to_patch_bounds_m[0,0] == pytest.approx(.1)
    assert not e.whole_tunnel_closure_verified
    assert not e.distance_to_patch_bounds_m.flags.writeable
