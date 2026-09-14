import numpy as np
import pytest
from mtare_topo.semantics.primitive_spatial_evidence import spatial_evidence


def evidence(axis, point=(2,0,0), direction=(1,0,0)):
    return spatial_evidence(np.asarray(axis,float).reshape(-1,3,3), list(range(len(axis))),
                            np.asarray([point],float), np.asarray([direction],float))


def test_parallel_and_stacked_same_direction_different_position():
    base=np.array([[0,0,0],[2,0,0],[4,0,0]],float)
    result=evidence([base,base+[0,3,0],base+[0,0,4]])
    rows=result.candidate_segments
    assert [rows[i].distance_m for i in (0,2,4)]==[0,3,4]
    assert all(r.unoriented_angle_deg==0 for r in rows)
    assert result.membership is None and not result.connectivity_verified


def test_finite_axis_not_infinite_line():
    result=evidence([[[0,0,0],[1,0,0],[2,0,0]]],point=(5,0,0))
    assert min(r.distance_m for r in result.candidate_segments)==3
    assert all(r.projection_clamped for r in result.candidate_segments)


def test_reflection_changes_spatial_alignment_not_angle():
    a=np.array([[[1,0,0],[2,0,0],[3,0,0]]],float)
    original=evidence(a);reflected=evidence(-a)
    assert min(r.distance_m for r in original.candidate_segments)==0
    assert min(r.distance_m for r in reflected.candidate_segments)==3
    assert all(r.unoriented_angle_deg==0 for r in reflected.candidate_segments)


def test_joint_rigid_transform_preserves_metric_and_rotates_offsets():
    a=np.array([[[0,0,0],[1,0,0],[2,0,0]]],float)
    p=np.array([[1,2,3]],float);d=np.array([[1,0,0]],float)
    r=np.array([[0,-1,0],[1,0,0],[0,0,1]],float);t=np.array([8,-2,6])
    before=spatial_evidence(a,[7],p,d)
    after=spatial_evidence(a@r.T+t,[7],p@r.T+t,d@r.T)
    for b,n in zip(before.candidate_segments,after.candidate_segments):
        assert n.distance_m==pytest.approx(b.distance_m)
        assert n.unoriented_angle_deg==pytest.approx(b.unoriented_angle_deg)
        np.testing.assert_allclose(n.offset_xyz_m,np.asarray(b.offset_xyz_m)@r.T)


def test_permutation_and_axis_reversal_preserve_evidence():
    a=np.array([[[0,0,0],[1,0,0],[2,0,0]],[[0,3,0],[1,3,0],[2,3,0]]],float)
    p=np.array([[1,0,0]],float);d=np.array([[1,0,0]],float)
    x=spatial_evidence(a,[5,8],p,d);y=spatial_evidence(a[::-1,::-1],[8,5],p,d)
    key=lambda r:(r.primitive_index,r.distance_m,r.unoriented_angle_deg)
    assert sorted(map(key,x.candidate_segments))==sorted(map(key,y.candidate_segments))


def test_duplicate_axes_keep_all_alternatives_not_forced_membership():
    a=np.array([[[0,0,0],[1,0,0],[2,0,0]]]*2,float)
    x=evidence(a)
    assert len(x.candidate_segments)==4 and x.membership is None


def test_pair_layout_changes_when_only_one_primitive_moves():
    a=np.array([[[0,0,0],[1,0,0],[2,0,0]],[[0,1,0],[1,1,0],[2,1,0]]],float)
    x=evidence(a);a[1]+=[0,0,3];y=evidence(a)
    assert x.relative_centers!=y.relative_centers


def test_empty_is_unknown():
    x=spatial_evidence(np.empty((0,3,3)),[],np.empty((0,3)),np.empty((0,3)))
    assert not x.candidate_segments and x.membership is None


@pytest.mark.parametrize('value',[float('nan'),float('inf')])
def test_invalid_is_not_silently_dropped(value):
    with pytest.raises(ValueError):evidence([[[0,0,0],[value,0,0],[2,0,0]]])


def test_degenerate_is_not_silently_dropped():
    with pytest.raises(ValueError):evidence([[[0,0,0],[0,0,0],[2,0,0]]])
