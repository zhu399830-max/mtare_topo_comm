import numpy as np
import pytest
from mtare_topo.representation.gse_paired_partitions import registered_returns,paired_partitions


def sensor():
    return np.zeros((5,16,720),np.float32),np.zeros((5,16,720),np.uint8),np.zeros((5,3),np.float32),np.zeros(5,np.float32)


def test_relative_yaw_translation_and_source_index():
    r,v,t,y=sensor();r[0,7,0]=2.;v[0,7,0]=1;t[0]=[1,2,3];y[0]=90
    p,f,index=registered_returns(r,v,t,y)
    assert f.tolist()==[0] and index.tolist()==[7*720]
    np.testing.assert_allclose(p[0],[1,2+2*np.cos(np.deg2rad(-1)),3+2*np.sin(np.deg2rad(-1))],atol=1e-6)


def test_shared_roi_and_empty():
    r,v,t,y=sensor()
    assert len(registered_returns(r,v,t,y)[0])==0
    r[4,7,0]=11;v[4,7,0]=1
    assert len(registered_returns(r,v,t,y)[0])==0
    r[4,7,0]=9
    assert len(registered_returns(r,v,t,y)[0])==1


def test_pose_and_mask_errors():
    r,v,t,y=sensor();t[-1,0]=1
    with pytest.raises(ValueError): registered_returns(r,v,t,y)
    t[-1]=0;v[0,0,0]=2
    with pytest.raises(ValueError): registered_returns(r,v,t,y)


def test_small_population_explicitly_unpartitioned():
    def forbidden(*args): raise AssertionError('native called on small fixture')
    p=np.zeros((3,3),np.float32);f=np.array([0,2,4])
    a,b,r=paired_partitions(p,f,geof_backend=forbidden,partition_backend=forbidden)
    assert a.point_count.sum()==3 and b is None
    assert r['status']=='INSUFFICIENT_GEOMETRY_NEIGHBORS'
    assert r['voxels'].frame_counts.sum()==3


def test_shared_original_mapping_for_both_methods():
    p=np.random.default_rng(3).uniform(-2,2,(100,3)).astype(np.float32);f=np.arange(100)%5
    def geof(x,n,k): return np.ones((len(x),4),np.float32)
    def partition(x,s,t,w,reg): return [list(range(len(x)))],np.zeros(len(x),np.uint32)
    a,b,r=paired_partitions(p,f,geof_backend=geof,partition_backend=partition)
    assert len(a.point_to_group)==len(b.point_to_group)==100
    assert np.array_equal(a.frame_support.sum(0),b.frame_support.sum(0))
    assert r['status']=='PARTITIONED_NOT_SEMANTICALLY_QUALIFIED'


def test_backprojection_exact_legacy_arithmetic():
    from mtare_topo.semantics.primitive_relation_nonlearning import _register_points
    rng=np.random.default_rng(12)
    r=rng.uniform(0,15,(5,16,720)).astype(np.float32)
    v=(rng.random(r.shape)>.2).astype(np.uint8)
    t=rng.normal(size=(5,3)).astype(np.float32);t[-1]=0
    y=rng.uniform(-180,180,5).astype(np.float32);y[-1]=0
    old,frames=_register_points(np.stack((r/50.,v),axis=1),t,y)
    old=old.astype(np.float32);inside=np.linalg.norm(old.astype(float),axis=1)<=10.
    points,slots,index=registered_returns(r,v,t,y)
    assert np.array_equal(points,old[inside])
    assert np.array_equal(slots,frames[inside])
    assert np.array_equal(index,np.flatnonzero(v.ravel())[inside])
