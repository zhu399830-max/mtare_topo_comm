import numpy as np
import pytest
from mtare_topo.representation.gse_partition_attributes import summarize_partition
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches


def test_exact_legacy_R1_attribute_agreement():
    xyz=np.random.default_rng(5).uniform(-1,1,(500,3))
    frames=np.arange(500)%5
    legacy=extract_surface_patches(xyz,np.ones(500,bool),frames)
    common=summarize_partition(xyz,frames,legacy.point_patch_index)
    for name in ('centers_m','normals','normal_valid','normal_uncertainty','bounds_min_m','bounds_max_m','roughness_m','point_count','frame_support'):
        assert np.array_equal(getattr(common,name),getattr(legacy,name)),name


def test_same_original_returns_not_equal_voxel_weighting():
    xyz=np.array([[0.,0,0],[0,0,0],[3,0,0]])
    r=summarize_partition(xyz,np.array([0,1,4]),np.zeros(3,int))
    assert r.centers_m[0,0]==1.
    assert r.point_count.tolist()==[3] and r.frame_support.sum()==3
    assert not r.normal_valid[0]


def test_permutation_preserves_geometry_and_lift():
    x=np.array([[1.,0,0],[0,1,0],[0,0,0],[9,9,9]])
    f=np.array([0,1,2,3]);g=np.array([7,7,7,10]);p=np.array([3,1,0,2])
    a=summarize_partition(x,f,g);b=summarize_partition(x[p],f[p],g[p])
    assert np.array_equal(a.centers_m,b.centers_m)
    assert np.array_equal(a.point_to_group[p],b.point_to_group)


def test_empty_and_missing_membership():
    r=summarize_partition(np.empty((0,3)),np.array([],int),np.array([],int))
    assert r.centers_m.shape==(0,3)
    with pytest.raises(ValueError):
        summarize_partition(np.zeros((1,3)),np.array([0]),np.array([-1]))


def test_capacity_fails_without_truncation():
    with pytest.raises(OverflowError):
        summarize_partition(np.zeros((4097,3)),np.zeros(4097,int),np.arange(4097))
