import numpy as np
import pytest
from mtare_topo.representation.gse_spg_extractor import extract_spg


def forbidden(*args):
    raise AssertionError('native backend must not be called')


@pytest.mark.parametrize('n,status',[(0,'EMPTY'),(1,'INSUFFICIENT_GEOMETRY_NEIGHBORS'),(45,'INSUFFICIENT_GEOMETRY_NEIGHBORS')])
def test_small(n,status):
    p=np.zeros((n,3),dtype='float32');p[:,0]=np.arange(n)
    r=extract_spg(p,np.zeros(n,dtype=int),geof_backend=forbidden,partition_backend=forbidden)
    assert r['status']==status and len(r['voxels'].point_to_voxel)==n


def test_capacity():
    with pytest.raises(OverflowError):
        extract_spg(np.zeros((57601,3),dtype='float32'),np.zeros(57601,dtype=int),geof_backend=forbidden,partition_backend=forbidden)


def cloud():
    return np.random.default_rng(0).uniform(-5,5,(128,3)).astype('float32')


def features(x,index,k):
    assert index.shape==(len(x)*45,) and k==45
    assert all(i not in row for i,row in enumerate(index.reshape(-1,45)))
    return np.ones((len(x),4),dtype='float32')


def test_mapping_and_no_all_pairs_backend():
    def partition(f,s,t,w,reg):
        assert len(s)==len(f)*10 and reg==.1 and np.isfinite(w).all()
        return [list(range(len(f)))],np.zeros(len(f),dtype='uint32')
    r=extract_spg(cloud(),np.arange(128)%5,geof_backend=features,partition_backend=partition)
    assert r['original_point_to_component'].shape==(128,)
    assert r['voxels'].frame_counts.sum()==128


def test_nonfinite_native_features_stop():
    with pytest.raises(ValueError,match='features'):
        extract_spg(cloud(),np.arange(128)%5,geof_backend=lambda x,*_: np.full((len(x),4),np.nan,dtype='float32'),partition_backend=forbidden)


def test_duplicate_component_members_stop():
    with pytest.raises(ValueError,match='duplicated'):
        extract_spg(cloud(),np.arange(128)%5,geof_backend=features,
            partition_backend=lambda f,*_: ([list(range(len(f)))+[0]],np.zeros(len(f),dtype='uint32')))
