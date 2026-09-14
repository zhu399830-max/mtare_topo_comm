import numpy as np
import pytest
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.gse_block_context import pool_block_context


def example():
    index=np.array([0,3,4,720,11519,11520,57599])
    xyz=np.zeros((len(index),3),np.float32)
    group=np.array([0,0,0,0,1,1,1])
    blocks=bind_block_points(xyz,index//11520,group)
    memory=np.repeat(np.arange(900,dtype=np.float32)[:,None],128,axis=1)
    return blocks,index,memory,np.ones(57600,bool)


def test_exact_frame_azimuth_boundaries_and_elevation_sharing():
    b,i,m,v=example();r=pool_block_context(b,i,m,v,chunk_size=2)
    assert r['sensor_token_index'].tolist()==[0,0,1,0,179,180,899]
    np.testing.assert_allclose(r['context'][:,0],[.25,(179+180+899)/3])
    assert r['point_count'].tolist()==[4,3]


def test_streaming_matches_full_gather_and_permutation():
    b,i,m,v=example();rng=np.random.default_rng(2);m=rng.normal(size=(900,128)).astype(np.float32)
    a=pool_block_context(b,i,m,v,chunk_size=1);c=pool_block_context(b,i,m,v,chunk_size=57600)
    assert np.array_equal(a['context'],c['context'])
    dense=m[a['sensor_token_index']].astype(float)
    expected=np.stack([dense[b.point_to_block==g].mean(0) for g in range(2)]).astype(np.float32)
    assert np.array_equal(a['context'],expected)
    p=np.array([6,5,4,3,2,1,0])
    other=bind_block_points(b.xyz_m[p],b.frame_index[p],b.point_to_block[p])
    np.testing.assert_allclose(pool_block_context(other,i[p],m,v)['context'],expected,atol=1e-7)


def test_wrong_source_indices_and_invalid_returns_fail():
    b,i,m,v=example();bad=i.copy();bad[0]=bad[1]
    with pytest.raises(ValueError):pool_block_context(b,bad,m,v)
    bad=i.copy();bad[-1]=40000
    with pytest.raises(ValueError):pool_block_context(b,bad,m,v)
    v[i[0]]=False
    with pytest.raises(ValueError):pool_block_context(b,i,m,v)


def test_nonfinite_and_empty():
    b,i,m,v=example();m[0,0]=np.nan
    with pytest.raises(ValueError):pool_block_context(b,i,m,v)
    empty=bind_block_points(np.empty((0,3),np.float32),np.empty(0,int),np.empty(0,int))
    r=pool_block_context(empty,np.empty(0,int),np.zeros((900,128),np.float32),v)
    assert r['context'].shape==(0,128)


def test_source_checkpoint_and_frame_mismatch_rejected():
    from dataclasses import replace
    from mtare_topo.representation.gse_candidate_context import ObservationBinding
    from mtare_topo.representation.gse_block_context import bind_stamped_block_context
    b,i,m,v=example()
    stamp=ObservationBinding('fixture',(1,2,3,4,5),'current_sensor','a'*64,'b'*64)
    bind_stamped_block_context(b,i,m,v,stamp,stamp)
    for bad in (replace(stamp,observation_id='other'),replace(stamp,input_sha256='c'*64),
                replace(stamp,encoder_sha256='d'*64),replace(stamp,frame_indices=(2,3,4,5,6))):
        with pytest.raises(ValueError):bind_stamped_block_context(b,i,m,v,stamp,bad)
