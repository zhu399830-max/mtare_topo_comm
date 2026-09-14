import numpy as np
import pytest
import torch
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.gse_block_point_encoder import BlockPointEncoder


def fixture():
    rng=np.random.default_rng(11)
    return rng.uniform(-2,2,(97,3)).astype(np.float32),np.arange(97)%5,np.arange(97)%3


def test_no_points_lost_and_same_coordinate_units():
    x,f,g=fixture();b=bind_block_points(x,f,g)
    assert np.array_equal(b.xyz_m,x) and np.array_equal(b.frame_index,f)
    assert len(b.point_to_block)==97 and len(b.block_ids)==3
    assert not b.xyz_m.flags.writeable


def test_point_permutation_and_group_relabeling():
    torch.manual_seed(11);model=BlockPointEncoder()
    x,f,g=fixture();a=model(bind_block_points(x,f,g))
    p=np.random.default_rng(5).permutation(len(x))
    # 0->30,1->10,2->20; sorted new groups are old1,old2,old0.
    remap=np.array([30,10,20]);b=model(bind_block_points(x[p],f[p],remap[g[p]]),chunk_size=7)
    torch.testing.assert_close(a['embedding'][[1,2,0]],b['embedding'],rtol=1e-5,atol=1e-6)
    torch.testing.assert_close(a['centers_m'][[1,2,0]],b['centers_m'])


@pytest.mark.parametrize('duplicates',[False,True])
def test_chunk_forward_and_parameter_gradients(duplicates):
    torch.manual_seed(7);model=BlockPointEncoder();x,f,g=fixture()
    if duplicates:x=np.repeat(x[:1],97,axis=0);f[:]=0;g[:]=0
    b=bind_block_points(x,f,g)
    a=model(b,chunk_size=57600)['embedding'];a.sum().backward()
    grads=[p.grad.clone() for p in model.parameters()];model.zero_grad()
    other=model(b,chunk_size=7)['embedding'];other.sum().backward()
    torch.testing.assert_close(a,other,rtol=1e-5,atol=1e-6)
    for p,expected in zip(model.parameters(),grads):
        torch.testing.assert_close(p.grad,expected,rtol=1e-4,atol=2e-5)


def test_empty_single_point_and_degenerate_preserved():
    model=BlockPointEncoder()
    b=bind_block_points(np.empty((0,3),np.float32),np.empty(0,int),np.empty(0,int))
    assert model(b)['embedding'].shape==(0,128)
    b=bind_block_points(np.zeros((1,3),np.float32),np.array([4]),np.array([9]))
    r=model(b);assert r['degenerate'].tolist()==[True]
    assert torch.isfinite(r['embedding']).all() and r['extent_m'].sum()==0


def test_geometry_changes_not_erased_by_equal_center():
    torch.manual_seed(3);model=BlockPointEncoder()
    x=np.array([[-1,0,0],[1,0,0],[0,-1,0],[0,1,0]],np.float32)
    y=x.copy();y[:,2]=[-.5,-.5,.5,.5]
    f=np.zeros(4,int);g=np.zeros(4,int)
    a=model(bind_block_points(x,f,g));b=model(bind_block_points(y,f,g))
    assert torch.equal(a['centers_m'],b['centers_m'])
    assert not torch.allclose(a['embedding'],b['embedding'])


def test_capacity_roi_and_unknown_membership_fail():
    with pytest.raises(ValueError):bind_block_points(np.zeros((57601,3),np.float32),np.zeros(57601,int),np.zeros(57601,int))
    with pytest.raises(ValueError):bind_block_points(np.array([[11,0,0]],np.float32),np.array([0]),np.array([0]))
    with pytest.raises(ValueError):bind_block_points(np.zeros((1,3),np.float32),np.array([0]),np.array([-1]))
