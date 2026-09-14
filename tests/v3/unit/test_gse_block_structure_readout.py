import numpy as np
import pytest
import torch
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.gse_block_point_encoder import BlockPointEncoder
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout
from mtare_topo.representation.gse_block_structure_loss import LocatedStructureTargets,located_structure_loss


def example():
    torch.manual_seed(11)
    rng=np.random.default_rng(2);x=rng.uniform(-2,2,(80,3)).astype(np.float32)
    b=bind_block_points(x,np.arange(80)%5,np.arange(80)%4)
    encoder=BlockPointEncoder();readout=BlockStructureReadout()
    features=encoder(b);context=torch.randn(4,128,requires_grad=True)
    return encoder,readout,features,context


def target(complete,positive=False,known_direction=True):
    return LocatedStructureTargets(torch.zeros((1 if positive else 0,3)),
        torch.tensor([[10.,0,0]]) if positive else torch.empty((0,3)),
        torch.tensor([[1.,0,0]]) if positive and known_direction else torch.full((1 if positive else 0,3),float('nan')),
        torch.full((1 if positive else 0,),known_direction,dtype=torch.bool),complete,complete)


def test_shapes_bounds_and_block_permutation():
    _,head,f,c=example();a=head(f,c);perm=torch.tensor([2,0,3,1])
    b=head({k:v[perm] for k,v in f.items()},c[perm])
    assert a.anchor_position_m.shape==(32,3) and a.opening_position_m.shape==(64,3)
    assert torch.all(torch.linalg.vector_norm(a.anchor_position_m,dim=1)<=10+1e-5)
    torch.testing.assert_close(a.opening_position_m,b.opening_position_m,atol=1e-5,rtol=1e-5)
    assert 'membership' in a.unknown_fields


def test_known_positive_gradients_reach_point_encoder_not_frozen_context():
    encoder,head,f,c=example();result=located_structure_loss(head(f,c),target(True,True))
    result['total'].backward()
    assert result['counts']==dict(anchor_positive=1,anchor_negative=31,opening_positive=1,opening_negative=63,opening_direction=1)
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in encoder.parameters())
    assert c.grad is None


def test_unknown_entire_observation_no_model_gradient():
    encoder,head,f,c=example();result=located_structure_loss(head(f,c),target(False))
    result['total'].backward()
    assert not result['has_supervision']
    assert all(p.grad is None for p in list(encoder.parameters())+list(head.parameters()))


def test_complete_empty_targets_penalize_all_queries():
    _,head,f,c=example();result=located_structure_loss(head(f,c),target(True))
    assert result['counts']['anchor_negative']==32 and result['counts']['opening_negative']==64
    assert result['terms']['anchor_presence']>0 and result['has_supervision']
    result['total'].backward();assert head.anchor.weight.grad is not None


def test_unknown_direction_nan_excluded():
    _,head,f,c=example();result=located_structure_loss(head(f,c),target(False,True,False))
    assert torch.isfinite(result['total']) and result['counts']['opening_direction']==0
    assert result['counts']['opening_negative']==0


def test_empty_input_and_capacity_fail():
    _,head,f,c=example();empty=head({k:v[:0] for k,v in f.items()},c[:0])
    assert not empty.observation_supported
    result=located_structure_loss(empty,target(True));assert not result['has_supervision']
    result['total'].backward()
    with pytest.raises(ValueError):located_structure_loss(empty,target(True,True))
    t=target(True);t=LocatedStructureTargets(torch.zeros((33,3)),t.openings_m,t.opening_direction,t.direction_known,True,True)
    with pytest.raises(OverflowError):located_structure_loss(head(f,c),t)
