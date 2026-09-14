import pytest
import torch
from mtare_topo.representation.observed_anchor_objective_v1 import observed_anchor_objective


def test_existence_assignment_and_unknown_gradient():
    p=torch.tensor([[0.,0,0],[1.,0,0],[5.,0,0]],requires_grad=True)
    l=torch.tensor([-4.,4.,0.],requires_grad=True)
    r=observed_anchor_objective(p,l,torch.zeros(1,3),confirmed_negative=torch.zeros(3,dtype=torch.bool))
    assert r['assignment'].tolist()==[1] and r['unknown_count']==2
    r['total'].backward()
    assert l.grad[0]==l.grad[2]==0 and l.grad[1]<0
    assert p.grad[0].abs().sum()==0 and p.grad[2].abs().sum()==0


def test_equal_group_weight_regardless_of_count():
    p=torch.arange(0,10,dtype=torch.float32)[:,None].expand(-1,3)/2.
    logits=torch.zeros(10,requires_grad=True)
    r=observed_anchor_objective(p,logits,torch.zeros(1,3),confirmed_negative=torch.ones(10,dtype=torch.bool))
    assert r['positive_count']==1 and r['negative_count']==9
    r['presence'].backward()
    assert torch.allclose(-logits.grad[0],logits.grad[1:].sum())
    assert not r['negative_mask'][0]


def test_unknown_only_does_not_touch_parameters():
    p=torch.zeros(2,3,requires_grad=True); l=torch.zeros(2,requires_grad=True)
    r=observed_anchor_objective(p,l,torch.empty(0,3),confirmed_negative=torch.zeros(2,dtype=torch.bool))
    assert not r['has_supervision'] and r['unknown_count']==2
    r['total'].backward()
    assert p.grad is None and l.grad is None


def test_negative_only_uses_single_group():
    p=torch.zeros(2,3); l=torch.zeros(2,requires_grad=True)
    r=observed_anchor_objective(p,l,torch.empty(0,3),confirmed_negative=torch.tensor([True,False]))
    r['total'].backward()
    assert l.grad[0]==.5 and l.grad[1]==0


def test_capacity_and_ambiguity_fail():
    with pytest.raises(ValueError,match='capacity'):
        observed_anchor_objective(torch.zeros(1,3),torch.zeros(1),torch.zeros(2,3),confirmed_negative=torch.zeros(1,dtype=torch.bool))
    with pytest.raises(ValueError,match='ambiguous'):
        observed_anchor_objective(torch.zeros(2,3),torch.zeros(2),torch.zeros(1,3),confirmed_negative=torch.zeros(2,dtype=torch.bool))
