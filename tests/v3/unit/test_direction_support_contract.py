from dataclasses import replace
import pytest
import torch
from mtare_topo.representation.direction_support_contract import DirectionSupportTargets,direction_support_loss,require_direction_training_contract

def target(values,known,qualified=True):
    v=torch.tensor(values,dtype=torch.float32);k=torch.tensor(known,dtype=torch.bool)
    return DirectionSupportTargets((11,22),v,k,tuple('synthetic_evidence' if x else '' for x in k.flatten().tolist()),qualified)

def test_shared_positive_not_mutually_exclusive():
    t=target([[1,1],[0,float('nan')]],[[True,True],[True,False]])
    x=torch.zeros(2,2,requires_grad=True);loss,counts=direction_support_loss(x,(11,22),t);loss.backward()
    assert (x.grad[0]<0).all() and x.grad[1,0]>0 and x.grad[1,1]==0
    assert counts==dict(positive=2,negative=1,unknown=1,shared_positive_rows=1)

def test_unknown_score_has_no_loss_effect():
    t=target([[1,float('nan')]],[[True,False]])
    a,_=direction_support_loss(torch.tensor([[0.,-100.]]),(11,22),t)
    b,_=direction_support_loss(torch.tensor([[0.,100.]]),(11,22),t)
    assert torch.equal(a,b)

def test_missing_query_binding_rejected():
    t=target([[1,0]],[[True,True]])
    with pytest.raises(ValueError,match='columns'):direction_support_loss(torch.zeros(1,2),(22,11),t)
    with pytest.raises(ValueError,match='ray IDs'):direction_support_loss(torch.zeros(1,2),('east','south'),replace(t,query_ray_ids=('east','south')))

def test_no_fake_positive_only_training():
    t=target([[1,float('nan')]],[[True,False]])
    with pytest.raises(ValueError,match='positive and negative'):require_direction_training_contract(torch.zeros(1,2),(11,22),t)

def test_missing_source_or_qualification_rejected():
    t=target([[1,0]],[[True,True]])
    with pytest.raises(ValueError,match='evidence'):direction_support_loss(torch.zeros(1,2),(11,22),replace(t,evidence_refs=('','x')))
    with pytest.raises(ValueError,match='not qualified'):require_direction_training_contract(torch.zeros(1,2),(11,22),replace(t,qualified=False))

def test_all_unknown_zero_grad():
    t=target([[float('nan'),float('nan')]],[[False,False]])
    x=torch.zeros(1,2,requires_grad=True);loss,_=direction_support_loss(x,(11,22),t);loss.backward()
    assert loss.item()==0 and torch.equal(x.grad,torch.zeros_like(x))
