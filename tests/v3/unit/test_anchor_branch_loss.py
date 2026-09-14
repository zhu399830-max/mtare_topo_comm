import torch
import pytest
from mtare_topo.representation.anchor_branch_loss import AnchorBranchPrediction,PartialAnchorBranches,anchor_branch_loss


def fixture():
    xyz=torch.tensor([[0.,0,0],[3.,0,0]],requires_grad=True)
    directions=torch.tensor([[[1.,0,0],[-1.,0,0],[0,1.,0]]]*2,requires_grad=True)
    p=AnchorBranchPrediction(xyz,torch.zeros(2,requires_grad=True),directions,torch.zeros((2,3),requires_grad=True))
    t=PartialAnchorBranches(xyz.detach().clone(),(torch.tensor([[1.,0,0]]),torch.tensor([[1.,0,0]])),False,(False,False))
    return p,t


def test_same_direction_different_anchor_and_unknown_gradient():
    p,t=fixture();r=anchor_branch_loss(p,t);r['total'].backward()
    assert r['anchor_assignment'].tolist()==[0,1]
    assert r['counts']['branch_presence']==2
    assert (p.branch_logits.grad[:,0]<0).all()
    assert (p.branch_logits.grad[:,1:]==0).all()
    assert not r['aperture_supervised']


def test_all_unknown_disconnected_even_warm_optimizer():
    p,_=fixture();optimizer=torch.optim.AdamW([p.presence_logits],lr=.1,weight_decay=.1)
    p.presence_logits.sum().backward();optimizer.step();optimizer.zero_grad(set_to_none=True)
    before=p.presence_logits.detach().clone()
    t=PartialAnchorBranches(torch.empty((0,3)),(),False,())
    r=anchor_branch_loss(p,t);r['total'].backward()
    assert not r['has_supervision'] and p.presence_logits.grad is None
    # Runner must skip all-unknown; even an accidental optimizer step has no gradient.
    optimizer.step();assert torch.equal(p.presence_logits,before)


def test_complete_branch_background_only_explicit():
    p,t=fixture();t=PartialAnchorBranches(t.position_m,t.directions,False,(True,False))
    r=anchor_branch_loss(p,t);r['total'].backward()
    assert (p.branch_logits.grad[0,1:]>0).all() and (p.branch_logits.grad[1,1:]==0).all()


def test_ambiguous_geometry_rejected_not_confidence_selected():
    p,t=fixture();p.position_m.data[1]=p.position_m.data[0]
    with pytest.raises(ValueError,match='ambiguous'):anchor_branch_loss(p,t)


def test_branch_target_permutation_and_opposite_directions():
    p,t=fixture();d=torch.tensor([[1.,0,0],[-1.,0,0]])
    a=PartialAnchorBranches(t.position_m,(d,d),False,(False,False))
    b=PartialAnchorBranches(t.position_m,(d.flip(0),d.flip(0)),False,(False,False))
    ra=anchor_branch_loss(p,a);rb=anchor_branch_loss(p,b)
    assert torch.equal(ra['total'],rb['total']) and ra['counts']['branch_direction']==4
