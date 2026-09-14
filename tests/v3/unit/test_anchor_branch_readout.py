import torch
import pytest
from mtare_topo.representation.anchor_branch_readout import AnchorBranchReadout
from mtare_topo.representation.anchor_branch_loss import PartialAnchorBranches,anchor_branch_loss


def test_forward_backward_to_observation_queries():
    torch.manual_seed(0);head=AnchorBranchReadout(branch_slots=4)
    q=torch.randn(2,128,requires_grad=True);p=head(q)
    t=PartialAnchorBranches(p.position_m.detach().clone()+.01,
        (torch.tensor([[1.,0,0]]),torch.tensor([[0.,1,0]])),False,(False,False))
    r=anchor_branch_loss(p,t);r['total'].backward()
    assert r['has_supervision'] and q.grad is not None and torch.isfinite(q.grad).all()
    assert q.grad.abs().sum()>0
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in head.parameters())
    torch.testing.assert_close(p.directions.norm(dim=-1),torch.ones(2,4))


def test_query_permutation_and_determinism():
    torch.manual_seed(0);head=AnchorBranchReadout(branch_slots=4);q=torch.randn(3,128)
    a=head(q);b=head(q[[2,0,1]])
    torch.testing.assert_close(a.directions[[2,0,1]],b.directions)
    torch.testing.assert_close(a.position_m[[2,0,1]],b.position_m)
    assert torch.equal(a.directions,head(q).directions)


@pytest.mark.parametrize('slots',[0,65,True])
def test_capacity_explicit(slots):
    with pytest.raises(ValueError):AnchorBranchReadout(branch_slots=slots)


def test_empty_or_nonfinite_observations_rejected():
    head=AnchorBranchReadout(branch_slots=4)
    for q in (torch.empty(0,128),torch.full((1,128),float('nan')),torch.empty(33,128)):
        with pytest.raises(ValueError):head(q)


def test_degenerate_direction_never_filled():
    head=AnchorBranchReadout(branch_slots=4)
    with torch.no_grad():head.branch.weight.zero_();head.branch.bias.zero_()
    with pytest.raises(ValueError,match='degenerate'):head(torch.ones(1,128))
