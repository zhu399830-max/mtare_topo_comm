import torch
from mtare_topo.representation.gse_membership_gradients import task_gradients,describe_gradients


def test_conflict_and_no_parameter_mutation():
    p=torch.nn.Parameter(torch.tensor([1.,2.]))
    before=p.detach().clone()
    names,g=task_gradients({'anchor_position':p.sum(),'membership':-2*p.sum()},[p])
    r=describe_gradients(names,g)
    assert r['total_gradient_opposes_location']
    assert r['location_other_dot']==-4
    assert torch.equal(before,p) and p.grad is None


def test_unused_parameter_and_aligned_gradients():
    p=torch.nn.Parameter(torch.tensor(2.));q=torch.nn.Parameter(torch.tensor(8.))
    names,g=task_gradients({'opening_position':p.square(),'membership':p.square()},[p,q])
    r=describe_gradients(names,g)
    assert not r['total_gradient_opposes_location']
    assert g[:,1].eq(0).all() and q.grad is None
