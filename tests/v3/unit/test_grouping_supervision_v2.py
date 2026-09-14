import numpy as np
import pytest
import torch
from mtare_topo.representation.grouping_supervision_v2 import duplicate_candidates, RADIUS_M
from mtare_topo.representation.observed_anchor_objective_v1 import observed_anchor_objective


def mask(q, refs=((0., 0., 0.),), indices=(0,), observed=None, conflict=None):
    n = len(q)
    return duplicate_candidates(q, refs, indices, dict(
        observed_inside_score_region_mask=[True]*n if observed is None else observed,
        possible_unconfirmed_reference_mask=[False]*n if conflict is None else conflict))


def test_four_meters_unchanged_not_one_meter():
    assert RADIUS_M == 4.
    assert mask([[1.1,0,0], [3.99,0,0], [4.,0,0], [4.01,0,0]]).tolist() == [True,True,True,False]


def test_unknown_unobserved_and_unconfirmed_competitor_are_not_negative():
    q = [[2.,0,0]]*3
    assert mask(q, observed=[False,True,True], conflict=[False,True,False]).tolist() == [False,False,True]
    assert not mask(q, indices=()).any()
    assert not mask(q, refs=(), indices=()).any()


def test_two_confirmed_instances_and_non_target_instance_remain_unknown():
    assert not mask([[2,0,0]], refs=[[0,0,0],[3,0,0]], indices=[0,1]).any()
    assert not mask([[8,0,0]], refs=[[0,0,0],[8,0,0]], indices=[0]).any()


def objective(neg):
    pos=torch.tensor([[.4,0,0],[2.,0,0],[0,7.,0]],dtype=torch.float64,requires_grad=True)
    logits=torch.tensor([2.,1.,2.],dtype=torch.float64,requires_grad=True)
    result=observed_anchor_objective(pos,logits,torch.zeros((1,3),dtype=torch.float64),
        confirmed_negative=torch.tensor(neg,dtype=torch.bool))
    result['total'].backward()
    return result,pos.grad,logits.grad


def test_duplicate_gets_presence_gradient_but_unknown_stays_zero():
    old,op,ol=objective([False,False,False])
    new,np_,nl=objective([True,True,False])
    assert old['assignment'].tolist() == new['assignment'].tolist() == [0]
    assert ol[1] == 0 and nl[1] > 0 and nl[0] < 0 and nl[2] == 0
    assert torch.equal(op,np_)  # regression target and normalization unchanged
    assert new['negative_mask'].tolist() == [False,True,False]  # matched exempt
    assert torch.equal(np_[1:],torch.zeros_like(np_[1:]))


def test_positive_position_gradient_points_toward_target_and_is_nonzero():
    _,grad,_=objective([False,False,False])
    assert grad[0,0] == pytest.approx(.1)


def test_duplicate_presence_gradient_finite_difference():
    _,_,g=objective([True,True,False])
    def f(x):
        return observed_anchor_objective(torch.tensor([[.4,0,0],[2,0,0],[0,7,0]],dtype=torch.float64),
            torch.tensor([2.,x,2.],dtype=torch.float64),torch.zeros((1,3),dtype=torch.float64),
            confirmed_negative=torch.tensor([True,True,False]))['total'].item()
    assert g[1].item() == pytest.approx((f(1.+1e-5)-f(1.-1e-5))/2e-5,rel=1e-7)
