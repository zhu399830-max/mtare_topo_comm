import numpy as np
import torch
import pytest
from mtare_topo.evaluation.gse_candidate_set_loss_control import one_to_one_hard_negative_control


def loss(x,p,t,complete=True):
    return one_to_one_hard_negative_control(x,np.asarray(p,dtype=float).reshape(-1,3),np.asarray(t,dtype=float).reshape(-1,3),complete_region=complete)


def test_duplicate_high_outputs_cost_more_than_single_high():
    p=[[0,0,0],[.1,0,0]];t=[[0,0,0]]
    one,_=loss(torch.tensor([5.,-5.]),p,t)
    two,r=loss(torch.tensor([5.,5.]),p,t)
    assert two>one and len(r['hard_negative_indices'])==1


def test_close_distinct_nodes_are_not_suppressed():
    p=[[0,0,0],[.1,0,0]]
    value,r=loss(torch.tensor([5.,5.]),p,p)
    assert value<.01 and len(r['matched_candidate_indices'])==2
    assert not r['spatial_suppression_used']


def test_easy_background_does_not_dilute_wrong_high_candidate():
    def grad(extra):
        x=torch.tensor([5.,5.]+[-8.]*extra,requires_grad=True)
        v,_=loss(x,[[0,0,0],[5,0,0]]+[[10+i,0,0] for i in range(extra)],[[0,0,0]])
        v.backward();return x.grad[1]
    torch.testing.assert_close(grad(0),grad(999),rtol=0,atol=0)


def test_all_reject_and_all_accept_both_worse_than_correct_set():
    p=[[0,0,0],[5,0,0]];t=[[0,0,0]]
    good,_=loss(torch.tensor([5.,-5.]),p,t)
    reject,_=loss(torch.tensor([-5.,-5.]),p,t)
    accept,_=loss(torch.tensor([5.,5.]),p,t)
    assert reject>good and accept>good


def test_unknown_negative_and_missing_coverage():
    x=torch.tensor([0.,5.],requires_grad=True)
    v,r=loss(x,[[0,0,0],[5,0,0]],[[0,0,0]],False);v.backward()
    assert x.grad[1]==0 and r['unmatched_unknown']==1
    with pytest.raises(ValueError):loss(torch.zeros(1),[[5,0,0]],[[0,0,0]])


def test_identical_features_have_unavoidable_conflicting_signal():
    # Two identical candidates produced by a shared deterministic scorer cannot
    # yield one high and one low logit, even with a valid set loss.
    shared=torch.tensor(0.,requires_grad=True)
    v,r=loss(shared.expand(2),[[0,0,0],[0,0,0]],[[0,0,0]])
    v.backward();assert shared.grad==0 and v.item()==pytest.approx(np.log(2))
    assert not r['training_qualified']
