import torch
import pytest
from mtare_topo.evaluation.gse_candidate_loss_controls import peak_and_hard_background_control
from mtare_topo.representation.gse_candidate_objective import balanced_candidate_loss


def gradients(extra,alternate):
    logits=torch.tensor([3.,3.]+[-8.]*extra,requires_grad=True)
    target=torch.tensor([1.,0.]+[0.]*extra);known=torch.ones(len(logits),dtype=torch.bool)
    if alternate:
        loss,_=peak_and_hard_background_control(logits,(target==1)[None],target==0)
    else:loss,_=balanced_candidate_loss(logits,target,known)
    loss.backward();return float(logits.grad[1])


def test_old_hard_error_diluted_but_max_not_diluted():
    assert gradients(999,False)==pytest.approx(gradients(0,False)/1000)
    assert gradients(999,True)==pytest.approx(gradients(0,True))


def test_flat_support_mean_is_old_stationary_point_not_peak_solution():
    targets=torch.tensor([.1,.2,.4,.7],dtype=torch.float64)
    scalar=torch.logit(targets.mean()).detach().requires_grad_()
    loss,_=balanced_candidate_loss(scalar.expand(4),targets,torch.ones(4,dtype=torch.bool))
    loss.backward();assert abs(float(scalar.grad))<1e-10
    scalar.grad=None
    loss,_=peak_and_hard_background_control(scalar.expand(4),torch.ones(1,4,dtype=torch.bool),torch.zeros(4,dtype=torch.bool))
    loss.backward();assert float(scalar.grad)<0


def test_unknown_candidate_receives_no_gradient():
    x=torch.tensor([0.,0.,20.],requires_grad=True)
    loss,_=peak_and_hard_background_control(x,torch.tensor([[True,False,False]]),torch.tensor([False,True,False]))
    loss.backward();assert x.grad[2]==0


def test_uncovered_target_fails_not_skipped():
    with pytest.raises(ValueError):
        peak_and_hard_background_control(torch.zeros(2),torch.zeros(1,2,dtype=torch.bool),torch.ones(2,dtype=torch.bool))


def test_duplicate_high_positives_exposes_incomplete_alternative():
    bag=torch.tensor([[True,True,False]]);background=torch.tensor([False,False,True])
    one,_=peak_and_hard_background_control(torch.tensor([5.,-5.,-5.]),bag,background)
    many,report=peak_and_hard_background_control(torch.tensor([5.,5.,-5.]),bag,background)
    assert one==many and not report['duplicate_positives_controlled'] and not report['training_qualified']
