import torch
from mtare_topo.representation.observed_anchor_objective_v1 import observed_anchor_objective
from mtare_topo.representation.geometry_match_presence_v1 import restore_presence


def objective(p,l,t,negative):
    return restore_presence(observed_anchor_objective(p,torch.zeros_like(l),t,confirmed_negative=negative),l)


def test_confidence_cannot_select_farther_prediction_but_actual_presence_is_trained():
    p=torch.tensor([[.2,0,0],[2.,0,0]],requires_grad=True)
    l=torch.tensor([-5.,5.],requires_grad=True);t=torch.zeros(1,3);neg=torch.tensor([False,True])
    a=objective(p,l,t,neg);b=observed_anchor_objective(p,l,t,confirmed_negative=neg)
    assert a['assignment'].tolist()==[0] and b['assignment'].tolist()==[1]
    a['total'].backward();assert l.grad[0]<0 and l.grad[1]>0
    assert p.grad[0,0]>0 and p.grad[1].abs().sum()==0


def test_same_assignment_exact_original_loss_and_gradients():
    p=torch.tensor([[.2,0,0],[5.,0,0],[6.,0,0]],requires_grad=True)
    l=torch.tensor([2.,-1.,0.],requires_grad=True);t=torch.zeros(1,3);neg=torch.tensor([False,True,False])
    a=objective(p,l,t,neg);b=observed_anchor_objective(p,l,t,confirmed_negative=neg)
    for k in ['total','presence','position']:torch.testing.assert_close(a[k],b[k])
    ga=torch.autograd.grad(a['total'],(p,l),retain_graph=True);gb=torch.autograd.grad(b['total'],(p,l))
    for x,y in zip(ga,gb):torch.testing.assert_close(x,y)
    assert ga[1][2]==0


def test_no_positive_still_trains_confirmed_negatives_not_unknown():
    p=torch.tensor([[1.,0,0],[2.,0,0]],requires_grad=True);l=torch.zeros(2,requires_grad=True)
    a=objective(p,l,torch.empty(0,3),torch.tensor([True,False]))
    a['total'].backward();assert a['has_supervision'] and l.grad[0]>0 and l.grad[1]==0


def test_all_unknown_has_no_supervision():
    a=objective(torch.zeros(2,3),torch.zeros(2),torch.empty(0,3),torch.zeros(2,dtype=torch.bool))
    assert not a['has_supervision'] and a['total']==0


def test_adapter_neutralizes_only_matching_and_preserves_actual_prediction(monkeypatch):
    from types import SimpleNamespace
    import mtare_topo.representation.geometry_match_presence_v1 as module
    from mtare_topo.representation.anchor_branch_loss import AnchorBranchPrediction
    from mtare_topo.representation.observed_anchor_detector_v1 import ObservedAnchorPrediction
    p=torch.tensor([[.2,0,0],[2.,0,0]],requires_grad=True);l=torch.tensor([-5.,5.],requires_grad=True)
    prediction=AnchorBranchPrediction(p,l,torch.zeros(2,1,3),torch.zeros(2,1))
    output=ObservedAnchorPrediction(prediction,torch.arange(2),p.detach(),torch.zeros_like(p))
    t=torch.zeros(1,3);observation=SimpleNamespace(loss_only={'target':SimpleNamespace(position_m=t)})
    def repaired(neutral,obs):
        assert neutral.prediction.position_m is p and torch.count_nonzero(neutral.prediction.presence_logits)==0
        return observed_anchor_objective(p,neutral.prediction.presence_logits,t,confirmed_negative=torch.tensor([False,True]))
    monkeypatch.setattr(module,'repaired_objective',repaired)
    result=module.center_objective(output,observation);result['total'].backward()
    assert result['assignment'].tolist()==[0] and l.grad[0]<0 and l.grad[1]>0
    assert output.prediction is prediction and output.prediction.presence_logits is l
