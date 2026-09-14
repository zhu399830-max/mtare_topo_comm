import torch
from dataclasses import replace
from mtare_topo.representation.gse_observed_membership_v1 import ObservedMembershipPrediction
from mtare_topo.representation import gse_membership_fit_execution as execution


def example_prediction(value):
    return ObservedMembershipPrediction(torch.zeros(1,3)+value*0,torch.ones(1)*10+value*0,
        torch.tensor([[10.,0,0]])+value*0,torch.ones(1)*10+value*0,
        torch.tensor([[1.,0,0]])+value*0,value.reshape(1,1),torch.tensor([0]),torch.tensor([0]))


def reference(label=True):
    return dict(anchors=[{'position_m':[0.,0,0]}],openings=[{'position_m':[10.,0,0]}],membership=[[label]])


def test_all_predictions_retained_and_unknown_not_background():
    p=example_prediction(torch.tensor(1.))
    p=replace(p,anchor_position_m=torch.tensor([[0.,0,0],[4.,0,0]]),
        anchor_presence_logits=torch.tensor([10.,10.]),membership_logits=torch.tensor([[1.,1.]]),
        anchor_source_indices=torch.tensor([0,1]))
    r=execution.evaluate_membership(p,reference(None))
    assert r['anchor']['unmatched_unconfirmed']==1
    assert len(r['all_predictions']['anchor_position_m'])==2
    assert r['membership']['unknown_reference_pairs']==1
    assert r['membership']['positive_total']==0
    assert not r['full_detection_precision_available']


def test_unlocalized_negative_cannot_count_as_correct():
    p=example_prediction(torch.tensor(-10.))
    p=replace(p,anchor_position_m=torch.tensor([[5.,0,0]]))
    r=execution.evaluate_membership(p,reference(False))['membership']
    assert r['negative_total']==1 and r['negative_correct']==0 and r['unlocalized_known']==1


def test_fixed_schedule_executes_once_without_selection(monkeypatch):
    m=torch.nn.Linear(1,1,bias=False)
    monkeypatch.setattr(execution,'predict_membership',lambda model,e:example_prediction(model.weight.flatten()[0]))
    calls=[]
    result=execution.fit_membership(m,[(None,None,reference()) for _ in range(16)],on_update=calls.append)
    assert result['actual_updates']==500 and len(calls)==500
    assert sum(len(x['sample_indices']) for x in calls)==2000
    assert len(result['initial'])==len(result['final'])==16
    assert all(sorted(sum(result['schedule'][i:i+4],[]))==list(range(16)) for i in range(0,500,4))
