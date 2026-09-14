import torch
import numpy as np
import pytest
from mtare_topo.representation.gse_structure_prediction_contract import PartialStructuralPrediction,match_for_partial_loss,evaluate_partial_localization
from mtare_topo.representation.gse_partial_structure_contract import PartialStructureTargets
from mtare_topo.representation.gse_structure_prediction_contract import partial_structure_loss,evaluate_partial_structure


def targets():
    return PartialStructureTargets(np.array([[0.,0,0],[4.,0,0]]),np.array([[10.,0,0]]),np.array([[1.,0.]],dtype=np.float32),np.ones((1,2),dtype=bool),(0,1,2,3,4))


def prediction():
    return PartialStructuralPrediction(torch.tensor([[4.,0,0],[0.,0,0],[8.,0,0]]),torch.tensor([[10.,0,0]]),torch.zeros(1,3))


def test_matching_ignores_relation_scores_and_reference_permutation():
    p=prediction();t=targets();a=match_for_partial_loss(p,t)
    p.section_structure_logits[:]=100
    b=match_for_partial_loss(p,t)
    assert a['structure'].tolist()==b['structure'].tolist()==[1,0]


def test_unmatched_output_not_invented_negative():
    r=evaluate_partial_localization(prediction(),targets())
    assert r['structure']['correct']==2 and r['structure']['unconfirmed_predictions']==1
    assert r['full_detection_precision_supported'] is False


def test_loss_capacity_shortfall_is_not_silent_reference_drop():
    p=PartialStructuralPrediction(torch.zeros(1,3),torch.zeros(1,3),torch.zeros(1,1))
    with pytest.raises(ValueError,match='capacity'):match_for_partial_loss(p,targets())


def test_evaluation_requires_actual_one_meter_localization():
    p=prediction();p.structure_positions_m[:]=100
    assert evaluate_partial_localization(p,targets())['structure']['correct']==0


def test_relation_supervision_follows_geometric_correspondence_only():
    p=prediction();p.section_structure_logits.requires_grad_()
    result=partial_structure_loss(p,targets());result['total'].backward()
    # True anchor0 matched query1, false anchor1 matched query0; query2 unknown.
    assert p.section_structure_logits.grad[0,1]<0
    assert p.section_structure_logits.grad[0,0]>0
    assert p.section_structure_logits.grad[0,2]==0


def test_unlocalized_negative_does_not_gain_credit_from_rejection():
    p=prediction();p.structure_positions_m[:]=100
    r=evaluate_partial_structure(p,targets())['relations']
    assert r['negative_total']==1 and r['negative_correct']==0 and r['unlocalized_known']==2
