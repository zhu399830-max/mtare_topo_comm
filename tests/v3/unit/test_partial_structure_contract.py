import numpy as np
import torch
import pytest
from mtare_topo.representation.gse_partial_structure_contract import bind_partial_targets,known_relation_loss


def record():
    return dict(schema='gse_surface_observed_targets_v1',coordinate_frame='current_sensor_m',
        source_frame_indices=[0,1,2,3,4],score_region=dict(anchors_complete=False,openings_complete=False,radius_m=10,center_m=[0.,0.,0.]),
        anchors=[dict(position_m=[0,0,0]),dict(position_m=[1,0,0]),dict(position_m=[2,0,0])],
        openings=[dict(position_m=[10,0,0],width_m=None,height_m=None)],membership=[[True,False,None]])


def test_unknown_cannot_silently_become_negative():
    target=bind_partial_targets(record());logits=torch.zeros(1,3,requires_grad=True)
    known_relation_loss(logits,target).backward()
    assert logits.grad[0,0]<0 and logits.grad[0,1]>0 and logits.grad[0,2]==0
    assert target.relation_known.tolist()==[[True,True,False]]
    assert not target.full_detection_precision_supported and not target.unmatched_is_background


def test_unknown_nonfinite_placeholder_does_not_enter_loss():
    target=bind_partial_targets(record());x=torch.tensor([[0.,0.,float('nan')]],requires_grad=True)
    loss=known_relation_loss(x,target);assert torch.isfinite(loss)
    loss.backward();assert x.grad[0,2]==0


def test_all_unknown_has_zero_loss_and_gradient():
    r=record();r['membership']=[[None,None,None]];x=torch.ones(1,3,requires_grad=True)
    loss=known_relation_loss(x,bind_partial_targets(r));loss.backward()
    assert loss==0 and torch.equal(x.grad,torch.zeros_like(x))


def test_full_background_or_filled_dimensions_rejected():
    r=record();r['score_region']['anchors_complete']=True
    with pytest.raises(ValueError,match='partial'):bind_partial_targets(r)
    r=record();r['openings'][0]['width_m']=2
    with pytest.raises(ValueError,match='dimension'):bind_partial_targets(r)


def test_numeric_label_not_boolean_and_shape_drift_rejected():
    r=record();r['membership']=[[1,False,None]]
    with pytest.raises(ValueError,match='bool'):bind_partial_targets(r)
    r=record();r['membership']=[[True]]
    with pytest.raises(ValueError,match='shape'):bind_partial_targets(r)
