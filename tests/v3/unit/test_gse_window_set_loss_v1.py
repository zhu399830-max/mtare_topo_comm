from dataclasses import replace,fields
import torch
import pytest
from tests.v3.unit.test_gse_surface_losses_v1 import prediction
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.evaluation.gse_fixture_loss_targets import fixture_loss_targets
from mtare_topo.representation.gse_window_surface_domain import window_surface_positions
from mtare_topo.representation.gse_window_set_loss_v1 import training_assignment,window_set_loss


def inputs():
    p=prediction();p=replace(p,**{f.name:getattr(p,f.name).detach().float().requires_grad_() for f in fields(p) if getattr(p,f.name).dtype==torch.float64})
    p=replace(p,opening_position_m=window_surface_positions(p.opening_position_m,p.observation_supported))
    case=next(c for c in matrix() if c['case_id']=='straight__circle__view2')
    return p,fixture_loss_targets(case,[0,1,2,3,4])


def test_training_assignment_uses_presence_but_does_not_change_scoring():
    xyz=torch.tensor([[[0.,0.,0.],[.1,0.,0.]]]);t=torch.zeros(1,1,3);valid=torch.ones(1,1,dtype=torch.bool)
    assert training_assignment(xyz,torch.tensor([[-5.,5.]]),t,valid).item()==1
    assert training_assignment(xyz,torch.tensor([[5.,-5.]]),t,valid).item()==0
    from mtare_topo.evaluation.gse_synthetic_field_scoring import match_positions
    assert match_positions([[0,0,0]],[[0,0,0],[.1,0,0]],1.)['pairs'][0][:2]==(0,0)


def test_complete_window_all_queries_supervised_and_backward_finite():
    p,t=inputs();loss=window_set_loss(p,t)
    assert loss.denominators['opening_presence']==64
    loss.total.backward();assert torch.isfinite(p.opening_presence_logits.grad).all()
    matched=loss.assignments['opening'][0].tolist()
    assert (p.opening_presence_logits.grad[0,matched]<0).all()
    other=[i for i in range(64) if i not in matched]
    assert (p.opening_presence_logits.grad[0,other]>0).all()


def test_unknown_window_only_supervises_known_targets():
    p,t=inputs();t=replace(t,opening_region_complete=torch.zeros_like(t.opening_region_complete))
    loss=window_set_loss(p,t);assert loss.denominators['opening_presence']==2
    loss.total.backward();assert (p.opening_presence_logits.grad!=0).sum()==2


def test_off_surface_predictions_rejected_not_discarded():
    p,t=inputs()
    with pytest.raises(ValueError):window_set_loss(replace(p,opening_position_m=p.opening_position_m*1.001),t)
