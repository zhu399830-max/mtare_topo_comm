import pytest
import torch
from mtare_topo.representation.center_position_diagnostic_v1 import (
    geometry_assignment,position_objective,independent_tensor_check)
from mtare_topo.representation.observed_anchor_detector_v1 import residual_positions
from mtare_topo.representation.observed_anchor_objective_v1 import observed_anchor_objective


def test_actual_decoder_formula_and_inverse():
    q=torch.tensor([[-8.,0,0],[0,0,8.]],dtype=torch.double)
    t=torch.tensor([[8.,0,0],[1.,2,3.]],dtype=torch.double)
    raw=torch.atanh((t-q)/20)
    p,r=residual_positions(q,raw)
    torch.testing.assert_close(p,t);torch.testing.assert_close(r,t-q)


def test_projection_radial_gradient_and_boundary_limit():
    q=torch.tensor([[9.,0,0]],dtype=torch.double)
    raw=torch.tensor([[1.,0,0]],dtype=torch.double,requires_grad=True)
    p,_=residual_positions(q,raw);assert p[0,0]==10
    p[0,0].backward();assert abs(raw.grad[0,0])<1e-12
    # An exact axial antipode requires atanh(1), not a finite raw residual.
    assert torch.isinf(torch.atanh(torch.tensor(1.)))
    p,_=residual_positions(torch.tensor([[-10.,0,0]],dtype=torch.double),
        torch.tensor([[8.,0,0]],dtype=torch.double))
    assert 0<10-p[0,0]<1e-4


def test_decoder_and_loss_gradcheck():
    q=torch.tensor([[1.,2.,0]],dtype=torch.double)
    t=torch.tensor([[2.,-1.,1]],dtype=torch.double)
    raw=torch.tensor([[.04,.03,-.02]],dtype=torch.double,requires_grad=True)
    assert torch.autograd.gradcheck(lambda x:position_objective(
        residual_positions(q,x)[0],t,torch.tensor([0])),(raw,))


def test_position_term_equals_existing_loss_and_no_confidence_gradient():
    p=torch.tensor([[1.,0,0],[6.,0,0]],requires_grad=True)
    t=torch.tensor([[2.,0,0]]);logits=torch.tensor([1.,0],requires_grad=True)
    old=observed_anchor_objective(p,logits,t,confirmed_negative=torch.tensor([False,True]))
    new=position_objective(p,t,old['assignment'])
    torch.testing.assert_close(old['position'],new)
    a=torch.autograd.grad(old['position'],p,retain_graph=True)[0]
    b=torch.autograd.grad(new,(p,logits),allow_unused=True)
    torch.testing.assert_close(a,b[0]);assert b[1] is None


def test_fixed_correspondence_does_not_switch_but_geometry_only_does():
    t=torch.tensor([[0.,0,0]])
    init=torch.tensor([[1.,0,0],[4.,0,0]])
    fixed=geometry_assignment(init,t)
    later=torch.tensor([[3.,0,0],[.1,0,0]],requires_grad=True)
    assert fixed.tolist()==[0] and geometry_assignment(later,t).tolist()==[1]
    position_objective(later,t,fixed).backward()
    assert later.grad[0,0]>0 and later.grad[1].abs().sum()==0


def test_multiple_targets_injective_and_empty_no_background():
    p=torch.tensor([[1.,0,0],[2.,0,0]],requires_grad=True)
    t=torch.tensor([[0.,0,0],[.1,0,0]])
    assert geometry_assignment(p,t).unique().numel()==2
    loss=position_objective(p,torch.empty(0,3),torch.empty(0,dtype=torch.long))
    loss.backward();assert p.grad.abs().sum()==0
    with pytest.raises(ValueError):position_objective(p,t,torch.tensor([0,0]))


def test_tensor_only_can_fit_without_network():
    r=independent_tensor_check(torch.tensor([[5.,0,0],[-3.,2.,1]]),torch.tensor([[1.,1.,0],[0.,0,0]]))
    assert r['passed'] and r['network_updates']==0


def test_real_forward_does_not_read_GT_and_existence_row_has_zero_gradient():
    import numpy as np
    from test_observed_anchor_detector_v1 import observation
    from mtare_topo.representation.grouping_center_training_v1 import forward
    from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
    blocks=observation()
    class Student:
        student_representations={'PRIMITIVE':dict(blocks=blocks,context=np.zeros((len(blocks.block_ids),128),np.float32))}
        def __getattr__(self,name):raise AssertionError('no target access '+name)
    model=build_model();out=forward(model,Student(),'PRIMITIVE')
    target=torch.zeros(1,3);a=geometry_assignment(out.prediction.position_m,target)
    loss=position_objective(out.prediction.position_m,target,a);loss.backward()
    assert model['head'].head.anchor.weight.grad[:3].abs().sum()>0
    assert model['head'].head.anchor.weight.grad[3].abs().sum()==0
    assert all(p.grad is None for n,p in model.named_parameters() if 'head.head.branch' in n)


def test_json_schedule_checks_values_not_list_tuple_type():
    import json
    from mtare_topo.representation.development_paired_training import batch_schedule
    from primitive_position_diagnostic_v1 import schedule_equal
    saved=json.loads(json.dumps(batch_schedule(16,updates=1000)))
    assert schedule_equal(saved)
    saved[0][0],saved[0][1]=saved[0][1],saved[0][0]
    assert not schedule_equal(saved)
