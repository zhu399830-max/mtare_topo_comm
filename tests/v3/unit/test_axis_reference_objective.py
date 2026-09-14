import math
import pytest
import torch
from mtare_topo.representation.gse_axis_reference_objective import axis_reference_logit_loss as objective


@pytest.mark.parametrize('dtype',[torch.float32,torch.float64])
@pytest.mark.parametrize('z',[-80.,-12.,0.,12.,80.])
@pytest.mark.parametrize('y',[0.,.25,.75,1.])
def test_exact_stable_gradient(dtype,z,y):
    a=torch.tensor([z],dtype=dtype,requires_grad=True);target=torch.tensor([y],dtype=dtype)
    loss=objective(a,target,torch.tensor([True]),torch.ones_like(a));loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(a.grad).all()
    assert torch.allclose(a.grad,a.detach().sigmoid()-target,atol=1e-7,rtol=1e-6)


def test_unknown_nan_never_becomes_background():
    z=torch.tensor([12.,float('nan')],requires_grad=True);y=torch.tensor([.25,float('nan')])
    loss=objective(z,y,torch.tensor([True,False]),torch.tensor([1.,float('nan')]))
    loss.backward();assert torch.isfinite(loss) and z.grad[1]==0 and z.grad[0]>.7


def test_all_unknown_zero_gradient():
    z=torch.tensor([float('nan')],requires_grad=True)
    loss=objective(z,z.detach(),torch.tensor([False]),z.detach());loss.backward()
    assert loss==0 and z.grad.item()==0


def test_constant_optimum_mean_not_median():
    # .6 weight at1, .4 at0: L1 median1, logit objective mean.6.
    z=torch.tensor(math.log(.6/.4),dtype=torch.float64,requires_grad=True)
    loss=objective(z.expand(2),torch.tensor([1.,0.],dtype=z.dtype),torch.ones(2,dtype=torch.bool),torch.tensor([.6,.4],dtype=z.dtype))
    loss.backward();assert abs(z.grad.item())<1e-12
    saturated=torch.tensor(12.,dtype=z.dtype,requires_grad=True)
    objective(saturated.expand(2),torch.tensor([1.,0.],dtype=z.dtype),torch.ones(2,dtype=torch.bool),torch.tensor([.6,.4],dtype=z.dtype)).backward()
    assert saturated.grad.item()>.39  # descent lowers an incorrect common logit


@pytest.mark.parametrize('field,value',[('target',1.1),('target',float('nan')),('weights',-1.)])
def test_invalid_known_targets_not_silently_clipped(field,value):
    data=dict(logits=torch.zeros(1),target=torch.ones(1),known=torch.ones(1,dtype=torch.bool),weights=torch.ones(1))
    data[field]=torch.tensor([value])
    with pytest.raises(ValueError):objective(**data)


def test_model_exposes_same_logits_without_new_parameters():
    from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder
    from test_structural_representation import inputs,context
    x,p=inputs();model=GeometryStructureEncoder('C');r=model(x,p,context()).relations
    assert torch.equal(r.axis_abs_dot[r.computation_valid],r.axis_logits[r.computation_valid].sigmoid())
    assert not any('axis_logits' in key for key in model.state_dict())


def test_opt_in_keeps_old_objective_and_requires_raw_logits():
    from dataclasses import replace
    from mtare_topo.representation.gse_conditional_geometry_loss import ConditionalGeometryLossTargets,conditional_geometry_loss
    from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder
    from test_structural_representation import inputs,context
    x,p=inputs();model=GeometryStructureEncoder('C');r=model(x,p,context()).relations;k=r.computation_valid
    y=torch.full_like(r.axis_abs_dot,.25);w=k.float()/k.sum()
    t=ConditionalGeometryLossTargets(y,torch.zeros_like(y),k,torch.zeros_like(k),w)
    old,_=conditional_geometry_loss(r,t)
    expected=(w[k]*(r.axis_abs_dot[k]-.25).abs()).sum()+(w[k]*r.height_difference_m[k].abs()/10).sum()
    assert torch.equal(old,expected)
    new,_=conditional_geometry_loss(r,t,axis_objective='conditional_axis_soft_target_logit_v1');new.backward()
    assert torch.isfinite(model.relation_head[-1].weight.grad).all()
    with pytest.raises(ValueError,match='raw axis logits'):
        conditional_geometry_loss(replace(r,axis_logits=None),t,axis_objective='conditional_axis_soft_target_logit_v1')
