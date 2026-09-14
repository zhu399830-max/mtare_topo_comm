"""Synthetic loss contracts; no dataset, optimizer or training run."""
from dataclasses import replace, fields

import pytest
import torch

from mtare_topo.representation.gse_surface_relation_model_v1 import SurfaceRelationPrediction
from mtare_topo.representation.gse_surface_losses_v1 import (
    SurfaceLossTargets, surface_relation_losses, LOSS_NAMES, UNSUPERVISED_HEADS,
)


@pytest.fixture(autouse=True)
def threads():
    old = torch.get_num_threads(); torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def prediction():
    def tensor(shape, value=0.):
        return torch.full(shape, value, dtype=torch.float64, requires_grad=True)
    anchor = torch.stack((torch.arange(32), torch.zeros(32), torch.zeros(32)), -1).double()[None].requires_grad_()
    opening = torch.stack((torch.arange(64), torch.ones(64), torch.zeros(64)), -1).double()[None].requires_grad_()
    direction = torch.zeros(1,64,3,dtype=torch.float64); direction[...,0] = 1.; direction.requires_grad_()
    return SurfaceRelationPrediction(anchor, tensor((1,32)), tensor((1,32,3),1.),
        opening, tensor((1,64)), direction, torch.ones(1,64,dtype=torch.bool),
        tensor((1,64,2),3.), tensor((1,64,2)), tensor((1,64)), tensor((1,64,3)),
        tensor((1,64,32)), tensor((1,64,32)), torch.ones(1,dtype=torch.bool))


def targets():
    d = torch.float64
    return SurfaceLossTargets(
        anchor_position_m=torch.tensor([[[.1,0.,0.],[2.2,0.,0.]]],dtype=d), anchor_valid=torch.ones(1,2,dtype=torch.bool),
        opening_position_m=torch.tensor([[[1.1,1.,0.]]],dtype=d), opening_valid=torch.ones(1,1,dtype=torch.bool),
        opening_direction=torch.tensor([[[0.,1.,0.]]],dtype=d), direction_valid=torch.ones(1,1,dtype=torch.bool),
        opening_dimensions_m=torch.tensor([[[2.,4.]]],dtype=d), dimension_valid=torch.ones(1,1,2,dtype=torch.bool),
        reachability_class=torch.zeros(1,1,dtype=torch.long), reachability_valid=torch.ones(1,1,dtype=torch.bool),
        physical_reference_valid=torch.ones(1,1,dtype=torch.bool), membership=torch.ones(1,1,2,dtype=d),
        membership_valid=torch.ones(1,1,2,dtype=torch.bool), score_region_center_m=torch.zeros(1,3,dtype=d),
        score_region_radius_m=torch.full((1,),10.,dtype=d), anchor_region_complete=torch.zeros(1,dtype=torch.bool),
        opening_region_complete=torch.zeros(1,dtype=torch.bool))


def test_geometry_binding_not_class_confidence_or_membership_selected():
    p, t = prediction(), targets(); expected = surface_relation_losses(p,t).assignments
    presence = torch.full_like(p.opening_presence_logits,100.); presence[0,1] = -100.
    reach = torch.full_like(p.reachability_logits,-100.); reach[...,0] = 100.; reach[0,1] = torch.tensor([-100.,100.,0.])
    q = replace(p,opening_presence_logits=presence,reachability_logits=reach,
                membership_logits=torch.randn_like(p.membership_logits)*100)
    actual = surface_relation_losses(q,t)
    assert torch.equal(actual.assignments['anchor'],expected['anchor'])
    assert torch.equal(actual.assignments['opening'],expected['opening'])
    assert actual.assignments['anchor'].tolist()==[[0,2]] and actual.assignments['opening'].tolist()==[[1]]
    assert actual.terms['reachability']>100 and actual.terms['opening_presence']>99


def test_every_core_task_has_independent_denominator_and_valid_gradients():
    p,t=prediction(),targets(); result=surface_relation_losses(p,t)
    assert result.denominators==dict(anchor_presence=2,anchor_position=2,opening_presence=1,opening_position=1,
        opening_direction=1,opening_dimensions=2,reachability=1,membership=2)
    assert result.has_supervision and set(result.terms)==set(LOSS_NAMES)
    result.total.backward()
    for name in ('anchor_position_m','anchor_presence_logits','opening_position_m','opening_presence_logits',
                 'opening_direction','opening_dimensions_m','reachability_logits','membership_logits'):
        value=getattr(p,name); assert torch.isfinite(value.grad).all() and value.grad.abs().sum()>0
    assert result.unsupervised_heads==UNSUPERVISED_HEADS
    assert result.reliability_status=='UNTRAINED_UNCALIBRATED'
    for name in UNSUPERVISED_HEADS: assert getattr(p,name).grad is None
    assert result.terms['anchor_position'].item()==pytest.approx(.015)
    assert result.terms['opening_dimensions'].item()==pytest.approx(.1)


def test_unknown_attributes_remain_disconnected_not_negatives():
    p,t=prediction(),targets()
    t=replace(t,direction_valid=torch.zeros_like(t.direction_valid),dimension_valid=torch.zeros_like(t.dimension_valid),
        reachability_valid=torch.zeros_like(t.reachability_valid),physical_reference_valid=torch.zeros_like(t.physical_reference_valid),
        membership_valid=torch.zeros_like(t.membership_valid),opening_direction=torch.full_like(t.opening_direction,float('nan')),
        opening_dimensions_m=torch.full_like(t.opening_dimensions_m,float('nan')),membership=torch.full_like(t.membership,float('nan')))
    result=surface_relation_losses(p,t);result.total.backward()
    for name in ('opening_direction','opening_dimensions_m','reachability_logits','membership_logits'):
        assert getattr(p,name).grad is None
    for name in ('opening_direction','opening_dimensions','reachability','membership'): assert result.denominators[name]==0
    assert p.opening_presence_logits.grad[0,1]!=0
    assert (p.opening_presence_logits.grad[0,torch.arange(64)!=1]==0).all()


@pytest.mark.parametrize('warm_optimizer', [False, True])
def test_all_unknown_dimension_task_does_not_activate_adamw(warm_optimizer):
    p,t=prediction(),targets()
    optimizer=torch.optim.AdamW([p.opening_dimensions_m], lr=1e-3, weight_decay=1e-4)
    if warm_optimizer:
        surface_relation_losses(p,t).total.backward()
        optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    before=p.opening_dimensions_m.detach().clone()
    state_before={k:v.clone() if torch.is_tensor(v) else v
                  for k,v in optimizer.state.get(p.opening_dimensions_m,{}).items()}
    t=replace(t,dimension_valid=torch.zeros_like(t.dimension_valid))
    result=surface_relation_losses(p,t)
    result.total.backward()
    assert result.has_supervision and result.denominators['opening_dimensions']==0
    assert p.opening_dimensions_m.grad is None
    optimizer.step()
    assert torch.equal(before,p.opening_dimensions_m.detach())
    assert all(torch.equal(v,optimizer.state[p.opening_dimensions_m][k])
               if torch.is_tensor(v) else v==optimizer.state[p.opening_dimensions_m][k]
               for k,v in state_before.items())
    if not warm_optimizer:
        assert p.opening_dimensions_m not in optimizer.state


def test_independent_multi_anchor_membership_not_softmax():
    p,t=prediction(),targets();result=surface_relation_losses(p,t);result.terms['membership'].backward()
    assert p.membership_logits.grad[0,1,0]<0 and p.membership_logits.grad[0,1,2]<0
    assert torch.count_nonzero(p.membership_logits.grad)==2


def test_positive_only_membership_has_constant_positive_degenerate_solution():
    # Reproduce the *loss* defect of a positive-only population, not a model
    # experiment or an argument for turning unknown targets into negatives.
    p,t=prediction(),targets()
    t=replace(t,membership_valid=torch.tensor([[[True,False]]]),
              membership=torch.tensor([[[1.,float('nan')]]],dtype=torch.float64))
    initial=surface_relation_losses(p,t)
    constant=replace(p,membership_logits=torch.full_like(p.membership_logits,20.,requires_grad=True))
    result=surface_relation_losses(constant,t)
    assert result.denominators['membership']==1
    assert result.terms['membership'].item()<3e-9
    assert initial.terms['membership'].item()>0.69
    assert all(torch.equal(result.assignments[k],initial.assignments[k]) for k in initial.assignments)
    result.terms['membership'].backward()
    assert constant.membership_logits.grad[0,1,0]<0
    assert constant.membership_logits.grad[0,1,2]==0
    assert torch.count_nonzero(constant.membership_logits.grad)==1


def test_explicit_negative_membership_penalizes_constant_positive_without_softmax():
    # Synthetic independently supplied binary labels. This does NOT establish
    # that any real unknown pair is negative or supply a teacher criterion.
    p,t=prediction(),targets()
    t=replace(t,membership=torch.tensor([[[1.,0.]]],dtype=torch.float64))
    constant=replace(p,membership_logits=torch.full_like(p.membership_logits,20.,requires_grad=True))
    result=surface_relation_losses(constant,t)
    assert result.denominators['membership']==2
    assert result.terms['membership'].item()>9.99
    result.terms['membership'].backward()
    assert constant.membership_logits.grad[0,1,0]<0
    assert constant.membership_logits.grad[0,1,2]>0
    assert torch.count_nonzero(constant.membership_logits.grad)==2


def test_unmatched_fp_only_inside_complete_scored_region():
    p,t=prediction(),targets();t=replace(t,anchor_region_complete=torch.ones(1,dtype=torch.bool),opening_region_complete=torch.ones(1,dtype=torch.bool))
    result=surface_relation_losses(p,t);result.total.backward()
    assert result.denominators['anchor_presence']==11 and result.denominators['opening_presence']==10
    assert p.anchor_presence_logits.grad[0,0]<0 and p.anchor_presence_logits.grad[0,1]>0
    assert p.opening_presence_logits.grad[0,2]>0 and p.opening_presence_logits.grad[0,12]==0
    assert p.anchor_presence_logits.grad[0,31]==0


def test_all_unknown_nan_padding_is_finite_differentiable_zero():
    p,t=prediction(),targets()
    t=replace(t,anchor_valid=torch.zeros_like(t.anchor_valid),opening_valid=torch.zeros_like(t.opening_valid),
        direction_valid=torch.zeros_like(t.direction_valid),dimension_valid=torch.zeros_like(t.dimension_valid),
        reachability_valid=torch.zeros_like(t.reachability_valid),physical_reference_valid=torch.zeros_like(t.physical_reference_valid),
        membership_valid=torch.zeros_like(t.membership_valid),anchor_position_m=torch.full_like(t.anchor_position_m,float('nan')),
        opening_position_m=torch.full_like(t.opening_position_m,float('nan')))
    p=replace(p,observation_supported=torch.zeros_like(p.observation_supported))
    result=surface_relation_losses(p,t);assert result.total.item()==0 and not result.has_supervision
    assert not any(result.denominators.values()); result.total.backward()
    for f in fields(p):
        value=getattr(p,f.name)
        if value.dtype!=torch.bool:
            if f.name in UNSUPERVISED_HEADS: assert value.grad is None
            else: assert value.grad is None


@pytest.mark.parametrize('kind',['local_tie','global_tie','duplicate_gt'])
def test_ambiguous_assignments_fail_not_slot_order(kind):
    p,t=prediction(),targets()
    if kind=='local_tie':
        t=replace(t,anchor_position_m=torch.tensor([[[.5,0.,0.],[2.2,0.,0.]]],dtype=torch.float64))
    elif kind=='duplicate_gt':
        t=replace(t,anchor_position_m=torch.zeros_like(t.anchor_position_m))
    else:
        positions=p.anchor_position_m.detach().clone();positions[0,:,0]+=2.
        p=replace(p,anchor_position_m=positions)
        t=replace(t,anchor_position_m=torch.tensor([[[0.,0.,0.],[1.,0.,0.]]],dtype=torch.float64))
    with pytest.raises(ValueError,match='ambiguous'):surface_relation_losses(p,t)


@pytest.mark.parametrize('kind',['physical','membership','direction','dimensions','mask_type','unknown_location','known_nan'])
def test_invalid_label_contract_rejected(kind):
    p,t=prediction(),targets()
    if kind=='physical':t=replace(t,physical_reference_valid=torch.zeros_like(t.physical_reference_valid))
    if kind=='membership':t=replace(t,membership=torch.full_like(t.membership,.5))
    if kind=='direction':t=replace(t,opening_direction=t.opening_direction*2)
    if kind=='dimensions':t=replace(t,opening_dimensions_m=-t.opening_dimensions_m)
    if kind=='mask_type':t=replace(t,anchor_valid=t.anchor_valid.long())
    if kind=='unknown_location':t=replace(t,opening_valid=torch.zeros_like(t.opening_valid))
    if kind=='known_nan':t=replace(t,anchor_position_m=torch.full_like(t.anchor_position_m,float('nan')))
    with pytest.raises(ValueError):surface_relation_losses(p,t)


def test_masks_cannot_be_omitted():
    with pytest.raises(TypeError):SurfaceLossTargets(anchor_position_m=torch.zeros(1,1,3))


def test_empty_target_lists_and_complete_background():
    p,t=prediction(),targets()
    t=SurfaceLossTargets(torch.empty(1,0,3,dtype=torch.float64),torch.empty(1,0,dtype=torch.bool),
        torch.empty(1,0,3,dtype=torch.float64),torch.empty(1,0,dtype=torch.bool),
        torch.empty(1,0,3,dtype=torch.float64),torch.empty(1,0,dtype=torch.bool),
        torch.empty(1,0,2,dtype=torch.float64),torch.empty(1,0,2,dtype=torch.bool),
        torch.empty(1,0,dtype=torch.long),torch.empty(1,0,dtype=torch.bool),torch.empty(1,0,dtype=torch.bool),
        torch.empty(1,0,0,dtype=torch.float64),torch.empty(1,0,0,dtype=torch.bool),
        t.score_region_center_m,t.score_region_radius_m,torch.ones(1,dtype=torch.bool),torch.ones(1,dtype=torch.bool))
    result=surface_relation_losses(p,t)
    assert result.denominators['anchor_presence']==11 and result.denominators['opening_presence']==10
    assert result.denominators['anchor_position']==0 and result.has_supervision


def test_target_capacity_overflow_is_explicit_not_matching_truncation():
    p,t=prediction(),targets()
    xyz=torch.zeros(1,33,3,dtype=torch.float64);xyz[0,:,0]=torch.arange(33)/4
    t=replace(t,anchor_position_m=xyz,anchor_valid=torch.ones(1,33,dtype=torch.bool),
        membership=torch.zeros(1,1,33,dtype=torch.float64),membership_valid=torch.zeros(1,1,33,dtype=torch.bool))
    with pytest.raises(OverflowError,match='never truncate'):surface_relation_losses(p,t)


def test_query_permutation_changes_ids_not_losses_or_pair_meaning():
    p,t=prediction(),targets();a=torch.arange(31,-1,-1);o=torch.arange(63,-1,-1)
    data={}
    for f in fields(p):
        value=getattr(p,f.name)
        if f.name.startswith('anchor_'):value=value[:,a]
        elif f.name in ('membership_logits','membership_validity_logits'):value=value[:,o][:,:,a]
        elif f.name!='observation_supported':value=value[:,o]
        data[f.name]=value
    before=surface_relation_losses(p,t);after=surface_relation_losses(SurfaceRelationPrediction(**data),t)
    for name in LOSS_NAMES:torch.testing.assert_close(before.terms[name],after.terms[name],rtol=0,atol=1e-14)
    assert after.assignments['anchor'].tolist()==[[31,29]]
    assert after.assignments['opening'].tolist()==[[62]]
