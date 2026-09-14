from copy import deepcopy
from dataclasses import fields, replace
import pytest
import torch
from mtare_topo.data.gse_synthetic_matrix import matrix
from mtare_topo.evaluation.gse_fixture_loss_targets import fixture_loss_targets
from mtare_topo.representation.gse_surface_losses_v1 import surface_relation_losses
from test_gse_surface_losses_v1 import prediction


def case(key): return next(c for c in matrix() if c['case_id']==key)


def double_target(t):
    return replace(t,**{f.name:getattr(t,f.name).double() for f in fields(t)
                       if getattr(t,f.name).dtype==torch.float32})


def test_complete_empty_anchor_region_penalizes_false_nodes_not_outside():
    t=double_target(fixture_loss_targets(case('straight__circle__view2'),[0,1,2,3,4]))
    p=prediction(); result=surface_relation_losses(p,t); result.total.backward()
    assert (p.anchor_presence_logits.grad[0,:11]>0).all()
    assert (p.anchor_presence_logits.grad[0,11:]==0).all()
    assert not t.membership_valid.any() and not t.reachability_valid.any()
    assert not t.dimension_valid.any()


def test_partial_background_does_not_get_promoted_by_loss():
    t=double_target(fixture_loss_targets(case('straight__circle__view2'),[0,1,2,3,4]))
    t=replace(t,anchor_region_complete=torch.zeros_like(t.anchor_region_complete),
              opening_region_complete=torch.zeros_like(t.opening_region_complete))
    p=prediction();result=surface_relation_losses(p,t);result.total.backward()
    assert result.denominators['anchor_presence']==0
    assert p.anchor_presence_logits.grad is None
    assert result.denominators['opening_presence']==2


def test_reject_unknown_or_modified_fixture_and_preserve_input():
    c=case('T__circle__view2'); before=deepcopy(c)
    t=fixture_loss_targets(c,[0,1,2,3,4]); assert c==before
    assert t.anchor_valid.sum()==1 and t.opening_valid.sum()==3
    with pytest.raises(ValueError): fixture_loss_targets(case('T__circle__view0'),[0,1,2,3,4])
    c['poses_world_m'][0][0]+=1
    with pytest.raises(ValueError): fixture_loss_targets(c,[0,1,2,3,4])
