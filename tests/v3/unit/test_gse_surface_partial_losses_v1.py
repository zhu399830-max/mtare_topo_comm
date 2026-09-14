"""Synthetic protocol/gradient checks, not real teacher accuracy evidence."""
from dataclasses import replace
import pytest
import torch
from test_gse_surface_losses_v1 import prediction, targets
from test_gse_reference_exclusion_binding_v1 import fixture
from mtare_topo.representation.gse_surface_partial_losses_v1 import partial_surface_losses


def setup():
    b,g,q,binding=fixture();p,t=prediction(),targets()
    positions=p.anchor_position_m.detach().clone();positions[0,31]=torch.from_numpy(q[0])
    p=replace(p,anchor_position_m=positions.requires_grad_())
    return p,t,dict(bundles=[b],grids=[g],expected_bindings=[binding])


def test_joint_positive_negative_and_unknown_gradients():
    p,t,kwargs=setup();r,e=partial_surface_losses(p,t,**kwargs)
    assert e[0]['used_unmatched_negative_query_indices']==[31]
    assert r.denominators['anchor_presence']==3
    assert r.denominators['opening_presence']==1
    r.total.backward()
    grad=p.anchor_presence_logits.grad[0]
    assert grad[0]<0 and grad[2]<0 and grad[31]>0
    assert torch.count_nonzero(grad)==3
    assert not e[0]['training_eligible'] and not e[0]['whole_region_complete']


def test_prediction_movement_requeries_not_reuses_lattice_mask():
    p,t,kwargs=setup()
    positions=p.anchor_position_m.detach().clone();positions[0,31]=torch.tensor([9.,9.,9.])
    p=replace(p,anchor_position_m=positions.requires_grad_())
    r,e=partial_surface_losses(p,t,**kwargs)
    assert e[0]['used_unmatched_negative_query_indices']==[]
    assert r.denominators['anchor_presence']==2


def test_conditional_negatives_respect_declared_score_region():
    p,t,kwargs=setup()
    pos=p.opening_position_m.detach().clone();pos[0,63]=p.anchor_position_m[0,31].detach()
    p=replace(p,opening_position_m=pos.requires_grad_())
    # The source-bound query is observed, but not in this declared score area.
    t=replace(t,score_region_radius_m=torch.tensor([4.],dtype=t.score_region_radius_m.dtype))
    r,e=partial_surface_losses(p,t,opening_matching_radius_m=1.,**kwargs)
    assert e[0]['reference_negative_mask'][31]
    assert e[0]['used_unmatched_negative_query_indices']==[]
    assert e[0]['opening_exclusion']['used_unmatched_negative_query_indices']==[]
    assert r.denominators['anchor_presence']==2
    assert r.denominators['opening_presence']==1
    r.total.backward()
    assert p.anchor_presence_logits.grad[0,31]==0
    assert p.opening_presence_logits.grad[0,63]==0


def test_opening_negative_requires_explicit_radius_and_own_inventory():
    p,t,kwargs=setup()
    pos=p.opening_position_m.detach().clone();pos[0,63]=p.anchor_position_m[0,31].detach()
    p=replace(p,opening_position_m=pos.requires_grad_())
    unchanged,_=partial_surface_losses(p,t,**kwargs)
    assert unchanged.denominators['opening_presence']==1
    r,e=partial_surface_losses(p,t,opening_matching_radius_m=1.,**kwargs)
    assert r.denominators['opening_presence']==2
    assert e[0]['opening_exclusion']['used_unmatched_negative_query_indices']==[63]
    r.total.backward()
    assert p.opening_presence_logits.grad[0,63]>0
    assert p.opening_presence_logits.grad[0,1]<0
    assert torch.count_nonzero(p.opening_presence_logits.grad)==2


def test_matched_query_is_never_both_positive_and_negative():
    p,t,kwargs=setup()
    # Artificial matching target isolates the loss behavior. This is not a
    # claim that the fixture's construction teacher places a real anchor here.
    pos=t.anchor_position_m.clone();pos[0,0]=p.anchor_position_m[0,31].detach()
    t=replace(t,anchor_position_m=pos)
    r,e=partial_surface_losses(p,t,**kwargs)
    assert r.assignments['anchor'][0,0]==31
    assert e[0]['used_unmatched_negative_query_indices']==[]
    r.total.backward();assert p.anchor_presence_logits.grad[0,31]<0


def test_rejects_contract_mix_and_wrong_source():
    p,t,kwargs=setup()
    with pytest.raises(ValueError,match='mix'):
        partial_surface_losses(p,replace(t,anchor_region_complete=torch.ones(1,dtype=torch.bool)),**kwargs)
    kwargs['expected_bindings'][0]['source']['frame_rows'][-1]=99
    with pytest.raises(ValueError,match='binding'):
        partial_surface_losses(p,t,**kwargs)


def test_frozen_positive_manifest_and_negative_source_are_shared():
    from copy import deepcopy
    from dataclasses import fields
    from test_gse_surface_target_adapter_v1 import record
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.representation.gse_surface_partial_losses_v1 import bound_partial_surface_losses
    p,_,kwargs=setup()
    p=replace(p,**{f.name:getattr(p,f.name).float() for f in fields(p)
                   if getattr(p,f.name).dtype==torch.float64})
    row=record();row['source_frame_indices']=[0,1,2,3,4]
    manifest=dict(source_binding=deepcopy(kwargs['expected_bindings'][0]),target_record_sha256=canonical_sha(row))
    produced=dict(record=row,**deepcopy(manifest))
    options=dict(produced_targets=[produced],manifest_rows=[manifest],bundles=kwargs['bundles'],grids=kwargs['grids'])
    result,evidence=bound_partial_surface_losses(p,**options)
    assert result.denominators['anchor_presence']==2
    assert evidence[0]['used_unmatched_negative_query_indices']==[31]
    produced['record']['anchors'][0]['position_m'][0]=2
    with pytest.raises(ValueError,match='positive record'):
        bound_partial_surface_losses(p,**options)
    produced['record']=deepcopy(row)
    # Even a self-consistent edited envelope cannot replace the frozen record.
    produced['target_record_sha256']=canonical_sha(produced['record'])
    with pytest.raises(ValueError,match='positive record'):
        bound_partial_surface_losses(p,**options)
