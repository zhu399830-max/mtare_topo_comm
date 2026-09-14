from dataclasses import replace
import pytest
import torch
from test_conditional_anchor_branch_loss import fixture
from mtare_topo.representation.conditional_observed_anchor_loss_v1 import conditional_observed_anchor_loss


def test_binding_balanced_presence_and_unknown_preserved():
    p,t,k=fixture(); r=conditional_observed_anchor_loss(p,t,**k)
    assert r['anchor_assignment'].tolist()==[1]
    assert r['anchor_group_counts']==dict(positive=1,negative=1,unknown=1)
    r['total'].backward()
    assert p.presence_logits.grad[0]==-p.presence_logits.grad[1]
    assert p.presence_logits.grad[2]==0 and p.branch_logits.grad is None


def test_branch_objective_still_only_for_matched_node():
    p,t,k=fixture()
    t=replace(t,directions=(torch.tensor([[1.,0,0]],dtype=torch.float64),),branches_complete=(True,))
    r=conditional_observed_anchor_loss(p,t,**k);r['total'].backward()
    assert p.branch_logits.grad[1,0]<0
    assert p.branch_logits.grad[0,0]==p.branch_logits.grad[2,0]==0


@pytest.mark.parametrize('damage',['positions','hash','source','indices','branch'])
def test_invalid_target_or_source_stops(damage):
    p,t,k=fixture()
    if damage=='positions': t.position_m[0,0]=1
    if damage=='hash': k['produced_targets']['target_record_sha256']='bad'
    if damage=='source': k['frozen_manifest']['source_binding']['source']['frame_rows'][-1]=8
    if damage=='indices': k['frozen_manifest']['target_reference_indices']=(0,0)
    if damage=='branch': t=replace(t,directions=(torch.tensor([[2.,0,0]],dtype=torch.float64),))
    with pytest.raises(ValueError): conditional_observed_anchor_loss(p,t,**k)
