from dataclasses import replace
import pytest
import torch
from test_conditional_anchor_branch_loss import fixture
from mtare_topo.representation.conditional_anchor_branch_loss import conditional_anchor_branch_loss
from mtare_topo.representation.conditional_branch_selection_v1 import conditional_branch_selection


def test_node_loss_and_unknown_gradients_unchanged():
    p,t,k=fixture()
    old=conditional_anchor_branch_loss(p,t,**k);new=conditional_branch_selection(p,t,**k)
    assert torch.equal(old['total'],new['total'])
    assert old['conditional_anchor_evidence']==new['conditional_anchor_evidence']
    new['total'].backward()
    assert p.presence_logits.grad[0]>0 and p.presence_logits.grad[1]<0 and p.presence_logits.grad[2]==0
    assert p.branch_logits.grad is None


def test_only_assigned_node_has_branch_gradients():
    p,t,k=fixture();t=replace(t,directions=(torch.tensor([[1.,0,0]],dtype=torch.float64),),branches_complete=(True,))
    old=conditional_anchor_branch_loss(p,t,**k);new=conditional_branch_selection(p,t,**k)
    for name in ('anchor_presence','anchor_position'):
        assert torch.equal(old['terms'][name],new['terms'][name])
    new['total'].backward()
    assert p.branch_logits.grad[1,0]<0
    assert p.branch_logits.grad[0,0]==0 and p.branch_logits.grad[2,0]==0


def test_source_binding_still_enforced():
    p,t,k=fixture();k['frozen_manifest']['target_record_sha256']='bad'
    with pytest.raises(ValueError):conditional_branch_selection(p,t,**k)


def test_invalid_original_target_not_hidden_by_node_adapter():
    p,t,k=fixture();t=replace(t,directions=(torch.tensor([[2.,0,0]],dtype=torch.float64),))
    with pytest.raises(ValueError,match='unit'):conditional_branch_selection(p,t,**k)
