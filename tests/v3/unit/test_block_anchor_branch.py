import torch
import pytest
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout
from mtare_topo.representation.block_anchor_branch import BlockAnchorBranch
from mtare_topo.representation.anchor_branch_loss import PartialAnchorBranches,anchor_branch_loss


def fixture(n=7):
    torch.manual_seed(0);legacy=BlockStructureReadout();model=BlockAnchorBranch(legacy,branch_slots=4)
    data=dict(embedding=torch.randn(n,128,requires_grad=True),centers_m=torch.randn(n,3),
        extent_m=torch.rand(n,3),degenerate=torch.zeros(n,dtype=torch.bool))
    context=torch.randn(n,128,requires_grad=True)
    return legacy,model,data,context


def test_exact_legacy_queries_no_shared_storage():
    legacy,model,data,context=fixture();saved={k:v.clone() for k,v in legacy.state_dict().items()}
    captured=[];hook=legacy.decoder.register_forward_hook(lambda m,i,o:captured.append(o.detach().clone()))
    legacy(data,context);hook.remove()
    torch.testing.assert_close(model.encode_queries(data,context),captured[0][0,:32],rtol=0,atol=0)
    assert legacy.queries.data_ptr()!=model.queries.data_ptr()
    with torch.no_grad():model.queries.add_(1)
    assert all(torch.equal(v,saved[k]) for k,v in legacy.state_dict().items())


def test_loss_reaches_block_features_not_frozen_context_or_legacy():
    legacy,model,data,context=fixture();p=model(data,context)
    t=PartialAnchorBranches(p.position_m[:1].detach().clone()+.01,(torch.tensor([[1.,0,0]]),),False,(False,))
    result=anchor_branch_loss(p,t);result['total'].backward()
    assert data['embedding'].grad is not None and data['embedding'].grad.abs().sum()>0
    assert context.grad is None and all(x.grad is None for x in legacy.parameters())
    assert model.adapter[0].weight.grad is not None and model.queries.grad is not None


def test_block_order_and_empty_observation():
    _,model,data,context=fixture();order=torch.arange(6,-1,-1)
    p=model(data,context);q=model({k:v[order] for k,v in data.items()},context[order])
    torch.testing.assert_close(p.position_m,q.position_m,atol=2e-5,rtol=2e-5)
    assert model({k:v[:0] for k,v in data.items()},context[:0]) is None


def test_teacher_fields_rejected():
    _,model,data,context=fixture();data['node_id']=torch.ones(7)
    with pytest.raises(ValueError,match='observation'):model(data,context)
