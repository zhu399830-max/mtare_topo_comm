from copy import deepcopy
import torch
import pytest
from mtare_topo.representation.block_relation_increment_v1 import block_relations,BlockRelationIncrementV1
from mtare_topo.representation.block_anchor_branch import BlockAnchorBranch
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout


def test_layout_permutation_and_height_information():
    c=torch.tensor([[0.,0,0],[1,0,0],[0,1,0],[0,0,2]])
    s=torch.ones_like(c);d=torch.zeros(4,dtype=torch.bool);p=torch.tensor([2,0,3,1])
    a=block_relations(c,s,d);b=block_relations(c[p],s[p],d[p])
    for new,old in enumerate(p):
        mask=b.neighbor_valid[0,new]
        assert torch.equal(p[b.neighbor_index[0,new,mask]],a.neighbor_index[0,old,a.neighbor_valid[0,old]])
        assert torch.allclose(b.relation[0,new],a.relation[0,old])
    assert a.relation[0,0,2,2]==.2
    with pytest.raises(ValueError):block_relations(torch.zeros_like(c),s,d)


def test_paired_parameters_forward_and_gradients():
    torch.manual_seed(0)
    legacy=BlockAnchorBranch(BlockStructureReadout(),branch_slots=64)
    c1=BlockRelationIncrementV1(legacy,relation_attributes=True)
    c0=deepcopy(c1);c0.relation_attributes=False
    assert sum(p.numel() for p in c0.parameters())==sum(p.numel() for p in c1.parameters())
    features=dict(embedding=torch.randn(5,128),centers_m=torch.randn(5,3),
                  extent_m=torch.ones(5,3),degenerate=torch.zeros(5,dtype=torch.bool))
    context=torch.randn(5,128)
    a=c0(features,context);b=c1(features,context)
    assert not torch.allclose(a.position_m,b.position_m)
    b.position_m.sum().backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in c1.layers.parameters())
    perm=torch.tensor([3,1,4,0,2])
    bp=c1({k:v[perm] for k,v in features.items()},context[perm])
    assert torch.allclose(b.position_m,bp.position_m,atol=2e-5)
