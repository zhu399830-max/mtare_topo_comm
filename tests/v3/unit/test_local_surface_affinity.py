from dataclasses import replace
import pytest
import torch
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfacePatchBatch
from mtare_topo.representation.gse_local_surface_affinity import LocalSurfaceAffinity,affinity_loss


def inputs():
    g=torch.Generator().manual_seed(9)
    x=torch.randn(1,3,128,generator=g)
    ids=torch.full((1,3,8),-1,dtype=torch.long)
    ids[0,:,0]=torch.tensor([1,0,1])
    p=SurfacePatchBatch(torch.randn(1,3,18,generator=g),torch.ones(1,3,dtype=torch.bool),
                       ids,ids>=0,torch.randn(1,3,8,9,generator=g))
    return x,p


def test_same_initialization_no_center_or_teacher_forward():
    a=LocalSurfaceAffinity('A');b=LocalSurfaceAffinity('B');c=LocalSurfaceAffinity('C')
    b.load_state_dict(a.state_dict());c.load_state_dict(a.state_dict())
    assert not any('center' in n or 'presence' in n for n,_ in a.named_parameters())
    x,p=inputs()
    assert not a(x,p).is_structural_membership


def test_geometry_and_relation_ablation_wiring():
    x,p=inputs();changed=replace(p,relation=p.relation+2)
    for variant in ('A','B','C'):
        torch.manual_seed(1);model=LocalSurfaceAffinity(variant)
        one=model(x,p).logits;two=model(x,changed).logits
        assert torch.equal(one,two)==(variant!='C')
    model=LocalSurfaceAffinity('A');u=p.unary.clone();u[...,3:]+=3
    assert torch.equal(model(x,p).logits,model(x,replace(p,unary=u)).logits)


def test_reverse_pair_symmetric_not_a_directed_topology_edge():
    x,p=inputs();pred=LocalSurfaceAffinity('C')(x,p)
    assert torch.equal(pred.logits[0,0,0],pred.logits[0,1,0])


def test_unknown_nan_does_not_enter_loss_or_gradient():
    x,p=inputs();model=LocalSurfaceAffinity('C');out=model(x,p);out.logits.retain_grad()
    known=torch.zeros_like(out.pair_valid);known[0,0,0]=True;known[0,1,0]=True
    values=torch.full_like(out.logits,float('nan'));values[known]=1
    loss,counts=affinity_loss(out,values,known);loss.backward()
    assert counts==dict(positive_entries=2,negative_entries=0,unknown_entries=1)
    assert out.logits.grad[0,2,0]==0 and any(v.grad is not None for v in model.parameters())


def test_no_known_targets_zero_loss_not_training_success():
    x,p=inputs();out=LocalSurfaceAffinity('A')(x,p)
    loss,counts=affinity_loss(out,torch.full_like(out.logits,float('nan')),torch.zeros_like(out.pair_valid))
    assert loss.item()==0 and counts['unknown_entries']==3


def test_padded_pair_cannot_receive_background_label():
    x,p=inputs();out=LocalSurfaceAffinity('A')(x,p);known=torch.ones_like(out.pair_valid)
    with pytest.raises(ValueError,match='invalid pair'):affinity_loss(out,torch.zeros_like(out.logits),known)


def test_observation_is_not_optimized():
    x,p=inputs();x.requires_grad_();model=LocalSurfaceAffinity('C')
    model(x,p).logits.sum().backward();assert x.grad is None
