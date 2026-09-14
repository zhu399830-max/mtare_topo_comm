from dataclasses import replace
import torch
import pytest
from mtare_topo.representation.gse_candidate_readout import SharedCandidateReadout,CandidateGeometryBatch
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfacePatchBatch


def inputs():
    torch.manual_seed(11);n=4
    idx=torch.arange(n).roll(1)[None,:,None].expand(1,n,8).clone()
    g=CandidateGeometryBatch(torch.randn(1,n,18),torch.ones(1,n,dtype=torch.bool),idx,
                        torch.ones(1,n,8,dtype=torch.bool),torch.randn(1,n,8,9),torch.ones(1,n,18,dtype=torch.bool))
    return torch.randn(1,n,3),torch.randn(1,n,128),g


def test_paths_have_same_layout_and_geometry_isolation():
    p,f,g=inputs();models=[SharedCandidateReadout(k) for k in 'ABC']
    for m in models[1:]:m.load_state_dict(models[0].state_dict())
    changed=replace(g,unary=g.unary+2,relation=g.relation+4)
    torch.testing.assert_close(models[0](p,f,g).presence_logits,models[0](p,f,changed).presence_logits,rtol=0,atol=0)
    relations=replace(g,relation=g.relation+4)
    torch.testing.assert_close(models[1](p,f,g).presence_logits,models[1](p,f,relations).presence_logits,rtol=0,atol=0)
    assert not torch.allclose(models[1](p,f,g).presence_logits,models[1](p,f,changed).presence_logits)
    assert not torch.allclose(models[2](p,f,g).presence_logits,models[2](p,f,relations).presence_logits)


@pytest.mark.parametrize('path',list('ABC'))
def test_permutation_and_fixed_positions_and_backward(path):
    p,f,g=inputs();m=SharedCandidateReadout(path);q=torch.tensor([2,0,3,1]);inv=q.argsort()
    perm=CandidateGeometryBatch(g.unary[:,q],g.valid[:,q],inv[g.neighbor_index[:,q]],g.neighbor_valid[:,q],g.relation[:,q],g.unary_known[:,q])
    a=m(p,f,g);b=m(p[:,q],f[:,q],perm)
    torch.testing.assert_close(a.presence_logits[:,q],b.presence_logits,atol=1e-6,rtol=1e-6)
    torch.testing.assert_close(a.position_m,p,atol=0,rtol=0)
    assert a.position_m.data_ptr()!=p.data_ptr()
    a.presence_logits.square().mean().backward()
    assert all(v.grad is not None and torch.isfinite(v.grad).all() for v in m.parameters())


def test_invalid_neighbor_not_silently_clamped():
    p,f,g=inputs();m=SharedCandidateReadout('C')
    with pytest.raises(ValueError):m(p,f,replace(g,neighbor_index=torch.full_like(g.neighbor_index,-1)))
    valid=g.valid.clone();valid[0,0]=False
    with pytest.raises(ValueError):m(p,f,replace(g,valid=valid))


def test_no_teacher_argument_or_backbone_gradient():
    p,f,g=inputs();f.requires_grad_();m=SharedCandidateReadout('C')
    with pytest.raises(TypeError):m(p,f,g,targets=p)
    m(p,f,g).presence_logits.sum().backward()
    assert f.grad is None


def test_unknown_unary_numeric_placeholder_not_used():
    p,f,g=inputs();m=SharedCandidateReadout('B');mask=torch.zeros_like(g.unary_known)
    a=replace(g,unary_known=mask)
    b=replace(a,unary=g.unary+1000)
    torch.testing.assert_close(m(p,f,a).presence_logits,m(p,f,b).presence_logits,rtol=0,atol=0)
