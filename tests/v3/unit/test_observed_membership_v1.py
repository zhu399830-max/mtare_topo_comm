import inspect
import torch
from dataclasses import replace
from mtare_topo.representation.gse_observed_membership_loss import observed_membership_loss
from mtare_topo.representation.gse_observed_membership_v1 import ObservedMembershipV1
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfacePatchBatch


def inputs():
    torch.manual_seed(17)
    points=torch.randn(1,12,3)
    features=torch.randn(1,12,128,requires_grad=True)
    return points,features,torch.ones(1,12,dtype=torch.bool)


def test_shapes_gradients_and_teacher_free_signature():
    x,f,v=inputs();m=ObservedMembershipV1('A')
    r=m(x,f,v)
    assert r.membership_logits.shape==(12,12)
    assert r.anchor_position_m.shape==(12,3)
    assert torch.allclose(r.opening_position_m.norm(dim=-1),torch.full((12,),10.))
    (r.membership_logits.square().mean()+r.anchor_position_m.square().mean()
        +r.opening_position_m.square().sum(dim=0)[0]).backward()
    assert m.member_anchor.weight.grad.abs().sum()>0
    assert m.anchor_position.weight.grad.abs().sum()>0
    assert m.opening_position.weight.grad.abs().sum()>0
    assert f.grad is None
    assert set(inspect.signature(m.forward).parameters)=={
        'points_xyz_m','frozen_point_context','valid','patches','sensor_token_index'}
    assert 'anchor_queries' not in dict(m.named_parameters())


def test_permutation_and_input_dependence():
    x,f,v=inputs();m=ObservedMembershipV1('A').eval()
    r=m(x,f,v);order=torch.arange(11,-1,-1)
    s=m(x[:,order],f[:,order],v[:,order])
    assert torch.allclose(r.anchor_position_m,s.anchor_position_m,atol=2e-5)
    assert torch.allclose(r.membership_logits,s.membership_logits,atol=2e-5)
    changed=m(-x,f,v)
    assert not torch.allclose(r.membership_logits,changed.membership_logits)


def test_empty_and_duplicate_observations():
    x,f,v=inputs();m=ObservedMembershipV1('A')
    assert m(x,f,torch.zeros_like(v)) is None
    r=m(x[:,:1].expand(-1,12,-1),f[:,:1].expand(-1,12,-1),v)
    assert r.membership_logits.shape==(1,1)


def test_unknown_has_no_membership_gradient_and_scores_do_not_match():
    x,f,v=inputs();m=ObservedMembershipV1('A');r=m(x,f,v)
    logits=r.membership_logits.detach().clone().requires_grad_()
    r=replace(r,membership_logits=logits)
    record=dict(anchors=[{'position_m':r.anchor_position_m[0].detach().tolist()}],
        openings=[{'position_m':r.opening_position_m[i].detach().tolist()} for i in (0,1)],
        membership=[[True],[None]])
    out=observed_membership_loss(r,record);out['total'].backward()
    assert torch.count_nonzero(logits.grad)==1
    assert out['unknown_relations']==1
    s=replace(r,anchor_presence_logits=-r.anchor_presence_logits*100,
        opening_presence_logits=-r.opening_presence_logits*100)
    assert observed_membership_loss(s,record)['assignments']==out['assignments']


def test_relation_ablation_same_weights_neighbors_and_output_head():
    x,f,v=inputs();b=ObservedMembershipV1('B');c=ObservedMembershipV1('C')
    c.load_state_dict(b.state_dict())
    index=torch.full((1,2,8),-1,dtype=torch.long)
    index[0,0,0]=1;index[0,1,0]=0
    patch=SurfacePatchBatch(torch.randn(1,2,18),torch.ones(1,2,dtype=torch.bool),
        index,index>=0,torch.zeros(1,2,8,9))
    rb=b(x,f,v,patch);rc=c(x,f,v,patch)
    assert torch.equal(rb.membership_logits,rc.membership_logits)
    changed=replace(patch,relation=torch.ones_like(patch.relation))
    assert torch.equal(rb.membership_logits,b(x,f,v,changed).membership_logits)
    assert not torch.allclose(rc.membership_logits,c(x,f,v,changed).membership_logits)


def test_gradient_isolation_does_not_change_forward_or_stop_head_learning():
    x,f,v=inputs();original=ObservedMembershipV1('A')
    isolated=ObservedMembershipV1('A',membership_gradient_to_shared=False)
    isolated.load_state_dict(original.state_dict())
    a=original(x,f,v);b=isolated(x,f,v)
    assert all(torch.equal(value,getattr(b,name)) for name,value in vars(a).items())
    b.membership_logits.sum().backward()
    assert isolated.member_anchor.weight.grad.abs().sum()>0
    assert isolated.member_opening.weight.grad.abs().sum()>0
    assert all(p.grad is None for name,p in isolated.named_parameters() if not name.startswith('member_'))
    isolated.zero_grad(set_to_none=True)
    isolated(x,f,v).anchor_position_m.square().sum().backward()
    assert isolated.raw_adapter[0].weight.grad.abs().sum()>0
