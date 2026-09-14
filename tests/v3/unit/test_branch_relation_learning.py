from dataclasses import replace
import pytest
import torch
from mtare_topo.representation.branch_relation_learning import LocalBranchEvidence,BranchRelationTargets,RayBranchRelationModel,branch_relation_loss,require_trainable_targets,nearest_segment_patches,observed_ray_pairs
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfacePatchBatch
from mtare_topo.representation.gse_structural_representation import StructuralPatchInput
from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
from mtare_topo.topology.branch_hypotheses import group_signed_relations
from mtare_topo.teacher.branch_relation_targets import targets_from_section_witnesses

def observation():
    torch.manual_seed(5);m=9;n=6
    index=torch.tensor([[[j for j in range(m) if j!=i] for i in range(m)]])
    batch=SurfacePatchBatch(torch.rand(1,m,18),torch.ones(1,m,dtype=torch.bool),index,index>=0,torch.rand(1,m,8,9))
    return LocalBranchEvidence(torch.arange(n),torch.rand(n,3)+1,torch.randn(n,128),torch.randn(1,m,128),
        StructuralPatchInput(batch,torch.tensor([[.5,10.]])),CausalFrameOrderContext('sensor_current',(0,1,2,3,4),4,'synthetic'))

def test_nearest_is_segment_distance_not_return_distance_and_limits8():
    centers=torch.tensor([[1.,1,0],[19,0,0],[5,2,0]])
    ids=nearest_segment_patches(torch.tensor([[20.,0,0]]),centers,torch.ones(3,dtype=torch.bool))
    assert ids.tolist()==[[0,2,1]]

def test_abc_share_seed_parameters_and_observed_pairs_are_label_free():
    models=[]
    for v in 'ABC':torch.manual_seed(0);models.append(RayBranchRelationModel(v))
    assert all(torch.equal(models[0].state_dict()[k],m.state_dict()[k]) for m in models[1:] for k in models[0].state_dict())
    assert observed_ray_pairs(torch.tensor([0,1,719,720])).shape[1]==2

def test_gradients_reach_relation_attention_and_geometry_without_encoder_grad():
    e=observation();e.observed_features.requires_grad_();e.patch_observed_features.requires_grad_()
    model=RayBranchRelationModel('C');pairs=torch.tensor([[0,1],[2,3],[4,5]])
    p=model(e,pairs);t=BranchRelationTargets(torch.tensor([1.,0,float('nan')]),torch.tensor([True,True,False]),('synthetic same','synthetic different',''),True)
    loss,counts=branch_relation_loss(p,t);loss.backward()
    assert counts==dict(positive=1,negative=1,unknown=1)
    for name in ('relation.0.weight','query.weight','patch_encoder.geometry.0.weight'):
        assert model.get_parameter(name).grad.abs().sum()>0
    assert e.observed_features.grad is None and e.patch_observed_features.grad is None

def test_unknown_values_do_not_become_negative_or_nonzero_gradient():
    e=observation();p=RayBranchRelationModel('A')(e,torch.tensor([[0,1]]));p.logits.retain_grad()
    t=BranchRelationTargets(torch.tensor([float('nan')]),torch.tensor([False]),('',))
    loss,counts=branch_relation_loss(p,t);loss.backward()
    assert loss.item()==0 and p.logits.grad.item()==0 and counts['negative']==0
    with pytest.raises(ValueError):require_trainable_targets(t)

def test_forward_rejects_targets_and_ray_permutation_preserves_logits():
    e=observation();model=RayBranchRelationModel('B').eval();pairs=torch.tensor([[0,1],[2,3]])
    original=model(e,pairs).logits
    order=torch.tensor([5,4,3,2,1,0]);inverse=torch.argsort(order)
    shuffled=replace(e,ray_ids=e.ray_ids[order],endpoints_m=e.endpoints_m[order],observed_features=e.observed_features[order])
    assert torch.allclose(original,model(shuffled,inverse[pairs]).logits,atol=1e-6)
    with pytest.raises(TypeError):model(e,pairs,targets='hidden')

def test_ideal_signed_relations_keep_two_groups_and_unknown_not_singleton_branch():
    h=group_signed_relations([0,1,2,3,4],[(0,1,.9),(2,3,.9),(1,2,-1),(3,4,None)])
    assert h.groups==((0,1),(2,3)) and h.unresolved_ray_ids==(4,) and not h.physical_connection_verified

def test_tied_conflict_is_unresolved_without_order_winner():
    edges=[(0,1,.8),(1,2,.8),(0,2,-.8)]
    a=group_signed_relations([0,1,2],edges);b=group_signed_relations([2,0,1],list(reversed(edges)))
    assert a==b and a.groups==() and a.unresolved_ray_ids==(0,1,2)

def test_different_section_ids_cannot_manufacture_negative_supervision():
    w=dict(supporting_ray_indices={'first':[0,1],'second':[2,3]})
    t,c=targets_from_section_witnesses([0,1,2,3],[[0,1],[1,2]],w)
    assert t.known.tolist()==[True,False] and c['qualified_negative_pairs']==0
    assert c['different_section_pairs_kept_unknown']==1
    with pytest.raises(ValueError):require_trainable_targets(t)
