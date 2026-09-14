import torch
import pytest
from dataclasses import replace
from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder,StructuralPatchInput,bind_structural_patches
from mtare_topo.representation.gse_structure_context import CausalStructureContext
from test_local_surface_affinity import inputs as legacy_inputs


def inputs():
    x,p=legacy_inputs()
    return x,StructuralPatchInput(p,torch.tensor([[.5,10.]]))


def context():
    return (CausalStructureContext(timestamp_s=4.,coordinate_frame='sensor_current',
        source_frame_ids=(0,1,2,3,4),source_timestamps_s=(0.,1.,2.,3.,4.),sequence_id='sequence'),)


def test_two_level_representation_and_no_source_or_center_head():
    x,p=inputs();model=GeometryStructureEncoder('C');out=model(x,p,context())
    assert out.local_tokens.shape==(1,3,128) and out.region_embedding.shape==(1,128)
    assert not out.is_place_identity and not out.relations.connectivity_certified
    assert not out.relations.uncertainty_calibrated
    assert not any(n.startswith('backbone.score.') or 'center' in n for n,_ in model.named_parameters())
    assert out.token_valid.all() and out.relations.computation_valid.sum()==3


def test_common_init_and_relation_wiring():
    x,p=inputs();torch.manual_seed(1);state=GeometryStructureEncoder('A').state_dict()
    for variant in 'ABC':
        model=GeometryStructureEncoder(variant);model.load_state_dict(state)
        one=model(x,p,context());two=model(x,replace(p,batch=replace(p.batch,relation=p.batch.relation+1)),context())
        assert torch.equal(one.local_tokens,two.local_tokens)==(variant!='C')


def test_permutation_preserves_pooled_structure():
    x,p=inputs();order=torch.tensor([2,0,1]);inv=torch.argsort(order)
    q=p.batch;old=q.neighbor_index[:,order];new=torch.where(old>=0,inv[old.clamp_min(0)],-1)
    perm=replace(p,batch=replace(q,unary=q.unary[:,order],valid=q.valid[:,order],neighbor_index=new,
                 neighbor_valid=q.neighbor_valid[:,order],relation=q.relation[:,order]))
    model=GeometryStructureEncoder('C');a=model(x,p,context());b=model(x[:,order],perm,context())
    assert torch.allclose(a.local_tokens[:,order],b.local_tokens,atol=1e-6)
    assert torch.allclose(a.region_embedding,b.region_embedding,atol=1e-6)


def test_empty_observation_not_confident_embedding():
    x,p=inputs();q=p.batch;p=replace(p,batch=replace(q,valid=torch.zeros_like(q.valid),neighbor_index=torch.full_like(q.neighbor_index,-1),neighbor_valid=torch.zeros_like(q.neighbor_valid)))
    out=GeometryStructureEncoder('C')(x,p,context())
    assert not out.region_valid.any() and not out.local_tokens.any() and not out.region_embedding.any()
    assert not out.relations.computation_valid.any() and not out.relations.regression_scale.any()


def test_features_frozen_and_heads_have_gradients():
    x,p=inputs();x.requires_grad_();model=GeometryStructureEncoder('C');out=model(x,p,context())
    out.relations.axis_abs_dot.sum().backward()
    assert x.grad is None and model.relation_head[-1].weight.grad is not None


def test_context_required_and_not_network_input():
    x,p=inputs();model=GeometryStructureEncoder('A')
    with pytest.raises(ValueError):model(x,p,())
    c=context();other=(replace(c[0],sequence_id='another_sequence'),)
    assert torch.equal(model(x,p,c).local_tokens,model(x,p,other).local_tokens)


def test_reciprocal_relations_have_physical_symmetry():
    x,p=inputs();r=GeometryStructureEncoder('C')(x,p,context()).relations
    assert torch.equal(r.axis_abs_dot[0,0,0],r.axis_abs_dot[0,1,0])
    assert torch.equal(r.correspondence_logits[0,0,0],r.correspondence_logits[0,1,0])
    assert torch.equal(r.height_difference_m[0,0,0],-r.height_difference_m[0,1,0])
    assert torch.equal(r.section_log_ratio[0,0,0],-r.section_log_ratio[0,1,0])


def test_real_extractor_metric_support_and_scale_rejection():
    import numpy as np
    from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
    from mtare_topo.representation.gse_surface_relation_model_v1 import collate_surface_patches
    xyz=np.array([[4.1,.1,.1],[4.2,.2,.1],[4.3,.2,.2]],dtype=float)
    for voxel,roi in ((.5,10.),(1.,5.)):
        patches=extract_surface_patches(xyz,np.ones(3,dtype=bool),np.array([0,1,2]),voxel_size_m=voxel,roi_radius_m=roi)
        batch=bind_structural_patches([patches]);x=torch.zeros(1,len(patches.centers_m),128)
        model=GeometryStructureEncoder('A')
        if roi!=10.:
            with pytest.raises(ValueError,match='scales required'):model(x,batch,context())
        else:
            out=model(x,batch,context())
            assert np.allclose(out.support_centers_m[0].numpy(),patches.centers_m)
            assert np.allclose(out.support_extents_m[0].numpy(),patches.bounds_max_m-patches.bounds_min_m)
            with pytest.raises(ValueError,match='scales required'):
                model(x,batch.batch,context())
