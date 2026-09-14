from dataclasses import replace
import numpy as np
import torch
import pytest
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from mtare_topo.representation.gse_common_observation import assemble_common_observation
from mtare_topo.representation.gse_common_structure_memory import CommonStructureMemory
from mtare_topo.representation.gse_surface_relation_model_v1 import collate_surface_patches


def observation():
    xyz=torch.zeros(1,57600,3);xyz[0,0]=torch.tensor([3.,0,0]);xyz[0,4]=torch.tensor([3.,1,0]);xyz[0,8]=torch.tensor([20.,0,0])
    mask=torch.zeros(1,57600,dtype=torch.bool);mask[0,[0,4,8]]=True
    index=torch.from_numpy(np.repeat(np.arange(5),11520)*180+np.tile(np.arange(720)//4,80))[None]
    return assemble_common_observation(CompactFrozenDualPathFeatures(xyz,torch.zeros(1,900,128),mask,index),np.zeros((5,3)))


def test_full_and_outside_ray_tokens_survive_without_patch_or_queries():
    torch.manual_seed(0);model=CommonStructureMemory('A');obs=observation()
    result=model(obs)
    assert result.features.shape==(1,1800,128)
    assert result.valid[0,2] and result.valid[0,902]
    assert result.patch_slice==(1800,1800) and not result.is_structure_prediction
    assert not any('quer' in n or 'presence' in n for n,_ in model.named_parameters())


def test_b_c_share_weights_and_observation_branch_but_different_relation_use():
    torch.manual_seed(0);b=CommonStructureMemory('B');c=CommonStructureMemory('C');c.load_state_dict(b.state_dict())
    obs=observation();p=collate_surface_patches([obs.surface_patches])
    altered=replace(p,relation=p.relation+1)
    with torch.no_grad():
        b0=b(obs,patch_batch=p);b1=b(obs,patch_batch=altered)
        c0=c(obs,patch_batch=p);c1=c(obs,patch_batch=altered)
    assert torch.equal(b0.features,b1.features)
    assert torch.equal(b0.features[:,:1800],c0.features[:,:1800])
    assert not torch.equal(c0.features[:,1800:],c1.features[:,1800:])


def test_measured_vs_clipped_flag_enters_ray_features_and_gradient():
    torch.manual_seed(0);model=CommonStructureMemory('A');obs=observation()
    flags=obs.local_rays.end_is_observed_return.copy();flags[-1]=True
    changed=replace(obs,local_rays=replace(obs.local_rays,end_is_observed_return=flags))
    first=model(obs);second=model(changed)
    assert not torch.equal(first.features[:,900:1800],second.features[:,900:1800])
    first.features[:,900:1800].square().sum().backward()
    assert any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.ray_adapter.parameters())


def test_bad_adjacency_cannot_be_clamped_into_false_edge():
    model=CommonStructureMemory('C');obs=observation();p=collate_surface_patches([obs.surface_patches])
    ids=p.neighbor_index.clone();ids[0,0,0]=-2
    with pytest.raises(ValueError,match='identity'):
        model(obs,patch_batch=replace(p,neighbor_index=ids))
