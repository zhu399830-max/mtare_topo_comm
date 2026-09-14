import numpy as np
import torch
import pytest
from dataclasses import replace
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from mtare_topo.representation.gse_common_observation import assemble_common_observation


def fixture():
    xyz=torch.zeros(1,57600,3);xyz[0,0]=torch.tensor([3.,0,0]);xyz[0,4]=torch.tensor([20.,0,0])
    valid=torch.zeros(1,57600,dtype=torch.bool);valid[0,[0,4]]=True
    index=torch.from_numpy(np.repeat(np.arange(5),11520)*180+np.tile(np.arange(720)//4,80))[None]
    context=torch.arange(900*128,dtype=torch.float32).reshape(1,900,128)
    return CompactFrozenDualPathFeatures(xyz,context,valid,index)


def test_outside_only_token_preserved_without_fake_surface():
    c=fixture();before=c.valid.clone()
    r=assemble_common_observation(c,np.zeros((5,3)))
    assert r.full_sensor_valid[:2].tolist()==[True,True]
    assert torch.equal(r.full_sensor_context,c.frozen_sensor_context[0])
    assert r.surface_return_indices.tolist()==[0]
    assert r.local_rays.ray_indices.tolist()==[0,4]
    assert r.local_rays.end_is_observed_return.tolist()==[True,False]
    assert r.ray_sensor_token_indices.tolist()==[0,1]
    assert r.surface_patches.ray_input_index.tolist()==[0,4]
    assert r.surface_patches.point_patch_index[4]==-1
    assert torch.equal(c.valid,before)


def test_all_outside_returns_still_have_observation_and_no_patches():
    c=fixture();c.points_xyz_m[0,0]=torch.tensor([15.,0,0])
    r=assemble_common_observation(c,np.zeros((5,3)))
    assert len(r.surface_patches.centers_m)==0
    assert len(r.local_rays.ray_indices)==2
    assert r.full_sensor_valid.sum()==2


def test_empty_returns_remain_unknown_not_free_space():
    c=fixture();c=replace(c,valid=torch.zeros_like(c.valid))
    r=assemble_common_observation(c,np.zeros((5,3)))
    assert not r.full_sensor_valid.any()
    assert not len(r.local_rays.ray_indices)
    assert not r.teacher_in_forward


def test_layout_or_origins_not_silently_substituted():
    c=fixture();bad=c.sensor_token_index.clone();bad[0,0]=99
    with pytest.raises(ValueError,match='layout'):
        assemble_common_observation(replace(c,sensor_token_index=bad),np.zeros((5,3)))
    with pytest.raises(ValueError,match='origins'):
        assemble_common_observation(c,np.zeros((4,3)))
