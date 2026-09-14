from dataclasses import replace
import numpy as np
import pytest
import torch
from mtare_topo.representation.gse_common_observation import CommonObservation
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.gse_patch_observed_features import pool_patch_observed_features


def example():
    points=np.zeros((57600,3),dtype=np.float32)
    points[0]=[1.1,.1,.1];points[4]=[1.2,.1,.1];points[8]=[20.,.1,.1]
    mask=np.zeros(57600,dtype=bool);mask[[0,4,8]]=True
    patches=extract_surface_patches(points,mask,np.repeat(np.arange(5),11520))
    context=torch.arange(900,dtype=torch.float32)[:,None].expand(-1,128).clone().requires_grad_()
    valid=torch.zeros(900,dtype=torch.bool);valid[:3]=True
    return CommonObservation(context,valid,points,np.array([0,4]),patches,None,None,None)


def test_mean_uses_original_point_token_mapping_not_outside_return():
    obs=example();result=pool_patch_observed_features(obs)
    assert result.shape==(1,1,128) and torch.equal(result,torch.full_like(result,.5))
    assert not result.requires_grad


def test_count_drift_rejected():
    obs=example();p=replace(obs.surface_patches,point_count=np.array([3]))
    with pytest.raises(ValueError,match='count'):pool_patch_observed_features(replace(obs,surface_patches=p))


def test_missing_context_rejected():
    obs=example();valid=obs.full_sensor_valid.clone();valid[1]=False
    with pytest.raises(ValueError,match='context'):pool_patch_observed_features(replace(obs,full_sensor_valid=valid))
