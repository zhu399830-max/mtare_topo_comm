from __future__ import annotations

import numpy as np
import pytest
import torch

from mtare_topo.evaluation.phase3_stability_contract import compare_outputs,distribution_summary,fixed_ray_column_mask


def test_fixed_mask_matches_v1r3_contract_without_mutating_input():
    source=torch.zeros((2,2,16,720),dtype=torch.float32);source[:,0]=.25;source[:,1]=1.0
    masked=fixed_ray_column_mask(source)
    assert torch.all(source[:,0]==.25) and torch.all(source[:,1]==1.0)
    assert torch.all(masked[:,0,:,::10]==1.0) and torch.all(masked[:,1,:,::10]==0.0)
    assert torch.all(masked[:,0,:,1::10]==.25) and torch.all(masked[:,1,:,1::10]==1.0)


def test_compare_outputs_uses_frozen_positive_roll_and_cosine():
    logits=torch.arange(720,dtype=torch.float32).repeat(2,1);z=torch.tensor([[1.,0.],[0.,1.]])
    base={"direction_logits":logits,"z_role":z}
    rotated={"direction_logits":torch.roll(logits,12,dims=-1),"z_role":z.clone()}
    masked={"direction_logits":logits,"z_role":z.clone()}
    values=compare_outputs(base,rotated,masked)
    assert np.max(values["rotation_direction_absolute_logit_error"])==0.0
    assert np.allclose(values["rotation_z_role_cosine"],1.0)
    assert np.allclose(values["masking_z_role_cosine"],1.0)


def test_distribution_summary_and_validation():
    result=distribution_summary(np.asarray([0.,1.,2.]))
    assert result["mean"]==1.0 and result["maximum"]==2.0 and result["minimum"]==0.0
    with pytest.raises(ValueError):distribution_summary(np.asarray([]))
    with pytest.raises(ValueError):fixed_ray_column_mask(torch.zeros((1,2,16,719)))
