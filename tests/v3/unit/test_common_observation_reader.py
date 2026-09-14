import hashlib,io
import numpy as np
import pytest
import torch
from mtare_topo.data.gse_common_observation_reader import decode_common_observation
from mtare_topo.representation.gse_common_observation import assemble_common_observation
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures


def arrays():
    xyz=torch.zeros(1,57600,3);xyz[0,0]=torch.tensor([3.,0,0]);xyz[0,4]=torch.tensor([20.,0,0])
    valid=torch.zeros(1,57600,dtype=torch.bool);valid[0,[0,4]]=True
    index=torch.from_numpy(np.repeat(np.arange(5),11520)*180+np.tile(np.arange(720)//4,80))[None]
    obs=assemble_common_observation(CompactFrozenDualPathFeatures(xyz,torch.zeros(1,900,128),valid,index),np.zeros((5,3)))
    a={k:getattr(obs,k) for k in ('registered_returns_xyz_m','surface_return_indices','ray_sensor_token_indices','ray_history_indices')}
    a.update(full_sensor_context=obs.full_sensor_context.numpy(),full_sensor_valid=obs.full_sensor_valid.numpy())
    a.update({'ray_'+k:v for k,v in vars(obs.local_rays).items() if isinstance(v,np.ndarray)})
    a.update({'patch_'+k:v for k,v in vars(obs.surface_patches).items() if isinstance(v,np.ndarray)})
    return a


def decode(a):
    f=io.BytesIO();np.savez_compressed(f,**a);b=f.getvalue()
    return decode_common_observation(b,expected_sha256=hashlib.sha256(b).hexdigest())


def test_roundtrip_separates_surface_and_crop_ray():
    obs=decode(arrays())
    assert obs.surface_return_indices.tolist()==[0]
    assert obs.local_rays.ray_indices.tolist()==[0,4]
    assert obs.local_rays.end_is_observed_return.tolist()==[True,False]
    assert obs.full_sensor_valid[:2].tolist()==[True,True]
    assert not obs.teacher_in_forward


def test_extra_teacher_field_rejected():
    a=arrays();a['teacher_node_id']=np.zeros(1)
    with pytest.raises(ValueError,match='fields'):decode(a)


def test_identity_and_hash_drift_rejected():
    with pytest.raises(ValueError,match='hash'):
        decode_common_observation(b'invalid',expected_sha256='0'*64)
    a=arrays();a['ray_sensor_token_indices']=np.array([0,999])
    with pytest.raises(ValueError,match='layout'):decode(a)


def test_fake_surface_and_full_mask_rejected():
    a=arrays();a['surface_return_indices']=np.array([0,4])
    with pytest.raises(ValueError,match='surface index'):decode(a)
    a=arrays();a['full_sensor_valid']=np.zeros(900,dtype=bool)
    with pytest.raises(ValueError,match='context mask'):decode(a)
