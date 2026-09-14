import hashlib
import io
from copy import deepcopy
import numpy as np
import pytest
import torch
from test_gse_reference_exclusion_binding_v1 import fixture
from mtare_topo.data.gse_surface_feature_input_v1 import SurfaceFeatureInput
from mtare_topo.data.gse_surface_feature_cache_v1 import bind_feature_cache
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points


def setup():
    b,_,_,_=fixture();s=b['student']
    image=np.stack((s['ranges_m']/np.float32(50),s['valid_mask'].astype(np.float32)),axis=1)
    source=SurfaceFeatureInput(image,s['relative_translation_current_sensor_m'],s['relative_yaw_current_sensor_deg'],b['source'],'a'*64)
    p,v=register_causal_lidar_points(torch.from_numpy(image[None]),torch.from_numpy(source.translation_m[None]),torch.from_numpy(source.yaw_deg[None]))
    arrays=dict(points_xyz_m=p.reshape(1,-1,3).numpy(),valid=v.reshape(1,-1).numpy(),frozen_sensor_context=np.zeros((1,900,128),np.float32))
    def pack(a):
        f=io.BytesIO();np.savez_compressed(f,**a);return f.getvalue()
    raw=pack(arrays)
    entry=dict(source=deepcopy(source.provenance),input_binding_sha256='a'*64,frozen_encoder_state_sha256='b'*64,sha256=hashlib.sha256(raw).hexdigest())
    return raw,entry,source,arrays,pack


def test_roundtrip_keeps_full_points_and_reconstructs_token_layout():
    raw,e,s,_,_=setup();c=bind_feature_cache(raw,manifest_entry=e,source_input=s,encoder_state_sha256='b'*64)
    assert c.points_xyz_m.shape==(1,57600,3) and c.valid.sum()==1
    assert c.sensor_token_index[0,-1]==899


def test_digest_identity_and_geometry_drift_rejected():
    raw,e,s,a,pack=setup()
    with pytest.raises(ValueError,match='digest'):
        bind_feature_cache(raw+b'x',manifest_entry=e,source_input=s,encoder_state_sha256='b'*64)
    with pytest.raises(ValueError,match='encoder'):
        bind_feature_cache(raw,manifest_entry=e,source_input=s,encoder_state_sha256='c'*64)
    a['points_xyz_m'][a['valid']]+=1;raw=pack(a);e['sha256']=hashlib.sha256(raw).hexdigest()
    with pytest.raises(ValueError,match='coordinates'):
        bind_feature_cache(raw,manifest_entry=e,source_input=s,encoder_state_sha256='b'*64)
