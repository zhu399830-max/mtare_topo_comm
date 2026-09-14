"""Decode authenticated compact-cache bytes; never pair by row or load weights."""
import hashlib
import io
import zipfile
import numpy as np
import torch
from .gse_surface_feature_input_v1 import SurfaceFeatureInput
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points


def bind_feature_cache(payload, *, manifest_entry, source_input, encoder_state_sha256, projection_device='cpu'):
    if type(source_input) is not SurfaceFeatureInput:
        raise ValueError('independently bound raw feature input required')
    if (type(payload) is not bytes or not 0<len(payload)<=4*1024**2
            or hashlib.sha256(payload).hexdigest()!=manifest_entry['sha256']):
        raise ValueError('bounded compact feature byte digest mismatch')
    if (manifest_entry['source']!=source_input.provenance
            or manifest_entry['input_binding_sha256']!=source_input.input_binding_sha256
            or manifest_entry['frozen_encoder_state_sha256']!=encoder_state_sha256):
        raise ValueError('feature source or frozen encoder mismatch')
    contract={'points_xyz_m':((1,57600,3),np.dtype('float32')),
              'frozen_sensor_context':((1,900,128),np.dtype('float32')),
              'valid':((1,57600),np.dtype('bool'))}
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        items=z.infolist()
        if (len(items)!=3 or {x.filename for x in items}!={k+'.npy' for k in contract}
                or sum(x.file_size for x in items)>4*1024**2):
            raise ValueError('bounded three-field feature NPZ required')
    values={}
    with np.load(io.BytesIO(payload),allow_pickle=False) as z:
        for key,(shape,dtype) in contract.items():
            a=z[key]
            if a.shape!=shape or a.dtype!=dtype or not np.isfinite(a).all():
                raise ValueError('compact array contract mismatch: '+key)
            values[key]=torch.from_numpy(a.copy())
    from .gse_feature_projection_v1 import project_input
    points,valid=project_input(source_input.range_valid,source_input.translation_m,source_input.yaw_deg,device=projection_device)
    if not torch.equal(valid,values['valid']) or not torch.equal(points[valid],values['points_xyz_m'][valid]):
        raise ValueError('feature coordinates do not belong to bound original scans')
    t=torch.arange(5)[:,None,None];az=torch.arange(720)[None,None]
    index=(t*180+az//4).expand(5,16,720).reshape(1,-1)
    return CompactFrozenDualPathFeatures(values['points_xyz_m'],values['frozen_sensor_context'],values['valid'],index)
