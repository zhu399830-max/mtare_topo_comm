import hashlib
import io
from copy import deepcopy
import numpy as np
import pytest
import torch
from test_gse_surface_partial_losses_v2 import opening_setup
from mtare_topo.data.gse_surface_feature_input_v1 import SurfaceFeatureInput
from mtare_topo.data.gse_surface_training_join_v1 import join_training_observation
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points


def args():
    _,kw=opening_setup();b=kw['bundles'][0];s=b['student']
    image=np.stack((s['ranges_m']/np.float32(50),s['valid_mask'].astype(np.float32)),axis=1)
    source=SurfaceFeatureInput(image,s['relative_translation_current_sensor_m'],s['relative_yaw_current_sensor_deg'],deepcopy(b['source']),'a'*64)
    p,v=register_causal_lidar_points(torch.from_numpy(image[None]),torch.from_numpy(source.translation_m[None]),torch.from_numpy(source.yaw_deg[None]))
    stream=io.BytesIO();np.savez_compressed(stream,points_xyz_m=p.reshape(1,-1,3).numpy(),valid=v.reshape(1,-1).numpy(),frozen_sensor_context=np.zeros((1,900,128),np.float32))
    raw=stream.getvalue();entry=dict(source=deepcopy(source.provenance),input_binding_sha256='a'*64,frozen_encoder_state_sha256='b'*64,sha256=hashlib.sha256(raw).hexdigest())
    return dict(feature_bytes=raw,feature_entry=entry,source_input=source,encoder_state_sha256='b'*64,
        produced_targets=kw['produced_targets'][0],target_manifest=kw['manifest_rows'][0],bundle=b,opening_matching_radius_m=1.)


def test_join_retains_raw_points_and_teacher_is_separate():
    a=args();example,context=join_training_observation(**a)
    assert example.compact.valid.sum()==1
    assert example.targets.opening_valid.sum()==1
    assert context.bundle is a['bundle']
    assert example.header['target_binding_sha256']==a['target_manifest']['target_record_sha256']
    assert not hasattr(example.compact,'construction_teacher_only')


def test_identity_and_raw_motion_mismatch_rejected():
    a=args();a['source_input'].provenance['source_sequence_id']=99
    with pytest.raises(ValueError,match='identity'):join_training_observation(**a)
    a=args()
    # Copy to avoid changing both aliases in the synthetic fixture.
    a['bundle']['student']['relative_translation_current_sensor_m']=np.ones((5,3),np.float32)
    with pytest.raises(ValueError,match='raw student'):join_training_observation(**a)


@pytest.mark.skipif(not torch.cuda.is_available(),reason='exact CUDA contract requires CUDA')
def test_gpu_feature_cache_keeps_canonical_cpu_teacher_grid():
    from mtare_topo.data.gse_feature_projection_v1 import project_input
    from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
    a=args();s=a['source_input']
    p,v=project_input(s.range_valid,s.translation_m,s.yaw_deg,device='cuda:0')
    f=io.BytesIO();np.savez_compressed(f,points_xyz_m=p.numpy(),valid=v.numpy(),frozen_sensor_context=np.zeros((1,900,128),np.float32))
    a['feature_bytes']=f.getvalue();a['feature_entry']['sha256']=hashlib.sha256(f.getvalue()).hexdigest()
    example,context=join_training_observation(**a,feature_projection_device='cuda:0')
    assert torch.equal(example.compact.points_xyz_m,p)
    cpu,mask=project_input(s.range_valid,s.translation_m,s.yaw_deg,device='cpu')
    origins=np.broadcast_to(s.translation_m[:,None,None,:],(5,16,720,3)).reshape(-1,3)
    grid=build_surface_ray_grid(origins,cpu[0].numpy(),mask[0].numpy(),np.repeat(np.arange(5),11520))
    assert context.grid.source_geometry_sha256==grid.source_geometry_sha256
    assert context.grid.content_sha256==grid.content_sha256
    assert context.feature_projection_device=='cuda:0'


def test_projection_backend_is_not_silently_selected():
    from mtare_topo.data.gse_feature_projection_v1 import project_input
    s=args()['source_input']
    with pytest.raises(ValueError,match='projection contract'):
        project_input(s.range_valid,s.translation_m,s.yaw_deg,device='auto')
