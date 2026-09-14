"""Source-bound in-memory fixture example assembly; runner owns run authority.

This module never selects data or loads weights. Only the four sensor arrays
enter the supplied frozen encoder. Declared oracle targets are loss-only.
"""
from dataclasses import fields, replace
import gzip
import hashlib
import io
import json
import numpy as np
import torch
from .gse_synthetic_fit_scope import declared_cases
from .gse_structure_review_v1 import canonical_sha
from mtare_topo.evaluation.gse_fixture_loss_targets import fixture_loss_targets
from mtare_topo.representation.gse_surface_patches_v1 import extract_surface_patches
from mtare_topo.representation.gse_surface_relation_model_v1 import collate_surface_patches
from mtare_topo.representation.gse_surface_training_v1 import SurfaceTrainingExample, _validate_header


def cache_header(row,target_sha256,encoder_state_sha256):
    # Cache schema spells the frame differently from the loss target schema.
    # Both describe current-sensor coordinates in metres; no numeric transform.
    header=dict(schema_version='gse_surface_cached_input_header_v1',observation_id=row['case_id'],
        frame_indices=row['frame_rows'],coordinate_frame='current_sensor',
        input_binding_sha256=canonical_sha(row),target_binding_sha256=target_sha256,
        frozen_encoder_state_sha256=encoder_state_sha256)
    _validate_header(header,encoder_state_sha256)
    return header


def build_example(input_bytes, source_bytes, row, *, encoder_adapter, encoder_state_sha256, device):
    for payload,key in ((input_bytes,'input_sha256'),(source_bytes,'source_record_sha256')):
        if hashlib.sha256(payload).hexdigest()!=row[key]:raise ValueError('source bytes drift')
    source=json.loads(gzip.decompress(source_bytes))
    candidates=[c for c in declared_cases() if c['case_id']==row['case_id']]
    if len(candidates)!=1 or canonical_sha(candidates[0])!=row['case_sha256'] or canonical_sha(source['case'])!=row['case_sha256']:
        raise ValueError('fixture declaration differs from frozen scope')
    if source['source']['frame_rows']!=row['frame_rows']:raise ValueError('frame provenance drift')
    with np.load(io.BytesIO(input_bytes),allow_pickle=False) as payload:
        ranges=payload['ranges_m'];valid=payload['valid_mask']
        translation=payload['relative_translation_current_sensor_m'];yaw=payload['relative_yaw_current_sensor_deg']
        if ranges.shape!=(5,16,720) or ranges.dtype!=np.float32 or valid.dtype!=np.uint8 or valid.shape!=ranges.shape:
            raise ValueError('frozen sensor shape/dtype drift')
        if not np.isin(valid,[0,1]).all():raise ValueError('nonbinary validity')
        image=np.stack((ranges/np.float32(50),valid.astype(np.float32)),axis=1)
        args=[torch.from_numpy(a.copy()[None]).to(device) for a in (image,translation,yaw)]
        compact=encoder_adapter.extract_compact_features(*args)
        compact=replace(compact,**{f.name:getattr(compact,f.name).detach().cpu() for f in fields(compact)})
        patch=extract_surface_patches(compact.points_xyz_m[0].numpy(),compact.valid[0].numpy(),
            np.repeat(np.arange(5),16*720),ray_origins_m=np.repeat(translation,16*720,axis=0))
    target=fixture_loss_targets(candidates[0],row['frame_rows'])
    target_hash=hashlib.sha256()
    for f in fields(target):
        value=getattr(target,f.name).contiguous().numpy()
        target_hash.update(canonical_sha([f.name,str(value.dtype),list(value.shape)]).encode())
        target_hash.update(value.tobytes())
    header=cache_header(row,target_hash.hexdigest(),encoder_state_sha256)
    return SurfaceTrainingExample(header,compact,collate_surface_patches([patch]),target)
