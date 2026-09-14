"""Exact sealed v1 cache migration: frame-name metadata only, tensors unchanged."""
import hashlib,io
from dataclasses import fields
import torch
from .gse_synthetic_fit_scope import compile_scope as original_scope
from .gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from mtare_topo.representation.gse_surface_relation_model_v1 import SurfacePatchBatch
from mtare_topo.representation.gse_surface_losses_v1 import SurfaceLossTargets
from mtare_topo.representation.gse_surface_training_v1 import SurfaceTrainingExample,_validate_header

CACHE='results/gate3_semantics/gate3_20260908_gse_synthetic_fit_v1_seed0/artifacts/shared_examples.pt'
CACHE_SHA='bc5df284fbde3728851e4a98bdf55e2bd140b6a5967927537b283d4520b7565e'


def compile_scope(root):
    result=original_scope(root)
    result['cached_examples']=dict(path=CACHE,sha256=CACHE_SHA,correction='coordinate_frame name only; current_sensor_m -> current_sensor')
    result['input_sha256'][CACHE]=CACHE_SHA
    return result


def load_examples(payload,scope,encoder_sha):
    if hashlib.sha256(payload).hexdigest()!=CACHE_SHA:raise ValueError('sealed cached tensors drift')
    values=torch.load(io.BytesIO(payload),map_location='cpu',weights_only=True)
    if len(values)!=45:raise ValueError('exact cached population required')
    result=[]
    for value,row in zip(values,scope['observations'],strict=True):
        h=dict(value['header'])
        if h['observation_id']!=row['case_id'] or h['frame_indices']!=row['frame_rows'] or h['input_binding_sha256']!=canonical_sha(row):
            raise ValueError('cached source identity differs')
        if h['coordinate_frame']!='current_sensor_m':raise ValueError('only original incorrect header accepted')
        h['coordinate_frame']='current_sensor';_validate_header(h,encoder_sha)
        compact=CompactFrozenDualPathFeatures(**value['compact']);patch=SurfacePatchBatch(**value['patches']);target=SurfaceLossTargets(**value['targets'])
        for item in (compact,patch,target):
            for f in fields(item):
                t=getattr(item,f.name)
                if not torch.is_tensor(t) or t.requires_grad:raise ValueError('detached cache tensors required')
        result.append(SurfaceTrainingExample(h,compact,patch,target))
    return result
