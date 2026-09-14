"""One sealed source-cache-target join; no model forward or optimizer."""
import gzip
import json
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.data.gse_surface_training_join_v1 import join_training_observation
from .gse_query_coverage_probe_v1 import PATH as TARGET_PATH,SHA as TARGET_SHA

TASK='S01_flat_tree_small_C01__c1_mixed'
INPUT_PATH='results/gate3_semantics/gate3_20260907_gse_supplement_input_v1_seed20260906/artifacts/inputs/'+TASK+'.npz'
INPUT_SHA='ad2187578ddb7e13cee8e0043c6218401ce133d3fe7182128a02635112252dcc'
BINDING='3991b17711d1b4e3e120b408f523809477b93c943453eec002fbdbeabcb853ba'
FEATURE_PATH='results/gate3_semantics/gate3_20260908_gse_supplement_features_v1_seed0/artifacts/features/'+BINDING+'.npz'
FEATURE_SHA='b3116ebb7532fde5bb82fb352d7130c1fb5d76060ba76e364c77f97e58637a72'
ENCODER_SHA='200f5c2fbe66d68961cf2aea06e21f747cb5b8d419536008491bb58d3641a8cb'


def inspect_join(bundle,root,opened):
    source=bundle['source']
    if (source['task'],source['source_sequence_id'],source['frame_rows'])!=(TASK,225,[289,290,291,292,293]):
        raise ValueError('exact observation225 required')
    def read(path,h):
        raw=read_pinned(root,path,h);opened[path]=h;return raw
    input_row=bind_feature_input(read(INPUT_PATH,INPUT_SHA),expected_sha256=INPUT_SHA,
        task=TASK,row=1,source_sequence_id=225,frame_rows=source['frame_rows'],observation_count=7)
    if input_row.input_binding_sha256!=BINDING:raise ValueError('input binding drift')
    target=json.loads(gzip.decompress(read(TARGET_PATH,TARGET_SHA)))['observations'][0]['new']
    manifest={k:target[k] for k in ('source_binding','target_record_sha256')}
    entry=dict(source=input_row.provenance,input_binding_sha256=BINDING,
        frozen_encoder_state_sha256=ENCODER_SHA,sha256=FEATURE_SHA)
    example,context=join_training_observation(feature_bytes=read(FEATURE_PATH,FEATURE_SHA),
        feature_entry=entry,source_input=input_row,encoder_state_sha256=ENCODER_SHA,
        produced_targets=target,target_manifest=manifest,bundle=bundle,opening_matching_radius_m=1.,feature_projection_device='cuda:0')
    import torch
    if torch.cuda.max_memory_reserved()>1024**3:raise MemoryError('1GiB projection cap')
    return dict(feature_projection_device='cuda:0',teacher_projection_device='cpu',cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved(),source=source,header=example.header,valid_points=int(example.compact.valid.sum()),
        patches=int(example.patches.valid.sum()),anchors=int(example.targets.anchor_valid.sum()),
        openings=int(example.targets.opening_valid.sum()),grid_content_sha256=context.grid.content_sha256,
        joined_observations=1,qualified_complete_labels=0,optimizer_steps=0,
        scientific_gate_pass=False,opening_radius_role='interface_diagnostic_not_calibrated_threshold')
