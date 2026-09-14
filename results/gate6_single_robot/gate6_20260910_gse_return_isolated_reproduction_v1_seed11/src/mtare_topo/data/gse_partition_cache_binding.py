"""Authenticated same45 block/context join; no inference or target forwarding.

Historical PT contains targets, but only header and compact student fields are
accessed. Never construct a target object or derive identity from geometry.
"""
import io
import json
from pathlib import Path
import numpy as np
import torch
from .gse_synthetic_fit_cache import CACHE,CACHE_SHA
from .gse_structure_review_v1 import canonical_sha
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.gse_block_context import bind_stamped_block_context
from mtare_topo.representation.gse_candidate_context import ObservationBinding

RUN='results/gate3_semantics/gate3_20260908_gse_spg_paired_extraction_v1_seed0'
SEAL_SHA='66e056fa4f235694d9fb7b39f155a6a8696324514febf47ca263acf584a0cadd'


def load_bound_partition_observations(root,methods=('r1','r2')):
    if not isinstance(methods,tuple) or not methods or len(set(methods))!=len(methods) or not set(methods)<={'r0','r1','r2'}:
        raise ValueError('explicit unique representation list required')
    root=Path(root).resolve()
    seal=read_pinned(root,RUN+'/artifacts/evidence_sha256.txt',SEAL_SHA)
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(path):return read_pinned(root,path,pins[path])
    card=json.loads(read(RUN+'/config/data_card.json'))
    rows=card['scope']['observations']
    cache=torch.load(io.BytesIO(read_pinned(root,CACHE,CACHE_SHA)),weights_only=True,map_location='cpu')
    if len(rows)!=45 or len(cache)!=45:raise ValueError('same45 required')
    expected_tokens=(torch.arange(5)[:,None,None]*180+torch.arange(720)[None,None]//4).expand(5,16,720).reshape(1,-1)
    encoder=None
    for row,cached in zip(rows,cache,strict=True):
        h=cached['header'];compact=cached['compact']
        if (h['observation_id']!=row['case_id'] or h['frame_indices']!=row['frame_rows']
                or h['input_binding_sha256']!=canonical_sha(row) or h['coordinate_frame']!='current_sensor_m'):
            raise ValueError('source header drift')
        if encoder is None:encoder=h['frozen_encoder_state_sha256']
        if h['frozen_encoder_state_sha256']!=encoder:raise ValueError('mixed encoder state')
        memory=compact['frozen_sensor_context'];valid=compact['valid'];token=compact['sensor_token_index']
        xyz=compact['points_xyz_m']
        if (memory.shape!=(1,900,128) or valid.shape!=(1,57600) or xyz.shape!=(1,57600,3)
                or valid.dtype!=torch.bool or memory.dtype!=torch.float32 or xyz.dtype!=torch.float32 or token.dtype!=torch.long
                or not torch.equal(token,expected_tokens)):
            raise ValueError('compact layout mismatch')
        if any(t.requires_grad or t.device.type!='cpu' for t in (memory,valid,token,xyz)):
            raise ValueError('detached CPU cache required')
        if not torch.isfinite(xyz).all():raise ValueError('invalid cached coordinates')
        stamp=ObservationBinding(row['case_id'],tuple(row['frame_rows']),'current_sensor',row['input_sha256'],encoder)
        cache_stamp=ObservationBinding(h['observation_id'],tuple(h['frame_indices']),'current_sensor',row['input_sha256'],h['frozen_encoder_state_sha256'])
        with np.load(io.BytesIO(read(RUN+'/artifacts/partitions/'+row['case_id']+'.npz')),allow_pickle=False) as p:
            points=p['points_xyz_m'];frames=p['frame_index'];source=p['source_flat_ray_index']
            bound_methods={}
            for method in methods:
                # R0 follows existing sensor tensor layout, without geometric
                # segmentation. It still receives XYZ and geometric pretraining.
                assignment=(source//11520)*180+(source%720)//4 if method=='r0' else p[method+'_point_to_group']
                blocks=bind_block_points(points,frames,assignment)
                pooled=bind_stamped_block_context(blocks,source,memory[0].numpy(),valid[0].numpy(),stamp,cache_stamp)
                if method!='r0' and not np.array_equal(pooled['point_count'],p[method+'_point_count']):raise ValueError('point counts changed')
                bound_methods[method]=dict(blocks=blocks,context=pooled['context'])
            delta=np.linalg.norm(xyz[0].numpy()[source].astype(float)-points.astype(float),axis=1)
            yield dict(observation_id=row['case_id'],binding=stamp,methods=bound_methods,source_flat_ray_index=source,
                maximum_projection_difference_m=float(delta.max()) if len(delta) else 0.,
                cache_sha256=CACHE_SHA,partition_sha256=pins[RUN+'/artifacts/partitions/'+row['case_id']+'.npz'])
