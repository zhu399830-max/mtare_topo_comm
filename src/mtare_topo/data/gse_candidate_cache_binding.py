"""Read authenticated existing candidate/cache evidence; no raw scan/weight IO.

Returns student geometry and context only. Historical cache contains targets,
but those fields are not accessed or returned by this loader. Scoring is separate.
"""
import hashlib,io,json,math
from pathlib import Path
import torch
from mtare_topo.data.gse_synthetic_fit_cache import CACHE,CACHE_SHA
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import CompactFrozenDualPathFeatures
from mtare_topo.representation.gse_candidate_context import ObservationBinding,bind_candidate_context

RUN='results/gate3_semantics/gate3_20260908_gse_patch_center_diagnostic_v1_seed0'
SEAL_SHA='767769ff1c0f395041b816121716a4394e9725e6a006c184a1f6e2b10da0353d'


def authenticated_read(root,relative,expected):
    data=(Path(root)/relative).read_bytes()
    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError('evidence hash mismatch: '+relative)
    return data


def load_bound_candidates(root):
    root=Path(root)
    seal=authenticated_read(root,RUN+'/artifacts/evidence_sha256.txt',SEAL_SHA)
    pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
    def read(path):return authenticated_read(root,path,pins[path])
    card=json.loads(read(RUN+'/config/data_card.json'))
    rows=card['scope']['observations']
    values=torch.load(io.BytesIO(authenticated_read(root,CACHE,CACHE_SHA)),weights_only=True,map_location='cpu')
    if len(rows)!=45 or len(values)!=45:raise ValueError('exact45 observation population required')
    encoder=None
    for row,cached in zip(rows,values,strict=True):
        h=cached['header']
        if (h['observation_id']!=row['case_id'] or h['frame_indices']!=row['frame_rows']
                or h['input_binding_sha256']!=canonical_sha(row) or h['coordinate_frame']!='current_sensor_m'):
            raise ValueError('cached observation binding mismatch')
        if encoder is None:encoder=h['frozen_encoder_state_sha256']
        if h['frozen_encoder_state_sha256']!=encoder:raise ValueError('mixed encoder states')
        record=json.loads(read(RUN+'/artifacts/'+row['case_id']+'.json'))
        if record['case_id']!=row['case_id']:raise ValueError('candidate row mismatch')
        # The known historical frame-name correction is metadata only.
        stamp=ObservationBinding(row['case_id'],tuple(row['frame_rows']),'current_sensor',row['input_sha256'],encoder)
        cache_stamp=ObservationBinding(h['observation_id'],tuple(h['frame_indices']),'current_sensor',row['input_sha256'],h['frozen_encoder_state_sha256'])
        compact=CompactFrozenDualPathFeatures(**cached['compact'])
        pair=[p for p in record['proposals']['pairs'] if p['position_m'] is not None]
        patches=record['patch_candidates']
        positions=torch.tensor([p['position_m'] for p in pair]+[p['position_m'] for p in patches],dtype=compact.points_xyz_m.dtype)
        if len(positions)!=record['score']['candidate_count']:raise ValueError('candidate count drift')
        context=bind_candidate_context(compact,positions,stamp,cache_stamp)
        # Same18-value layout as old patch unary, with independent known bits.
        # Unstored extent/roughness/normal uncertainty remain unknown for ALL.
        unary=torch.zeros(len(positions),18,dtype=positions.dtype)
        known=torch.zeros(len(positions),18,dtype=torch.bool)
        unary[:,:3]=positions/10.;known[:,:3]=True
        for i,p in enumerate(patches,start=len(pair)):
            normal_valid=p['normal_valid']
            if type(normal_valid)!=bool:raise ValueError('normal validity must be explicit')
            if normal_valid:
                unary[i,3:6]=torch.tensor(p['normal']);known[i,3:6]=True
            unary[i,6]=float(normal_valid);known[i,6]=True
            unary[i,11]=math.log1p(p['point_count'])/math.log(57601);known[i,11]=True
            unary[i,12:17]=torch.tensor(p['frame_support'])>0;known[i,12:17]=True
        yield dict(observation_id=row['case_id'],binding=stamp,positions_m=positions,
                   context=context,unary=unary,unary_known=known,axis_pair_count=len(pair),
                   patch_count=len(patches),candidate_source_sha256=pins[RUN+'/artifacts/'+row['case_id']+'.json'],
                   cache_sha256=CACHE_SHA)
