"""Compile student-cache-only307 partition population from frozen manifests."""
import json
from collections import Counter
from mtare_topo.governance_surface_material import read_pinned

FEATURE_RUN='results/gate3_semantics/gate3_20260908_gse_supplement_features_v1_seed0'
GRID_RUN='results/gate3_semantics/gate3_20260909_gse_development_grids_v1_seed20260906'
MANIFESTS={FEATURE_RUN+'/artifacts/feature_manifest.json':'9e9d12c056c8ad138a66710da93e58554b430af59f717a55eabfe11b6cd051bc',
           GRID_RUN+'/artifacts/grid_manifest.json':'d3d3ddf06a56b2942dba93176d1808919850377e95cb044a53508c95980c8909'}
ENCODER='200f5c2fbe66d68961cf2aea06e21f747cb5b8d419536008491bb58d3641a8cb'


def compile_scope(root):
    docs={p:json.loads(read_pinned(root,p,h)) for p,h in MANIFESTS.items()}
    features=docs[FEATURE_RUN+'/artifacts/feature_manifest.json']['observations']
    grids=docs[GRID_RUN+'/artifacts/grid_manifest.json']
    if grids['status']!='COMPLETE' or len(grids['observations'])!=307:
        raise ValueError('complete exact selection required')
    def key(s):return (s['task'],s['source_sequence_id'],tuple(s['frame_rows']))
    index={key(r['source']):r for r in features}
    if len(index)!=len(features):raise ValueError('duplicate original feature source')
    rows=[];seen=set();frames=set();splits=Counter()
    for grid in grids['observations']:
        source=grid['binding']['source'];k=key(source)
        if k in seen or k not in index:raise ValueError('selection/cache mismatch')
        row=index[k]
        if row['frozen_encoder_state_sha256']!=ENCODER or row['source']['coordinate_frame']!='current_sensor':
            raise ValueError('encoder/coordinate drift')
        if not row['path'].startswith('artifacts/features/') or '..' in row['path'].split('/'):
            raise ValueError('invalid cache path')
        seen.add(k);frames.update((source['task'],f) for f in source['frame_rows']);splits[grid['split']]+=1
        rows.append(dict(source=source,split=grid['split'],cache_path=FEATURE_RUN+'/'+row['path'],
                         cache_sha256=row['sha256'],feature_entry=row))
    if len(frames)!=1529 or splits!=dict(fit=250,calibration=27,development=30):
        raise ValueError('population/split drift')
    return dict(schema='development_partition_scope_v1',manifests_sha256=MANIFESTS,observations=rows,
                counts=dict(observations=307,unique_variant_frames=1529,splits=dict(splits)),
                method='Existing fixed voxel and official SPG native adapters; same original ROI returns and ray indices; no encoding/teacher/model input.',
                fallback='Fail and seal; no missing representation fallback, point truncation or rerun.',
                caps=dict(host_ram_bytes=4*1024**3,output_bytes=2*1024**3,wall_time_s=3600,gpu_bytes=0),
                optimizer_steps=0,new_semantic_labels=0)
