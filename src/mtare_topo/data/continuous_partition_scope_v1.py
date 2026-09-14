"""Same native partition contract, exact30 frozen continuous feature caches."""
import json
from mtare_topo.governance_surface_material import read_pinned
from .development_partition_scope import ENCODER

FEATURE_RUN='results/gate3_semantics/gate3_20260909_gse_continuous_features_v1_seed0'
SEAL_SHA='39f48ff6e49d9e543c1cd75b3d6678c830523002e8c658ab7c213310562b3d25'


def compile_scope(root):
    seal=FEATURE_RUN+'/artifacts/evidence_sha256.txt'
    raw=read_pinned(root,seal,SEAL_SHA)
    pins={p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    path=FEATURE_RUN+'/artifacts/feature_manifest.json'
    manifest=json.loads(read_pinned(root,path,pins[path]))
    rows=[];seen=set();frames=set()
    expected_tasks={'S09_flat_complex_C04__'+v for v in ('ellipse','rounded_rectangle','c1_mixed')}
    expected={(t,s,tuple(range(4012+i,4017+i))) for t in expected_tasks for i,s in enumerate(range(158206,158216))}
    for row in manifest['observations']:
        source={k:row['source'][k] for k in ('task','source_sequence_id','frame_rows')}
        key=(source['task'],source['source_sequence_id'],tuple(source['frame_rows']))
        if key in seen or key not in expected:raise ValueError('source identity duplicate or out of scope')
        if row['frozen_encoder_state_sha256']!=ENCODER or row['source']['coordinate_frame']!='current_sensor':
            raise ValueError('original encoder/current sensor contract drift')
        p=row['path']
        if not p.startswith('artifacts/features/') or '..' in p.split('/') or pins.get(FEATURE_RUN+'/'+p)!=row['sha256']:
            raise ValueError('cache seal/path mismatch')
        seen.add(key);frames.update((source['task'],f) for f in source['frame_rows'])
        rows.append(dict(source=source,split='fit',cache_path=FEATURE_RUN+'/'+p,cache_sha256=row['sha256'],feature_entry=row))
    if seen!=expected or len(frames)!=42:raise ValueError('incomplete30 continuous population')
    return dict(schema='continuous_partition_scope_v1',manifests_sha256={seal:SEAL_SHA,path:pins[path]},observations=rows,
        counts=dict(observations=30,unique_variant_frames=42,splits=dict(fit=30)),
        method='Reuse unchanged produce_partitions with official native SPG and original voxel side output; identical ROI points/ray indices; no training or tuning.',
        fallback='Fail and seal without representation substitution, point truncation, resampling or retry.',
        caps=dict(host_ram_bytes=4*1024**3,output_bytes=2*1024**3,wall_time_s=3600,gpu_bytes=0),optimizer_steps=0,new_semantic_labels=0)
