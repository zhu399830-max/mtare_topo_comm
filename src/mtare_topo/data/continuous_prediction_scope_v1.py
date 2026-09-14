"""Metadata-only exact join of existing continuous caches and final weights."""
import json
from .continuous_partition_scope_v1 import compile_scope as feature_scope
from mtare_topo.governance_surface_material import read_pinned

PARTITIONS='results/gate3_semantics/gate3_20260909_gse_continuous_partitions_v1_seed0'
TRAIN='results/gate3_semantics/gate3_20260909_gse_block_relation_train_v1_seed0'
INITIAL='118f9f420b7675fea79ae2b43f58c0d7a98b4b404ec4516b205f94ffa7329ab9'


def compile_scope(root):
    source=feature_scope(root)
    metadata=dict(source['manifests_sha256'])
    def sealed(run, digest):
        path=run+'/artifacts/evidence_sha256.txt'
        raw=read_pinned(root,path,digest);metadata[path]=digest
        return {p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    pins=sealed(PARTITIONS,'fa19584a529a3c9067903b52aaadf3a66007ef6a77a36bb3d859599de0d7fabb')
    path=PARTITIONS+'/artifacts/partition_manifest.json'; metadata[path]=pins[path]
    manifest=json.loads(read_pinned(root,path,pins[path]))
    def key(s):return s['task'],s['source_sequence_id'],tuple(s['frame_rows'])
    indexed={key(row['source']):row for row in manifest['observations']}
    if manifest['status']!='COMPLETE' or len(indexed)!=30 or len(manifest['observations'])!=30:
        raise ValueError('complete unique30 partitions required')
    rows=[]
    for row in source['observations']:
        part=indexed.pop(key(row['source']))
        file=part['file']
        if '/' in file or not file.endswith('.npz'):raise ValueError('closed partition filename required')
        path=PARTITIONS+'/artifacts/partitions/'+file
        if pins.get(path)!=part['sha256'] or part['cache_sha256']!=row['cache_sha256']:
            raise ValueError('partition cache binding drift')
        rows.append(dict(row,partition_path=path,partition_sha256=part['sha256']))
    if indexed:raise ValueError('unmatched partition')
    weights=sealed(TRAIN,'e24094e4a26a400f5099c158fb70a00701b53514af3fcce0e307b2bedca3a482')
    checkpoints={name:dict(path=TRAIN+'/checkpoints/'+name+'_final.pt',
        sha256=weights[TRAIN+'/checkpoints/'+name+'_final.pt'],relation_attributes=name=='c1') for name in ('c0','c1')}
    return dict(schema='continuous_prediction_scope_v1',metadata_sha256=metadata,observations=rows,
        checkpoints=checkpoints,initial_sha256=INITIAL,
        counts=dict(parent_maps=1,physical_traversals=1,geometry_variants=3,observations=30,
                    predictions=60,unique_variant_frames=42,split='fit'),
        teacher_payload=False,optimizer_steps=0,calibration=False,graph_evaluation=False)
