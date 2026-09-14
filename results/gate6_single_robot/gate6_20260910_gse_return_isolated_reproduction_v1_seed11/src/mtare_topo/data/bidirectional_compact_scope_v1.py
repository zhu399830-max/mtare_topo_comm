"""Join existing860 compact features and reusable paired partitions."""
import json
from collections import Counter
from mtare_topo.governance_surface_material import read_pinned
from .bidirectional_feature_scope_v1 import INVENTORY,INVENTORY_SHA
from .development_partition_scope import ENCODER

FEATURE_RUN='results/gate3_semantics/gate3_20260909_gse_bidirectional_features_v1_seed0'
FEATURE_SEAL='e1e91cd9945bbf4951f2503d0a056afb706d0bc5ff2d8b0d0166ce1111353657'
PARTITION_RUN='results/gate3_semantics/gate3_20260909_gse_development_partitions_v1_seed0'
PARTITION_MANIFEST_SHA='0cd19b29b06815cc62b26a8e2ec09a36498c844de7fa1be8881ff0fc7693da81'

def compile_scope(root):
    opened={}
    def read(p,h):
        raw=read_pinned(root,p,h);opened[p]=h;return raw
    inv=json.loads(read(INVENTORY,INVENTORY_SHA))
    seal=read(FEATURE_RUN+'/artifacts/evidence_sha256.txt',FEATURE_SEAL)
    pins={p:h for h,p in (l.split('  ',1) for l in seal.decode().splitlines())}
    p=FEATURE_RUN+'/artifacts/feature_manifest.json';features=json.loads(read(p,pins[p]))['observations']
    identity=lambda s:(s['task'],s['source_sequence_id'],tuple(s['frame_rows']))
    fi={identity(f['source']):f for f in features}
    if len(fi)!=817 or len(features)!=817:raise ValueError('817 unique features required')
    p=PARTITION_RUN+'/artifacts/partition_manifest.json';pm=json.loads(read(p,PARTITION_MANIFEST_SHA))
    if pm['status']!='COMPLETE':raise ValueError('incomplete partition source')
    pi={identity(r['source']):r for r in pm['observations']}
    if len(pi)!=len(pm['observations']):raise ValueError('duplicate partition source')
    rows=[];seen=set()
    for r in inv['rows']:
        s=r['source'];key=identity(s)
        if key in seen:raise ValueError('duplicate inventory source')
        seen.add(key);cached=r['feature_cache']
        if cached:
            if key in fi:raise ValueError('new/old feature overlap')
            f=cached['feature_entry'];p=cached['path'];h=cached['sha256']
        else:
            f=fi.pop(key);p=FEATURE_RUN+'/'+f['path'];h=f['sha256']
            if pins[p]!=h:raise ValueError('feature seal mismatch')
        if identity(f['source'])!=key or f['frozen_encoder_state_sha256']!=ENCODER or h!=f['sha256']:raise ValueError('feature identity/state mismatch')
        source={k:s[k] for k in ('task','source_sequence_id','frame_rows')}
        part=pi.get(key)
        if part and (part['cache_sha256']!=h or part['split']!=s['split']):raise ValueError('partition cache/split mismatch')
        reuse=None if part is None else dict(path=PARTITION_RUN+'/artifacts/partitions/'+part['file'],sha256=part['sha256'])
        rows.append(dict(source=source,split=s['split'],cache_path=p,cache_sha256=h,feature_entry=f,reused_partition=reuse))
    if fi or len(rows)!=860:raise ValueError('extra/missing features')
    return dict(schema='bidirectional_compact_scope_v1',observations=rows,metadata_sha256=opened,
                counts=dict(observations=len(rows),splits=dict(Counter(r['split'] for r in rows)),
                            reusable_partitions=sum(r['reused_partition'] is not None for r in rows),
                            missing_partitions=sum(r['reused_partition'] is None for r in rows)),
                limitations=['reference-conditioned fit/calibration only','no new labels','metadata reuse pending payload checks'],model_calls=0)
