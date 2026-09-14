"""Read only existing calibration targets and verify reusable cache hashes."""
import collections
import gzip
import hashlib
import json
from pathlib import Path
from mtare_topo.data.development_paired_scope import compile_scope

ROOT=Path(__file__).resolve().parents[2]


def audit():
    scope=compile_scope(ROOT)
    rows=[r for r in scope['observations'] if r['split']=='calibration']
    if len(rows)!=27:raise ValueError('old calibration population drift')
    def read(p,h):
        raw=(ROOT/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('source drift: '+p)
        return raw
    versions=collections.Counter();entities=set();tasks=set();frames=set();out=[];total=0
    for r in rows:
        source=r['source'];parent=source['task'].split('__')[0]
        if not parent.endswith('_C07'):raise ValueError('not C07 calibration')
        target=json.loads(gzip.decompress(read(r['target_path'],r['target_sha256'])))['produced_targets']
        if target['source_binding']!=r['source_binding']:raise ValueError('target binding mismatch')
        ids=[a['node_id_teacher_only'] for a in target['teacher_provenance']['anchors']]
        entities.update((parent,node) for node in ids);versions[target['producer_version']]+=1
        assets={}
        for kind in ('cache','partition','grid'):
            path=r[kind+'_path'];sha=r[kind+'_sha256'];data=read(path,sha)
            assets[kind]=dict(path=path,sha256=sha,bytes=len(data));total+=len(data)
        tasks.add(source['task']);frames.update((source['task'],f) for f in source['frame_rows'])
        out.append(dict(source=source,parent=parent,target_path=r['target_path'],target_sha256=r['target_sha256'],
                        nodes=ids,positions=[a['position_m'] for a in target['record']['anchors']],
                        full_training_gate_eligible=target['full_training_gate_eligible'],assets=assets))
    return dict(status='EXISTING_CALIBRATION_ASSETS_NOT_NEW_LABEL_QUALIFICATION',
                observations=len(rows),parents=len({r['parent'] for r in out}),tasks=len(tasks),
                unique_variant_frames=len(frames),independent_junctions=len(entities),
                entities=[list(e) for e in sorted(entities)],versions=dict(versions),
                verified_assets=3*len(rows),verified_asset_bytes=total,rows=out,
                missing='V8 labels, multi-position coverage and background/threshold adequacy not established',
                teacher_calls=0,model_calls=0)


if __name__=='__main__':print(json.dumps(audit(),indent=2))
