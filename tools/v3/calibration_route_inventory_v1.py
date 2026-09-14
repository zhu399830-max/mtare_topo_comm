"""Expand existing calibration identities using sealed interval metadata only."""
import collections
import hashlib
import json
from pathlib import Path
from mtare_topo.governance_surface_selection import INVENTORY,SEAL_SHA256
from mtare_topo.data.gse_surface_feature_scope_v1 import INPUT as ORIGINAL,INPUT_SEAL as ORIGINAL_SEAL
from mtare_topo.data.gse_supplement_teacher_scope_v1 import INPUT as SUPPLEMENT,INPUT_SEAL as SUPPLEMENT_SEAL

ROOT=Path(__file__).resolve().parents[2]


def inventory():
    opened={}
    def read(p,h):
        raw=(ROOT/p).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=h:raise ValueError('metadata drift: '+p)
        opened[p]=h;return raw
    def seal(run,h):
        raw=read(run+'/artifacts/evidence_sha256.txt',h)
        return {p:h for h,p in (l.split('  ',1) for l in raw.decode().splitlines())}
    old=json.loads(read('docs/figures/gse_graph/old_calibration_reuse_20260909.json',
                       '84e6532086a623a8edd77aaaeef9b75674a00d89de49fc5395dc32bad9528014'))
    pins=seal(INVENTORY,SEAL_SHA256);reports={};selected={};rows=[]
    for source in old['rows']:
        parent=source['parent'];s=source['source']
        if parent not in reports:
            p=INVENTORY+'/artifacts/'+parent+'_identity_intervals.json';reports[parent]=json.loads(read(p,pins[p]))
        matches=[]
        for interval in reports[parent]['intervals']:
            for variant in interval['variants']:
                if variant['task']!=s['task']:continue
                for i,sid in enumerate(variant['source_sequence_ids']):
                    if sid==s['source_sequence_id'] and variant['frame_rows'][i]==s['frame_rows']:
                        matches.append((interval,variant))
        if len(matches)!=1:raise ValueError('calibration source does not identify unique interval')
        interval,v=matches[0];k=(v['task'],interval['traversal_id'])
        selected[k]=(interval,v)
    for (task,traversal),(interval,v) in sorted(selected.items()):
        for i,sid in enumerate(v['source_sequence_ids']):
            rows.append(dict(task=task,traversal_id=traversal,source_sequence_id=sid,frame_rows=v['frame_rows'][i],
                             sequence_row=v['sequence_rows'][i],decision_route_arc_m=v['decision_arc_m'][i],split='calibration'))
    key=lambda r:(r['task'],r['source_sequence_id'],tuple(r['frame_rows']))
    wanted={key(r):r for r in rows}
    if len(wanted)!=len(rows):raise ValueError('duplicate expanded source')
    cached={}
    for run,h in ((ORIGINAL,ORIGINAL_SEAL),(SUPPLEMENT,SUPPLEMENT_SEAL)):
        pin=seal(run,h);p=run+'/artifacts/input_manifest.json';doc=json.loads(read(p,pin[p]))
        shards={s['task']:s for s in doc['task_shards']};groups=collections.defaultdict(list)
        for r in doc['observations']:groups[r['task']].append(r)
        for task,items in groups.items():
            for i,r in enumerate(items):
                k=key(r)
                if k not in wanted:continue
                if r['split']!='calibration':raise ValueError('split drift')
                shard=shards[task];p=run+'/'+shard['path']
                if pin[p]!=shard['sha256']:raise ValueError('cache manifest mismatch')
                cached.setdefault(k,dict(path=p,sha256=pin[p],row=i,container_rows=len(items)))
    # These routes were already shown disjoint from rank0; inspect its saved
    # reuse plan metadata as well, rather than assuming no additional cache.
    p='configs/v3/gate3/multiview_input_reuse_inventory_v1.json'
    doc=json.loads(read(p,'171946ee973097316b148bd2a700d4c784fd2d5ba5aaf10cc077eeaa7bd4554a'))
    rank0={key(r['source']) for r in doc['reuse']}|{key(r) for r in doc['missing']}
    overlap=set(wanted)&rank0
    return dict(status='METADATA_CANDIDATE_NOT_LABEL_OR_EXPORT_AUTHORITY',counts=dict(original=27,
        parent_maps=len(reports),variant_intervals=len(selected),physical_traversals=len({k[1] for k in selected}),
        observations=len(rows),unique_variant_frames=len({(r['task'],f) for r in rows for f in r['frame_rows']}),
        cached=len(cached),missing=len(rows)-len(cached),rank0_overlap=len(overlap)),
        rows=[dict(r,cached_input=cached.get(key(r))) for r in rows],metadata_sha256=opened,
        selection='all existing positions of exactly the variant traversals containing old27; no score selection',
        warning='historical positive-conditioned route selection; not unbiased evaluation; no implicit other-variant addition',
        payload_reads=0,teacher_calls=0)


if __name__=='__main__':print(json.dumps(inventory(),indent=2))
