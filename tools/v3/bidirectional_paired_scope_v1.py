"""Exact metadata join for authenticated860 partial-supervision observations."""
import json
from pathlib import Path
from collections import Counter
from bidirectional_grid_inventory_v1 import compile_inventory as grid_inventory, identity
from bidirectional_feature_inventory_v1 import INPUT_RUNS, BASE
from mtare_topo.data.bidirectional_compact_scope_v1 import compile_scope as compact_scope
from mtare_topo.governance_surface_material import read_pinned

GRID_RUN=BASE+'gate3_20260909_gse_bidirectional_grids_v1_seed20260906'
PART_RUN=BASE+'gate3_20260909_gse_bidirectional_partitions_v1r_seed0'
GRID_SHA='5f922d3fc42b3152db041c80363898b46fe21d73fee1fae2a6b72e0ffca86c46'
PART_SHA='09bc298cbf75b944b3e7fb725de1f3feceb3b472ac646367b806ac86b3d0d908'
OLD_INDEX='docs/figures/gse_graph/v8_saved_paired_audit_20260909.json'
OLD_SHA='42026d9bf0c38af1c185ae1a2d291842dbd48e993e9fe414d657668ea9b23d65'


def unique(rows, source):
    out={identity(source(r)):r for r in rows}
    if len(out)!=len(rows):raise ValueError('duplicate source')
    return out


def artifact(run, folder, filename):
    if '/' in filename or filename in ('', '.', '..'):raise ValueError('invalid artifact filename')
    return run+'/artifacts/'+folder+filename


def compile_scope(root):
    root=Path(root);opened={}
    def read(p,h):
        raw=read_pinned(root,p,h);opened[p]=h;return raw
    compact=compact_scope(root);inventory=grid_inventory(root)
    opened.update(compact['metadata_sha256']);opened.update(inventory['metadata_sha256'])
    gm=json.loads(read(GRID_RUN+'/artifacts/grid_manifest.json',GRID_SHA))
    pm=json.loads(read(PART_RUN+'/artifacts/partition_manifest.json',PART_SHA))
    if gm['status']!='COMPLETE' or pm['status']!='COMPLETE':raise ValueError('incomplete assets')
    gi=unique(gm['observations'],lambda r:r['binding']['source'])
    pi=unique(pm['observations'],lambda r:r['source'])
    ci=unique(compact['observations'],lambda r:r['source'])
    old=json.loads(read(OLD_INDEX,OLD_SHA));targets={}
    for r in old['rows']:
        key=identity(r['source'])
        if key in targets:raise ValueError('duplicate original target')
        targets[key]=dict(path=r['v8_path'],sha256=r['v8_sha256'],split=r['source']['split'])
    if len(targets)!=141:raise ValueError('original141 index required')
    for name, seal_sha in list(INPUT_RUNS.items())[1:]:
        run=BASE+name;seal=read(run+'/artifacts/evidence_sha256.txt',seal_sha)
        pins={p:h for h,p in (line.split('  ',1) for line in seal.decode().splitlines())}
        p=run+'/artifacts/target_manifest.json';manifest=json.loads(read(p,pins[p]))
        for r in manifest['observations']:
            key=identity(r['source']);p=artifact(run,'',r['evidence_file'])
            if key in targets or pins[p]!=r['storage']['compressed_sha256']:raise ValueError('target overlap/hash drift')
            targets[key]=dict(path=p,sha256=pins[p],split=r['split'])
    if set(targets)!=set(ci) or len(ci)!=860:raise ValueError('target/feature population differs')
    rows=[]
    for r in inventory['observations']:
        key=identity(r['source']);c=ci[key];t=targets[key]
        if c['split']!=r['split'] or t['split']!=r['split']:raise ValueError('split differs')
        grid=r['reused_grid'];part=c['reused_partition']
        if grid:
            if key in gi:raise ValueError('recomputed old grid')
        else:
            g=gi.pop(key)
            if g['split']!=r['split']:raise ValueError('new grid split')
            grid=dict(path=artifact(GRID_RUN,'grids/',g['file']),sha256=g['sha256'],binding=g['binding'])
        if part:
            if key in pi:raise ValueError('recomputed old partition')
        else:
            p=pi.pop(key)
            if p['split']!=r['split'] or p['cache_sha256']!=c['cache_sha256']:raise ValueError('partition cache/split drift')
            part=dict(path=artifact(PART_RUN,'partitions/',p['file']),sha256=p['sha256'])
        if grid['binding']['source']!=r['source']:raise ValueError('grid source drift')
        rows.append(dict(source=r['source'],split=r['split'],source_card=r['source_card'],
            cache_path=c['cache_path'],cache_sha256=c['cache_sha256'],feature_entry=c['feature_entry'],
            partition_path=part['path'],partition_sha256=part['sha256'],
            grid_path=grid['path'],grid_sha256=grid['sha256'],source_binding=grid['binding'],
            target_path=t['path'],target_sha256=t['sha256']))
    if pi or gi:raise ValueError('unconsumed assets')
    return dict(schema='bidirectional_paired_scope_v1',observations=rows,
        source_cards=inventory['source_cards'],metadata_sha256=opened,
        source_file_sha256=inventory['source_file_sha256'],input_sha256=inventory['input_sha256'],
        counts=dict(observations=len(rows),splits=dict(Counter(r['split'] for r in rows))),
        restrictions=['C01_C07_only','partial_reference_not_full_detection','no_independent_development_population',
                      'teacher_never_forward','no_export_or_training_authority','raw_payload_adapter_checks_pending'])


if __name__=='__main__':
    print(json.dumps(compile_scope(Path(__file__).resolve().parents[2]),separators=(',',':')))
