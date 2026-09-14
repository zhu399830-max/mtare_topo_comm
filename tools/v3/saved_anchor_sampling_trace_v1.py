"""Metadata-only provenance of existing307 targets and available source windows."""
import json
import hashlib
from collections import Counter
from _bootstrap import PROJECT_ROOT as ROOT
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import INVENTORY,SEAL_SHA256
from mtare_topo.governance_surface_input import SELECTION,SELECTION_SEAL
from mtare_topo.data.gse_supplement_input_scope_v1 import SUPPLEMENT,SUPPLEMENT_SEAL
from mtare_topo.data.development_paired_scope import compile_scope


def main():
    output=ROOT/'docs/figures/gse_graph/saved_anchor_sampling_trace_20260909'
    if output.exists():raise FileExistsError('immutable diagnostic')
    scope=compile_scope(ROOT);opened=dict(scope['manifest_sha256'])
    def read(p,h):
        raw=read_pinned(ROOT,p,h);opened[p]=h;return raw
    def seal(run,h):
        raw=read(run+'/artifacts/evidence_sha256.txt',h)
        return {p:h for h,p in (line.split('  ',1) for line in raw.decode().splitlines())}
    ip=seal(INVENTORY,SEAL_SHA256);sp=seal(SUPPLEMENT,SUPPLEMENT_SEAL);bp=seal(SELECTION,SELECTION_SEAL)
    path=SELECTION+'/artifacts/selection_manifest.json';background=json.loads(read(path,bp[path]))
    def key(s):return s['task'],s['source_sequence_id'],tuple(s['frame_rows'])
    bk={key(r):r for r in background['observations']}
    needed={key(r['source']):r for r in scope['observations']};summaries={};all_rows=[]
    for parent in sorted({r['source']['task'].split('__')[0] for r in needed.values()}):
        path=INVENTORY+'/artifacts/'+parent+'_identity_intervals.json';inv=json.loads(read(path,ip[path]))
        path=SUPPLEMENT+'/artifacts/'+parent+'.json';sup=json.loads(read(path,sp[path]))
        roles={}
        for sel in sup['selections']:
            for s in sel['sources']:roles.setdefault(key(s),[]).append(dict(node=sel['node_id_teacher_only'],kind=sel['kind'],decision_index=sel['decision_index']))
        for traversal in inv['intervals']:
            for variant in traversal['variants']:
                ids=variant['source_sequence_ids'];frames=variant['frame_rows'];arcs=variant['decision_arc_m']
                if not len(ids)==len(frames)==len(arcs):raise ValueError('metadata alignment mismatch')
                for i,(sid,f) in enumerate(zip(ids,frames)):
                    k=(variant['task'],sid,tuple(f))
                    if k not in needed:continue
                    row=needed.pop(k)
                    source_role=('both' if k in bk and k in roles else 'original' if k in bk else 'supplement' if k in roles else 'unaccounted')
                    if source_role=='unaccounted':raise ValueError('source not in either sealed sampling manifest')
                    all_rows.append(dict(source=row['source'],split=row['split'],traversal=traversal['traversal_id'],
                        sampling_role=source_role,supplement_roles=roles.get(k,[]),
                        decision_index=i,decision_arc_m=arcs[i],available_windows=len(ids),
                        available_before=i,available_after=len(ids)-1-i,first_arc_m=arcs[0],last_arc_m=arcs[-1],
                        available_source_sequence_ids=ids,available_frame_rows=frames))
    if needed or len(all_rows)!=307:raise ValueError('incomplete307 source join')
    for split in ('fit','calibration','development'):
        rows=[r for r in all_rows if r['split']==split]
        traversals={r['traversal']:r for r in rows}
        summaries[split]=dict(observations=len(rows),sampling_roles=dict(Counter(r['sampling_role'] for r in rows)),
            first_window=sum(r['decision_index']==0 for r in rows),last_window=sum(r['available_after']==0 for r in rows),
            arc4=sum(r['decision_arc_m']==4 for r in rows),other_arc_values=sorted({r['decision_arc_m'] for r in rows if r['decision_arc_m']!=4}),
            independent_directed_traversals=len(traversals),logical_windows_on_same_traversals=sum(r['available_windows'] for r in traversals.values()),
            unique_variant_windows=len({(r['source']['task'],s) for r in rows for s in r['available_source_sequence_ids']}))
    output.mkdir(parents=True)
    result=dict(status='METADATA_PROVENANCE_NOT_NEW_DATA_OR_LABELS',summaries=summaries,rows=all_rows,source_sha256=opened,
        scan_reads=0,teacher_generation=0,model_calls=0,
        limitation='Other source windows exist, but their label support, pose quality and negative completeness are not established by index availability.')
    (output/'sampling_trace.json').write_text(json.dumps(result,ensure_ascii=False,allow_nan=False)+'\n')
    for p,h in opened.items():
        if hashlib.sha256((ROOT/p).read_bytes()).hexdigest()!=h:raise ValueError('source drift')
    script=ROOT/'tools/v3/saved_anchor_sampling_trace_v1.py'
    (output/'analysis_source_sha256.json').write_text(json.dumps({str(script.relative_to(ROOT)):hashlib.sha256(script.read_bytes()).hexdigest()})+'\n')
    (output/'evidence_sha256.txt').write_text(''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(output.iterdir()) if p.is_file()))
    print(json.dumps(summaries,ensure_ascii=False))


if __name__=='__main__':main()
