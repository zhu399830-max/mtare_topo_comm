"""Source-paired V6/V8 saved-output diagnosis; never invokes a teacher."""
import collections
import gzip
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
REST='results/gate3_semantics/gate3_20260909_gse_v8_remaining19_v1_seed20260906'
REST_SEAL='7a7e1a5b777aaf38dec48efc9612ee595bb1649ded99dafc2c28758b485158a2'


def audit():
    def read(path,sha):
        raw=(ROOT/path).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=sha:raise ValueError('input drift: '+path)
        return raw
    def key(s):return s['task'],s['source_sequence_id']
    union=json.loads(read('configs/v3/gate3/v8_completed_union_v1.json',
                         '622b98c37920ae57fc06816eb62400ec599bf1fd7f2320bd3030956dd6357c46'))
    seal=read(REST+'/artifacts/evidence_sha256.txt',REST_SEAL)
    pins={p:h for h,p in (l.split('  ',1) for l in seal.decode().splitlines())}
    rows=list(union['reused'])
    logpath=REST+'/logs/observations.jsonl'
    done=[json.loads(l) for l in read(logpath,pins[logpath]).decode().splitlines() if json.loads(l)['state']=='COMPLETED']
    if [r['source'] for r in done]!=union['missing']:raise ValueError('remainder mismatch')
    for r in done:
        p=REST+'/artifacts/'+r['evidence_file'];rows.append(dict(source=r['source'],path=p,sha256=pins[p]))
    manifest=json.loads(read('configs/v3/gate3/v8_multiview_mechanism_manifest_v1.json',
                            '6350cc865d29c1970a1cd523b3ea7c7f01b84ee7ab13f02f0e32bfa0526113bd'))
    old={ (r['task'],r['source_sequence_id']):r for r in manifest['observations']}
    if len(rows)!=141 or {key(r['source']) for r in rows}!=set(old):raise ValueError('not complete141')
    result=[];counts=collections.Counter();reasons=collections.Counter();tasks={}
    for r in rows:
        s=r['source'];v6ref=old[key(s)]
        before=json.loads(gzip.decompress(read(v6ref['evidence_path'],v6ref['evidence_sha256'])))
        after=json.loads(gzip.decompress(read(r['path'],r['sha256'])))
        if before['raw_interfaces']['source']!=s or after['raw_interfaces']['source']!=s:raise ValueError('source mismatch')
        if after['archive_sha256']!=union['archive_sha256']:raise ValueError('V8 archive mismatch')
        a=before['produced_targets'];b=after['produced_targets']
        if b['geometry_evidence_settings']!=union['geometry_settings']:raise ValueError('V8 parameters mismatch')
        na=a['teacher_provenance']['terminal_anchor_start'];nb=b['teacher_provenance']['terminal_anchor_start']
        state=f'{bool(na)}->{bool(nb)}';counts[state]+=1
        refs=after['raw_interfaces']['nearby_junction_references']
        unknown=b['unknown_candidates']['anchors']
        if not nb:
            if not refs:reasons['NO_NEARBY_REFERENCE']+=1
            elif unknown:
                for u in unknown:reasons[u.get('reason','UNSPECIFIED')]+=1
            else:reasons['REFERENCE_WITHOUT_ANCHOR_OR_UNKNOWN_RECORD']+=1
        ids_a=[v['node_id_teacher_only'] for v in a['teacher_provenance']['anchors']]
        ids_b=[v['node_id_teacher_only'] for v in b['teacher_provenance']['anchors']]
        deltas=[]
        for i,node in enumerate(ids_a):
            if node in ids_b:
                p=a['record']['anchors'][i]['position_m'];q=b['record']['anchors'][ids_b.index(node)]['position_m']
                deltas.append(sum((x-y)**2 for x,y in zip(p,q))**.5)
        item=dict(source=s,v8_path=r['path'],v8_sha256=r['sha256'],
                  v6_path=v6ref['evidence_path'],v6_sha256=v6ref['evidence_sha256'],
                  old_nodes=ids_a,new_nodes=ids_b,old_positions=a['record']['anchors'][:na],
                  new_positions=b['record']['anchors'][:nb],nearby_references=refs,unknown_anchors=unknown,
                  shared_node_position_deltas_m=deltas)
        result.append(item)
        task=tasks.setdefault(s['task'],dict(arc=[],old=[],new=[],references=[]))
        task['arc'].append(s['decision_route_arc_m']);task['old'].append(na);task['new'].append(nb);task['references'].append(refs)
        del before,after,a,b
    return dict(schema='v8_saved_paired_audit_v1',observations=141,transitions=dict(counts),
                zero_anchor_reasons=dict(reasons),tasks=tasks,rows=result,
                max_shared_node_position_delta_m=max((d for r in result for d in r['shared_node_position_deltas_m']),default=None),
                interpretation='label availability not accuracy; jointly changed V8 mechanisms; no new teacher/model calls')


if __name__=='__main__':print(json.dumps(audit(),indent=2))
