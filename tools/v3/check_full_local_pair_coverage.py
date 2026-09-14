"""Read archived coordinates and exact sequence indices, never sensor returns."""
from _bootstrap import PROJECT_ROOT as ROOT
from check_local_pair_window_coverage import sha, write
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import sys
import time
import traceback

SLUG='gse_full_local_pair_coverage_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed20260906'
OLD='configs/v3/gate3/data_cards/gse_local_pair_window_coverage_v1.json'
FIELDS=('sensor_xyz_m','traversal_index','local_frame_index','frame_row','source_global_sequence_index')


def freeze():
    from mtare_topo.governance_surface_input import SEALS, expected_tasks
    if (ROOT/RUN).exists():raise FileExistsError('existing run')
    old=json.loads((ROOT/OLD).read_text())['scope'];tasks=expected_tasks()
    files={OLD:sha(ROOT/OLD)};pins={};entries=[];total_bytes=0
    prefixes={tasks[e['task']]['sensor']+'/'+f+'/' for e in old['entries'] for f in FIELDS[:3]}
    prefixes.update(tasks[e['task']]['teacher']+'/'+f+'/' for e in old['entries'] for f in FIELDS[3:])
    for spec in SEALS.values():
        raw=(ROOT/spec['path']).read_bytes()
        if hashlib.sha256(raw).hexdigest()!=spec['sha256']:raise ValueError('source seal drift')
        files[spec['path']]=spec['sha256']
        for line in raw.decode().splitlines():
            h,p=line.split(None,1)
            if any(p.startswith(prefix) for prefix in prefixes):pins[p]=h
    for original in old['entries']:
        entry={k:original[k] for k in ('task','parent','split','construction','reference_pairs')}
        files[entry['construction']]=old['source_sha256'][entry['construction']]
        entry['arrays']={}
        for field in FIELDS:
            prefix=tasks[entry['task']]['sensor' if field in FIELDS[:3] else 'teacher']+'/'+field
            header_path=prefix+'/.zarray';raw=(ROOT/header_path).read_bytes()
            if hashlib.sha256(raw).hexdigest()!=pins[header_path]:raise ValueError('header drift')
            header=json.loads(raw)
            chunks=['.'.join(map(str,indices)) for indices in itertools.product(*[range(math.ceil(s/c)) for s,c in zip(header['shape'],header['chunks'])])]
            files[header_path]=pins[header_path]
            for key in chunks:files[prefix+'/'+key]=pins[prefix+'/'+key]
            entry['arrays'][field]=dict(prefix=prefix,header=header,chunk_keys=chunks)
            total_bytes+=len(chunks)*math.prod(header['chunks'])*int(header['dtype'][2:])
        if entry['arrays']['frame_row']['header']['shape']!=[original['observations'][0]['source_sequence_count'],5]:
            raise ValueError('original sequence count mismatch')
        entries.append(entry)
    counts={}
    for split in old['counts']:
        es=[e for e in entries if e['split']==split]
        counts[split]=dict(parents=len({e['parent'] for e in es}),tasks=len(es),
            source_frames=sum(e['arrays']['sensor_xyz_m']['header']['shape'][0] for e in es),
            sequences=sum(e['arrays']['frame_row']['header']['shape'][0] for e in es))
    scope=dict(entries=entries,counts=counts,source_sha256=files,fields=list(FIELDS),radius_m=10,
        reference_pairs=77,decoded_padded_bytes=total_bytes,scan_reads=False,training=False,labels_generated=0,
        sampling='all archived five-frame sequences of same44 parents; original sequence table, never arbitrary adjacent rows',
        purpose='necessary coordinate co-domain only; no visibility or membership labels',
        split='original parent-disjoint fit/calibration/development C01-C07; historical development, not strict test')
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        authorized_gates=[3],authorized_operations=['audit'],scope='Explicit full-coordinate/index expansion within same44 parents, no scan or target payloads',
        confirmation_reference='User active goal and continuous authorization; announced full-coordinate next step',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    write(ROOT/CARD,dict(schema_version='gse_full_local_pair_coordinate_card_v1',scope=scope,approval=approval),'w')
    source=[Path(__file__),ROOT/'tools/v3/check_local_pair_window_coverage.py',ROOT/'tools/v3/_bootstrap.py',ROOT/CARD,
        ROOT/'src/mtare_topo/governance.py',ROOT/'src/mtare_topo/governance_branch_place_replay.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=20260906,operation='audit',
        data_card=CARD,config_path=CARD,user_authorization=approval,
        command=[sys.executable,'tools/v3/check_full_local_pair_coverage.py','--execute'],
        question='Do the full archived causal sequences cover the77 local reference pairs, unlike old sparse sampling?',
        method='Exact archived sequence indices, original10m sensor ball; verify within-traversal continuity',
        baseline='Old1743-window coordinate audit; no detector comparison',
        estimated_cost=dict(compute='CPU only; under4GiB, zero scan/model payloads',disk_gb=.1,wall_time_hours=.1),
        acceptance_criteria=['Exact declared frames/sequences','No sequence crosses traversal or skips local frame','Keep all pairs including uncovered','Coordinate opportunities are not labels'],
        expected_evidence=['all pair statistics, covered sequence identities, metadata hashes, environment, logs, seal'],
        input_sha256=files,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in source}),'w')
    print(json.dumps(dict(counts=counts,decoded_bytes=total_bytes,spec=SPEC)))


def execute():
    import numpy as np
    import numcodecs
    from mtare_topo.governance_branch_place_replay import validate_full_local_pair_card
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec and validate_full_local_pair_card(card).passed
    for p,h in dict(spec['source_sha256'],**spec['input_sha256']).items():
        if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
    def decode(plan):
        header=plan['header'];shape=header['shape'];chunk=header['chunks']
        if header['filters'] or header['order']!='C':raise ValueError('unsupported codec layout')
        result=np.empty(shape,dtype=header['dtype']);codec=numcodecs.get_codec(header['compressor'])
        for key in plan['chunk_keys']:
            block=np.frombuffer(codec.decode((ROOT/(plan['prefix']+'/'+key)).read_bytes()),dtype=header['dtype']).reshape(chunk)
            starts=[int(k)*c for k,c in zip(key.split('.'),chunk)]
            sizes=[min(c,n-start) for c,n,start in zip(chunk,shape,starts)]
            result[tuple(slice(start,start+size) for start,size in zip(starts,sizes))]=block[tuple(slice(0,size) for size in sizes)]
        return result
    started=time.monotonic();error=None;summary={};write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    try:
        pairs={};totals={k:dict(frames=0,sequences=0,covered_sequences=0) for k in s['counts']}
        with (out/'artifacts/covered_sequences.jsonl').open('x') as log:
            for e in s['entries']:
                a={f:decode(e['arrays'][f]) for f in FIELDS};xyz=a['sensor_xyz_m'];frames=a['frame_row']
                if frames.dtype.kind not in 'iu' or np.any(frames<0) or np.any(frames>=len(xyz)) or not np.isfinite(xyz).all():raise ValueError('invalid source coordinates/indices')
                if not np.all(a['traversal_index'][frames]==a['traversal_index'][frames[:,0]][:,None]):raise ValueError('cross-traversal sequence')
                if not np.all(np.diff(a['local_frame_index'][frames],axis=1)==1):raise ValueError('nonconsecutive local frames')
                if len(np.unique(a['source_global_sequence_index']))!=len(frames):raise ValueError('duplicate sequence identity')
                nodes={n['node_id']:n for n in json.loads((ROOT/e['construction']).read_text())['base_construction']['composition_operations']}
                hits=[[] for _ in range(len(frames))]
                for pair in e['reference_pairs']:
                    key=(e['parent'],pair['junction'],pair['terminal'])
                    distances=np.maximum(*(np.linalg.norm(xyz[frames[:,-1]]-nodes[pair[k]]['anchor_xyz_m'],axis=1) for k in ('junction','terminal')))
                    result=pairs.setdefault(key,dict(parent=e['parent'],split=e['split'],junction=pair['junction'],terminal=pair['terminal'],covered_sequences=0,minimum_required_radius_m=None))
                    nearest=float(distances.min());previous=result['minimum_required_radius_m']
                    result['minimum_required_radius_m']=nearest if previous is None else min(previous,nearest)
                    indices=np.flatnonzero(distances<=10);result['covered_sequences']+=len(indices)
                    for i in indices:hits[i].append(dict(junction=pair['junction'],terminal=pair['terminal']))
                for i,values in enumerate(hits):
                    if values:log.write(json.dumps(dict(parent=e['parent'],split=e['split'],task=e['task'],sequence_row=i,
                        source_global_sequence_index=int(a['source_global_sequence_index'][i]),frame_rows=frames[i].tolist(),
                        traversal_index=int(a['traversal_index'][frames[i,-1]]),reference_pairs=values,visibility='NOT_EVALUATED'))+'\n')
                t=totals[e['split']];t['frames']+=len(xyz);t['sequences']+=len(frames);t['covered_sequences']+=sum(bool(h) for h in hits)
        for split,t in totals.items():
            if t['frames']!=s['counts'][split]['source_frames'] or t['sequences']!=s['counts'][split]['sequences']:raise ValueError('population mismatch')
            selected=[r for r in pairs.values() if r['split']==split]
            t['covered_pairs']=sum(r['covered_sequences']>0 for r in selected)
            t['covered_parents']=len({r['parent'] for r in selected if r['covered_sequences']})
        write(out/'artifacts/pair_coverage.json',list(pairs.values()))
        write(out/'artifacts/environment.json',dict(python=sys.version,numpy=np.__version__,numcodecs=numcodecs.__version__))
        summary=dict(status='GATE_MIXED',splits=totals,labels_generated=0,scan_reads=0,training_steps=0,
            meaning='Original sequence geometry coverage only; not visibility or membership qualification')
    except Exception:error=traceback.format_exc()
    summary.update(error=error,elapsed_s=time.monotonic()-started)
    if error:summary['status']='GATE_FAIL'
    write(out/'metrics/summary.json',summary);write(out/'logs/raw.json',summary)
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
