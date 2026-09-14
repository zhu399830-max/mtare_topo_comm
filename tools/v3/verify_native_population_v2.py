"""Read-only post-completion evidence check; never starts/resumes native work."""
from _bootstrap import PROJECT_ROOT as ROOT
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile
from mtare_topo.evaluation.native_candidate_graph_v2 import validate_native_graph

RUN='results/gate3_semantics/gate3_20260909_gse_native_graph_population_v2_seed0'


def digest_file(p):
    with p.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()


def verify():
    run=ROOT/RUN
    state=json.loads((run/'RUN_STATE.json').read_text())
    if state['state']!='COMPLETED':raise ValueError('not a completed population; do not infer completion')
    seal=run/'artifacts/evidence_sha256.txt'
    paths=set()
    for line in seal.read_text().splitlines():
        h,name=line.split('  ',1);p=ROOT/name
        if not p.resolve().is_relative_to(run.resolve()) or p==seal or name in paths:
            raise ValueError('invalid seal path')
        if digest_file(p)!=h:raise ValueError('seal hash mismatch: '+name)
        paths.add(name)
    actual={str(p.relative_to(ROOT)) for p in run.rglob('*') if p.is_file() and p!=seal}
    if paths!=actual:raise ValueError('seal does not cover exact evidence set')
    spec=json.loads((run/'config/run_spec.json').read_text())
    with zipfile.ZipFile(run/'artifacts/source_snapshot.zip') as z:
        if set(z.namelist())!=set(spec['source_sha256']):raise ValueError('source archive set')
        for name,h in spec['source_sha256'].items():
            if hashlib.sha256(z.read(name)).hexdigest()!=h:raise ValueError('source snapshot hash')
    card=json.loads((run/'config/data_card.json').read_text())
    rows=card['scope']['observations']
    manifest=json.loads((run/'artifacts/native_graph_manifest.json').read_text())
    entries=manifest['observations']
    events=[json.loads(line) for line in (run/'logs/native_observations.jsonl').read_text().splitlines()]
    if len(rows)!=307 or len(entries)!=307 or len(events)!=614:raise ValueError('population count')
    if manifest['cross_observation_edges']!=0 or manifest['optimizer_steps']!=0:
        raise ValueError('scope drift')
    totals=Counter(); splits=Counter(); seen=set()
    for i,(row,e) in enumerate(zip(rows,entries)):
        for key in ('source','split','parent_id','input_path','input_sha256','input_row'):
            if e[key]!=row[key]:raise ValueError('identity/partition mismatch')
        identity=json.dumps(e['source'],sort_keys=True)
        if identity in seen:raise ValueError('duplicate observation')
        seen.add(identity);splits[e['split']]+=1
        if events[2*i]['event']!='START' or events[2*i]['source']!=e['source']:
            raise ValueError('start attribution')
        if events[2*i+1]!=dict(event='COMPLETE',**e):raise ValueError('completion trace mismatch')
        for field,hfield in [('graph_file','graph_sha256'),('stderr_file','stderr_sha256')]:
            p=run/e[field]
            if not p.resolve().is_relative_to(run.resolve()) or digest_file(p)!=e[hfield]:
                raise ValueError('output binding')
        g=validate_native_graph((run/e['graph_file']).read_bytes())
        metrics=dict(nodes=g['nodes'],edges=g['edges'],rays=g['rays'],
            self_loop_edges=sum(a['endpoints'][0]==a['endpoints'][1] for a in g['edge_audit']),
            unknown_sample_edges=sum(a['unknown_samples']>0 for a in g['edge_audit']),
            low_clearance_sample_edges=sum(a['below_0_4m_samples']>0 for a in g['edge_audit']))
        for k,v in metrics.items():
            if e[k]!=v:raise ValueError('graph summary drift')
            totals[k]+=v
    expected_reads={r['input_path']:r['input_sha256'] for r in rows}
    if manifest['source_reads_sha256']!=expected_reads:raise ValueError('input read manifest drift')
    if splits!=Counter(fit=250,calibration=27,development=30):raise ValueError('split drift')
    summary=json.loads((run/'metrics/summary.json').read_text())
    if summary['observations']!=307 or summary['error'] is not None:raise ValueError('terminal summary')
    result=dict(status='SEALED_POPULATION_VERIFIED_NOT_SEMANTIC_OR_SAFETY_PASS',
        observations=307,files_verified=len(paths),splits=dict(splits),totals=dict(totals),
        seal_sha256=digest_file(seal),elapsed_s=summary['elapsed_s'],
        check_scope='Evidence hashes,archived source,identity/order,raw graph fields and risk counts; does not rerender scans or prove geometry accuracy.')
    print(json.dumps(result,indent=2));return result


if __name__=='__main__':verify()
