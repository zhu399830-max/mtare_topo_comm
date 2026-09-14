"""One exact all839 coordinate-only containment recheck; no ray loads."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,gzip,hashlib,json,math,resource,signal,sys,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write

SLUG='gse_saved_positive_origins_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed20260906'
INVENTORY='docs/figures/gse_local_pair_support_pilot_v1/existing_continuation_opportunities.json'
OLD_CARD='configs/v3/gate3/data_cards/gse_supplement_joint_v3.json'


def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True).encode()).hexdigest()


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    old=json.loads((ROOT/OLD_CARD).read_text())['scope'];inventory=json.loads((ROOT/INVENTORY).read_text())
    candidates=[r for r in inventory['candidates'] if r['original_membership'] is True]
    tasks={e['task']:e for e in old['entries']};files={OLD_CARD:sha(ROOT/OLD_CARD),INVENTORY:sha(ROOT/INVENTORY)};entries=[];decoded=0
    for task in sorted({r['source']['task'] for r in candidates}):
        selected=[r for r in candidates if r['source']['task']==task];e=tasks[task]
        if any(r['source'] not in e['observations'] for r in selected):raise ValueError('source not in archived task')
        prefix=e['sensor_prefix']+'/sensor_xyz_m';header=prefix+'/.zarray'
        if sha(ROOT/header)!=old['file_sha256'][header]:raise ValueError('coordinate header drift')
        h=json.loads((ROOT/header).read_text());rows=sorted({f for r in selected for f in r['source']['frame_rows']})
        if h['chunks'][1:]!=[3] or h['dtype']!='<f8' or h['filters'] or h['order']!='C':raise ValueError('fixed float64 coordinate layout required')
        keys=[str(i)+'.0' for i in sorted({r//h['chunks'][0] for r in rows})]
        files.update({p:old['file_sha256'][p] for p in [header,e['construction_path'],*(prefix+'/'+k for k in keys)]})
        files.update({r['evidence_path']:r['evidence_sha256'] for r in selected})
        entries.append(dict(task=task,selected=selected,construction=e['construction_path'],prefix=prefix,header=h,chunk_keys=keys,rows=rows))
        decoded+=len(keys)*math.prod(h['chunks'])*8
    scope=dict(entries=entries,input_sha256=files,observations=len(candidates),tasks=len(entries),
        parents=len({r['source']['parent_id'] for r in candidates}),unique_variant_frames=sum(len(e['rows']) for e in entries),
        split_counts={s:sum(r['source']['split']==s for r in candidates) for s in ('fit','calibration','development')},
        decoded_coordinate_bytes=decoded,field_spacing_m=.025,ray_reads=False,labels_changed=False,training=False,
        limits=dict(address_space_bytes=3*1024**3,wall_seconds=1800,output_bytes=1024**3))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',authorized_gates=[3],authorized_operations=['data_export'],
        scope_sha256=digest(scope),scope='Exact839 existing positives: coordinate containment prerequisite only',
        confirmation_reference='User active goal and current PLAN explicitly authorize frozen839 coordinate/target/construction recheck after24 software tests; no training or label changes.')
    write(ROOT/CARD,dict(schema_version='gse_saved_positive_origins_card_v1',scope=scope,approval=a))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=20260906,operation='data_export',data_card=CARD,config_path=CARD,user_authorization=a,
        command=[sys.executable,'tools/v3/check_saved_positive_origins.py','--execute'],
        question='Do the original839 positive witness frames retain unique-source containment at common .025 precision?',
        method='Reuse pinned old target and witness records; read only original sensor_xyz and constructions; taskwise existing sparse field; never add witness frames',
        baseline='Original .01-supported frames; not a model comparison or complete label qualification',
        fallback='Fail and seal; no retries, dropped observations, new positives or ray loads',
        estimated_cost=dict(compute='CPU210 task fields,4195 selected positions; zero GPU',host_ram_gb=3,gpu_vram_gb=0,disk_gb=1,wall_time_hours=.5),
        acceptance_criteria=['All839 declared observations, no replacement','Every outcome retained including lost support','Original unknown/negative labels untouched; no training qualification claim'],
        expected_evidence=['per-observation retained frames,source snapshot,source hashes,logs,environment,seal'],input_sha256=files,source_sha256=sources))
    print(json.dumps({k:v for k,v in scope.items() if k not in ('entries','input_sha256')}))


def execute():
    import numpy as np
    import numcodecs
    from mtare_topo.governance_branch_place_replay import validate_saved_positive_origins_card
    from mtare_topo.data.gse_structure_review_v1 import canonical_sha
    from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
    from mtare_topo.teacher.swept_superellipse_field import SweptSuperellipseProvenanceField
    from mtare_topo.teacher.gse_terminal_continuation_contract import recheck_saved_positive_origin
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED' or json.loads((run/'config/run_spec.json').read_text())!=spec or not validate_saved_positive_origins_card(card).passed:raise ValueError('fresh exact run required')
    start=time.monotonic();outputs=[];error=None
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*args):raise TimeoutError('1800s cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(1800)
    resource.setrlimit(resource.RLIMIT_AS,(3*1024**3,3*1024**3))
    try:
        for p,h in dict(spec['input_sha256'],**spec['source_sha256']).items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        with (run/'logs/tasks.jsonl').open('x') as log, (run/'artifacts/origin_rechecks.jsonl').open('x') as evidence:
            for e in s['entries']:
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>1024**3:raise OSError('output cap')
                h=e['header'];codec=numcodecs.get_codec(h['compressor']);coords={}
                for key in e['chunk_keys']:
                    a=np.frombuffer(codec.decode((ROOT/(e['prefix']+'/'+key)).read_bytes()),dtype=h['dtype']).reshape(h['chunks']);first=int(key.split('.')[0])*h['chunks'][0]
                    for row in e['rows']:
                        if first<=row<first+len(a):coords[row]=a[row-first].copy()
                if set(coords)!=set(e['rows']):raise ValueError('missing source coordinate')
                doc=json.loads((ROOT/e['construction']).read_text());_,primitives=load_p1a_realized_construction(doc)
                field=SweptSuperellipseProvenanceField(primitives,spacing_m=.025)
                distances=field.operand_signed_distances_sparse(np.asarray([coords[r] for r in e['rows']]))
                lookup={r:i for i,r in enumerate(e['rows'])}
                for r in e['selected']:
                    target=json.loads(gzip.decompress((ROOT/r['evidence_path']).read_bytes()))['produced_targets']
                    if target['source_binding']['construction_sha256']!=canonical_sha(doc) or target['record']['source_frame_indices']!=r['source']['frame_rows']:raise ValueError('target construction/frame mismatch')
                    outcome=recheck_saved_positive_origin(target,opening_index=r['opening_index'],terminal_index=r['terminal_index'],source_ids=[p.primitive_id for p in primitives],
                        origin_distances=distances[[lookup[f] for f in r['source']['frame_rows']]],field_spacing_m=.025)
                    result=dict(source=r['source'],terminal_node=r['terminal_node'],opening_source=r['opening_source'],**outcome)
                    line=json.dumps(result)+'\n'
                    if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())+len(line.encode())+65536>1024**3:raise OSError('output cap with finalization reserve')
                    evidence.write(line);evidence.flush();outputs.append(result)
                progress=dict(task=e['task'],completed=len(outputs),elapsed_s=time.monotonic()-start)
                log.write(json.dumps(progress)+'\n');log.flush();print(json.dumps(progress),flush=True)
                del field,distances,primitives,coords,doc
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    write(run/'artifacts/origin_rechecks.json',dict(records_file='origin_rechecks.jsonl',completed=len(outputs)))
    counts={split:dict(observations=sum(r['source']['split']==split for r in outputs),
        retained=sum(r['source']['split']==split and bool(r['retained_witness_frames']) for r in outputs),
        retained_parents=len({r['source']['parent_id'] for r in outputs if r['source']['split']==split and r['retained_witness_frames']})) for split in ('fit','calibration','development')}
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(outputs),counts=counts,elapsed_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,labels_changed=0,training_steps=0,ray_reads=0,python=sys.version,numpy=np.__version__)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
