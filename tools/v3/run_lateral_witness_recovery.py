"""Single immutable three-case archived surface intersection diagnostic."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,gzip,hashlib,json,os,resource,subprocess,sys,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write
SLUG='gse_lateral_witness_recovery_v1'
SCOPE='configs/v3/gate3/gse_lateral_witness_recovery_scope_v1.json'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python'
SINGLE_THREAD=False
SAVED_CONDITIONS=False

def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    base_scope='configs/v3/gate3/gse_lateral_witness_recovery_scope_v1.json'
    scope=json.loads((ROOT/base_scope).read_text())
    if SAVED_CONDITIONS:
        old='results/gate3_semantics/gate3_20260911_gse_lateral_witness_recovery_single_thread_v1_seed0'
        pins={p:h for h,p in (line.split('  ',1) for line in (ROOT/old/'artifacts/evidence_sha256.txt').read_text().splitlines())}
        for i,e in enumerate(scope['entries']):
            path=old+f'/artifacts/case_{i}.json.gz'
            if sha(ROOT/path)!=pins[path]:raise ValueError('saved intersection drift')
            e['recovered']=dict(path=path,sha256=pins[path])
        write(ROOT/SCOPE,scope)
    scope['limits']=dict(worker_address_space_bytes=3*1024**3,parent_rss_bytes=1024**3,wall_seconds=1800,output_bytes=1024**3)
    if SINGLE_THREAD:scope['execution_threads']=dict(scene_build=1,intersection_query=1)
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['audit'],
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest(),scope='Three frozen cases,21440 rays; saved intersection predicate verification only' if SAVED_CONDITIONS else 'Three frozen lateral witnesses,21440 rays,archived intersections only',confirmation_reference='User continuous development authorization; current PLAN permits saved original-condition verification, no new labels or training' if SAVED_CONDITIONS else 'User explicitly established continuous goal; PLAN specifies one bounded archived recovery, no labels or training')
    write(ROOT/CARD,dict(schema_version='gse_lateral_recovery_card_v1',scope=scope,approval=approval))
    files={SCOPE:sha(ROOT/SCOPE),scope['archive']['path']:scope['archive']['sha256']}
    for e in scope['entries']:files.update(e['files']);files[e['reference']['path']]=e['reference']['sha256']
    if SAVED_CONDITIONS:
        for e in scope['entries']:files[e['recovered']['path']]=e['recovered']['sha256']
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='audit',data_card=CARD,user_authorization=approval,
        command=['python3','tools/v3/run_lateral_witness_recovery.py','--execute']+(['--saved-conditions'] if SAVED_CONDITIONS else ['--single-thread'] if SINGLE_THREAD else ['--streaming-correction'] if SLUG.endswith('v1r1') else []),question='Do saved intersections reproduce original lateral containment/direction claims?' if SAVED_CONDITIONS else 'Can original saved lateral witness rays recover archived side-surface intersection records?',
        method='Archived field .025 and original containment/direction predicates on saved entries; no mesh or intersection calls' if SAVED_CONDITIONS else 'Three isolated CPU workers, all archived operands, selected original rays, no target producer',baseline='Original saved lateral witness identity; not a model comparison',fallback='Seal failure and missing records; no retry or label change',
        estimated_cost=dict(compute='CPU3 serial archived mesh cases',host_ram_gb=4,gpu_vram_gb=0,disk_gb=1,wall_time_hours=.5),
        acceptance_criteria=['All3 exact cases,original ray IDs and source hashes','Report missing entries, never silently omit or label negative','No teacher target calls or training','Keep immutable config,environment,raw logs,entries,summary,seal'],
        expected_evidence=['3 compressed intersection records,environment,source snapshot,logs,counts,RUN_STATE,seal'],input_sha256=files,source_sha256=sources))

def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());scope=card['scope']
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('drift '+p)
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
        packages=subprocess.check_output([PYTHON,'-m','pip','freeze'],env=env,text=True,timeout=30)
        write(run/'config/runtime_environment.json',dict(parent_python=sys.version,worker_python=PYTHON,worker_binary_sha256=sha(__import__('pathlib').Path(PYTHON).resolve()),packages=packages))
        for i,e in enumerate(scope['entries']):
            remaining=scope['limits']['wall_seconds']-(time.monotonic()-start)
            if remaining<=0:raise TimeoutError('total wall limit')
            output=run/f'artifacts/case_{i}.json.gz'
            command=[PYTHON,str(ROOT/'tools/v3/recover_lateral_witness_worker.py'),'--root',str(ROOT),'--scope',str(ROOT/SCOPE),'--index',str(i),'--output',str(output)]
            if SAVED_CONDITIONS:command[1]=str(ROOT/'tools/v3/check_saved_lateral_conditions.py')
            if SINGLE_THREAD:
                if scope.get('execution_threads')!=dict(scene_build=1,intersection_query=1):raise ValueError('thread scope mismatch')
                command.append('--single-thread')
            print(json.dumps(dict(started_case=i,ray_count=e['ray_count'])),flush=True)
            with (run/f'logs/case_{i}.log').open('x') as log:subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=remaining)
            with gzip.open(output,'rt') as f:result=json.load(f)
            if result['queried_rays']!=e['original_ray_indices'] or result['archive_sha256']!=scope['archive']['sha256']:raise ValueError('result binding drift')
            if SINGLE_THREAD and result.get('thread_execution')!=dict(scene_builds=1,intersection_queries=1,nthreads=1):raise ValueError('thread execution mismatch')
            row=dict(case=i,queried=e['ray_count'],entries=len(result['entries']),missing=len(result['missing_entry_ray_indices']),peak_rss_bytes=result['peak_rss_bytes'])
            rows.append(row);print(json.dumps(row),flush=True)
            if SAVED_CONDITIONS and not result['agrees_with_saved_claims']:raise ValueError('saved lateral condition mismatch')
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>scope['limits']['parent_rss_bytes']:raise MemoryError('parent RSS cap')
            if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>scope['limits']['output_bytes']:raise MemoryError('output cap')
    except BaseException:error=traceback.format_exc()
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),cases=rows,elapsed_s=time.monotonic()-start,teacher_target_calls=0,training_steps=0,new_labels=0)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary),flush=True);return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');group=p.add_mutually_exclusive_group();group.add_argument('--streaming-correction',action='store_true');group.add_argument('--single-thread',action='store_true');group.add_argument('--saved-conditions',action='store_true');a=p.parse_args()
    if a.streaming_correction:
        SLUG='gse_lateral_witness_recovery_v1r1'
        CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
        SPEC='configs/v3/gate3/'+SLUG+'.json'
        RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
    if a.single_thread:
        SINGLE_THREAD=True
        SLUG='gse_lateral_witness_recovery_single_thread_v1'
        CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
        SPEC='configs/v3/gate3/'+SLUG+'.json'
        RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
    if a.saved_conditions:
        SAVED_CONDITIONS=True
        SLUG='gse_saved_lateral_conditions_v1'
        SCOPE='configs/v3/gate3/'+SLUG+'_scope.json'
        CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
        SPEC='configs/v3/gate3/'+SLUG+'.json'
        RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
