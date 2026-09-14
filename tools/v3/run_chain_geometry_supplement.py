"""Freeze and execute exactly one missing two-source geometry supplement."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,gzip,hashlib,json,os,subprocess,time,traceback,sys
from ai_junction_pilot import sha,write
NAME='gse_chain_geometry_supplement_v1'
CARD='configs/v3/gate3/data_cards/'+NAME+'.json';SPEC='configs/v3/gate3/'+NAME+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+NAME+'_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/cano_e1_zarr2187_v1/bin/python'
MAPPING='docs/figures/gse_supervision_acquisition_pilot_v1/observed_chain_field_mapping.json'

def environment():
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONNOUSERSITE='1');env.pop('PYTHONPATH',None)
    return env

def freeze():
    mapping=json.loads((ROOT/MAPPING).read_text())['minimal_supplement']
    original=json.loads((ROOT/'configs/v3/gate3/gse_lateral_witness_recovery_scope_v1.json').read_text())
    archive=original['archive']
    if archive['sha256']!=mapping['archive_sha256'] or sha(ROOT/archive['path'])!=archive['sha256']:raise ValueError('archive differs')
    files={p:h for p,h in mapping['input_sha256'].items() if not p.endswith('.gz')}
    scope=dict(entry=dict(task=mapping['task'],parent=mapping['parent'],split='fit',frame_rows=mapping['frames'],
        original_ray_indices=mapping['frame_ray_indices'],files=files),sources=mapping['source_ids'],archive=archive,
        observations=1,parents=1,frames=5,raw_ray_slots=57600,selected_rays=2774,
        spacing='Original consecutive frames301..305; no assertion of independent adjacent rays or known metre spacing',
        geometry_settings={k:mapping['geometry_settings'][k] for k in ('axial_spacing_m','angular_segments','field_spacing_m')},
        limits=dict(worker_address_space_bytes=3*1024**3,wall_seconds=600,output_bytes=100*1024**2),
        teacher_calls=0,new_labels=0,training=False,unknown_is_background=False,
        authorization='Standing goal; PLAN explicitly permits bounded missing-source evidence supplement only')
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['audit'],
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest(),scope=scope['authorization'],
        confirmation_reference='User approved continued source verification and fixed fit population; PLAN explicitly permits exactly this bounded two-source supplement')
    write(ROOT/CARD,dict(schema_version='gse_chain_geometry_card_v1',scope=scope,approval=approval))
    inputs={CARD:sha(ROOT/CARD),MAPPING:sha(ROOT/MAPPING),archive['path']:archive['sha256'],**mapping['input_sha256']}
    src=['tools/v3/run_chain_geometry_supplement.py','tools/v3/recover_chain_geometry_worker.py','tools/v3/recover_lateral_witness_worker.py','tools/v3/ai_junction_pilot.py','src/mtare_topo/governance_membership_fit.py','src/mtare_topo/governance.py']
    runtime=dict(worker_binary_sha256=sha(__import__('pathlib').Path(PYTHON).resolve()),packages=subprocess.check_output([PYTHON,'-m','pip','freeze'],env=environment(),text=True,timeout=30))
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=approval,
      command=['python3','tools/v3/run_chain_geometry_supplement.py','--execute'],question='Do original2774 rays have missing source transitions at the degree2 connection?',
      method='Original archived closed meshes for exactly2 sources, one single-thread scene/query, no hit filtering; original field at5 sensor origins',
      baseline='Existing source evidence omitted intermediate node; not model comparison',fallback='Seal failure/no-hit/ambiguity; no retry or invented labels',
      estimated_cost=dict(compute='CPU1 case2 meshes2774 rays',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=1/6),
      acceptance_criteria=['All2774 ray records including no hits','Exact archive/data/settings and single-thread execution','No teacher targets or labels; preserve ambiguity'],
      expected_evidence=['All intersections, origins field, environment,command,source,logs,summary,seal'],input_sha256=inputs,source_sha256={p:sha(ROOT/p) for p in src},runtime=runtime))

def execute():
    run=ROOT/RUN;s=json.loads((ROOT/SPEC).read_text());c=json.loads((ROOT/CARD).read_text());start=time.monotonic();error=None;result=None
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    try:
        for p,h in {**s['input_sha256'],**s['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('drift '+p)
        current=dict(worker_binary_sha256=sha(__import__('pathlib').Path(PYTHON).resolve()),packages=subprocess.check_output([PYTHON,'-m','pip','freeze'],env=environment(),text=True,timeout=30))
        if current!=s['runtime']:raise ValueError('environment drift')
        write(run/'config/runtime_environment.json',dict(parent_python=sys.version,**current))
        import zipfile
        with zipfile.ZipFile(run/'artifacts/source.zip','x') as z:
            for p in s['source_sha256']:z.write(ROOT/p,p)
        command=[PYTHON,str(ROOT/'tools/v3/recover_chain_geometry_worker.py'),'--root',str(ROOT),'--card',str(ROOT/CARD),'--output',str(run/'artifacts/intersections.json.gz')]
        write(run/'config/worker_command.json',command)
        with (run/'logs/worker.log').open('x') as log:subprocess.run(command,env=environment(),stdout=log,stderr=subprocess.STDOUT,check=True,timeout=c['scope']['limits']['wall_seconds'])
        with gzip.open(run/'artifacts/intersections.json.gz','rt') as f:result=json.load(f)
        if [r['ray_index'] for r in result['rays']]!=c['scope']['entry']['original_ray_indices']:raise ValueError('ray identity drift')
        if result['sources']!=c['scope']['sources']:raise ValueError('source drift')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>c['scope']['limits']['output_bytes']:raise ValueError('output cap')
    except BaseException:error=traceback.format_exc()
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,elapsed_s=time.monotonic()-start,
        completed_rays=len(result['rays']) if result else 0,hits=sum(len(r['hits']) for r in result['rays']) if result else 0,
        no_hit_rays=sum(not r['hits'] for r in result['rays']) if result else None,peak_worker_rss_bytes=result['peak_rss_bytes'] if result else None,
        teacher_calls=0,new_labels=0,training_steps=0)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
