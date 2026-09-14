"""Freeze and execute one non-overwritable fixed16 partial-reference fit."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write

SLUG='gse_membership_fit_gradient_isolated_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    bp='configs/v3/gate3/gse_membership_fit_input_binding_v1.json'
    binding=json.loads((ROOT/bp).read_text())
    ck=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_supplement_features_v1.json').read_text())['scope']['checkpoint']
    files={bp:sha(ROOT/bp),ck['path']:ck['sha256'],**binding['metadata_sha256']}
    initial='results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0/artifacts/initial.pt'
    files[initial]=sha(ROOT/initial)
    for e in binding['entries']:
        files[e['student_path']]=e['student_sha256'];files[e['target_path']]=e['target_sha256']
    s=dict(binding=binding,checkpoint=ck,updates=500,seed=0,variant='C',learning_rate=.001,
        weight_decay=.0001,accumulation=4,teacher_in_forward=False,unknown_is_background=False,
        initial_checkpoint=initial,membership_gradient_to_shared=False,
        limits=dict(wall_seconds=7200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=4*1024**3),
        evaluation='Initial/final only;1m and0.5; partial reference recall and localized known relations; all outputs saved',
        decision='All anchor/opening reference recall and both localized membership class recalls >=0.9; prerequisite only, never full detection PASS')
    approval=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['training'],scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='One fixed16/500-update gradient-routing correction from saved original initialization; no label/loss-weight changes',
        confirmation_reference='Active user goal authorizes minimum fitting and model implementation without repeated routine approval; scope announced before freeze')
    write(ROOT/CARD,dict(schema_version='gse_membership_fit_card_v1',scope=s,approval=approval))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='training',data_card=CARD,
        user_authorization=approval,question='Can the fixed16 partial-reference structure membership task be fitted?',
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/membership_fit_v1.py','--execute'],
        research_question='Can observed queries fit saved local opening-structure relationships on the fixed16 partial-reference windows?',
        method='Original forward and loss, but membership gradients update only the original relation head, not shared location features',
        baseline='Original v1r1 same initialization/16 windows/500 updates; mechanism diagnosis, not geometry superiority',fallback='Fail and seal; stop this corrective configuration, no added steps or label changes',
        estimated_cost=dict(compute='single CUDA GPU500 updates/micro1x4 plus16 frozen feature extractions',host_ram_gb=32,gpu_vram_gb=28,disk_gb=4,wall_time_hours=2),
        acceptance_criteria=[s['decision'],'Unknowns never background; all predictions saved; no GT forward','Original and final weights retained'],
        expected_evidence=['inputs and label hashes,source snapshot,initial/final weights and predictions,per-update log,resource metrics,seal'],
        input_sha256=files,source_sha256=sources))
    print(json.dumps(dict(card=CARD,spec=SPEC,windows=16,updates=500)))


def execute():
    import torch
    from mtare_topo.governance_membership_fit import validate_card
    from mtare_topo.data.gse_membership_fit_reader import load_student_window,load_partial_reference
    from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    from mtare_topo.representation.gse_observed_membership_v1 import ObservedMembershipV1
    from mtare_topo.representation.gse_membership_fit_execution import prepare_membership_example,fit_membership
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if (not validate_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED'
        or json.loads((run/'config/run_spec.json').read_text())!=spec):raise ValueError('fresh bound run required')
    start=time.monotonic();error=None;steps=0;model=None;result=None
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*args):raise TimeoutError('fixed7200s limit')
    signal.signal(signal.SIGALRM,expire);signal.alarm(7200)
    def check():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes']:raise MemoryError('RAM cap')
        if torch.cuda.max_memory_reserved()>s['limits']['gpu_bytes']:raise MemoryError('GPU cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise OSError('output cap')
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('input/source drift '+p)
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU substitution')
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        write(run/'config/runtime_environment.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        ck=s['checkpoint'];bound=load_surface_encoder((ROOT/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
        model=ObservedMembershipV1('C',membership_gradient_to_shared=s['membership_gradient_to_shared']).cuda()
        model.load_state_dict(torch.load(ROOT/s['initial_checkpoint'],weights_only=True,map_location='cuda'),strict=True)
        torch.save(model.state_dict(),run/'artifacts/initial.pt')
        adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).cuda();examples=[]
        for i,e in enumerate(s['binding']['entries']):
            student=load_student_window(ROOT,e);target=load_partial_reference(ROOT,e)
            examples.append(prepare_membership_example(adapter,student,target['record'],device='cuda'))
            write(run/f'artifacts/reference_{i:02d}.json',target)
            check();print(json.dumps(dict(prepared=i+1,elapsed_s=time.monotonic()-start)),flush=True)
        with (run/'logs/updates.jsonl').open('x') as log:
            def progress(row):
                nonlocal steps
                steps=row['step'];row['elapsed_s']=time.monotonic()-start
                log.write(json.dumps(row)+'\n');log.flush();check()
                if steps%25==0:print(json.dumps(dict(step=steps,elapsed_s=row['elapsed_s'])),flush=True)
            result=fit_membership(model,examples,on_update=progress)
        if module_state_sha256(bound.backbone)!=bound.state_sha256:raise ValueError('frozen encoder drift')
        write(run/'artifacts/predictions.json',result)
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    if model is not None:torch.save(model.state_dict(),run/'artifacts/final.pt')
    rates={}
    if result:
        for role in ('anchor','opening'):
            count=sum(r[role]['references'] for r in result['final'])
            rates[role]=sum(r[role]['correct'] for r in result['final'])/count if count else 0.
        for side in ('positive','negative'):
            count=sum(r['membership'][side+'_total'] for r in result['final'])
            rates[side]=sum(r['membership'][side+'_correct'] for r in result['final'])/count if count else 0.
    summary=dict(status='GATE_MIXED' if not error and rates and min(rates.values())>=.9 else 'GATE_FAIL',error=error,
        actual_updates=steps,partial_reference_rates=rates,elapsed_s=time.monotonic()-start,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,full_detection_qualified=False,method_advantage_proven=False)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary)
    write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(run))+'\n')
    print(json.dumps(summary),flush=True)
    return int(bool(error))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose freeze or execute')
