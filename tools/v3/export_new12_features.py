"""One shared frozen observation/patch cache for new12 A/B/C; zero labels."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from ai_junction_pilot import sha,write
NAME='gse_new12_features_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
SOURCE='configs/v3/gate3/gse_new12_surface_source_scope_v1.json'
ENCODER_CARD='configs/v3/gate3/data_cards/gse_common_observation_export_v1.json'
PYTHON='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'


def freeze():
    b=json.loads((ROOT/SOURCE).read_text());ck=json.loads((ROOT/ENCODER_CARD).read_text())['scope']['checkpoint']
    es=[e['identity'] for e in b['entries']]
    s=dict(entries=es,parents=12,frames=60,valid_returns=684440,valid_returns_per_observation=b['valid_returns_per_observation'],
        checkpoint=ck,spacing=b['spacing'],selection_bias=b['selection_bias'],training_steps=0,teacher_payload_reads=0,
        execution_context='controlled outside-sandbox CUDA access; no driver or system modification',
        limits=dict(wall_seconds=720,host_bytes=4*1024**3,gpu_bytes=28*1024**3,output_bytes=512*1024**2))
    a=dict(status='APPROVED',approved_by='user-standing-new-route-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['data_export'],scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        scope='Same fixed new12 common frozen features only; no old16 payload or labels',
        confirmation_reference='User active goal explicitly authorizes fixed new12 feature binding and same-input A/B/C; standing execution authorization')
    write(ROOT/CARD,dict(schema_version='gse_new12_features_card_v1',scope=s,approval=a))
    files={SOURCE:sha(ROOT/SOURCE),ENCODER_CARD:sha(ROOT/ENCODER_CARD),CARD:sha(ROOT/CARD),ck['path']:ck['sha256']}
    files.update({e['student_path']:e['student_sha256'] for e in es})
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_new12_features.py','--execute'],
        question='Can new12 A/B/C share exactly the same frozen observation features and observation-derived patch inventory?',
        method='Frozen seed0 epoch2 encoder; original common observation; compact patch mean context without teacher filtering',
        baseline='Identical cache shared by A/B/C; export itself is not a model comparison',
        fallback='Seal partial failure; do not modify weights, drop patches or retry',
        acceptance_criteria=['12 exact observations and valid-return counts','Encoder state unchanged','No teacher input or optimizer','All observed patches retained'],
        expected_evidence=['12 NPZ common features and patch features, log, environment, source snapshot, SHA seal'],
        estimated_cost=dict(compute='12 frozen GPU forwards plus CPU patch pooling',host_ram_gb=4,gpu_vram_gb=28,disk_gb=.5,wall_time_hours=.2),
        input_sha256=files,source_sha256=sources))


def execute(*, run_path=RUN, spec_path=SPEC, card_path=CARD, validator=None, student_iterator=None):
    import torch,numpy as np
    from mtare_topo.governance_new12_features import validate_card
    from mtare_topo.data.gse_membership_fit_reader import load_student_window
    from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    from mtare_topo.representation.gse_common_observation import prepare_common_observation
    from mtare_topo.representation.gse_patch_observed_features import pool_patch_observed_features
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    run=ROOT/run_path;spec=json.loads((ROOT/spec_path).read_text());card=json.loads((ROOT/card_path).read_text());s=card['scope']
    validate_card = validator or validate_card
    if not validate_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None;peak=0
    def expire(*_):raise TimeoutError('fixed feature export wall cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(s['limits']['wall_seconds'])
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),numpy=np.__version__,execution_context=s['execution_context']))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        ck=s['checkpoint'];bound=load_surface_encoder((ROOT/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
        adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).cuda()
        with (run/'logs/windows.jsonl').open('x') as log:
            students = (student_iterator(ROOT,s['entries']) if student_iterator else
                        (load_student_window(ROOT,{**e,'layout':'saved_single_window','decoded_observations':1}) for e in s['entries']))
            for i,e in enumerate(s['entries']):
                student=next(students)
                valid_count=int(student.valid_mask.sum())
                if s['valid_returns_per_observation'] is not None and valid_count!=s['valid_returns_per_observation'][i]:raise ValueError('valid count drift')
                obs=prepare_common_observation(adapter,student,device='cuda')
                arrays={k:getattr(obs,k) for k in ('registered_returns_xyz_m','surface_return_indices','ray_sensor_token_indices','ray_history_indices')}
                arrays.update(full_sensor_context=obs.full_sensor_context.cpu().numpy(),full_sensor_valid=obs.full_sensor_valid.cpu().numpy(),
                              patch_observed_features=pool_patch_observed_features(obs).cpu().numpy())
                arrays.update({'ray_'+k:v for k,v in vars(obs.local_rays).items() if isinstance(v,np.ndarray)})
                arrays.update({'patch_'+k:v for k,v in vars(obs.surface_patches).items() if isinstance(v,np.ndarray)})
                np.savez_compressed(run/f'artifacts/window_{i:02d}.npz',**arrays)
                row=dict(observation=i,task=e['task'],valid_returns=valid_count,measured_local_returns=len(obs.surface_return_indices),patches=len(obs.surface_patches.centers_m),elapsed_s=time.monotonic()-start)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                peak=torch.cuda.max_memory_reserved()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes'] or peak>s['limits']['gpu_bytes']:raise MemoryError('memory cap')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output cap')
                del obs,arrays,student
            if next(students,None) is not None:raise ValueError('extra student observation')
        if module_state_sha256(bound.backbone)!=bound.state_sha256:raise ValueError('encoder mutation')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),windows=rows,training_steps=0,teacher_payload_reads=0,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_reserved_bytes=peak)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
