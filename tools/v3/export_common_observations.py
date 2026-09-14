"""One frozen16 feature export. No target loads or model updates."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write
from membership_fit_v1 import PYTHON

SLUG='gse_common_observation_export_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
OLD='results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0/config/data_card.json'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    old=json.loads((ROOT/OLD).read_text())['scope'];b=old['binding'];ck=old['checkpoint']
    files={OLD:sha(ROOT/OLD),ck['path']:ck['sha256']}
    for e in b['entries']:files[e['student_path']]=e['student_sha256']
    s=dict(binding=b,checkpoint=ck,training_steps=0,teacher_payload_reads=0,
        limits=dict(wall_seconds=720,host_bytes=4*1024**3,gpu_bytes=28*1024**3,output_bytes=512*1024**2))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['data_export'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),scope='Same16 causal inputs and exact old frozen encoder only',
        confirmation_reference='Active goal and current PLAN common full-context/surface/ray interface, standing development authorization; no target payloads or training')
    write(ROOT/CARD,dict(schema_version='gse_common_observation_card_v1',scope=s,approval=a))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_common_observations.py','--execute'],
        question='Can a common observation retain full scan context, local surfaces and all supported local ray segments without teacher input?',
        method='Frozen seed0 epoch2 encoder, no point-context expansion; separate local surfaces and crop-ray endpoints',
        baseline='Original ROI surface mask, reported as coverage only',fallback='Seal failure; no limits/data changes or automatic retries',
        estimated_cost=dict(compute='GPU16 frozen encoder forwards plus CPU patches',host_ram_gb=4,gpu_vram_gb=28,disk_gb=.5,wall_time_hours=.2),
        acceptance_criteria=['Exact16 outputs; backbone state unchanged','Outside-return rays not turned into surfaces or openings','Zero target reads and optimizer updates'],
        expected_evidence=['16 common-input NPZ,coverage counts,environment,logs,source snapshot,seal'],input_sha256=files,source_sha256=sources))


def execute():
    import torch
    import numpy as np
    from mtare_topo.governance_membership_fit import validate_common_observation_card
    from mtare_topo.data.gse_membership_fit_reader import load_student_window
    from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    from mtare_topo.representation.gse_common_observation import prepare_common_observation
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_common_observation_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None;gpu_peak=0
    def expire(*args):raise TimeoutError('720 second cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(720)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(python=sys.version,torch=torch.__version__,gpu=torch.cuda.get_device_name(0),numpy=np.__version__))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        ck=s['checkpoint'];bound=load_surface_encoder((ROOT/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
        adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).cuda()
        with (run/'logs/windows.jsonl').open('x') as log:
            for i,e in enumerate(s['binding']['entries']):
                student=load_student_window(ROOT,e)
                obs=prepare_common_observation(adapter,student,device='cuda')
                arrays={k:getattr(obs,k) for k in ('registered_returns_xyz_m','surface_return_indices','ray_sensor_token_indices','ray_history_indices')}
                arrays.update(full_sensor_context=obs.full_sensor_context.cpu().numpy(),full_sensor_valid=obs.full_sensor_valid.cpu().numpy())
                arrays.update({'ray_'+k:v for k,v in vars(obs.local_rays).items() if isinstance(v,np.ndarray)})
                arrays.update({'patch_'+k:v for k,v in vars(obs.surface_patches).items() if isinstance(v,np.ndarray)})
                np.savez_compressed(run/f'artifacts/window_{i:02d}.npz',**arrays)
                row=dict(observation=i,task=e['source']['task'],full_valid_sensor_tokens=int(obs.full_sensor_valid.sum()),
                    measured_local_returns=len(obs.surface_return_indices),local_segments=len(obs.local_rays.ray_indices),
                    clipped_non_surface_endpoints=int((~obs.local_rays.end_is_observed_return).sum()),patches=len(obs.surface_patches.centers_m),
                    elapsed_s=time.monotonic()-start)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                gpu_peak=torch.cuda.max_memory_reserved()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes'] or gpu_peak>s['limits']['gpu_bytes']:raise MemoryError('memory limit')
                if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output limit')
                del obs,arrays,student
        if module_state_sha256(bound.backbone)!=bound.state_sha256:raise ValueError('encoder state mutation')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),windows=rows,training_steps=0,teacher_payload_reads=0,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_reserved_bytes=gpu_peak)
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
