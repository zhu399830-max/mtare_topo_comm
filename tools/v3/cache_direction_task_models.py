"""Cache all frozen GPU outputs for the sealed three-fragment task pilot."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha, write
from pathlib import Path
from types import SimpleNamespace
import argparse, hashlib, json, resource, signal, sys, time, traceback, zipfile

NAME='gse_direction_task_model_cache_v1'
SPEC=f'configs/v3/gate3/{NAME}.json'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260912_{NAME}_seed0'
SOURCE='results/gate3_semantics/gate3_20260912_gse_direction_task_pilot_inputs_v1_seed20260912'

def freeze():
    manifest=json.loads((ROOT/SOURCE/'artifacts/manifest.json').read_text())
    original='configs/v3/gate3/data_cards/gse_direction_task_pilot_inputs_v1.json'
    models=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_conditional_development_evaluation_v1.json').read_text())['scope']['models']
    encoder=json.loads((ROOT/'configs/v3/gate3/gse_conditional_development_manifest_v1.json').read_text())['encoder_checkpoint']
    scope=dict(source_run=SOURCE,source_scope=json.loads((ROOT/original).read_text())['scope'],
        entries=[dict(student_path=f"{SOURCE}/{e['student_path']}",sha256=e['student_sha256'],
                      task=e['source']['task'],frame_rows=e['source']['frame_rows'],fragment_id=e['source']['fragment_id']) for e in manifest],
        encoder=encoder,models=models,observations=12,unique_frames=24,independent_fragments=3,
        variants=list('ABC'),training_steps=0,labels_generated=0,teacher_payload_reads=0,
        purpose='Portable frozen features and all outputs, not task accuracy or method benefit',
        limits=dict(host_bytes=4*1024**3,gpu_bytes=28*1024**3,output_bytes=1024**3,wall_seconds=900))
    approval=dict(status='APPROVED',approved_by='user-standing-development-execution',approved_at='2026-09-12',
        authorized_operations=['data_export'],authorized_gates=[3],scope='Same sealed three fit fragments, frozen inference only',
        confirmation_reference='User requests immediate execution before workstation becomes unavailable; standing scoped development authorization',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=scope,approval=approval)
    from mtare_topo.governance_task_cache import validate_card
    assert validate_card(card).passed
    pins={original:sha(ROOT/original),SOURCE+'/artifacts/manifest.json':sha(ROOT/SOURCE/'artifacts/manifest.json')}
    for entry in [*scope['entries'],encoder,*[models[m] for m in 'ABC']]:
        path=entry.get('student_path',entry.get('path'))
        assert sha(ROOT/path)==entry['sha256'],path
        pins[path]=entry['sha256']
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260912',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1',sys.executable,str(Path(__file__).resolve()),'--execute'],
        question='Preserve GPU-dependent observations and frozen ABC outputs for task correspondence work',
        method='12 shared encoder forwards plus36 frozen ABC forwards; complete observed patches',baseline='Same inputs for A/B/C',
        fallback='Stop on drift or resource failure; no alternate model or labels',
        estimated_cost=dict(compute='48 frozen forwards, zero optimizer',host_ram_gb=4,gpu_vram_gb=28,disk_gb=1,wall_time_hours=.25),
        acceptance_criteria=['All12 windows complete','Weights unchanged','No teacher payload in forward','Full causal support and unknown outputs retained'],
        expected_evidence=['Common features, allABC outputs, source snapshot, raw log, environment, hashes'],
        input_sha256=pins,execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    import numpy as np, torch
    from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    from mtare_topo.representation.gse_common_observation import prepare_common_observation
    from mtare_topo.representation.gse_patch_observed_features import pool_patch_observed_features
    from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder,bind_structural_patches
    from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    from mtare_topo.governance_task_cache import validate_card
    spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope'];out=ROOT/RUN
    assert validate_card(card).passed
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    for path,h in {**spec['input_sha256'],**spec['execution_source_sha256']}.items():
        assert sha(ROOT/path)==h,'drift '+path
    start=time.monotonic();rows=[];error=None
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*_):raise TimeoutError('wall cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(s['limits']['wall_seconds'])
    try:
        torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(out/'config/runtime_environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)))
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for path in spec['execution_source_sha256']:z.write(ROOT/path,path)
        ck=s['encoder'];bound=load_surface_encoder((ROOT/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
        adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).cuda().eval()
        models={};initial={}
        for m in 'ABC':
            saved=torch.load(ROOT/s['models'][m]['path'],map_location='cpu',weights_only=True)
            assert saved['variant']==m and saved['step']==2000
            model=GeometryStructureEncoder(m);model.load_state_dict(saved['model'],strict=True)
            models[m]=model.cuda().eval().requires_grad_(False);initial[m]=module_state_sha256(model)
        with torch.inference_mode(),(out/'logs/windows.jsonl').open('x') as log:
            for i,e in enumerate(s['entries']):
                with np.load(ROOT/e['student_path'],allow_pickle=False) as data:
                    assert set(data.files)=={'ranges_m','valid_mask','relative_translation_current_sensor_m','relative_yaw_current_sensor_deg'}
                    student=SimpleNamespace(**{k:data[k] for k in data.files})
                obs=prepare_common_observation(adapter,student,device='cuda');x=pool_patch_observed_features(obs)
                patches=bind_structural_patches([obs.surface_patches],device='cuda')
                frames=tuple(e['frame_rows']);ctx=(CausalFrameOrderContext('sensor_current',frames,frames[-1],e['task']),)
                arrays={k:getattr(obs,k) for k in ('registered_returns_xyz_m','surface_return_indices','ray_sensor_token_indices','ray_history_indices')}
                arrays.update(full_sensor_context=obs.full_sensor_context.cpu().numpy(),full_sensor_valid=obs.full_sensor_valid.cpu().numpy(),patch_observed_features=x.cpu().numpy())
                arrays.update({'patch_'+k:v for k,v in vars(obs.surface_patches).items() if isinstance(v,np.ndarray)})
                arrays.update({'ray_'+k:v for k,v in vars(obs.local_rays).items() if isinstance(v,np.ndarray)})
                np.savez_compressed(out/f'artifacts/common_{i:02d}.npz',**arrays)
                for m,model in models.items():
                    result=model(x,patches,ctx)
                    outputs={k:v.cpu().numpy() for k,v in vars(result).items() if torch.is_tensor(v)}
                    outputs.update({'relation_'+k:v.cpu().numpy() for k,v in vars(result.relations).items() if torch.is_tensor(v)})
                    assert all(np.isfinite(v).all() for v in outputs.values())
                    np.savez_compressed(out/f'artifacts/{m}_{i:02d}.npz',**outputs)
                row=dict(index=i,fragment=e['fragment_id'],patches=len(obs.surface_patches.centers_m),elapsed_s=time.monotonic()-start)
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=s['limits']['host_bytes']
                assert torch.cuda.max_memory_reserved()<=s['limits']['gpu_bytes']
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=s['limits']['output_bytes']
        assert module_state_sha256(bound.backbone)==bound.state_sha256
        assert all(module_state_sha256(models[m])==initial[m] for m in 'ABC')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),windows=rows,training_steps=0,teacher_payload_reads=0,
        method_advantage_proven=False,elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved())
    write(out/'metrics/summary.json',summary);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
