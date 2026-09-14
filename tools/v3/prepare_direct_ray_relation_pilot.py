"""Direct-task student interface and weak-target availability, no optimizer."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write
from pathlib import Path
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
NAME='gse_direct_ray_relation_pilot_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
CACHE='results/gate3_semantics/gate3_20260912_gse_direction_task_model_cache_v1_seed0'
SOURCE='results/gate3_semantics/gate3_20260912_gse_direction_task_pilot_inputs_v1_seed20260912'
WITNESS='results/gate3_semantics/gate3_20260913_gse_direction_task_witness_replay_v2_seed0'

def freeze():
    sourcecard='configs/v3/gate3/data_cards/gse_direction_task_model_cache_v1.json'
    source_scope=json.loads((ROOT/sourcecard).read_text())['scope']
    scope=dict(source_scope=source_scope,observations=12,unique_frames=24,parents=3,split='fit',
        model='C fresh shared seed0 initialization, frozen encoder cache; no old final model initialization',
        nearest_geometry='8 nearest observed patch centers to clipped current ray segment, fixed10m/.5m',
        pair_policy='valid observed rays only; grid offsets (0,1),(1,0),(0,8),(0,32),(0,128), azimuth wrap',
        target_policy='same unique section supports conditional positive; different IDs and all unsupported/competing remain unknown; no qualified negative certificates in archived schema',
        training_steps=0,model_forwards=12,labels_training_qualified=False,
        limits=dict(host_bytes=8*1024**3,gpu_bytes=28*1024**3,output_bytes=2*1024**3,wall_seconds=900))
    approval=dict(status='APPROVED',approved_by='user-explicit-direct-branch-plan',approved_at='2026-09-13',
        authorized_operations=['data_export'],authorized_gates=[3],scope='Same12 observations: student interface, initial forward and weak relation availability only',
        confirmation_reference='PLEASE IMPLEMENT THIS PLAN: GSE-Graph推进方案：直接学习支路关系，验证后再接探索闭环',
        scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=scope,approval=approval)
    from mtare_topo.governance_direct_branch_pilot import validate_card
    assert validate_card(card).passed
    pins={sourcecard:sha(ROOT/sourcecard)}
    for folder in (CACHE,SOURCE,WITNESS):
        seal=ROOT/folder/'artifacts/evidence_sha256.txt';pins[str(seal.relative_to(ROOT))]=sha(seal)
        for line in seal.read_text().splitlines():
            h,p=line.split('  ',1);assert sha(ROOT/p)==h,p;pins[p]=h
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1',sys.executable,str(Path(__file__).resolve()),'--execute'],
        question='Does direct per-ray input execute correctly, and do existing witnesses support both training classes?',
        method='Fresh C ray attention/relations; observed pair graph; independent archived weak witnesses read after forward',
        baseline='A/B/C identical initialization tested synthetically; no actual trained baseline claim',
        fallback='Stop fit when qualified positives/negatives absent; no IDs-to-negatives conversion',
        estimated_cost=dict(compute='12 initialC forwards on5090,0optimizer',host_ram_gb=8,gpu_vram_gb=28,disk_gb=2,wall_time_hours=.25),
        acceptance_criteria=['All original valid current rays retained','Nearest8 point-to-segment selection','No teacher in forward; model unchanged','Separate weakpositive, identifier-only unknown, and qualified negative counts'],
        expected_evidence=['Initialization checkpoint, ray pairs, initial predictions, weak masks, source snapshot, environment, logs, summary, seal'],
        input_sha256=pins,execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    import numpy as np,torch
    from dataclasses import fields
    from mtare_topo.representation.branch_relation_learning import LocalBranchEvidence,RayBranchRelationModel,observed_ray_pairs
    from mtare_topo.representation.gse_structural_representation import bind_structural_patches
    from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches
    from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    from mtare_topo.teacher.branch_relation_targets import targets_from_section_witnesses
    from mtare_topo.governance_direct_branch_pilot import validate_card
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());scope=card['scope']
    assert validate_card(card).passed
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    for p,h in {**spec['input_sha256'],**spec['execution_source_sha256']}.items():assert sha(ROOT/p)==h,p
    start=time.monotonic();rows=[];error=None
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def timeout(*_):raise TimeoutError('wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(scope['limits']['wall_seconds'])
    try:
        torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(scope['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(out/'config/runtime_environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0)))
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(ROOT/p,p)
        model=RayBranchRelationModel('C').cuda().eval();initial=module_state_sha256(model)
        torch.save(dict(model=model.state_dict(),step=0,seed=0,variant='C',purpose='INITIALIZATION_NOT_TRAINED'),out/'artifacts/initial.pt')
        with torch.inference_mode(),(out/'logs/observations.jsonl').open('x') as log:
            for i,e in enumerate(scope['source_scope']['entries']):
                with np.load(ROOT/e['student_path'],allow_pickle=False) as data:valid=data['valid_mask'][-1].astype(bool).reshape(-1)
                ids=np.flatnonzero(valid)
                with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as data:
                    kwargs={f.name:data['patch_'+f.name] for f in fields(SurfacePatches) if 'patch_'+f.name in data}
                    kwargs.update(voxel_size_m=.5,roi_radius_m=10.)
                    patch=SurfacePatches(**kwargs)
                    def tensor(x):return torch.from_numpy(np.array(x,copy=True)).float().cuda()
                    frameids=tuple(e['frame_rows'])
                    evidence=LocalBranchEvidence(torch.from_numpy(ids).long().cuda(),tensor(data['registered_returns_xyz_m'][4*11520+ids]),
                        tensor(data['full_sensor_context'][4*180+(ids%720)//4]),tensor(data['patch_observed_features']),
                        bind_structural_patches([patch],device='cuda'),CausalFrameOrderContext('sensor_current',frameids,frameids[-1],e['task']))
                pairs=observed_ray_pairs(evidence.ray_ids);prediction=model(evidence,pairs)
                assert torch.isfinite(prediction.logits).all() and torch.isfinite(prediction.ray_tokens).all()
                np.savez_compressed(out/f'artifacts/prediction_{i:02d}.npz',ray_ids=ids,pairs=pairs.cpu().numpy(),logits=prediction.logits.cpu().numpy(),
                    ray_tokens=prediction.ray_tokens.cpu().numpy(),nearest_patch_indices=prediction.nearest_patch_indices.cpu().numpy())
                # No relation target or construction is read until the above
                # student-only forward result is fixed on disk.
                witness=json.loads((ROOT/WITNESS/f'artifacts/witness_{i:02d}.json').read_text())
                targets,counts=targets_from_section_witnesses(ids.tolist(),pairs.cpu().tolist(),witness)
                np.savez_compressed(out/f'artifacts/weak_targets_{i:02d}.npz',values=targets.values.numpy(),known=targets.known.numpy())
                row=dict(observation=i,task=e['task'],rays=len(ids),pairs=len(pairs),patches=len(patch.centers_m),**counts)
                rows.append(row);write(out/f'artifacts/target_evidence_{i:02d}.json',dict(source=f'{WITNESS}/artifacts/witness_{i:02d}.json',source_sha256=sha(ROOT/WITNESS/f'artifacts/witness_{i:02d}.json'),**counts))
                log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=scope['limits']['host_bytes']
                assert torch.cuda.max_memory_reserved()<=scope['limits']['gpu_bytes']
                assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=scope['limits']['output_bytes']
        assert module_state_sha256(model)==initial
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),observations=rows,training_steps=0,
        fit_ready=False,fit_blocker='Archived witnesses lack observation-supported negative branch relations; differing source IDs cannot supply them',
        weak_positive_pairs=sum(r['weak_positive_pairs'] for r in rows),qualified_negative_pairs=sum(r['qualified_negative_pairs'] for r in rows),
        different_section_pairs_kept_unknown=sum(r['different_section_pairs_kept_unknown'] for r in rows),
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        model_trained=False,method_advantage_proven=False)
    write(out/'metrics/summary.json',summary);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='observations'}));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
