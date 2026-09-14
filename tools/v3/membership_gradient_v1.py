"""Bound fixed16 final-checkpoint gradients; no optimizer or new checkpoint."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write
from membership_fit_v1 import PYTHON

SLUG='gse_membership_gradient_missing5_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
OLD='results/gate3_semantics/gate3_20260911_gse_membership_fit_v1r1_seed0'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    old=json.loads((ROOT/OLD/'config/run_spec.json').read_text())
    training=json.loads((ROOT/OLD/'config/data_card.json').read_text())
    indices=list(range(11,16))
    chosen=[training['scope']['binding']['entries'][i] for i in indices]
    ck=training['scope']['checkpoint']
    files={ck['path']:ck['sha256']}
    for e in chosen:
        files[e['student_path']]=e['student_sha256'];files[e['target_path']]=e['target_sha256']
    for p in ('artifacts/final.pt','artifacts/predictions.json','artifacts/evidence_sha256.txt','config/data_card.json'):
        files[OLD+'/'+p]=sha(ROOT/OLD/p)
    scope=dict(training_card=training,optimizer_steps=0,checkpoint_role='final_only',observation_indices=indices,
        selected_windows=5,decoded_observations=sum(e['decoded_observations'] for e in chosen),
        numerical_policy='Record original1e-5 exceedances and decision differences, never reclassify original run; no training decision from unreproduced outputs',
        final_checkpoint=OLD+'/artifacts/final.pt',saved_predictions=OLD+'/artifacts/predictions.json',
        limits=dict(wall_seconds=600,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=1024**3))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',
        authorized_gates=[3],authorized_operations=['audit'],scope_sha256=hashlib.sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest(),
        scope='Final checkpoint and only missing observations11..15 from fixed16; no updates',
        confirmation_reference='Active goal and current PLAN permit one bounded final-checkpoint task-gradient diagnosis; no retraining')
    write(ROOT/CARD,dict(schema_version='gse_membership_gradient_card_v1',scope=scope,approval=a))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='audit',data_card=CARD,
        user_authorization=a,question='Do task gradients oppose localization at the saved final checkpoint?',
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/membership_gradient_v1.py','--execute'],
        method='Missing5 per-window shared-gradient matrices and numerical/decision differences; no update',baseline='Same-checkpoint saved predictions and original1e-5 numerical criterion',
        fallback='Fail and seal; no checkpoint change or automatic training',
        estimated_cost=dict(compute='singleGPU16 forward/backward diagnostics',host_ram_gb=32,gpu_vram_gb=28,disk_gb=1,wall_time_hours=1/6),
        acceptance_criteria=['Record exact deltas, original1e-5 failures and scoring/assignment changes without suppressing outputs','Exactly missing5 records; model and encoder tensors unchanged','No optimizer steps; original failed run unchanged'],
        expected_evidence=['gradient matrices,source snapshot,per-window predictions difference,environment,logs,seal'],input_sha256=files,source_sha256=sources))


def execute():
    import torch
    from mtare_topo.governance_membership_fit import validate_gradient_card
    from mtare_topo.data.gse_membership_fit_reader import load_student_window,load_partial_reference
    from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    from mtare_topo.representation.gse_observed_membership_v1 import ObservedMembershipV1
    from mtare_topo.representation.gse_membership_fit_execution import prepare_membership_example,predict_membership,evaluate_membership
    from mtare_topo.representation.gse_observed_membership_loss import observed_membership_loss
    from mtare_topo.representation.gse_membership_gradients import task_gradients,describe_gradients
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope'];training=s['training_card']['scope']
    if not validate_gradient_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;rows=[];aggregate=None
    def expire(*args):raise TimeoutError('600second cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        model=ObservedMembershipV1('C').cuda();model.load_state_dict(torch.load(ROOT/s['final_checkpoint'],weights_only=True,map_location='cuda'));model.eval()
        initial_sha=module_state_sha256(model)
        ck=training['checkpoint'];bound=load_surface_encoder((ROOT/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
        adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).cuda()
        shared=[(n,p) for n,p in model.named_parameters() if n.startswith(('raw_adapter','patch_adapter','patch_layers','decoder','query_position','query_role'))]
        saved=json.loads((ROOT/s['saved_predictions']).read_text())['final']
        with (run/'logs/windows.jsonl').open('x') as log:
            for i in s['observation_indices']:
                e=training['binding']['entries'][i]
                student=load_student_window(ROOT,e);target=load_partial_reference(ROOT,e)
                example=prepare_membership_example(adapter,student,target['record'],device='cuda')
                pred=predict_membership(model,example);deltas={}
                for name,value in vars(pred).items():
                    old=torch.tensor(saved[i]['all_predictions'][name],device=value.device,dtype=value.dtype)
                    deltas[name]=float((value.detach()-old).abs().max())
                    if not torch.isfinite(value).all():raise ValueError('nonfinite saved prediction comparison')
                evaluation=evaluate_membership(pred,target['record'])
                same_decisions=(evaluation['membership']==saved[i]['membership'] and all(
                    evaluation[role]['reference_to_prediction']==saved[i][role]['reference_to_prediction'] or
                    {str(k):v for k,v in evaluation[role]['reference_to_prediction'].items()}==saved[i][role]['reference_to_prediction']
                    for role in ('anchor','opening')))
                loss=observed_membership_loss(pred,target['record'])
                names,g=task_gradients(loss['terms'],[p for _,p in shared])
                if aggregate is None:aggregate=torch.zeros_like(g)
                aggregate+=g/len(s['observation_indices'])
                row=dict(observation=i,prediction_delta=deltas,original_numerical_check_passed=max(deltas.values())<=1e-5,
                    decisions_unchanged=same_decisions,losses={k:float(v.detach()) for k,v in loss['terms'].items()},**describe_gradients(names,g))
                rows.append(row);log.write(json.dumps(row)+'\n');log.flush()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes'] or torch.cuda.max_memory_reserved()>s['limits']['gpu_bytes']:raise MemoryError('resource cap')
                print(json.dumps(dict(completed=i+1,elapsed_s=time.monotonic()-start)),flush=True)
                del example,pred,loss,g
        if initial_sha!=module_state_sha256(model) or bound.state_sha256!=module_state_sha256(bound.backbone):raise ValueError('model mutation')
        write(run/'artifacts/gradients.json',dict(windows=rows,mean_subset_gradient=describe_gradients(names,aggregate),shared_parameters=[n for n,p in shared]))
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),optimizer_steps=0,
        opposing_windows=sum(r['total_gradient_opposes_location'] for r in rows),elapsed_s=time.monotonic()-start,
        original_numerical_failures=sum(not r['original_numerical_check_passed'] for r in rows),
        decision_changes=sum(not r['decisions_unchanged'] for r in rows),
        mean_subset_gradient=None if error else describe_gradients(names,aggregate))
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(run))+'\n')
    print(json.dumps(summary),flush=True);return int(bool(error))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose freeze or execute')
