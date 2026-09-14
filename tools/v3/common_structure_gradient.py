"""One fixed final-checkpoint diagnostic; no optimizer and no new model."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,time,traceback
from check_local_pair_window_coverage import sha,write
from membership_fit_v1 import PYTHON
SLUG='gse_common_structure_gradient_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
OLD='results/gate3_semantics/gate3_20260911_gse_common_structure_fit_v1_seed0'

def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    training=json.loads((ROOT/OLD/'config/data_card.json').read_text())
    old=json.loads((ROOT/OLD/'config/run_spec.json').read_text())
    files=dict(old['input_sha256'])
    for p in ('artifacts/final.pt','artifacts/final_predictions.json','artifacts/evidence_sha256.txt'):
        files[OLD+'/'+p]=sha(ROOT/OLD/p)
    s=dict(training_card=training,optimizer_steps=0,checkpoint=OLD+'/artifacts/final.pt',saved_predictions=OLD+'/artifacts/final_predictions.json',
        observation_indices=list(range(16)),limits=dict(wall_seconds=600,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=1024**3))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['audit'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),scope='Fixed common-readout final checkpoint16 gradients; zero updates',confirmation_reference='Active goal and PLAN authorize bounded new configuration gradient/correspondence diagnosis')
    write(ROOT/CARD,dict(schema_version='gse_common_gradient_card_v1',scope=s,approval=a))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='audit',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/common_structure_gradient.py','--execute'],
        question='Do current shared task gradients oppose structure localization?',method='Fixed final16 task gradients and saved-forward correspondence; zero updates',baseline='Saved final predictions at same weights',fallback='Seal failure; no automatic retraining',
        estimated_cost=dict(compute='16 GPU forwards and task gradients',host_ram_gb=32,gpu_vram_gb=28,disk_gb=1,wall_time_hours=1/6),
        acceptance_criteria=['All16 records, unchanged weights, zero updates','Record numerical deltas and exact decision/assignment differences; do not suppress failures','Gradient directions are not actual Adam update or causal proof'],
        expected_evidence=['per-window gradients,aggregate,forward deltas,matching deltas,environment,logs,seal'],input_sha256=files,source_sha256=sources))

def execute():
    import torch
    from mtare_topo.data.gse_common_observation_reader import load_common_observation
    from mtare_topo.representation.gse_common_structure_readout import CommonStructureReadout
    from mtare_topo.representation.gse_partial_structure_contract import bind_partial_targets
    from mtare_topo.representation.gse_structure_prediction_contract import PartialStructuralPrediction,partial_structure_loss,evaluate_partial_structure
    from mtare_topo.representation.gse_membership_gradients import task_gradients,describe_gradients
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    from mtare_topo.governance_membership_fit import validate_common_gradient_card
    spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope'];run=ROOT/RUN
    if not validate_common_gradient_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh validated run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None;aggregate=None
    def expire(*args):raise TimeoutError('600 seconds')
    signal.signal(signal.SIGALRM,expire);signal.alarm(600)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('drift '+p)
        torch.manual_seed(0);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        model=CommonStructureReadout('C').cuda();model.load_state_dict(torch.load(ROOT/s['checkpoint'],weights_only=True,map_location='cuda'));model.eval()
        initial=module_state_sha256(model)
        shared=[(n,p) for n,p in model.named_parameters() if n.startswith(('memory.','query_adapter.','role','decoder.'))]
        saved=json.loads((ROOT/s['saved_predictions']).read_text())
        for i,e in enumerate(s['training_card']['scope']['task']['entries']):
            obs=load_common_observation(ROOT,e['feature_path'],e['feature_sha256'],device='cuda')
            target=bind_partial_targets(json.loads((ROOT/e['reference_path']).read_text())['record'])
            pred=model(obs).prediction;loss=partial_structure_loss(pred,target)
            old=PartialStructuralPrediction(**{k:torch.tensor(v,device='cuda',dtype=torch.float32) for k,v in saved[i]['predictions'].items()})
            oldloss=partial_structure_loss(old,target)
            delta={k:float((v.detach()-getattr(old,k)).abs().max()) for k,v in vars(pred).items() if torch.is_tensor(v)}
            names,g=task_gradients(loss['terms'],[p for _,p in shared])
            if aggregate is None:aggregate=torch.zeros_like(g)
            aggregate+=g/16
            gram=describe_gradients(names,g);si=names.index('structure_position')
            row=dict(observation=i,prediction_delta=delta,decisions_equal=evaluate_partial_structure(pred,target)==evaluate_partial_structure(old,target),
                matches_equal=all((loss['matches'][k]==oldloss['matches'][k]).all().item() for k in loss['matches']),
                structure_total_dot=float(sum(gram['gram'][si])),**gram)
            rows.append(row);write(run/f'artifacts/window_{i:02d}.json',row)
            if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes'] or torch.cuda.max_memory_reserved()>s['limits']['gpu_bytes']:raise MemoryError('memory limit')
            del obs,pred,loss,g,old,oldloss
        if initial!=module_state_sha256(model):raise ValueError('weights changed')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed=len(rows),optimizer_steps=0,elapsed_s=time.monotonic()-start,
        decision_changes=sum(not r['decisions_equal'] for r in rows),assignment_changes=sum(not r['matches_equal'] for r in rows),structure_opposing=sum(r['structure_total_dot']<0 for r in rows),
        aggregate=None if error else describe_gradients(names,aggregate))
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
