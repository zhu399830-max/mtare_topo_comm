"""Finite same16 position-only isolation; no detection/unknown scoring rerun."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import io
import json
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
import zipfile
import numpy as np
import torch
from surface_features_v1 import PYTHON,sha,environment
from development_grids_v1 import write
from grouping_center_fit_v1 import SelectedReader
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_position_diagnostic_v1 import (
    SCHEMA,PARENT,AUTH,slug,policy,compile_scope,validate_card)
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.grouping_center_training_v1 import forward
from mtare_topo.representation.development_paired_training import state_sha256,batch_schedule
from mtare_topo.representation.center_position_diagnostic_v1 import (
    geometry_assignment,position_objective,independent_tensor_check)


def schedule_equal(saved):
    return saved==[list(batch) for batch in batch_schedule(16,updates=1000)]


def freeze(mode):
    name=slug(mode);cardpath='configs/v3/gate3/data_cards/'+name+'.json'
    specpath='configs/v3/gate3/'+name+'.json';runpath='results/gate3_semantics/gate3_20260909_'+name+'_seed0'
    if any((ROOT/p).exists() for p in (cardpath,specpath,runpath)):raise FileExistsError('immutable no overwrite')
    scope=compile_scope(ROOT,mode);p=policy(mode)
    approval=dict(status='APPROVED',approved_by='user-latest-exact-scope-and-standing-authorization',approved_at='2026-09-09',
        authorized_operations=['training'],authorized_gates=[3],scope_sha256=digest(scope),confirmation_reference=AUTH,
        scope='Same16 fit observations and sealed initial queries/targets; independent decoder tensors then one PRIMITIVE fixed position diagnostic; conditional dynamic same-start only; no protected data or new labels.')
    card=dict(schema_version=SCHEMA,card_id=name,operation='training',scope=scope,scope_sha256=digest(scope),policy=p,approval=approval)
    if not validate_card(card).passed:raise ValueError('invalid card')
    write(ROOT/cardpath,card,'x')
    files=sorted({str(f.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for f in (ROOT/folder).rglob('*.py')}
        |{cardpath,'tests/v3/unit/test_center_position_diagnostic_v1.py'})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,'tools/v3/primitive_position_diagnostic_v1.py',
        '--spec',str(ROOT/specpath),'--run-dir',str(ROOT/runpath)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=name,seed=0,operation='training',
        data_card=cardpath,config_path=cardpath,user_authorization=approval,command=command,
        question='Does actual decoder and position-only '+mode+' one-to-one regression restore the same12 center locations?',
        method=json.dumps(p,sort_keys=True),baseline='Same sealed PRIMITIVE initial weights and real16 inputs; '+('decoder-independent tensors' if mode=='fixed' else 'completed fixed-position diagnostic'),
        fallback='Stop on numerical/precondition failure or final localization failure; no added budgets or existence restoration.',
        wall_time_cap_s=43200,estimated_cost=dict(compute='one5090 same16 1000scheduled micro1accum4 updates and11evaluations',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=['Actual12 inverse<=1e-5m and independent residual tensor1000 maximum error<=.1m before model training.',
            'At final1000,11of12 assigned centers within1m; finite gradients; exact initial state/input/query/schedule.',
            'No presence or confidence matching cost, no negative/unknown penalties, no GT forward.',
            'Dynamic starts only after fixed pass and from original step0; no detector precision or method-win claims.'],
        expected_evidence=['decoder numerical check,12fixed correspondences,11position checkpoints and errors,gradient logs,raw predictions,exact inputs,summary,source and environment hashes,seal'],
        source_sha256={f:sha(ROOT/f) for f in files},environment=environment())
    write(ROOT/specpath,spec,'x');print(json.dumps(dict(spec=specpath,counts=scope['counts'],policy=p)),flush=True)


@torch.no_grad()
def evaluate(model,examples,fixed,run,step,mode,previous):
    model.eval();rows=[];all_errors=[];switches=0
    for o in examples:
        key=digest(o.source);out=forward(model,o,'PRIMITIVE');p=out.prediction
        target=o.loss_only['target'].position_m.to(p.position_m)
        assignment=fixed[key] if mode=='fixed' else geometry_assignment(p.position_m,target)
        errors=(p.position_m[assignment]-target).norm(dim=-1).cpu().tolist()
        old=previous.get(key,assignment.tolist());switches+=sum(a!=b for a,b in zip(old,assignment.tolist()))
        previous[key]=assignment.tolist();all_errors.extend(errors)
        rows.append(dict(source=o.source,assignment=assignment.tolist(),fixed_step0_assignment=fixed[key].tolist(),
            error_m=errors,within1m=[e<=1 for e in errors],position_m=p.position_m[assignment].cpu().tolist()))
        np.savez_compressed(run/'artifacts'/f'prediction_{step:04d}_{key}.npz',position_m=p.position_m.cpu().numpy(),
            presence_logits=p.presence_logits.cpu().numpy(),query_positions_m=out.query_positions_m.cpu().numpy(),
            query_indices=out.query_source_indices.cpu().numpy(),residual_m=out.residual_m.cpu().numpy())
    summary=dict(step=step,references=len(all_errors),within1m=sum(e<=1 for e in all_errors),
        mean_m=float(np.mean(all_errors)),max_m=max(all_errors),assignment_switches_since_last_evaluation=switches,
        detection_score=False)
    write(run/'metrics'/f'evaluation_{step:04d}.json',dict(summary=summary,observations=rows),'x')
    print(json.dumps(summary),flush=True);return summary


def execute(spec,run):
    run=run.resolve(strict=True);card=load_json(ROOT/spec['data_card']);p=card['policy'];mode=p['mode']
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
        or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    start=time.monotonic();error=None;steps=0;reader=None;final=None;history=[];stage='startup';tensor=None
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def expire(*_):raise TimeoutError('fixed wall budget exceeded')
    oldsignal=signal.signal(signal.SIGALRM,expire);signal.alarm(p['wall_time_s'])
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>p['host_ram_bytes']:raise MemoryError('host cap')
        if torch.cuda.max_memory_reserved()>p['gpu_bytes']:raise MemoryError('GPU cap')
        if sum(f.stat().st_size for f in run.rglob('*') if f.is_file())>p['output_bytes']:raise OSError('disk cap')
    try:
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for path,h in spec['source_sha256'].items():z.writestr(path,read_pinned(ROOT,path,h))
        write(run/'config/environment.json',spec['environment'])
        bound=card['scope']['parent_bound_sha256']
        def parent(rel):return read_pinned(ROOT,PARENT+'/'+rel,bound[PARENT+'/'+rel])
        stage='decoder_independent_tensor_check';fixed_cpu={};queries=[];targets=[];origins=[]
        # Read only original initial queries and target tensors, not later population statistics.
        for item in card['scope']['selection']:
            source=item['source'];key=digest(source)
            with np.load(io.BytesIO(parent('artifacts/prediction_0000_'+key+'.npz'))) as z:q=torch.from_numpy(z['query_positions_m'].copy())
            with np.load(io.BytesIO(parent('artifacts/input_'+key+'.npz'))) as z:t=torch.from_numpy(z['target_positions_m'].copy())
            a=geometry_assignment(q,t);fixed_cpu[key]=a
            if len(t):queries.append(q[a]);targets.append(t)
            origins.append(dict(source=source,assignment=a.tolist(),queries_m=q[a].tolist(),targets_m=t.tolist()))
        if mode=='fixed':
            prerequisite=card['scope']['decoder_prerequisite'];prior=prerequisite['run']
            tensor=json.loads(read_pinned(ROOT,prior+'/metrics/decoder_tensor_check.json',prerequisite['tensor_sha256']))
            if tensor['correspondences']!=origins:raise ValueError('fixed correspondence drift')
            tensor=dict(tensor,reused_from=prior,no_new_tensor_optimization=True)
        else:
            from mtare_topo.governance_position_diagnostic_v1 import sealed_index
            prerequisite=card['scope']['fixed_prerequisite'];prior=prerequisite['run']
            index=sealed_index(ROOT,prior,prerequisite['seal_sha256'])
            tensor=json.loads(read_pinned(ROOT,prior+'/metrics/decoder_tensor_check.json',index[prior+'/metrics/decoder_tensor_check.json']))
            if tensor['correspondences']!=origins:raise ValueError('fixed correspondences changed')
            tensor=dict(tensor,reused_from=prior,no_new_tensor_optimization=True)
        write(run/'metrics/decoder_tensor_check.json',tensor,'x')
        print(json.dumps(dict(stage=stage,passed=tensor['passed'],max_abs_delta_m=tensor['max_abs_delta_m'],
            inverse_max_m=max(tensor['analytic_inverse_error_m']),tensor_final_max_m=tensor['final_max_m'])),flush=True)
        if not tensor['passed']:raise ValueError('decoder/tensor precondition failed; model not trained')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(p['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        (run/'checkpoints').mkdir();stage='load_same16';reader=SelectedReader(card['scope']);examples=[]
        for o,_ in reader.selected():
            key=digest(o.source);r=o.student_representations['PRIMITIVE']['blocks']
            payload=parent('artifacts/input_'+key+'.npz')
            with np.load(io.BytesIO(payload)) as z:
                for name,actual in [('xyz_m',r.xyz_m),('frame_index',r.frame_index),('primitive_assignment',r.point_to_block),
                    ('target_positions_m',o.loss_only['target'].position_m.numpy())]:
                    if not np.array_equal(z[name],actual):raise ValueError('actual parent input drift '+name)
            (run/'artifacts'/('input_'+key+'.npz')).write_bytes(payload)
            examples.append(o);print(json.dumps(dict(stage=stage,loaded=len(examples))),flush=True);guard()
        examples.sort(key=lambda o:digest(o.source))
        if len(examples)!=16:raise ValueError('population drift')
        write(run/'config/population.json',[o.source for o in examples],'x')
        schedule=json.loads(parent('config/schedule.json'))
        if not schedule_equal(schedule):raise ValueError('schedule drift')
        write(run/'config/schedule.json',schedule,'x')
        model=build_model(device='cuda');checkpoint=torch.load(io.BytesIO(parent('checkpoints/step_0000.pt')),map_location='cuda',weights_only=False)
        model.load_state_dict(checkpoint['model'],strict=True);initial=state_sha256(model)
        # Restored state must also equal current unchanged zero-residual initialization.
        pristine=build_model(device='cuda')
        if state_sha256(pristine)!=initial:raise ValueError('initialization implementation drift')
        del pristine
        fixed={k:v.to('cuda') for k,v in fixed_cpu.items()}
        with torch.no_grad():
            for o in examples:
                out=forward(model,o,'PRIMITIVE');key=digest(o.source)
                with np.load(io.BytesIO(parent('artifacts/prediction_0000_'+key+'.npz'))) as z:
                    if (not np.array_equal(out.query_positions_m.cpu().numpy(),z['query_positions_m'])
                        or not np.array_equal(out.query_source_indices.cpu().numpy(),z['query_indices'])):raise ValueError('query drift')
        torch.save(dict(model=model.state_dict(),updates=0,initial_sha256=initial),run/'checkpoints/step_0000.pt')
        write(run/'metrics/input_initialization_check.json',dict(same16=True,same_queries=True,same_schedule=True,
            same_initial_state=True,initial_sha256=initial,GT_in_forward=False,supervision_repair_source_unchanged=True),'x')
        previous={};history.append(evaluate(model,examples,fixed,run,0,mode,previous))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
        stage=mode+'_position_only_training'
        with (run/'logs/updates.jsonl').open('x') as log:
            for step,batch in enumerate(schedule,1):
                model.train();optimizer.zero_grad(set_to_none=True);details=[];active=0
                for i in batch:
                    o=examples[i];out=forward(model,o,'PRIMITIVE');pos=out.prediction.position_m
                    target=o.loss_only['target'].position_m.to(pos)
                    assignment=fixed[digest(o.source)] if mode=='fixed' else geometry_assignment(pos,target)
                    loss=position_objective(pos,target,assignment)
                    if not torch.isfinite(loss):raise FloatingPointError('position loss')
                    if len(target):(loss/4.).backward();active+=1
                    details.append(dict(source=o.source,position=float(loss.detach()),assignment=assignment.tolist(),targets=len(target)))
                gradients={n:float(v.grad.norm()) for n,v in model.named_parameters() if v.grad is not None}
                if any(not np.isfinite(v) for v in gradients.values()):raise FloatingPointError('gradient')
                if any('head.head.branch' in n for n in gradients):raise ValueError('branch gradient')
                for v in (model['head'].head.anchor.weight,model['head'].head.anchor.bias):
                    if v.grad is not None and torch.count_nonzero(v.grad[3]):raise ValueError('existence row gradient')
                if active:optimizer.step();steps+=1
                log.write(json.dumps(dict(step=step,optimizer_steps=steps,active_observations=active,
                    observations=details,gradient_norms=gradients,existence_gradient=0,branch_gradient=0),allow_nan=False)+'\n');log.flush()
                if step%100==0:
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=steps,scheduled_step=step,initial_sha256=initial),run/'checkpoints'/f'step_{step:04d}.pt')
                    final=evaluate(model,examples,fixed,run,step,mode,previous);history.append(final);guard()
                elif step%20==0:print(json.dumps(dict(stage=stage,step=step,elapsed_s=time.monotonic()-start)),flush=True);guard()
        for path,h in {**reader.opened,**spec['source_sha256'],**bound}.items():
            if sha(ROOT/path)!=h:raise ValueError('source or input drift '+path)
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,oldsignal)
        passed=bool(not error and final and final['step']==1000 and final['references']==12 and final['within1m']>=11)
        summary=dict(status='SOFTWARE_OR_DATA_FAIL' if error else ('POSITION_PASS' if passed else 'POSITION_FAIL'),
            mode=mode,stage=stage,error=error,optimizer_steps=steps,final=final,history=history,
            elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),method_advantage_proven=False,
            existence_loss=False,existence_matching_cost=False,branches=False,graph=False,
            next=('dynamic position-only from original initial state allowed' if mode=='fixed' and passed else 'stop; append evidence, no further loss restoration or expansion'))
        write(run/'metrics/summary.json',summary);write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(f)+'  '+str(f.relative_to(ROOT))+'\n' for f in sorted(run.rglob('*')) if f.is_file() and f!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--freeze',choices=['fixed','dynamic']);parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path)
    args=parser.parse_args()
    if args.freeze:freeze(args.freeze)
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:parser.error('--freeze or --spec/--run-dir required')
