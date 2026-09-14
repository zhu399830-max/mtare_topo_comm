"""One immutable same-R2 C0/C1 relation increment training diagnostic."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter,defaultdict
from dataclasses import fields
import json
from pathlib import Path
import resource
import shlex
import signal
import sys
import time
import traceback
import zipfile
import numpy as np
import torch
from surface_features_v1 import PYTHON,environment,sha
from development_grids_v1 import write
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_block_relation_train_v1 import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.data.development_paired_scope import compile_scope
from mtare_topo.data.development_paired_reader import DevelopmentPairedReader
from mtare_topo.representation.development_paired_training import state_sha256,batch_schedule,forward_observation
from mtare_topo.representation.block_relation_training_v1 import build_relation_model
from mtare_topo.representation.development_corrective_training_v1 import train_update
from mtare_topo.evaluation.development_paired_scoring import score_observation

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
MATCHING='configs/v3/gate3/branch_paired_matching_policy_v1.json'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC)):raise FileExistsError('no refreeze')
    scope=compile_scope(ROOT)
    initial=build_relation_model(relation_attributes=True,seed=0,device='cpu')
    if (sum(p.numel() for p in initial.parameters())!=POLICY['parameters']
            or state_sha256(initial)!=POLICY['initial_state_sha256']):raise ValueError('shared initialization drift')
    approval=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-09',
        scope_sha256=digest(scope),authorized_operations=['training'],authorized_gates=[3],
        scope='Exactly307 existing observations250fit27calibration30development; C0/C1 sameR2 each2000updates micro1accum4; no upstream encoder training, no new labels or test data.',
        confirmation_reference='User standing autonomous development authorization; current PLAN and GSE_BLOCK_RELATION_INCREMENT_V1 authorize a same-R2 explicit-relation paired diagnostic. Exact307, original corrective losses and independent scoring unchanged; no new labels or protected worlds.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,MATCHING,'docs/GSE_BLOCK_RELATION_INCREMENT_V1.md'})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0','CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,
             'tools/v3/block_relation_train_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=0,operation='training',
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='Do explicit relative block attributes improve partial junction/branch recovery over matched-neighbor C0 on identical R2 observations?',
        method='Existing point encoder/head plus reused three message layers789256parameters; C0/C1 same k8 R2 blocks; C0 zeros explicit attributes, C1 uses them; both retain coordinates/context. Seed0 AdamW1e-3wd1e-4 micro1accum4 each2000 updates, original corrective loss, initial/final evaluation only.',
        baseline='C0 retains original geometry and identical neighbor graph; only explicit9 relation attributes ablated. Equal nominal parameters, not equal effective relation-input capacity; partial structure diagnostic only.',
        fallback='Fail and seal on drift, invalid targets, ambiguity, numeric/resource failure; no rerun, additional steps or seeds. No empirical gain means no expansion.',
        wall_time_cap_s=43200,estimated_cost=dict(compute='5090D4000totalupdates plus initial/final307observations per method; zero encoder inference',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=['Exact307 and951 observed branches; reference completeness291,partial16; fit-only250updates population; same initial weights and schedule.', POLICY['advance_condition'],
            'Preserve every raw prediction; source-bound partial scores and parent counts at4/1/2m and10/5/15deg; unknown separate; no full detectorF1 or graph claim.',
            'Both final checkpoints,optimizer states,source/command/environment,logs,metrics,seals; no additional budget or protected worlds.'],
        expected_evidence=['Population and source manifests,initial state,shared schedule,initial/final raw predictions and parent-scored diagnostics,4000update logs,two final states,previews,summary,RUN_STATE,seal.'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],initial_sha256=state_sha256(initial),executed=False)))


@torch.no_grad()
def evaluate(model,examples,method,run,stage,guard):
    model.eval();totals=defaultdict(Counter);label=method+'_'+stage
    directory=run/'artifacts'/label;directory.mkdir()
    with (run/'metrics'/(label+'.jsonl')).open('x') as log:
        for index,o in enumerate(examples):
            p=forward_observation(model,o,'r2')
            name=digest(o.source)+'.npz'
            with (directory/name).open('xb') as f:
                np.savez_compressed(f,**{v.name:getattr(p,v.name).detach().cpu().numpy() for v in fields(p)})
            scores=[]
            for radius in POLICY['position_radii_m']:
                for angle in POLICY['direction_angles_deg']:
                    score=score_observation(p,o,matching_radius_m=radius,matching_angle_deg=angle)
                    parent=o.source['task'].split('__')[0];key=(o.split,parent,radius,angle)
                    for kind in ('anchors','branches'):
                        for metric,value in score[kind].items():
                            if metric in ('tp','fn','known_fp','unresolved_predictions'):totals[key][kind+'_'+metric]+=value
                    scores.append(score)
            log.write(json.dumps(dict(source=o.source,prediction_file=label+'/'+name,scores=scores),allow_nan=False)+'\n');log.flush()
            guard()
            if (index+1)%25==0:print(json.dumps(dict(phase=label,evaluated=index+1,total=len(examples))),flush=True)
    rows=[dict(split=k[0],parent=k[1],position_radius_m=k[2],direction_angle_deg=k[3],counts=dict(v)) for k,v in sorted(totals.items())]
    write(run/'metrics'/(label+'_parents.json'),dict(rows=rows,full_detection_f1=None,graph_score=False),'x')
    return dict(observations=len(examples),parent_metric_rows=len(rows),full_detection_f1=None)


def execute(spec,run):
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    start=time.monotonic();reader=None;error=None;results={};steps={k:0 for k in POLICY['methods']};phase='startup'
    def expire(*args):raise TimeoutError('12h whole-run limit')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(43200)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['host_ram_bytes']:raise MemoryError('host RAM cap')
        if torch.cuda.max_memory_reserved()>POLICY['gpu_bytes']:raise MemoryError('GPU cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['output_bytes']:raise OSError('evidence cap')
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/environment.json',spec['environment'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA required, no CPU training fallback')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(POLICY['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        (run/'checkpoints').mkdir(exist_ok=False)
        phase='load307';reader=DevelopmentPairedReader(ROOT,card['scope']);examples=[];inventory=[]
        for o in reader.observations():
            examples.append(o);t=o.loss_only
            inventory.append(dict(source=o.source,split=o.split,anchors=len(t['target'].position_m),branches=t['observed_branch_count'],
                                  reference_branches=t['reference_branch_count'],reference_complete=all(t['target'].branches_complete)))
            guard()
            if len(examples)%20==0:print(json.dumps(dict(phase=phase,loaded=len(examples),total=307)),flush=True)
        if (len(examples)!=307 or sum(v['branches'] for v in inventory)!=951
                or sum(v['reference_complete'] for v in inventory)!=291):raise ValueError('fixed target population mismatch')
        fit=[o for o in examples if o.split=='fit']
        if len(fit)!=250:raise ValueError('fit population mismatch')
        write(run/'config/population.json',inventory,'x')
        schedule=batch_schedule(len(fit));write(run/'config/schedule.json',schedule,'x')
        initial=build_relation_model(relation_attributes=True,device='cpu');initial_sha=state_sha256(initial)
        if initial_sha!=POLICY['initial_state_sha256']:raise ValueError('initialization drift')
        torch.save(initial.state_dict(),run/'checkpoints/initial.pt');del initial
        for method in POLICY['methods']:
            model=build_relation_model(relation_attributes=(method=='c1'),device='cuda')
            if state_sha256(model)!=initial_sha:raise ValueError('paired state differs')
            phase=method+'_initial';results[phase]=evaluate(model,examples,method,run,'initial',guard)
            optimizer=torch.optim.AdamW(model.parameters(),lr=POLICY['learning_rate'],weight_decay=POLICY['weight_decay'])
            phase=method+'_training'
            with (run/'logs'/(method+'_updates.jsonl')).open('x') as log:
                for index,batch in enumerate(schedule):
                    result=train_update(model,optimizer,[fit[i] for i in batch],'r2')
                    if not result['optimizer_step']:raise ValueError('fixed observed task produced empty supervision')
                    steps[method]+=1;log.write(json.dumps(dict(update=index+1,**result),allow_nan=False)+'\n');log.flush();guard()
                    if (index+1)%10==0:print(json.dumps(dict(phase=phase,update=index+1,total=2000,elapsed_s=time.monotonic()-start)),flush=True)
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=steps[method],initial_sha256=initial_sha),run/'checkpoints'/(method+'_final.pt'))
            phase=method+'_final';results[phase]=evaluate(model,examples,method,run,'final',guard)
            del model,optimizer;torch.cuda.empty_cache()
        for p,h in reader.opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during training')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else 'PAIRED_DIAGNOSTIC_COMPLETE',error=error,phase=phase,updates=steps,
                     results=results,elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
                     scientific_gate_pass=False,full_detection_eligible=False)
        write(run/'metrics/summary.json',summary);write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt';seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
