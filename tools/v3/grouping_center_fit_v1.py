"""One fixed16 centre-only diagnostic. Pair expansion conditional, never on fail."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import resource
import shlex
import signal
import subprocess
import time
import traceback
import zipfile
import numpy as np
import torch
from surface_features_v1 import PYTHON,sha,environment
from development_grids_v1 import write
from grouping_fit_scope_v1 import compile_scope
from bidirectional_paired_reader_v1 import BidirectionalPairedReader
from bidirectional_paired_scope_v1 import identity
from mtare_topo.data.multiview_teacher_reader_v1 import MultiviewTeacherReader
from mtare_topo.data.development_compact_blocks import read_compact_points
from mtare_topo.data.development_partition_scope import ENCODER
from mtare_topo.governance_grouping_fit_v1 import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.grouping_pair_v1 import spatial_representation
from mtare_topo.representation.grouping_center_training_v1 import build_model,forward,center_objective
from mtare_topo.representation.development_paired_training import state_sha256,batch_schedule
from mtare_topo.evaluation.grouping_center_scoring_v1 import score_prediction
from observed_detector_pilot_v1 import preview

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
SCRIPT='tools/v3/grouping_center_fit_v1.py'
GROUPING='PRIMITIVE'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC)):raise FileExistsError('immutable no refreeze')
    scope=compile_scope(ROOT)
    approval=dict(status='APPROVED',approved_by='user-taskbook-and-standing-scope',approved_at='2026-09-09',
        authorized_operations=['training'],authorized_gates=[3],scope_sha256=digest(scope),
        confirmation_reference=POLICY.get('authorization','20260909 taskbook c26dd993: implement fixed about16 fit up to1000 center-only, and at most one evidenced localized correction/retest; prior standing exact-scope authority.'),
        scope='Only fixed16 fit samples trained/scored; original860 metadata/task-level source scope retained for authenticated reader incidental decoding. No cal payload/target inference, protected worlds, new labels, pair expansion or actual navigation.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,'docs/GSE_GROUPING_MINIMAL_TASK_20260909.txt'}|set(POLICY.get('extra_sources',[])))
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,SCRIPT,'--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=0,operation='training',
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question=POLICY.get('question','Can fixed16 PRIMITIVE fit known-domain centres at1m without branch gradients before fair grouping comparison?'),
        method='Original793992parameter observed detector from seed0; identical cached causal sensor context; center-only position/10+balanced presence;1000updates micro1accum4 lr.001 wd0; checkpoint and evaluation each100. Initialization: '+POLICY.get('initialization','original seeded weights')+'; supervision: '+POLICY['positive_negative']+'; selected grouping: '+GROUPING,
        baseline=POLICY.get('baseline','SPATIAL matched-M FPS grouping wired and preprocessing verified, no baseline training before fit health passes. No main-comparison claim from this diagnostic.'),
        fallback='If final known1m precision or recall<.9 stop pair/branch training; diagnose gradients,coordinates,matching,masks/scales from saved evidence. No model search.',
        wall_time_cap_s=POLICY['wall_time_s'],estimated_cost=dict(compute='5090 bounded16samples1000updates plus11evaluations;CPU matched-M preprocessing',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=['Fixed16 training-only12positive4withoutpositive from5parents, no new labels or heldout evaluation.',
            'Known-domain1m centre P/R>=.90 at final1000;threshold.5 diagnostic only;duplicates FP,unknown separate;all gradients logged;branch loss and matching disabled.',
            'Same raw points,query positions,matched M and same token pooling; no imported old500 checkpoint.',
            'Preserve all11checkpoints,predictions,source/evidence masks,counts,XY/XZ,schedule,environment,code snapshot,logs,seal. MAP_EVIDENCE_MISSING if no continuous label-qualified replay.'],
        expected_evidence=['fixed16manifest;unified grouping interface checks;raw predictions and scores every100;final checkpoint;failure localization;command and source hashes;summary'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    write(ROOT/SPEC,spec,'x')
    print(json.dumps(dict(spec=SPEC,counts=scope['counts'],selection=scope['selection'],training_started=False)),flush=True)


class SelectedReader(BidirectionalPairedReader):
    def __init__(self,scope):
        super().__init__(ROOT,scope['original_scope']);self.small_scope=scope

    def selected(self):
        rows={identity(r['source']):r for r in self.small_scope['selected_rows']};seen=set()
        for item in self.scope['source_cards']:
            wanted={k:r for k,r in rows.items() if r['source_card']==item['path']}
            if not wanted:continue
            path,h=item['path'],item['sha256']
            card=json.loads(self._read(path,h))
            def compiler(root,path=path,h=h):return json.loads(read_pinned(root,path,h))['scope']
            reader=MultiviewTeacherReader(ROOT,card['scope'],scope_compiler=compiler)
            try:
                for task in sorted({r['source']['task'] for r in wanted.values()}):
                    bundles=reader.read_task(task)
                    for bundle in bundles:
                        key=identity(bundle['source'])
                        if key not in wanted:continue
                        if key in seen:raise ValueError('duplicate selected observation')
                        row=wanted[key];o=self._observation(bundle,row);seen.add(key)
                        compact=read_compact_points(self._read(row['cache_path'],row['cache_sha256']),
                            manifest_row=row['feature_entry'],expected_source=row['source'],encoder_sha256=ENCODER)
                        start=time.monotonic();spatial=spatial_representation(compact,o.student_representations['r2'])
                        seconds=time.monotonic()-start
                        o.student_representations.update(PRIMITIVE=o.student_representations['r2'],SPATIAL=spatial)
                        yield o,seconds
                    del bundles,bundle
            finally:self.opened.update(reader.opened)
        if seen!=set(rows):raise ValueError('missing selected observation')


def aggregate(rows):
    counts=Counter()
    for r in rows:counts.update({k:r[k] for k in ('tp','fp','fn','ignored','output_count')})
    tp,fp,fn=counts['tp'],counts['fp'],counts['fn']
    return dict(counts,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
        f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.,known_fp_per_observation=fp/len(rows))


@torch.no_grad()
def evaluate(model,examples,run,step):
    model.eval();records=[];main=[]
    for o in examples:
        started=time.monotonic();output=forward(model,o,GROUPING);torch.cuda.synchronize()
        duration=time.monotonic()-started;p=output.prediction;key=digest(o.source)
        np.savez_compressed(run/'artifacts'/f'prediction_{step:04d}_{key}.npz',
            position_m=p.position_m.cpu().numpy(),presence_logits=p.presence_logits.cpu().numpy(),
            query_positions_m=output.query_positions_m.cpu().numpy(),query_indices=output.query_source_indices.cpu().numpy(),
            residual_m=output.residual_m.cpu().numpy())
        scores={str(radius):score_prediction(p,o,radius=radius) for radius in [1.,.5,2.,4.]}
        main.append(scores['1.0']);records.append(dict(source=o.source,scores=scores,inference_s=duration))
        if step in (0,1000):
            # Branches intentionally disabled: do not draw random untrained branches.
            hidden=replace(p,branch_logits=torch.full_like(p.branch_logits,-100.))
            preview(o,hidden,run/'previews'/f'{step:04d}_{key}.png')
    summary=aggregate(main);summary.update(step=step,stage='known_domain_fit_not_generalization')
    write(run/'metrics'/f'evaluation_{step:04d}.json',dict(summary=summary,observations=records),'x')
    print(json.dumps(summary),flush=True)
    return summary


def execute(spec,run):
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec)
            or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh matching run required')
    started=time.monotonic();reader=None;error=None;steps=0;final=None;stage='startup';evaluations=[]
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def expire(*_):raise TimeoutError('whole run budget exceeded')
    old=signal.signal(signal.SIGALRM,expire);signal.alarm(POLICY['wall_time_s'])
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['host_ram_bytes']:raise MemoryError('host cap')
        if torch.cuda.max_memory_reserved()>POLICY['gpu_bytes']:raise MemoryError('GPU cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['output_bytes']:raise OSError('evidence cap')
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        for args,name in [(['git','status','--short'],'worktree_status.txt'),(['git','diff','--binary'],'tracked_worktree.diff')]:
            result=subprocess.run(args,cwd=ROOT,capture_output=True,text=True)
            (run/'config'/name).write_text(result.stdout+result.stderr)
        write(run/'config/environment.json',spec['environment'])
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(POLICY['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        (run/'checkpoints').mkdir();(run/'previews').mkdir(exist_ok=True)
        stage='load_fixed16';reader=SelectedReader(card['scope']);examples=[];inventory=[]
        for o,group_seconds in reader.selected():
            examples.append(o);r=o.student_representations
            inventory.append(dict(source=o.source,anchors=len(o.loss_only['target'].position_m),
                spatial_preprocessing_s=group_seconds,primitive_preprocessing='reused authenticated existing SPG packet; timing not remeasured',
                points=len(r['PRIMITIVE']['blocks'].xyz_m),blocks=len(r['PRIMITIVE']['blocks'].block_ids)))
            # Save the exact input/layout/pose used for every figure, not a later reread.
            sensor=o.loss_only['bundle']['sensor_teacher_only'];key=digest(o.source)
            np.savez_compressed(run/'artifacts'/f'input_{key}.npz',xyz_m=r['PRIMITIVE']['blocks'].xyz_m,
                frame_index=r['PRIMITIVE']['blocks'].frame_index,
                primitive_assignment=r['PRIMITIVE']['blocks'].point_to_block,
                spatial_assignment=r['SPATIAL']['blocks'].point_to_block,
                target_positions_m=o.loss_only['target'].position_m.numpy(),
                sensor_xyz_m=sensor['sensor_xyz_m'],yaw_deg=sensor['yaw_deg'])
            print(json.dumps(dict(stage=stage,loaded=len(examples),source=o.source)),flush=True);guard()
        examples=sorted(examples,key=lambda o:digest(o.source))
        if len(examples)!=16 or sum(len(o.loss_only['target'].position_m) for o in examples)!=12:raise ValueError('fixed16 target drift')
        write(run/'config/population.json',inventory,'x')
        schedule=batch_schedule(16,updates=1000);write(run/'config/schedule.json',schedule,'x')
        model=build_model(device='cuda');initial_sha=state_sha256(model)
        # Both groupings must expose exactly the same observation queries and model state.
        model.eval()
        with torch.no_grad():
            a=forward(model,examples[0],'SPATIAL');b=forward(model,examples[0],'PRIMITIVE')
            if not torch.equal(a.query_positions_m,b.query_positions_m) or not torch.equal(a.query_source_indices,b.query_source_indices):raise ValueError('query mismatch')
        del a,b
        torch.save(dict(model=model.state_dict(),updates=0),run/'checkpoints/step_0000.pt')
        evaluations.append(evaluate(model,examples,run,0))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=0.)
        stage='center_only_training'
        with (run/'logs/updates.jsonl').open('x') as log:
            for step,batch in enumerate(schedule,1):
                model.train();optimizer.zero_grad(set_to_none=True);details=[];active=0
                for i in batch:
                    o=examples[i];output=forward(model,o,GROUPING);r=center_objective(output,o)
                    if not torch.isfinite(r['total']):raise FloatingPointError('nonfinite loss')
                    if r['has_supervision']:(r['total']/4.).backward();active+=1
                    details.append(dict(source=o.source,position=float(r['position'].detach()),presence=float(r['presence'].detach()),
                        positives=r['positive_count'],negatives=r['negative_count'],unknown=r['unknown_count'],assignment=r['assignment'].tolist()))
                    del output,r
                gradients={name:float(p.grad.norm()) for name,p in model.named_parameters() if p.grad is not None}
                if any(not np.isfinite(v) for v in gradients.values()):raise FloatingPointError('nonfinite gradient')
                if any('head.head.branch' in name for name in gradients):raise ValueError('branch gradient in centre-only phase')
                if active:optimizer.step();steps+=1
                log.write(json.dumps(dict(step=step,optimizer_steps=steps,observations=details,gradient_norms=gradients),allow_nan=False)+'\n');log.flush()
                if step%100==0:
                    torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=steps,scheduled_step=step,initial_sha256=initial_sha),run/'checkpoints'/f'step_{step:04d}.pt')
                    final=evaluate(model,examples,run,step);evaluations.append(final);guard()
                elif step%20==0:print(json.dumps(dict(stage=stage,step=step,elapsed_s=time.monotonic()-started)),flush=True);guard()
        for p,h in reader.opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input drift during run')
        for p,h in spec['source_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('source drift during run')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,old)
        budget_complete=(steps==1000 or (POLICY.get('allow_empty_batches',False)
            and steps>0 and len(evaluations)==11 and final and final.get('step')==1000))
        passed=bool(final and final['precision']>=.9 and final['recall']>=.9 and budget_complete and not error)
        summary=dict(status='SOFTWARE_OR_DATA_FAIL' if error else ('FIT_PASS' if passed else 'FIT_FAIL'),stage=stage,error=error,
            optimizer_steps=steps,method=GROUPING,final=final,evaluations=evaluations,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),pair_training_started=False,
            map_status='MAP_EVIDENCE_MISSING',map_reason='Only16 fixed sparse fit observations; not a continuous route with independently qualified visited-edge references.',
            method_advantage_proven=False,next=('main comparison, branches and real mapping remain paused by user; report fixed error outcomes' if POLICY.get('paused') else 'freeze grouped development pair only after fitpass' if passed else 'diagnose saved centre gradients/masks/matching; no pair expansion'))
        write(run/'metrics/summary.json',summary)
        write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
        with (run/'metrics/main_metrics.csv').open('x') as out:
            out.write('stage,method,updates,tp,fp,fn,ignored,precision,recall,f1\n')
            for r in evaluations:out.write('fit,'+GROUPING+','+','.join(str(r[k]) for k in ('step','tp','fp','fn','ignored','precision','recall','f1'))+'\n')
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);args=p.parse_args()
    if args.freeze:freeze()
    elif args.spec and args.run_dir:raise SystemExit(execute(load_json(args.spec),args.run_dir))
    else:p.error('--freeze or --spec and --run-dir required')
