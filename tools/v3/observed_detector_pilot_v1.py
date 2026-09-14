"""Single immutable observation-anchored500batch pilot; never launches pairs."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter,defaultdict
from dataclasses import fields
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
from surface_features_v1 import PYTHON,environment,sha
from development_grids_v1 import write
from bidirectional_paired_scope_v1 import compile_scope
from bidirectional_paired_reader_v1 import BidirectionalPairedReader
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_observed_detector_pilot_v1 import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.representation.observed_anchor_training_v1 import build_model,forward_observation,train_update
from mtare_topo.representation.development_paired_training import state_sha256,batch_schedule
from mtare_topo.representation.anchor_branch_loss import AnchorBranchPrediction
from mtare_topo.evaluation.development_paired_scoring import score_observation
from mtare_topo.evaluation.observed_detector_feasibility_v1 import fit_template,fixed_correspondence_shuffle,decision

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
DOCUMENT='docs/GSE_OBSERVATION_ANCHORED_EXECUTION_V1.md'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC)):raise FileExistsError('no refreeze')
    scope=compile_scope(ROOT);model=build_model()
    if (sum(p.numel() for p in model.parameters())!=POLICY['parameters']
            or state_sha256(model)!=POLICY['initial_state_sha256']):raise ValueError('initialization drift')
    approval=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-09',
        scope_sha256=digest(scope),authorized_operations=['training'],authorized_gates=[3],
        scope='Same860 existing observations282fit578cal; only500 no-explicit-relation scheduled batches micro1accum4 seed0. No new labels, encoder inference, C08-C10, paired training or graph execution.',
        confirmation_reference='User explicitly approved observation-anchored detector plan and continuing goal:500batch feasibility before any2000pair; standing autonomous authorization on unchanged282/578 source population.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=approval)
    report=validate_card(card)
    if not report.passed:raise ValueError(report.errors)
    write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3')
                  for p in (ROOT/folder).rglob('*.py')}|{CARD,DOCUMENT})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,'tools/v3/observed_detector_pilot_v1.py',
        '--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260909',slug=SLUG,seed=0,operation='training',
        data_card=CARD,config_path=CARD,user_authorization=approval,command=command,
        question='Can observation-anchored no-explicit-relation detection fit partial junction and branch references without fixed-template or all-reject failure?',
        method='3DETR mechanism adaptation, not reproduction; FPS32 real observations, coordinate encoding,20m bounded residual10m sphere; frozen context,new point adapter/detector793992parameters;500seed0AdamW1e-3wd1e-4micro1accum4,balanced conditional presence and unchanged branch objective.',
        baseline='Fit-only slot-mean output template and fixed input-target derangement; initial checkpoint; calibration reported but not independent test. No geometry relation pair authorized by this run.',
        fallback='Stop expansion if any fixed pilot criterion fails. Preserve final weights and failed evidence. Software defect correction only with evidence, no repeated completed updates. Center heatmap needs separate partial-label specification, not training authority.',
        wall_time_cap_s=POLICY['wall_time_s'],estimated_cost=dict(compute='5090D500scheduledbatches;initial/final860scores;fit-only saved-outputcounterfactuals;zero encoder rerender',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=['Exact282fit578cal,186/598fit references,562/1813all; zero protected worlds and unchanged source order.',
            'Fit anchor and branch partial recall>=0.90,combined knownFP/observation<=0.1,all unresolved counts separate; template nearest-error advantage>=0.20m; shuffled fit anchor recall drop>=0.10.',
            'Initial/final only;0.5fixed thresholds;all1/2/4m and5/10/15deg scores; no checkpoint/threshold selection or graph claim.',
            'Preserve queries/sourceindices,residuals,all predictions,matching/evidence counts,updates,final weights,exactsource hashes,command,environment,identity-fixed XY/XZ previews,summary,RUN_STATE,seal.'],
        expected_evidence=['Two860population raw prediction sets and9scores per row;500scheduled batch log;final state;fittemplate and282shuffled source-bound scores;fixedidentity previews;immutable seal.'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    write(ROOT/SPEC,spec,'x')
    print(json.dumps(dict(spec=SPEC,counts=scope['counts'],initial_sha256=state_sha256(model),executed=False)))


def counts(score):
    return Counter({kind+'_'+metric:int(score[kind][metric]) for kind in ('anchors','branches')
                    for metric in ('tp','fn','known_fp','unresolved_predictions')})


def preview(observation,prediction,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    blocks=observation.student_representations['r2']['blocks'];xyz=blocks.xyz_m
    target=observation.loss_only['target'].position_m.cpu().numpy()
    position=prediction.position_m.cpu().numpy()
    selected=prediction.presence_logits.sigmoid().cpu().numpy()>=.5
    fig,axes=plt.subplots(2,3,figsize=(13,8))
    for row,(u,v) in enumerate(((0,1),(0,2))):
        for col in range(3):
            ax=axes[row,col]
            ax.scatter(xyz[:,u],xyz[:,v],s=.2,c=blocks.point_to_block if col==1 else '0.65',
                       cmap='tab20' if col==1 else None,rasterized=True)
            if col==2:
                ax.scatter(position[~selected,u],position[~selected,v],s=10,c='silver',marker='.')
                ax.scatter(position[selected,u],position[selected,v],s=35,c='red',marker='x',label='prediction >=0.5')
                ax.scatter(target[:,u],target[:,v],s=65,facecolors='none',edgecolors='black',label='partial reference')
                for q in np.flatnonzero(selected):
                    ds=prediction.directions[q][prediction.branch_logits[q].sigmoid()>=.5].cpu().numpy()
                    for d in ds:ax.plot([position[q,u],position[q,u]+d[u]],[position[q,v],position[q,v]+d[v]],c='red',lw=.7)
                ax.legend(fontsize=6)
            ax.set(xlim=(-10,10),ylim=(-10,10),aspect='equal',xlabel='X (m)',ylabel=('Y' if v==1 else 'Z')+' (m)')
            ax.set_title(('Observed points','Existing geometric blocks','Predictions / partial references')[col])
    fig.suptitle(json.dumps(observation.source,sort_keys=True)+'\nUnknown background is not confirmed absence',fontsize=7)
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


@torch.no_grad()
def evaluate(model,examples,run,stage,guard,preview_sources):
    model.eval();totals=defaultdict(Counter);saved=[]
    directory=run/'artifacts'/stage;directory.mkdir()
    with (run/'metrics'/(stage+'.jsonl')).open('x') as log:
        for index,o in enumerate(examples):
            output=forward_observation(model,o)
            if output is None:raise ValueError('empty source requires accounting; no guessed output')
            p=AnchorBranchPrediction(**{f.name:getattr(output.prediction,f.name).detach().cpu() for f in fields(output.prediction)})
            name=digest(o.source)+'.npz'
            arrays={f.name:getattr(p,f.name).numpy() for f in fields(p)}
            arrays.update(query_source_indices=output.query_source_indices.detach().cpu().numpy(),
                          query_positions_m=output.query_positions_m.detach().cpu().numpy(),residual_m=output.residual_m.detach().cpu().numpy())
            with (directory/name).open('xb') as f:np.savez_compressed(f,**arrays)
            scores=[]
            for radius in POLICY['position_radii_m']:
                for angle in POLICY['direction_angles_deg']:
                    score=score_observation(p,o,matching_radius_m=radius,matching_angle_deg=angle)
                    totals[(o.split,o.source['task'].split('__')[0],radius,angle)].update(counts(score));scores.append(score)
            log.write(json.dumps(dict(source=o.source,prediction_file=stage+'/'+name,scores=scores),allow_nan=False)+'\n');log.flush()
            if digest(o.source) in preview_sources:preview(o,p,run/'previews'/(stage+'_'+digest(o.source)+'.png'))
            saved.append(p);del output;guard()
            if (index+1)%25==0:print(json.dumps(dict(phase=stage,evaluated=index+1,total=len(examples))),flush=True)
    rows=[dict(split=k[0],parent=k[1],position_radius_m=k[2],direction_angle_deg=k[3],counts=dict(v)) for k,v in sorted(totals.items())]
    write(run/'metrics'/(stage+'_parents.json'),dict(rows=rows,full_detection_f1=None,graph_score=False),'x')
    return saved,rows


def counterfactuals(examples,predictions,rows,run,guard):
    indices=[i for i,o in enumerate(examples) if o.split=='fit']
    fit=[examples[i] for i in indices];pred=[predictions[i] for i in indices]
    template=fit_template([dict(split='fit',position_m=p.position_m.numpy()) for p in pred])
    write(run/'artifacts/fit_template.json',{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in template.items()},'x')
    donors=fixed_correspondence_shuffle([o.source for o in fit]);error=template_error=0.;n=0;shuffled=Counter()
    with (run/'metrics/fit_counterfactuals.jsonl').open('x') as log:
        for i,(o,p) in enumerate(zip(fit,pred)):
            target=o.loss_only['target'].position_m.cpu().numpy().astype(float)
            actual=np.linalg.norm(target[:,None]-p.position_m.numpy()[None],axis=-1).min(axis=1)
            fixed=np.linalg.norm(target[:,None]-template['position_m'][None],axis=-1).min(axis=1)
            error+=float(actual.sum());template_error+=float(fixed.sum());n+=len(target)
            scores=[]
            for radius in POLICY['position_radii_m']:
                for angle in POLICY['direction_angles_deg']:
                    s=score_observation(pred[int(donors[i])],o,matching_radius_m=radius,matching_angle_deg=angle)
                    scores.append(s)
                    if radius==4. and angle==10.:shuffled.update(counts(s))
            log.write(json.dumps(dict(kind='counterfactual_not_deployment_prediction',recipient=o.source,
                donor=fit[int(donors[i])].source,actual_nearest_errors_m=actual.tolist(),template_nearest_errors_m=fixed.tolist(),scores=scores),allow_nan=False)+'\n');log.flush();guard()
    if n!=186:raise ValueError('fit reference count drift')
    main=Counter()
    for r in rows:
        if r['split']=='fit' and r['position_radius_m']==4. and r['direction_angle_deg']==10.:main.update(r['counts'])
    return decision(observations=len(fit),anchor_tp=main['anchors_tp'],anchor_fn=main['anchors_fn'],
        branch_tp=main['branches_tp'],branch_fn=main['branches_fn'],anchor_known_fp=main['anchors_known_fp'],
        branch_known_fp=main['branches_known_fp'],unresolved_anchors=main['anchors_unresolved_predictions'],
        unresolved_branches=main['branches_unresolved_predictions'],candidate_mean_error_m=error/n,
        template_mean_error_m=template_error/n,shuffled_anchor_tp=shuffled['anchors_tp'],shuffled_anchor_fn=shuffled['anchors_fn'])


def execute(spec,run):
    run=run.resolve(strict=True)
    if (run!=ROOT/'results/gate3_semantics'/build_run_id(spec)
            or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED'
            or load_json(run/'config/run_spec.json')!=spec):raise ValueError('fresh exact run required')
    start=time.monotonic();reader=None;error=None;phase='startup';steps=0;result=None
    def expire(*args):raise TimeoutError('whole-run resource limit')
    previous=signal.signal(signal.SIGALRM,expire);signal.alarm(POLICY['wall_time_s'])
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
        if not torch.cuda.is_available():raise RuntimeError('CUDA required; no CPU training substitute')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(POLICY['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        (run/'checkpoints').mkdir();(run/'previews').mkdir(exist_ok=True)
        phase='load860';reader=BidirectionalPairedReader(ROOT,card['scope']);examples=[];inventory=[]
        for o in reader.observations():
            examples.append(o);t=o.loss_only
            inventory.append(dict(source=o.source,split=o.split,anchors=len(t['target'].position_m),branches=t['observed_branch_count']))
            guard()
            if len(examples)%20==0:print(json.dumps(dict(phase=phase,loaded=len(examples),total=860)),flush=True)
        if (len(examples)!=860 or sum(r['anchors'] for r in inventory)!=562 or sum(r['branches'] for r in inventory)!=1813):
            raise ValueError('fixed population mismatch')
        fit=[o for o in examples if o.split=='fit']
        if len(fit)!=282 or len(examples)-len(fit)!=578:raise ValueError('split mismatch')
        write(run/'config/population.json',inventory,'x')
        schedule=batch_schedule(282,updates=500);write(run/'config/schedule.json',schedule,'x')
        preview_ids={digest(o.source) for split in ('fit','calibration') for o in sorted(
            [e for e in examples if e.split==split],key=lambda e:json.dumps(e.source,sort_keys=True))[:3]}
        write(run/'config/preview_source_ids.json',sorted(preview_ids),'x')
        model=build_model(device='cuda');initial_sha=state_sha256(model)
        if initial_sha!=POLICY['initial_state_sha256']:raise ValueError('initial state drift')
        torch.save(model.state_dict(),run/'checkpoints/initial.pt')
        phase='initial';initial,_=evaluate(model,examples,run,phase,guard,preview_ids);del initial
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);phase='training'
        with (run/'logs/updates.jsonl').open('x') as log:
            for index,batch in enumerate(schedule):
                r=train_update(model,optimizer,[fit[i] for i in batch]);steps+=int(r['optimizer_step'])
                log.write(json.dumps(dict(batch=index+1,**r),allow_nan=False)+'\n');log.flush();guard()
                if (index+1)%10==0:print(json.dumps(dict(phase=phase,batch=index+1,total=500,elapsed_s=time.monotonic()-start)),flush=True)
        torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),updates=steps,
            scheduled_batches=500,initial_sha256=initial_sha),run/'checkpoints/final.pt')
        phase='final';predictions,rows=evaluate(model,examples,run,phase,guard,preview_ids)
        phase='counterfactuals';result=counterfactuals(examples,predictions,rows,run,guard)
        for p,h in reader.opened.items():
            if sha(ROOT/p)!=h:raise ValueError('input changed during run')
        for p,h in spec['source_sha256'].items():
            if sha(ROOT/p)!=h:raise ValueError('bound source changed during run')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);signal.signal(signal.SIGALRM,previous)
        summary=dict(status='FAILED' if error else ('PILOT_PASS' if result['pilot_pass'] else 'PILOT_FAIL'),
            phase=phase,error=error,optimizer_steps=steps,decision=result,elapsed_s=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
            scientific_gate_pass=False,independent_evaluation=False,graph_advantage=False,pair_training_started=False)
        write(run/'metrics/summary.json',summary);write(run/'artifacts/source_reads_sha256.json',reader.opened if reader else {})
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
