"""One immutable synthetic interface fitting run; never a real-world result."""
import _bootstrap
import argparse,json,resource,shlex,signal,sys,time,traceback,zipfile
from pathlib import Path
from dataclasses import asdict
import numpy as np
import torch
from surface_features_v1 import PYTHON,environment,sha,write
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_synthetic_fit import SCHEMA,SLUG,POLICY,validate_card
from mtare_topo.data.gse_synthetic_fit_scope import compile_scope
ROOT=Path(__file__).resolve().parents[2]
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0'
ENTRY='tools/v3/synthetic_fit_v1.py'


def freeze():
    if (ROOT/CARD).exists() or (ROOT/SPEC).exists():raise FileExistsError('no refreeze')
    s=compile_scope(ROOT)
    a=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-08',
        authorized_operations=['training'],authorized_gates=[3],scope_sha256=digest(s),
        scope='45 archived synthetic fixtures only, shared seed0epoch2 encoder; ABC500 each; no real-world training/test reads.',
        confirmation_reference='User standing autonomous execution authority; current PLAN authorizes one bounded synthetic fit after resource verification.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=s,scope_sha256=digest(s),policy=POLICY,approval=a)
    if not validate_card(card).passed:raise ValueError('card validation failed')
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0',
        'CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,ENTRY,'--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    files=sorted(str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py'))
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260908',slug=SLUG,seed=0,operation='training',
        data_card=CARD,config_path=CARD,user_authorization=a,command=command,
        question='Can the frozen-encoder structural interface fit independently scored synthetic geometry?',
        method='ABC shared initialization/order, fixed500 updates micro1x4 AdamW; declared fixture complete geometry oracle; unknown attributes masked.',
        baseline='A raw observations, B patch unary, C patch relations; in-sample fit only, no research ranking.',
        fallback='Fail and seal source/numeric/resource errors; final fit failure is reported, no retry or checkpoint selection.',
        estimated_cost=dict(compute=('5090D: reuse45 sealed caches; zero encoder forwards; ABC500 updates each' if 'cached_examples' in s else '5090D:45 shared encoder forwards and ABC500 updates each'),host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        wall_time_cap_s=43200,acceptance_criteria=['Exact45 fixed synthetic inputs; zero real-world/test reads.',
            'Anchor and opening F1 >=0.90 at fixed confidence0.5 and1m matching on declared complete sphere; initial/final only.',
            'All ABC reported; no fitted threshold, best epoch or real-world gate advance.',
            'Preserve predictions,weights,history,source bindings,environment and seal; no retry.'],
        expected_evidence=['Shared compact features, raw initial/final predictions, final ABC weights, schedule, per-step logs, per-case scoring, summary and seal.'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment(),
        scoring=dict(confidence=.5,match_radius_m=1.,score_radius_m=10.,outside_predictions='count separately; outside complete region unscored'))
    if s.get('training_contract')=='window_surface_set_v1':
        spec['method']='ABC shared initialization/order; sphere-window openings, training distance/10-probability matching and no-object0.1; fixed500 updates; masked unknown attributes.'
        spec['scoring']['outside_predictions']='Sphere schema verified; every selected opening scored, no radial discard; anchors retain complete sphere.'
        spec['scoring']['version']='window_surface_scoring_v1'
    for path,value in ((CARD,card),(SPEC,spec)):
        with (ROOT/path).open('x') as stream:json.dump(value,stream,ensure_ascii=False,indent=2)
    print(json.dumps(dict(spec=SPEC,scope_sha256=digest(s),observations=45)),flush=True)


def score_predictions(result,examples,*,window_surface=False):
    from mtare_topo.evaluation.gse_synthetic_field_scoring import match_positions
    report={}
    for branch,stages in result.evaluations.items():
        report[branch]={}
        for stage,values in stages.items():
            total={kind:dict(tp=0,fp=0,fn=0,outside=0) for kind in ('anchor','opening')};rows=[]
            for example,value in zip(examples,values,strict=True):
                row=dict(case_id=example.header['observation_id'])
                for kind in total:
                    p=value['prediction'];xyz=getattr(p,kind+'_position_m')[0].numpy()
                    selected=getattr(p,kind+'_presence_logits')[0].sigmoid().numpy()>=.5
                    inside=np.linalg.norm(xyz,axis=1)<=10.
                    t=example.targets;truth=getattr(t,kind+'_position_m')[0][getattr(t,kind+'_valid')[0]].numpy()
                    if window_surface and kind=='opening':
                        from mtare_topo.evaluation.gse_window_surface_scoring import score_complete_window_openings
                        match=score_complete_window_openings(truth,xyz,getattr(p,kind+'_presence_logits')[0].sigmoid().numpy(),complete_window=bool(t.opening_region_complete[0]))
                        match['outside']=0
                    else:
                        match=match_positions(truth,xyz[selected&inside],1.)
                        match['outside']=int((selected&~inside).sum())
                    row[kind]=match
                    for key in total[kind]:total[kind][key]+=match[key]
                rows.append(row)
            for counts in total.values():
                den=2*counts['tp']+counts['fp']+counts['fn'];counts['f1']=2*counts['tp']/den if den else None
            report[branch][stage]=dict(totals=total,per_case=rows)
    return report


def execute(spec,run):
    run=run.resolve(strict=True)
    if run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh exact run required')
    if load_json(run/'config/run_spec.json')!=spec:raise ValueError('spec differs')
    started=time.monotonic();error=None;summary={};signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('12h resource cap')));signal.alarm(43200)
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def check():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['host_ram_bytes']:raise MemoryError('host cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['output_bytes']:raise RuntimeError('output cap')
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if Path(sys.executable)!=Path(PYTHON) or environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec['source_sha256'].items():archive.writestr(p,read_pinned(ROOT,p,h))
        s=compile_scope(ROOT)
        if s!=card['scope']:raise ValueError('scope drift')
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
        from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
        from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
        from mtare_topo.data.gse_synthetic_fit_example import build_example
        from mtare_topo.representation.gse_surface_training_v1 import train_paired_surface_models,SurfaceTrainingBudget
        ck=s['checkpoint'];bound=load_surface_encoder(read_pinned(ROOT,ck['path'],ck['sha256']),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
        adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).to('cuda:0')
        write(run/'config/environment.json',dict(**spec['environment'],gpu=torch.cuda.get_device_name(0),torch=torch.__version__))
        examples=[]
        with (run/'logs/progress.jsonl').open('x') as log:
            def progress(row):
                check();row=dict(row,elapsed_s=time.monotonic()-started)
                log.write(json.dumps(row)+'\n');log.flush();write(run/'metrics/progress.json',row)
                if row.get('attempted_update',0)%25==0:print(json.dumps(row),flush=True)
            if 'cached_examples' in s:
                from mtare_topo.data.gse_synthetic_fit_cache import load_examples
                cache=s['cached_examples']
                examples=load_examples(read_pinned(ROOT,cache['path'],cache['sha256']),s,bound.state_sha256)
                progress(dict(stage='sealed_cache_loaded',completed=len(examples),encoder_forward_windows=0))
            else:
                for row in s['observations']:
                    example=build_example(read_pinned(ROOT,row['input_path'],row['input_sha256']),
                        read_pinned(ROOT,row['source_record_path'],row['source_record_sha256']),row,
                        encoder_adapter=adapter,encoder_state_sha256=bound.state_sha256,device='cuda:0')
                    examples.append(example);progress(dict(stage='features',case_id=row['case_id'],completed=len(examples)))
            torch.save([asdict(e) for e in examples],run/'artifacts/shared_examples.pt')
            result=train_paired_surface_models(examples,budget=SurfaceTrainingBudget(updates=500,device='cuda:0'),
                frozen_encoder=bound.backbone,progress=progress,contract=s.get('training_contract','legacy'))
            torch.save(asdict(result),run/'artifacts/paired_result.pt')
            for branch,state in result.final_states.items():torch.save(state,run/('artifacts/'+branch+'_final.pt'))
            scores=score_predictions(result,examples,window_surface=s.get('training_contract')=='window_surface_set_v1');write(run/'metrics/scores.json',scores)
            passed={b:all(scores[b]['final']['totals'][k]['f1'] is not None and scores[b]['final']['totals'][k]['f1']>=.9 for k in ('anchor','opening')) for b in scores}
            summary=dict(status='SYNTHETIC_FIT_COMPLETE',fit_pass_by_method=passed,counts=result.counts,scientific_gate_pass=False)
        for p,h in spec['source_sha256'].items():read_pinned(ROOT,p,h)
        check()
    except Exception:error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0);summary.update(error=error,elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,real_world_training=False)
        if error:summary['status']='FAILED'
        write(run/'metrics/summary.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name))
        seal=run/'artifacts/evidence_sha256.txt'
        seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path)
    a=p.parse_args()
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('freeze or spec/run-dir required')
