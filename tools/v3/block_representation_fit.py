"""One immutable same45 R0/R1/R2 fit with common initialized weights."""
import _bootstrap
import argparse,io,json,resource,shlex,signal,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from surface_features_v1 import PYTHON,environment,sha,write
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_block_fit import SCHEMA,SLUG,POLICY,scope,validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_partition_cache_binding import load_bound_partition_observations
from mtare_topo.data.gse_synthetic_fit_scope import declared_cases
from mtare_topo.data.gse_block_targets import located_targets
from mtare_topo.evaluation.gse_synthetic_field_scoring import expected_geometry
from mtare_topo.evaluation.gse_block_structure_scoring import prediction_record,score_record
from mtare_topo.representation.gse_block_point_encoder import BlockPointEncoder
from mtare_topo.representation.gse_block_structure_readout import BlockStructureReadout
from mtare_topo.representation.gse_block_structure_loss import located_structure_loss

ROOT=Path(__file__).resolve().parents[2]
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0'
METHODS=('r0','r1','r2')
COORDINATE_VERSION=1


def configure_version(version):
    global COORDINATE_VERSION,SCHEMA,SLUG,CARD,SPEC,RUN
    if version not in (1,2):raise ValueError('unknown coordinate version')
    COORDINATE_VERSION=version
    SCHEMA='v3_block_fit_card_v'+str(version);SLUG='gse_block_representation_fit_v'+str(version)
    CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
    RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC)):raise FileExistsError('no refreeze')
    s=scope(ROOT,COORDINATE_VERSION)
    a=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-08',
        scope='Exact45 synthetic shared block R0/R1/R2 each500 updates; no real-world or backbone training',
        scope_sha256=digest(s),authorized_operations=['training'],authorized_gates=[3],
        confirmation_reference='User standing full autonomous authority; PLAN exact population, software and GPU checks complete, single fit next')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=s,scope_sha256=digest(s),approval=a)
    if not validate_card(card).passed:raise ValueError('invalid card')
    with (ROOT/CARD).open('x') as f:json.dump(card,f,indent=2)
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,POLICY})
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260908',slug=SLUG,seed=0,operation='training',
        data_card=CARD,config_path=CARD,user_authorization=a,
        command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,
            'tools/v3/block_representation_fit.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)],
        question='Does changing grouping with identical all-point geometry/context/readout permit fitting located structures?',
        method='Shared464011parameter state; R0sensor,R1voxel,R2SPG; each500updates,AdamW1e-3wd1e-4,micro1accum4',
        baseline='R0 sensor-layout grouping includes XYZ and inherited geometry pretraining; primary R2 versus R1',
        fallback='Seal error/negative results, no automatic retry or additional seeds; no graph claim',
        estimated_cost=dict(compute='5090D1500 total updates,zero encoder inference',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=['All45 initial/final raw scoring,score>=0.5,noNMS;anchor/opening1mF1>=0.90 is fit only',
            'Same initialization and schedule; all backgrounds retained; no threshold/best-checkpoint selection','No research Gate advance'],
        expected_evidence=['Input manifest,schedule,initial state,raw predictions,independent scores,allfinalweights,optimizer,logs,environment,source snapshot,seal'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    if COORDINATE_VERSION==2:
        spec['coordinate_version']=2
        spec['command']+=['--coordinate-version','2']
        spec['question']='Does correcting the verified hard radial plateau restore fitting, under exactly the old initial state and schedule?'
        spec['method']+='; only smooth anchor coordinates changed; no composition or graph contribution claim'
    with (ROOT/SPEC).open('x') as f:json.dump(spec,f,indent=2)
    print(json.dumps(dict(spec=SPEC,scope_sha256=digest(s),population=s['population'])),flush=True)


def model_pair():
    if COORDINATE_VERSION==2:
        from mtare_topo.representation.gse_block_structure_readout_v2 import BlockStructureReadoutV2
        head_class=BlockStructureReadoutV2
    else:head_class=BlockStructureReadout
    return torch.nn.ModuleDict(dict(point=BlockPointEncoder(),head=head_class()))


def forward(model,example,method):
    item=example['methods'][method]
    context=torch.tensor(item['context'],device='cuda')
    return model['head'](model['point'](item['blocks'],chunk_size=2048),context)


@torch.no_grad()
def evaluate(model,examples,method,run,label):
    model.eval();rows=[];totals={k:dict(tp=0,fp=0,fn=0) for k in ('anchor1','anchor2','anchor4','opening1')}
    for e in examples:
        record=prediction_record(forward(model,e,method));score=score_record(record,e['reference'])
        if score['status']!='SCORED_COMPLETE_SYNTHETIC_REFERENCE':raise ValueError('scoring coverage drift')
        for key,value in [('anchor'+str(int(r)),score['anchors'][str(r)]) for r in (1.,2.,4.)]+[('opening1',score['openings'])]:
            for c in ('tp','fp','fn'):totals[key][c]+=value[c]
        rows.append(dict(observation_id=e['observation_id'],prediction=record,score=score))
    for v in totals.values():
        den=2*v['tp']+v['fp']+v['fn'];v['f1']=2*v['tp']/den if den else None
    write(run/('artifacts/'+label+'_predictions.json'),rows)
    write(run/('metrics/'+label+'.json'),totals)
    return totals


def execute(spec,run):
    if spec.get('coordinate_version',1)!=COORDINATE_VERSION:raise ValueError('coordinate CLI/spec drift')
    run=run.resolve(strict=True)
    if run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run required')
    if load_json(run/'config/run_spec.json')!=spec:raise ValueError('spec drift')
    start=time.monotonic();error=None;results={};steps={k:0 for k in METHODS}
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('12h cap')));signal.alarm(43200)
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>32*1024**3 or torch.cuda.max_memory_allocated()>28*1024**3:raise MemoryError('memory cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>30*1024**3:raise OSError('evidence cap')
    try:
        card=load_json(ROOT/spec['data_card'])
        if not validate_card(card).passed or card!=load_json(run/'config/data_card.json'):raise ValueError('card drift')
        if Path(sys.executable)!=Path(PYTHON) or environment()!=spec['environment']:raise ValueError('environment drift')
        if (run/'config/command.txt').read_text()!=shlex.join(spec['command'])+'\n':raise ValueError('command drift')
        for p,h in card['scope']['input_sha256'].items():read_pinned(ROOT,p,h)
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(28*1024**3/torch.cuda.get_device_properties(0).total_memory)
        cases={c['case_id']:c for c in declared_cases()};examples=[];rows=[]
        for e in load_bound_partition_observations(ROOT,methods=METHODS):
            ref=expected_geometry(cases[e['observation_id']]);target=located_targets(ref,device='cuda')
            rows.append(dict(case_id=e['observation_id'],points=len(e['source_flat_ray_index']),groups={m:len(v['blocks'].block_ids) for m,v in e['methods'].items()},
                anchors=len(target.anchors_m),openings=len(target.openings_m),directions=int(target.direction_known.sum()),reference_sha256=digest(ref)))
            e.update(reference=ref,target=target);examples.append(e)
        if len(examples)!=45 or digest(rows)!=card['scope']['population']['rows_sha256']:raise ValueError('population drift')
        write(run/'config/population.json',rows)
        rng=np.random.default_rng(0);schedule=np.concatenate([rng.permutation(45) for _ in range(45)])[:2000].tolist()
        write(run/'config/schedule.json',schedule)
        torch.manual_seed(0);initial={k:v.clone() for k,v in model_pair().state_dict().items()};torch.save(initial,run/'artifacts/shared_initial.pt')
        if COORDINATE_VERSION==2:
            prior=card['scope']['policy']['coordinate_correction']['prior_run']
            pins=card['scope']['input_sha256']
            old_initial=torch.load(io.BytesIO(read_pinned(ROOT,prior+'/artifacts/shared_initial.pt',pins[prior+'/artifacts/shared_initial.pt'])),weights_only=True,map_location='cpu')
            if initial.keys()!=old_initial.keys() or any(not torch.equal(v,old_initial[k]) for k,v in initial.items()):raise ValueError('paired initial state drift')
            if schedule!=json.loads(read_pinned(ROOT,prior+'/config/schedule.json',pins[prior+'/config/schedule.json'])):raise ValueError('paired schedule drift')
            write(run/'config/paired_control.json',dict(initial_state_equal=True,schedule_equal=True,coordinate_version=2))
        write(run/'config/environment.json',dict(**spec['environment'],gpu=torch.cuda.get_device_name(0),torch=torch.__version__))
        for method in METHODS:
            model=model_pair();model.load_state_dict(initial)
            if sum(p.numel() for p in model.parameters())!=464011:raise ValueError('parameter count drift')
            if any(not torch.equal(v,initial[k]) for k,v in model.state_dict().items()):raise ValueError('initial state mismatch')
            model=model.cuda();torch.cuda.reset_peak_memory_stats()
            results[method]=dict(initial=evaluate(model,examples,method,run,method+'_initial'))
            optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
            model.train()
            for step in range(500):
                optimizer.zero_grad(set_to_none=True);value=0.;counts={}
                for index in schedule[4*step:4*step+4]:
                    e=examples[index];loss=located_structure_loss(forward(model,e,method),e['target'])
                    if not torch.isfinite(loss['total']):raise ValueError('nonfinite loss')
                    (loss['total']/4).backward();value+=float(loss['total'].detach())/4
                    for k,v in loss['counts'].items():counts[k]=counts.get(k,0)+v
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('gradient failure')
                guard();optimizer.step();steps[method]+=1
                if any(not torch.isfinite(p).all() for p in model.parameters()):raise ValueError('nonfinite state')
                record=dict(method=method,step=step+1,loss=value,counts=counts)
                with (run/'logs/training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if (step+1)%25==0:print(json.dumps(record),flush=True)
            results[method]['final']=evaluate(model,examples,method,run,method+'_final')
            results[method]['peak_allocated_bytes']=torch.cuda.max_memory_allocated()
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in model.state_dict().items()},optimizer=optimizer.state_dict()),run/('artifacts/'+method+'_final.pt'))
            guard();del model,optimizer;torch.cuda.empty_cache()
        for p,h in spec['source_sha256'].items():read_pinned(ROOT,p,h)
        for p,h in card['scope']['input_sha256'].items():read_pinned(ROOT,p,h)
        if environment()!=spec['environment']:raise ValueError('post-run environment drift')
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0)
        summary=dict(status='FAILED' if error else 'SYNTHETIC_BLOCK_FIT_COMPLETE',error=error,methods=results,optimizer_steps=steps,
            fit_pass={m:all(v.get('final',{}).get(k,{}).get('f1',0)>=.9 for k in ('anchor1','opening1')) for m,v in results.items()},
            scientific_gate_pass=False,encoder_inference=0,elapsed_s=time.monotonic()-start,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(run/'metrics/summary.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name))
        seal=run/'artifacts/evidence_sha256.txt';seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--coordinate-version',type=int,choices=(1,2),default=1);a=p.parse_args()
    configure_version(a.coordinate_version)
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('freeze or spec/run-dir required')
