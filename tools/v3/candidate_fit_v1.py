"""One immutable paired fit; synthetic interface evidence, not generalization."""
import _bootstrap
import argparse,json,resource,shlex,signal,sys,time,traceback,zipfile
from pathlib import Path
import numpy as np
import torch
from surface_features_v1 import PYTHON,environment,sha,write
from mtare_topo.governance import load_json,build_run_id
from mtare_topo.governance_candidate_fit import SCHEMA,SLUG,scope,validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_candidate_training_examples import load_training_examples
from mtare_topo.representation.gse_candidate_readout import SharedCandidateReadout
from mtare_topo.representation.gse_candidate_objective import balanced_candidate_loss
from mtare_topo.evaluation.gse_candidate_selection import score_candidate_selection

ROOT=Path(__file__).resolve().parents[2]
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0';ENTRY='tools/v3/candidate_fit_v1.py'
CORRECTIVE=False


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC)):raise FileExistsError('no refreeze')
    s=scope(ROOT,CORRECTIVE)
    a=dict(status='APPROVED',approved_by='user-standing-scope-authorization',approved_at='2026-09-08',scope='Exact45 sealed synthetic candidate ABC500 fit, no real-world or backbone training',scope_sha256=digest(s),authorized_operations=['training'],authorized_gates=[3],confirmation_reference='Standing autonomous user authority; PLAN candidate GPU resource check completed and paired fit next')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=s,scope_sha256=digest(s),approval=a)
    assert validate_card(card).passed
    files=sorted(str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py'))
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260908',slug=SLUG,seed=0,operation='training',data_card=CARD,config_path=CARD,user_authorization=a,
        command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,ENTRY,'--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)],
        question='Can shared observation-derived candidates be selected without arbitrary free-query positions?',
        method='Frozen cached encoder, shared355457-parameter ABC initialization,500 updates micro1 accumulation4, AdamW .001 wd.0001',
        baseline='A shared geometric candidates plus observation features; B unary attributes; C sparse relations. Fit only.',
        fallback='Fail and seal drift/numeric/resource errors; preserve negative fit result, no retry/tuning',
        estimated_cost=dict(compute='5090D,1500 optimizer updates total,zero encoder inference',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),
        acceptance_criteria=[s['acceptance'],'All45 and full background scored before/after fixed suppression','No checkpoint selection or independent-world claim'],
        expected_evidence=['Shared schedule,input hashes,source snapshot,environment,raw initial/final predictions,checkpoints,optimizer states,loss log,per-case scores,summary,seal'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    if CORRECTIVE:
        spec['command'].insert(spec['command'].index('--spec'),'--set-loss')
        spec['method']+='; unique corrective one-to-one hard-negative objective; raw selection primary, no output spatial suppression'
    for p,v in ((CARD,card),(SPEC,spec)):
        with (ROOT/p).open('x') as f:json.dump(v,f,indent=2)
    print(json.dumps(dict(spec=SPEC,scope_sha256=digest(s))))


@torch.no_grad()
def evaluate(model,examples,run,label):
    model.eval();rows=[];total={stage:{t:dict(matched=0,missed=0,false_positives=0) for t in ('1.0','2.0','4.0')} for stage in ('raw','after_suppression')}
    probabilities=[]
    for e in examples:
        args,_,_=e.on_device('cuda');p=model(*args).presence_logits[0].sigmoid().cpu().numpy()
        score=score_candidate_selection(e.positions_m.numpy(),p,e.expected_positions_m,complete_region=True)
        rows.append(dict(observation_id=e.observation_id,score=score));probabilities.append(dict(observation_id=e.observation_id,positions_m=e.positions_m,probabilities=torch.from_numpy(p)))
        for stage in total:
            for t in total[stage]:
                for k in total[stage][t]:total[stage][t][k]+=score[stage]['scores'][t][k]
    for stage in total:
        for counts in total[stage].values():
            tp=counts['matched'];den=2*tp+counts['missed']+counts['false_positives'];counts['f1']=2*tp/den if den else None
    write(run/('metrics/'+label+'.json'),dict(totals=total,per_case=rows));torch.save(probabilities,run/('artifacts/'+label+'_predictions.pt'))
    return total


def execute(spec,run):
    run=run.resolve(strict=True)
    if run!=ROOT/'results/gate3_semantics'/build_run_id(spec) or load_json(run/'RUN_STATE.json')['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh exact run required')
    if load_json(run/'config/run_spec.json')!=spec:raise ValueError('spec drift')
    start=time.monotonic();error=None;results={};steps={k:0 for k in 'ABC'}
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    signal.signal(signal.SIGALRM,lambda *_: (_ for _ in ()).throw(TimeoutError('wall cap')));signal.alarm(43200)
    def guard():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>32*1024**3 or torch.cuda.max_memory_allocated()>28*1024**3:raise MemoryError('memory cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>30*1024**3:raise RuntimeError('disk cap')
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
        examples=load_training_examples(ROOT)
        if sum(len(e.positions_m) for e in examples)!=32469:raise ValueError('candidate count drift')
        rng=np.random.default_rng(0);schedule=np.concatenate([rng.permutation(45) for _ in range(45)])[:2000].tolist()
        write(run/'config/schedule.json',schedule)
        torch.manual_seed(0);initial=SharedCandidateReadout('A').state_dict();torch.save(initial,run/'artifacts/shared_initial.pt')
        for path in 'ABC':
            m=SharedCandidateReadout(path);m.load_state_dict(initial);m=m.cuda();torch.cuda.reset_peak_memory_stats()
            results[path]=dict(initial=evaluate(m,examples,run,path+'_initial'))
            optimizer=torch.optim.AdamW(m.parameters(),lr=.001,weight_decay=.0001)
            m.train()
            for step in range(500):
                optimizer.zero_grad(set_to_none=True);loss_sum=0.
                for index in schedule[4*step:4*step+4]:
                    args,y,known=examples[index].on_device('cuda');prediction=m(*args)
                    if card['scope'].get('corrective_version'):
                        from mtare_topo.representation.gse_candidate_set_objective import candidate_set_loss
                        loss,counts=candidate_set_loss(prediction.presence_logits,examples[index].positions_m.numpy(),examples[index].expected_positions_m)
                    else:
                        loss,counts=balanced_candidate_loss(prediction.presence_logits,y,known)
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    (loss/4).backward();loss_sum+=float(loss.detach())/4
                if any(p.grad is None or not torch.isfinite(p.grad).all() for p in m.parameters()):raise ValueError('gradient failure')
                guard();optimizer.step();steps[path]+=1
                if any(not torch.isfinite(p).all() for p in m.parameters()):raise ValueError('parameter failure')
                record=dict(method=path,step=step+1,loss=loss_sum)
                with (run/'logs/training.jsonl').open('a') as f:f.write(json.dumps(record)+'\n')
                if (step+1)%50==0:print(json.dumps(record),flush=True)
            results[path]['final']=evaluate(m,examples,run,path+'_final')
            torch.save(dict(state_dict={k:v.detach().cpu() for k,v in m.state_dict().items()},optimizer=optimizer.state_dict()),run/('artifacts/'+path+'_final.pt'))
            results[path]['peak_allocated_bytes']=torch.cuda.max_memory_allocated();guard()
            del m,optimizer;torch.cuda.empty_cache()
        for p,h in spec['source_sha256'].items():read_pinned(ROOT,p,h)
        for p,h in card['scope']['input_sha256'].items():read_pinned(ROOT,p,h)
    except Exception:error=traceback.format_exc();(run/'logs/error.log').write_text(error)
    finally:
        signal.alarm(0)
        summary=dict(status='FAILED' if error else 'SYNTHETIC_CANDIDATE_FIT_COMPLETE',error=error,methods=results,optimizer_steps=steps,
            fit_pass={k:v.get('final',{}).get('raw' if CORRECTIVE else 'after_suppression',{}).get('1.0',{}).get('f1',0)>=.9 for k,v in results.items()},
            scientific_gate_pass=False,encoder_inference=0,elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(run/'metrics/summary.json',summary);write(run/'config/environment.json',spec['environment'])
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name))
        seal=run/'artifacts/evidence_sha256.txt';seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);p.add_argument('--set-loss',action='store_true');a=p.parse_args()
    if a.set_loss:
        CORRECTIVE=True;SLUG='gse_candidate_set_fit_v1'
        CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
        RUN='results/gate3_semantics/gate3_20260908_'+SLUG+'_seed0'
    if a.freeze:freeze()
    elif a.spec and a.run_dir:raise SystemExit(execute(load_json(a.spec),a.run_dir))
    else:p.error('freeze or spec/run-dir required')
