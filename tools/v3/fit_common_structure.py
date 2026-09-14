"""One fixed500-update partial-reference fit of the common structural readout."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from check_local_pair_window_coverage import sha,write
from membership_fit_v1 import PYTHON

SLUG='gse_common_structure_fit_v1'
CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260911_'+SLUG+'_seed0'
TASK='configs/v3/gate3/gse_common_partial_structure_task_v1.json'


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    task=json.loads((ROOT/TASK).read_text());files={TASK:sha(ROOT/TASK)}
    for e in task['entries']:
        for k in ('feature','reference'):files[e[k+'_path']]=e[k+'_sha256']
    s=dict(task=task,updates=500,accumulation=4,seed=0,variant='C',learning_rate=.001,weight_decay=.0001,partial_reference_only=True,
        limits=dict(wall_seconds=7200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=4*1024**3))
    a=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['training'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),scope='One C common-readout500-update fit, same16 original partial references; not full detection or independent validation',
        confirmation_reference='Active goal authorizes bounded new-interface fitting after data/input/output contract tests; no old head restored, no new labels or worlds')
    write(ROOT/CARD,dict(schema_version='gse_common_structure_fit_card_v1',scope=s,approval=a))
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')};sources[CARD]=sha(ROOT/CARD)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=SLUG,seed=0,operation='training',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/fit_common_structure.py','--execute'],
        question='Can the common ray/surface/context readout fit existing partial structures and known relations?',
        method='C readout from random seed0;500 AdamW updates,batch4; no encoder forward or presence loss',
        baseline='Own fixed initialization; not a historical fair comparison',fallback='Seal failure, no extra updates/seeds or automatic correction',
        estimated_cost=dict(compute='singleGPU500 updates with cached encoder features',host_ram_gb=32,gpu_vram_gb=28,disk_gb=4,wall_time_hours=2),
        acceptance_criteria=['Final structure and section1m reference recalls >=0.9','Known positive and negative relation recovery each >=0.9; unlocalized not correct','Initial/final all predictions and unknowns saved; no checkpoint selection','Only partial fitting evidence, never full detection qualification'],
        expected_evidence=['initial/final checkpoints,all predictions,500-step schedule,losses,counts,source snapshot,environment,seal'],input_sha256=files,source_sha256=sources))


def execute():
    import torch
    import numpy as np
    from mtare_topo.governance_membership_fit import validate_common_fit_card
    from mtare_topo.data.gse_common_observation_reader import load_common_observation
    from mtare_topo.representation.gse_partial_structure_contract import bind_partial_targets
    from mtare_topo.representation.gse_common_structure_readout import CommonStructureReadout
    from mtare_topo.representation.gse_surface_relation_model_v1 import collate_surface_patches
    from mtare_topo.representation.gse_structure_prediction_contract import partial_structure_loss,evaluate_partial_structure
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_common_fit_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;initial=None;final=None;updates=0;gpu_peak=0
    def expire(*args):raise TimeoutError('7200second cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(7200)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('source drift '+p)
        torch.manual_seed(0);torch.cuda.manual_seed_all(0)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(torch=torch.__version__,gpu=torch.cuda.get_device_name(0),python=sys.version))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        model=CommonStructureReadout('C').cuda();torch.save(model.state_dict(),run/'artifacts/initial.pt')
        examples=[]
        for e in s['task']['entries']:
            obs=load_common_observation(ROOT,e['feature_path'],e['feature_sha256'],device='cuda')
            target=bind_partial_targets(json.loads((ROOT/e['reference_path']).read_text())['record'])
            patch=collate_surface_patches([obs.surface_patches],device='cuda')
            examples.append((obs,target,patch))
        def snapshot():
            model.eval();rows=[]
            with torch.no_grad():
                for obs,target,patch in examples:
                    r=model(obs,patch_batch=patch)
                    if r is None:raise ValueError('no observation readout for supervised sample')
                    row=evaluate_partial_structure(r.prediction,target)
                    row['predictions']={k:v.detach().cpu().tolist() for k,v in vars(r.prediction).items() if torch.is_tensor(v)}
                    row['structure_query_ray_indices']=r.structure_query_ray_indices.tolist();row['section_query_ray_indices']=r.section_query_ray_indices.tolist()
                    rows.append(row)
            return rows
        initial=snapshot();write(run/'artifacts/initial_predictions.json',initial)
        generator=torch.Generator().manual_seed(0);schedule=[]
        while len(schedule)<500:
            order=torch.randperm(16,generator=generator).tolist();schedule.extend([order[i:i+4] for i in range(0,16,4)])
        write(run/'artifacts/schedule.json',schedule)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        with (run/'logs/updates.jsonl').open('x') as log:
            for step,batch in enumerate(schedule,1):
                model.train();optimizer.zero_grad(set_to_none=True);losses=[]
                for i in batch:
                    obs,target,patch=examples[i];r=model(obs,patch_batch=patch)
                    if r is None:raise ValueError('missing supervised readout')
                    loss=partial_structure_loss(r.prediction,target);(loss['total']/4).backward()
                    losses.append({k:float(v.detach()) for k,v in loss['terms'].items()})
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):raise ValueError('nonfinite gradient')
                optimizer.step();updates=step
                log.write(json.dumps(dict(step=step,sample_indices=batch,loss_terms=losses,elapsed_s=time.monotonic()-start))+'\n');log.flush()
                gpu_peak=torch.cuda.max_memory_reserved()
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes'] or gpu_peak>s['limits']['gpu_bytes']:raise MemoryError('memory budget')
                if step%50==0:print(json.dumps(dict(step=step,elapsed_s=time.monotonic()-start)),flush=True)
        torch.save(model.state_dict(),run/'artifacts/final.pt');final=snapshot();write(run/'artifacts/final_predictions.json',final)
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output budget')
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    def totals(rows):
        if rows is None:return None
        return dict(structures=sum(r['structure']['correct'] for r in rows),sections=sum(r['section']['correct'] for r in rows),
            positive=sum(r['relations']['positive_correct'] for r in rows),negative=sum(r['relations']['negative_correct'] for r in rows),
            unlocalized_known=sum(r['relations']['unlocalized_known'] for r in rows),
            unconfirmed_structures=sum(r['structure']['unconfirmed_predictions'] for r in rows),unconfirmed_sections=sum(r['section']['unconfirmed_predictions'] for r in rows))
    t=totals(final);passed=error is None and t['structures']/24>=.9 and t['sections']/24>=.9 and t['positive']/22>=.9 and t['negative']/13>=.9
    summary=dict(status='GATE_PASS' if passed else 'GATE_FAIL',scope='PARTIAL_REFERENCE_TRAINING_FIT_ONLY_NOT_DETECTION_BASELINE',error=error,updates=updates,
        initial=totals(initial),final=t,elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_reserved_bytes=gpu_peak,
        full_detection_qualified=False,independent_advantage_validated=False)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('choose action')
