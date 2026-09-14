"""Fixed-budget A/B/C source organization fit; no structure/topology claim."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,copy,hashlib,json,resource,signal,time,traceback,zipfile
from ai_junction_pilot import sha,write
from export_new12_affinity_targets import FEATURE,RUN as TARGET,PYTHON
NAME='gse_new12_affinity_fit_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'


def freeze():
    target_card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new12_affinity_targets_v1.json').read_text())
    s=dict(population=target_card['scope'],feature_run=FEATURE,target_run=TARGET,variants=['A','B','C'],seed=0,
        updates_per_variant=2000,microbatch=1,accumulation=4,lr=.001,weight_decay=.0001,
        threshold=.5,evaluation_steps=[0,2000],encoder_frozen=True,fit_only=True,
        schedule='numpy default_rng(0) shuffled cycles of 12, 8000 observations shared by all groups',
        loss='equal mean positive/negative BCE per observation, four observations averaged per update; unknown masked',
        limitations=['12 selected fit parents, no independent generalization claim','C vs B changes messages and relations, not relation attributes alone','same parameter layout; actual computation differs','operand identity is not junction identity'],
        limits=dict(wall_seconds=43200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=30*1024**3))
    a=dict(status='APPROVED',approved_by='user-standing-new-route-authorization',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['training'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
        confirmation_reference='Active user goal explicitly authorizes fixed new12 same-input/init/budget A/B/C training; standing execution authorization',scope='One fixed seed0 A/B/C local operand-affinity fit; no graph/test/model search')
    write(ROOT/CARD,dict(schema_version='gse_new12_affinity_fit_card_v1',scope=s,approval=a))
    inputs={CARD:sha(ROOT/CARD)}
    for base in (FEATURE,TARGET):
        if json.loads((ROOT/base/'RUN_STATE.json').read_text())['state']!='COMPLETED':raise ValueError('input not complete')
        seal=ROOT/base/'artifacts/evidence_sha256.txt';pins={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
        inputs[str(seal.relative_to(ROOT))]=sha(seal)
        for i in range(12):
            p=f'{base}/artifacts/window_{i:02d}.npz'
            if sha(ROOT/p)!=pins[p]:raise ValueError('cache drift')
            inputs[p]=pins[p]
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='training',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/train_new12_affinity.py','--execute'],
        question='Can paired observation, unary geometry and relational models fit frozen local operand affinity?',
        method='Frozen encoder patch features, shared initial state, AdamW, 2000 updates per A/B/C',
        baseline='A observed+coordinates; B adds geometry; constant same-source and different-source references',
        fallback='Seal failure and latest completed state; no added updates, seed, unknown-as-negative or model search',
        acceptance_criteria=['All three fixed budgets and full initial/final outputs','Common initialization and schedule hashes','Both class recalls and unknown counts per parent','No claim of independent structure or graph success'],
        expected_evidence=['initial/final predictions and weights, losses, schedule, class metrics, source archive, runtime, seal'],
        estimated_cost=dict(compute='RTX5090D, 6000 optimizer updates total',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),input_sha256=inputs,source_sha256=sources))


def execute():
    import numpy as np,torch
    from mtare_topo.governance_new12_affinity_fit import validate_card
    from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches
    from mtare_topo.representation.gse_surface_relation_model_v1 import collate_surface_patches
    from mtare_topo.representation.gse_local_surface_affinity import LocalSurfaceAffinity,affinity_loss
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    from mtare_topo.evaluation.gse_affinity_metrics import affinity_metrics
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    if not validate_card(card).passed or json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh valid run required')
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();error=None;results={};cache=[];model=None;optimizer=None;step=0;variant=None
    def limits():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes']:raise MemoryError('host cap')
        if torch.cuda.max_memory_reserved()>s['limits']['gpu_bytes']:raise MemoryError('GPU cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output cap')
    def timeout(*_):raise TimeoutError('12h total cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(s['limits']['wall_seconds'])
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('input drift '+p)
        torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.backends.cuda.matmul.allow_tf32=False
        torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.use_deterministic_algorithms(True)
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),execution_context='controlled outside-sandbox CUDA',python=__import__('sys').version))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        for i in range(12):
            with np.load(ROOT/FEATURE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as f,np.load(ROOT/TARGET/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as t:
                patches=SurfacePatches(**{k:f['patch_'+k] for k in SurfacePatches.__dataclass_fields__ if k not in ('voxel_size_m','roi_radius_m')},voxel_size_m=.5,roi_radius_m=10.)
                p=collate_surface_patches([patches],device='cuda');x=torch.tensor(f['patch_observed_features'],device='cuda')
                if not np.array_equal(p.neighbor_index[0].cpu().numpy(),t['neighbor_index']):raise ValueError('target adjacency drift')
                cache.append((x,p,torch.tensor(t['values'][None],device='cuda'),torch.tensor(t['known'][None],device='cuda')))
        rng=np.random.default_rng(0);schedule=np.concatenate([rng.permutation(12) for _ in range(667)])[:8000].reshape(2000,4)
        np.save(run/'artifacts/schedule.npy',schedule)
        initial=LocalSurfaceAffinity('A');state=copy.deepcopy(initial.state_dict());initial_hash=module_state_sha256(initial)
        torch.save(state,run/'artifacts/initial.pt');del initial
        def evaluate(model,tag):
            rows=[];model.eval()
            with torch.no_grad():
                for i,(x,p,y,k) in enumerate(cache):
                    out=model(x,p);probs=out.logits.sigmoid()[0].cpu().numpy();yn=y[0].cpu().numpy();kn=k[0].cpu().numpy();valid=out.pair_valid[0].cpu().numpy()
                    row=dict(observation=i,model=affinity_metrics(probs,yn,kn,valid),constant_same=affinity_metrics(np.ones_like(probs),yn,kn,valid),constant_different=affinity_metrics(np.zeros_like(probs),yn,kn,valid))
                    rows.append(row)
                    np.savez_compressed(run/f'artifacts/{tag}_{i:02d}.npz',probabilities=probs,neighbor_index=out.neighbor_index[0].cpu().numpy(),valid=valid)
            write(run/f'metrics/{tag}.json',rows);model.train();return rows
        with (run/'logs/updates.jsonl').open('x') as log:
            for variant in s['variants']:
                model=LocalSurfaceAffinity(variant);model.load_state_dict(state)
                if module_state_sha256(model)!=initial_hash:raise ValueError('initialization mismatch')
                model.cuda();optimizer=torch.optim.AdamW(model.parameters(),lr=s['lr'],weight_decay=s['weight_decay'])
                initial_rows=evaluate(model,variant+'_step0');variant_start=time.monotonic()
                for step,batch in enumerate(schedule,1):
                    optimizer.zero_grad(set_to_none=True);total=0.
                    for i in batch:
                        x,p,y,k=cache[int(i)];out=model(x,p);loss,_=affinity_loss(out,y,k)
                        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
                        (loss/4).backward();total+=float(loss.detach())/4
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('nonfinite gradient')
                    optimizer.step()
                    row=dict(variant=variant,step=step,loss=total,elapsed_s=time.monotonic()-start)
                    log.write(json.dumps(row)+'\n')
                    if step==1 or step%100==0:
                        log.flush();print(json.dumps(row),flush=True);limits()
                torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=step,variant=variant,initial_state_sha256=initial_hash),run/f'artifacts/{variant}_final.pt')
                final_rows=evaluate(model,variant+'_step2000')
                results[variant]=dict(initial=initial_rows,final=final_rows,updates=step,elapsed_s=time.monotonic()-variant_start,parameters=sum(p.numel() for p in model.parameters()))
                write(run/'metrics/completed_variants.json',results,'w');del model,optimizer;model=None;optimizer=None
    except BaseException:
        error=traceback.format_exc()
        if model is not None:
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict() if optimizer else None,attempted_step=step,variant=variant),run/'artifacts/interrupted.pt')
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed_variants=list(results),results=results,fit_only=True,structure_or_graph_success=False,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}));return int(error is not None)


if __name__=='__main__':
    raise SystemExit('SUPERSEDED_BY_IMPLICIT_STRUCTURE_PLAN: source-only training is retained as a draft, not an authorized main-task run. See docs/PLAN.md; no freeze or execution performed.')
