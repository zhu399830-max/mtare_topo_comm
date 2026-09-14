"""Fixed same-init A/B/C conditional-geometry fit, not independent validation."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse, copy, hashlib, json, resource, signal, time, traceback, zipfile
from ai_junction_pilot import sha, write
from export_conditional_geometry_targets import FEATURE, RUN as TARGET, PYTHON
NAME='gse_new12_conditional_fit_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'
CHECK=f'configs/v3/gate3/{NAME}_resources.json'


def resource_check():
    import torch
    from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder,StructuralPatchInput
    from mtare_topo.representation.gse_surface_relation_model_v1 import SurfacePatchBatch
    from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
    rows=[];torch.set_num_threads(1)
    for variant in 'ABC':
        torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats();start=time.monotonic()
        x=torch.zeros(1,4096,128,device='cuda');ids=(torch.arange(4096,device='cuda')[:,None]+torch.arange(1,9,device='cuda'))%4096;ids=ids[None]
        p=SurfacePatchBatch(torch.zeros(1,4096,18,device='cuda'),torch.ones(1,4096,dtype=torch.bool,device='cuda'),ids,torch.ones_like(ids,dtype=torch.bool),torch.zeros(1,4096,8,9,device='cuda'))
        model=GeometryStructureEncoder(variant).cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        r=model(x,StructuralPatchInput(p,torch.tensor([[.5,10.]],device='cuda')),(CausalFrameOrderContext('sensor',(0,1,2,3,4),4,'synthetic'),)).relations
        (r.axis_abs_dot.mean()+r.height_difference_m.square().mean()).backward();opt.step();torch.cuda.synchronize()
        peak=torch.cuda.max_memory_reserved();assert peak<28*1024**3
        rows.append(dict(variant=variant,patches=4096,microbatch=1,peak_gpu_bytes=peak,elapsed_s=time.monotonic()-start))
        del model,opt,r,p,x,ids
    write(ROOT/CHECK,dict(scope='synthetic capacity only, zero dataset reads',torch=torch.__version__,gpu=torch.cuda.get_device_name(0),rows=rows))
    print(json.dumps(rows))


def freeze():
    population=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_new12_conditional_geometry_v1.json').read_text())['scope']
    check=json.loads((ROOT/CHECK).read_text());assert [r['variant'] for r in check['rows']]==list('ABC')
    s=dict(population=population,feature_run=FEATURE,target_run=TARGET,variants=list('ABC'),seed=0,updates=2000,microbatch=1,accumulation=4,
        lr=.001,weight_decay=.0001,encoder_frozen=True,fit_only=True,evaluation_steps=[0,2000],
        schedule='default_rng(0), shuffled cycles of12, first8000 indices shared',
        loss='per observation: unordered component pair means, same/cross category equal means, axis L1 + height L1/10m; four observations averaged',
        limits=dict(wall_seconds=43200,host_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=30*1024**3))
    a=dict(status='APPROVED',approved_by='user-confirmed-conditional-supervision-and-standing-execution',approved_at='2026-09-11',authorized_gates=[3],authorized_operations=['training'],
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),scope='Fixed12 conditional geometry A/B/C seed0 fit, same budget; no independent evaluation claim',
        confirmation_reference='User confirmed conditional construction supervision; approved implicit plan A/B/C2000 updates and standing execution authorization')
    pins={CHECK:sha(ROOT/CHECK)}
    for base in (FEATURE,TARGET):
        assert json.loads((ROOT/base/'RUN_STATE.json').read_text())['state']=='COMPLETED'
        seal=ROOT/base/'artifacts/evidence_sha256.txt';known={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())};pins[str(seal.relative_to(ROOT))]=sha(seal)
        for i in range(12):
            p=f'{base}/artifacts/window_{i:02d}.npz';assert sha(ROOT/p)==known[p];pins[p]=known[p]
    write(ROOT/CARD,dict(schema_version='gse_conditional_fit_card_v1',scope=s,approval=a));pins[CARD]=sha(ROOT/CARD)
    sources={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=3,date='20260911',slug=NAME,seed=0,operation='training',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONPATH=src',PYTHON,'tools/v3/train_conditional_geometry.py','--execute'],
        question='Can same-init A/B/C fit conditional geometry beyond same-axis zero-height shortcuts?',method=s['loss'],
        baseline='Same init A raw, B unary geometry, C neighbor messages/relations; constant axis1 height0, initial/final only',
        fallback='Seal any failure; no extra steps/seeds or checkpoint selection; no independent graph/structure claim',
        acceptance_criteria=['12 observations and fixed masks unchanged','Same8000 observations and2000 updates per variant','Finite losses/gradients and resource limits','Report same/cross errors, constant baseline, every parent; numerical completion is not research success'],
        expected_evidence=['initial/final predictions and weights, shared schedule, per-update log, same/cross per-parent errors, runtime, environment, code snapshot, seal'],
        estimated_cost=dict(compute='One5090, three fixed fit runs',host_ram_gb=32,gpu_vram_gb=28,disk_gb=30,wall_time_hours=12),input_sha256=pins,source_sha256=sources))


def execute(*, spec_path=SPEC, card_path=CARD, run_path=RUN, validator=None):
    import numpy as np
    import torch
    from mtare_topo.governance_conditional_geometry import validate_fit_card
    from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches
    from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder,bind_structural_patches
    from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
    from mtare_topo.representation.gse_conditional_geometry_loss import compile_conditional_loss_targets,conditional_geometry_loss
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    spec=json.loads((ROOT/spec_path).read_text());card=json.loads((ROOT/card_path).read_text());s=card['scope'];run=ROOT/run_path
    assert (validator or validate_fit_card)(card).passed and json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    start=time.monotonic();error=None;results={};cache=[];model=None;opt=None;step=0;variant=None
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def limits():
        if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes']:raise MemoryError('host cap')
        if torch.cuda.max_memory_reserved()>s['limits']['gpu_bytes']:raise MemoryError('GPU cap')
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise MemoryError('output cap')
    def expire(*_):raise TimeoutError('12h run cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(43200)
    try:
        for p,h in {**spec['input_sha256'],**spec['source_sha256']}.items():
            if sha(ROOT/p)!=h:raise ValueError('input/source drift '+p)
        torch.set_num_threads(1);torch.manual_seed(0);torch.cuda.manual_seed_all(0);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(run/'config/runtime_environment.json',dict(torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),python=__import__('sys').version))
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['source_sha256']:z.write(ROOT/p,p)
        for i,e in enumerate(s['population']['entries']):
            with np.load(ROOT/FEATURE/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as f,np.load(ROOT/TARGET/f'artifacts/window_{i:02d}.npz',allow_pickle=False) as y:
                patches=SurfacePatches(**{k:f['patch_'+k] for k in SurfacePatches.__dataclass_fields__ if k not in ('voxel_size_m','roi_radius_m')},voxel_size_m=.5,roi_radius_m=10.)
                p=bind_structural_patches([patches],device='cuda');x=torch.tensor(f['patch_observed_features'],device='cuda')
                assert np.array_equal(p.batch.neighbor_index[0].cpu().numpy(),y['neighbor_index'])
                assert np.array_equal(f['surface_return_indices'],y['original_roi_indices'])
                frames=tuple(e['frame_rows']);ctx=(CausalFrameOrderContext('sensor_current',frames,frames[-1],e['task']),)
                cache.append((x,p,ctx,compile_conditional_loss_targets(y,'cuda')))
        rng=np.random.default_rng(0);schedule=np.concatenate([rng.permutation(12) for _ in range(667)])[:8000].reshape(2000,4)
        if 'original_schedule' in s:
            saved=np.load(ROOT/s['original_schedule'],allow_pickle=False)
            if not np.array_equal(schedule,saved):raise ValueError('original schedule mismatch')
            schedule=saved
        np.save(run/'artifacts/schedule.npy',schedule)
        initial=GeometryStructureEncoder('A')
        if 'original_initial' in s:initial.load_state_dict(torch.load(ROOT/s['original_initial'],map_location='cpu',weights_only=True),strict=True)
        state=copy.deepcopy(initial.state_dict());initial_hash=module_state_sha256(initial);torch.save(state,run/'artifacts/initial.pt');del initial
        def evaluate(model,tag):
            model.eval();rows=[]
            with torch.no_grad():
                for i,(x,p,ctx,t) in enumerate(cache):
                    r=model(x,p,ctx).relations;metrics={}
                    for category,mask in [('all',t.known),('same',t.known&t.same),('cross',t.known&~t.same)]:
                        count=int(mask.sum());w=t.weights[mask];w=w/w.sum() if count else w
                        metrics[category]=dict(count=count,axis_mae=float((w*(r.axis_abs_dot[mask]-t.axis[mask]).abs()).sum()) if count else None,
                            height_mae_m=float((w*(r.height_difference_m[mask]-t.height[mask]).abs()).sum()) if count else None,
                            constant_axis_mae=float((w*(1-t.axis[mask]).abs()).sum()) if count else None,
                            constant_height_mae_m=float((w*t.height[mask].abs()).sum()) if count else None)
                    rows.append(dict(observation=i,parent=s['population']['entries'][i]['parent'],metrics=metrics))
                    np.savez_compressed(run/f'artifacts/{tag}_{i:02d}.npz',axis=r.axis_abs_dot.cpu().numpy(),height=r.height_difference_m.cpu().numpy(),valid=r.computation_valid.cpu().numpy(),neighbors=r.neighbor_index.cpu().numpy())
            write(run/f'metrics/{tag}.json',rows);model.train();return rows
        with (run/'logs/updates.jsonl').open('x') as log:
            for variant in s['variants']:
                model=GeometryStructureEncoder(variant);model.load_state_dict(state);assert module_state_sha256(model)==initial_hash
                model.cuda();opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001);first=evaluate(model,variant+'_step0');t0=time.monotonic()
                for step,batch in enumerate(schedule,1):
                    opt.zero_grad(set_to_none=True);total=0.
                    for i in batch:
                        x,p,ctx,t=cache[int(i)];loss,_=conditional_geometry_loss(model(x,p,ctx).relations,t,axis_objective=s.get('axis_objective','conditional_axis_l1_v1'))
                        if not bool(torch.isfinite(loss)):raise ValueError('nonfinite loss')
                        (loss/4).backward();total+=float(loss.detach())/4
                    if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in model.parameters()):raise ValueError('nonfinite gradient')
                    opt.step();row=dict(variant=variant,step=step,loss=total,elapsed_s=time.monotonic()-start);log.write(json.dumps(row)+'\n')
                    if step==1 or step%100==0:log.flush();print(json.dumps(row),flush=True);limits()
                torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict(),step=step,variant=variant,initial_sha256=initial_hash),run/f'artifacts/{variant}_final.pt')
                results[variant]=dict(initial=first,final=evaluate(model,variant+'_step2000'),updates=step,elapsed_s=time.monotonic()-t0,parameters=sum(p.numel() for p in model.parameters()))
                write(run/'metrics/completed_variants.json',results,'w');del model,opt;model=None;opt=None
    except BaseException:
        error=traceback.format_exc()
        if model is not None:torch.save(dict(model=model.state_dict(),optimizer=opt.state_dict() if opt else None,step=step,variant=variant),run/'artifacts/interrupted.pt')
    finally:signal.alarm(0)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed_variants=list(results),results=results,fit_only=True,structure_or_graph_success=False,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_bytes=torch.cuda.max_memory_reserved() if torch.cuda.is_initialized() else 0)
    write(run/'metrics/summary.json',summary);write(run/'logs/raw.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (run/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(run.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='results'}));return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    for action in ('resource-check','freeze','execute'):g.add_argument('--'+action,action='store_true')
    a=p.parse_args()
    if a.resource_check:resource_check()
    elif a.freeze:freeze()
    else:raise SystemExit(execute())
