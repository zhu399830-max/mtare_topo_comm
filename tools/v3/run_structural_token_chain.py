"""Frozen A/B/C structure models on one causal fragment, without merge claims."""
from _bootstrap import PROJECT_ROOT as ROOT
from pathlib import Path
from types import SimpleNamespace
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
from run_short_observation_chain import sha,write
CODE=Path(__file__).resolve().parents[2]
NAME='gse_structural_token_chain_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260912_{NAME}_seed0'
SOURCE='results/gate3_semantics/gate3_20260912_gse_short_observation_chain_v1_seed0'
EVAL='configs/v3/gate3/data_cards/gse_conditional_development_evaluation_v1.json'
META='configs/v3/gate3/gse_conditional_development_manifest_v1.json'

def freeze():
    models=json.loads((ROOT/EVAL).read_text())['scope']['models']
    encoder=json.loads((ROOT/META).read_text())['encoder_checkpoint']
    sourcecard='configs/v3/gate3/data_cards/gse_short_observation_chain_v1.json'
    s=dict(source_run=SOURCE,source_scope=json.loads((ROOT/sourcecard).read_text())['scope'],
        models=models,encoder=encoder,variants=list('ABC'),training_steps=0,observations=8,
        candidate_policy='Identical sealed range_sectors from all8windows; no geometric fit frontend rerun',
        binding='Original candidate ray to .5m/10m observed patch; mean local token weighted by unique supporting returns',
        retrieval='All cosine scores to preceding window; unknown retained; ties kept; no probability, threshold or merge',
        limitations=['Existing models supervised on conditional axes/heights, not task identity','Engineering integration, not trained opening predictor',
            'Same small generated straight fragment, no heldout benefit or real exploration','No teacher reference or correspondence accuracy score'],
        limits=dict(host_bytes=4*1024**3,gpu_bytes=28*1024**3,wall_seconds=900,output_bytes=1024**3))
    pins={EVAL:sha(ROOT/EVAL),META:sha(ROOT/META),sourcecard:sha(ROOT/sourcecard)}
    seal=ROOT/SOURCE/'artifacts/evidence_sha256.txt';pins[str(seal.relative_to(ROOT))]=sha(seal)
    wanted={SOURCE+'/artifacts/inputs.npz',SOURCE+'/artifacts/graph.json',*[SOURCE+f'/artifacts/window_{i:02d}.json' for i in range(8)]}
    known={p:h for h,p in (l.split('  ',1) for l in seal.read_text().splitlines())}
    for p in wanted:
        if sha(ROOT/p)!=known[p]:raise ValueError('source drift')
        pins[p]=known[p]
    for r in [encoder,*[models[m] for m in 'ABC']]:
        if sha(ROOT/r['path'])!=r['sha256']:raise ValueError('model drift')
        pins[r['path']]=r['sha256']
    a=dict(status='APPROVED',approved_by='user-explicit-new-method-execution',approved_at='2026-09-12',
        authorized_operations=['data_export'],authorized_gates=[3],
        scope='Frozen ABC token-to-task integration on the same12frames/8windows; zero training or new labels',
        confirmation_reference='User 要跑啊新的方法,所以现在还是没验证出来吗; standing development implementation authorization',
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=s,approval=a)
    from mtare_topo.governance_token_chain import validate_card
    assert validate_card(card).passed
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(CODE)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (CODE/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260912',slug=NAME,seed=0,operation='data_export',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8','PYTHONDONTWRITEBYTECODE=1',sys.executable,str(Path(__file__)),'--execute'],
        question='Can actual learned ABC local tokens reach task retrieval on unchanged causal observations, and where is support absent?',
        method='Shared frozen encoder, ABC final2000 models, source-ray bound task pooling, identical all-pair retrieval interface',
        baseline='A observed features versus B unary geometry versus C relation messages; final weights only',
        fallback='Seal failure, no retuning, labels or unverified merges',
        estimated_cost=dict(compute='8 common encoder plus24ABC forwards on5090; no optimizer',host_ram_gb=4,gpu_vram_gb=28,disk_gb=1,wall_time_hours=.25),
        acceptance_criteria=['Same32proposals for each variant, unknown retained','8source windows and unchanged model states','Raw tokens and all relation outputs retained','No correspondence probability/accuracy or graph advantage asserted'],
        expected_evidence=['Common features, ABC tokens, task descriptors, all retrieval scores, task-attributed metric graph, per-window logs, environment, source snapshot and seal'],
        input_sha256=pins,execution_code_root=str(CODE),execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    import numpy as np,torch
    from mtare_topo.representation.gse_surface_encoder_checkpoint_v1 import load_surface_encoder
    from mtare_topo.representation.gse_dual_path_encoder_adapter_v1 import FrozenDualPathEncoderAdapterV1
    from mtare_topo.representation.gse_common_observation import prepare_common_observation
    from mtare_topo.representation.gse_patch_observed_features import pool_patch_observed_features
    from mtare_topo.representation.gse_structural_representation import GeometryStructureEncoder,bind_structural_patches
    from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    from mtare_topo.topology.structural_task_tokens import bind_task_tokens,retrieval_candidates
    from mtare_topo.governance_token_chain import validate_card
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text());s=card['scope']
    assert validate_card(card).passed and json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    for p,h in spec['input_sha256'].items():
        if sha(ROOT/p)!=h:raise ValueError('input drift '+p)
    for p,h in spec['execution_source_sha256'].items():
        if sha(CODE/p)!=h:raise ValueError('source drift '+p)
    start=time.monotonic();error=None;counts={m:dict(candidates=0,supported=0,unknown=0,retrieval_rows=0) for m in 'ABC'};completed=0
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    def expire(*_):raise TimeoutError('wall cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(900)
    try:
        torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        write(out/'config/runtime_environment.json',dict(python=sys.version,numpy=np.__version__,torch=torch.__version__,gpu=torch.cuda.get_device_name(0)))
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(CODE/p,p)
        ck=s['encoder'];bound=load_surface_encoder((ROOT/ck['path']).read_bytes(),expected_sha256=ck['sha256'],expected_epoch=ck['epoch'])
        adapter=FrozenDualPathEncoderAdapterV1(bound.backbone,torch.nn.Identity()).cuda().eval()
        models={};initial={}
        for m in 'ABC':
            saved=torch.load(ROOT/s['models'][m]['path'],map_location='cpu',weights_only=True)
            assert saved['variant']==m and saved['step']==2000
            model=GeometryStructureEncoder(m);model.load_state_dict(saved['model'],strict=True);model.cuda().eval().requires_grad_(False)
            models[m]=model;initial[m]=module_state_sha256(model)
        data=np.load(ROOT/SOURCE/'artifacts/inputs.npz',allow_pickle=False);poses=data['poses'];previous={m:[] for m in 'ABC'};history={m:[] for m in 'ABC'}
        with torch.inference_mode(),(out/'logs/windows.jsonl').open('x') as log:
            for j in range(8):
                i=j+4;record=json.loads((ROOT/SOURCE/f'artifacts/window_{j:02d}.json').read_text());keys=record['geometry']['source_frame_keys']
                trans=((poses[j:i+1,:3,3]-poses[i,:3,3])@poses[i,:3,:3]).astype(np.float32)
                rotations=np.array([poses[i,:3,:3].T@p[:3,:3] for p in poses[j:i+1]])
                yaw=np.rad2deg(np.arctan2(rotations[:,1,0],rotations[:,0,0])).astype(np.float32);trans[-1]=0;yaw[-1]=0
                student=SimpleNamespace(ranges_m=data['ranges_m'][j:i+1],valid_mask=data['valid_mask'][j:i+1],relative_translation_current_sensor_m=trans,relative_yaw_current_sensor_deg=yaw)
                obs=prepare_common_observation(adapter,student,device='cuda');x=pool_patch_observed_features(obs);p=bind_structural_patches([obs.surface_patches],device='cuda')
                frames=tuple(map(int,data['frame_rows'][j:i+1]));ctx=(CausalFrameOrderContext('sensor_current',frames,frames[-1],s['source_scope']['identity']['task']),)
                proposals=record['method_proposals']['range_sectors'];np.savez_compressed(out/f'artifacts/common_{j:02d}.npz',patch_observed_features=x.cpu().numpy(),patch_centers=obs.surface_patches.centers_m,point_patch_index=obs.surface_patches.point_patch_index)
                row=dict(window=j,patches=len(obs.surface_patches.centers_m),methods={})
                for m,model in models.items():
                    r=model(x,p,ctx);tokens=r.local_tokens[0].cpu().numpy()
                    tasks=bind_task_tokens(proposals,keys,obs.surface_patches.point_patch_index,tokens)
                    retrieval=retrieval_candidates(previous[m],tasks);previous[m]=tasks
                    item=dict(order=frames[-1],metric_anchor=record['decision']['node'],tasks=tasks,retrieval=retrieval)
                    history[m].append(item);write(out/f'artifacts/{m}_tasks_{j:02d}.json',item)
                    arrays=dict(local_tokens=tokens,region_embedding=r.region_embedding.cpu().numpy())
                    for k,v in vars(r.relations).items():
                        if torch.is_tensor(v):arrays[k]=v.cpu().numpy()
                    if any(not np.isfinite(v).all() for v in arrays.values()):raise ValueError('nonfinite output')
                    np.savez_compressed(out/f'artifacts/{m}_tokens_{j:02d}.npz',**arrays)
                    n=sum(t['descriptor'] is not None for t in tasks);c=counts[m];c['candidates']+=len(tasks);c['supported']+=n;c['unknown']+=len(tasks)-n;c['retrieval_rows']+=sum(bool(t['scores']) for t in retrieval)
                    row['methods'][m]=dict(supported=n,unknown=len(tasks)-n)
                completed+=1;log.write(json.dumps(row)+'\n');log.flush();print(json.dumps(row),flush=True)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>s['limits']['host_bytes'] or torch.cuda.max_memory_reserved()>s['limits']['gpu_bytes']:raise MemoryError('memory cap')
                if sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>s['limits']['output_bytes']:raise RuntimeError('output cap')
        assert module_state_sha256(bound.backbone)==bound.state_sha256
        assert all(module_state_sha256(models[m])==initial[m] for m in models)
        graph=json.loads((ROOT/SOURCE/'artifacts/graph.json').read_text())
        for m in 'ABC':write(out/f'artifacts/{m}_task_graph.json',dict(metric_graph=graph,task_observation_history=history[m],new_verified_merges=0,closed_loop=False,task_correspondence_quality_measured=False))
    except BaseException:error=traceback.format_exc()
    finally:signal.alarm(0)
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,completed_windows=completed,counts=counts,training_steps=0,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),
        learned_models_executed=True,graph_advantage_proven=False,task_correspondence_accuracy_measured=False)
    write(out/'metrics/summary.json',result);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result));return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
