"""One500-update conditional AI-core fit; all outputs preserved for grouping."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import sha,write
import argparse,hashlib,json,resource,signal,sys,time,traceback,zipfile
NAME='gse_branch_core_fit_v1';CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
REVIEW='results/gate3_semantics/gate3_20260913_gse_ai_branch_review_v1r1_seed0'
CACHE='results/gate3_semantics/gate3_20260912_gse_direction_task_model_cache_v1_seed0'
INITIAL='results/gate3_semantics/gate3_20260913_gse_direct_ray_relation_pilot_v1_seed0/artifacts/initial.pt'

def freeze():
    old=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_branch_review_v1r1.json').read_text())
    s=dict(entries=old['scope']['entries'],observations=12,parents=3,unique_frames=24,updates=500,effective_batch=4,micro_batch=1,
        same_pairs=98699,different_pairs=4202,unknown_pairs=1712750,variant='C',seed=0,pair_policy='dyadic_v2',
        full_branch_qualified=False,independent_evaluation=False,unknown_is_negative=False,
        supervision='Fixed nonblind AI observed surface-core proposals, only conditional fitting use; historical training_qualified flags untouched',
        sampling='All12 original overlapping five-frame windows,3 fit fragments; fixed repeated order0..11; no adjacent-frame independence claim',
        teacher='AI review.json and ray-bound core labels, no construction IDs; human_gold_count0; no objective full-opening truth',
        split='C01/C06 fit only; no C07-C10 or robot benchmark payloads',
        optimizer=dict(name='AdamW',lr=.001,weight_decay=.0001),
        loss='Separate mean positive and negative BCE, equal weighting; mean across4 observations; unknown excluded only from loss',
        inference='All valid rays and dyadic pairs retained; relation threshold.5; grouping p>=.9 attraction+1,p<=.1 repulsion-1,otherwise unknown; same conservative signed backend; no GT output filter',
        limits=dict(gpu_bytes=28*1024**3,host_bytes=32*1024**3,output_bytes=4*1024**3,wall_seconds=7200))
    approval=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',authorized_operations=['training'],authorized_gates=[3],
        scope='One conditional partial-reference C500 fit and unfiltered grouped evaluation, not formal full-branch or ABC experiment',
        confirmation_reference='Assistant explicitly proposed conditional AI-core bounded fit and full output grouping; user: 做啊',
        scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=s,approval=approval)
    from mtare_topo.governance_branch_core_fit import validate_card
    assert validate_card(card).passed
    pins={INITIAL:sha(ROOT/INITIAL)}
    for folder in (REVIEW,CACHE):
        seal=ROOT/folder/'artifacts/evidence_sha256.txt';pins[str(seal.relative_to(ROOT))]=sha(seal)
        for line in seal.read_text().splitlines():
            h,p=line.split('  ',1);assert sha(ROOT/p)==h;pins[p]=h
    for e in s['entries']:pins[e['student_path']]=sha(ROOT/e['student_path'])
    write(ROOT/CARD,card);pins[CARD]=sha(ROOT/CARD)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='training',data_card=CARD,user_authorization=approval,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',sys.executable,'tools/v3/run_branch_core_fit.py','--execute'],
        question='Can fixed partial AI core relations be fitted and survive unfiltered grouping?',
        method='C per-ray geometry relations; same initial state; frozen encoder cache; one500-update block',
        baseline='step0 and existing perfect partial-core backend outputs; not A/B ranking',
        fallback='No more epochs,seeds or threshold search; separate partial fit from grouping failure',
        estimated_cost=dict(compute='5090 C500 updates,2000 microforwards,initial/final evaluation,12 backward resource checks',host_ram_gb=32,gpu_vram_gb=28,disk_gb=4,wall_time_hours=2),
        acceptance_criteria=['All500 updates complete','Known class macroF1>=.90','All12 partial cores survive grouping without cross-core merges or core fragmentation/labeled singleton','Unknown outputs reported, not filtered; no generalization claim'],
        expected_evidence=['Initial/final full predictions,12case grouped outputs,checkpoints every100updates,optimizer/RNG,raw loss log,metrics,source snapshot,seal'],
        input_sha256=pins,execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    import numpy as np,torch
    from dataclasses import fields,asdict
    from mtare_topo.representation.branch_relation_learning import LocalBranchEvidence,RayBranchRelationModel,observed_ray_pairs,BranchRelationTargets,branch_relation_loss
    from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches
    from mtare_topo.representation.gse_structural_representation import bind_structural_patches
    from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    from mtare_topo.topology.branch_hypotheses import group_signed_relations
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());s=json.loads((ROOT/CARD).read_text())['scope']
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    for p,h in {**spec['input_sha256'],**spec['execution_source_sha256']}.items():assert sha(ROOT/p)==h,p
    write(out/'RUN_STATE.json',dict(state='RUNNING',stage='resource_check',updates=0),'w');start=time.monotonic();step=0;error=None;results={}
    def timeout(*_):raise TimeoutError('run wall cap')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(s['limits']['wall_seconds'])
    try:
        torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        torch.cuda.set_per_process_memory_fraction(s['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        model=RayBranchRelationModel('C').cuda();original=torch.load(ROOT/INITIAL,map_location='cpu',weights_only=False)['model']
        model.load_state_dict(original);initial_hash=module_state_sha256(model)
        write(out/'config/training_environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),initial_state_sha256=initial_hash))
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(ROOT/p,p)
        optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.0001)
        population=[]
        for i,e in enumerate(s['entries']):
            with np.load(ROOT/REVIEW/f'artifacts/ai_core_{i:02d}.npz',allow_pickle=False) as d:ids=d['ray_ids'].copy();labels=d['core_labels'].copy()
            with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as d:
                kwargs={f.name:d['patch_'+f.name].copy() for f in fields(SurfacePatches) if 'patch_'+f.name in d};kwargs.update(voxel_size_m=.5,roi_radius_m=10.)
                patch=SurfacePatches(**kwargs)
                def tensor(x):return torch.from_numpy(np.array(x,copy=True)).float().cuda()
                frames=tuple(e['frame_rows'])
                evidence=LocalBranchEvidence(torch.from_numpy(ids).long().cuda(),tensor(d['registered_returns_xyz_m'][4*11520+ids]),tensor(d['full_sensor_context'][4*180+(ids%720)//4]),tensor(d['patch_observed_features']),bind_structural_patches([patch],device='cuda'),CausalFrameOrderContext('sensor_current',frames,frames[-1],e['task']))
            pairs=observed_ray_pairs(evidence.ray_ids,policy='dyadic_v2');pn=pairs.cpu().numpy();a,b=labels[pn[:,0]],labels[pn[:,1]]
            known=(a>=0)&(b>=0);values=np.where(known,(a==b).astype(np.float32),np.nan)
            assert (values==0).any() and (values==1).any()
            # Scoped partial-diagnostic authorization does not rewrite the
            # historical training_qualified flag or qualify full branch labels.
            targets=BranchRelationTargets(tensor(values),torch.from_numpy(known).cuda(),tuple('AI_CORE_PARTIAL_DIAGNOSTIC' if v else '' for v in known),False)
            population.append((evidence,pairs,targets,labels,pn,ids))
        def cap():
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<=s['limits']['host_bytes']
            assert torch.cuda.max_memory_reserved()<=s['limits']['gpu_bytes']
            assert sum(p.stat().st_size for p in out.rglob('*') if p.is_file())<=s['limits']['output_bytes']
        def checkpoint(n):
            torch.save(dict(model=model.state_dict(),optimizer=optimizer.state_dict(),step=n,seed=0,torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all(),purpose='PARTIAL_AI_CORE_FIT_NOT_GENERALIZATION'),out/f'artifacts/step_{n:04d}.pt')
        # Full forward+backward on every shape, no optimizer update. Then clear
        # all gradients and verify the initialized weights remain unchanged.
        model.train()
        for evidence,pairs,targets,*_ in population:
            optimizer.zero_grad(set_to_none=True);pred=model(evidence,pairs);loss,_=branch_relation_loss(pred,targets);loss.backward();assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None);cap()
        optimizer.zero_grad(set_to_none=True);assert module_state_sha256(model)==initial_hash
        write(out/'metrics/resource_probe.json',dict(passed=True,shapes=12,optimizer_updates=0,peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),elapsed_s=time.monotonic()-start))
        def evaluate(n):
            model.eval();rows=[]
            with torch.no_grad():
                for i,(evidence,pairs,targets,labels,pn,ids) in enumerate(population):
                    pred=model(evidence,pairs);prob=pred.logits.sigmoid().cpu().numpy();assert np.isfinite(prob).all()
                    np.savez_compressed(out/f'artifacts/prediction_{n:04d}_{i:02d}.npz',ray_ids=ids,pairs=pn,logits=pred.logits.cpu().numpy(),probabilities=prob)
                    # All observed pairs participate. No label/GT mask here.
                    signed=np.where(prob>=.9,1,np.where(prob<=.1,-1,0))
                    hypotheses=group_signed_relations(ids.tolist(),[(int(ids[a]),int(ids[b]),int(v)) for (a,b),v in zip(pn,signed) if v])
                    write(out/f'artifacts/groups_{n:04d}_{i:02d}.json',asdict(hypotheses))
                    # Reference read only for scoring after raw output/grouping.
                    y=targets.values.cpu().numpy();known=targets.known.cpu().numpy();guess=prob>=.5
                    tp=int((known&(y==1)&guess).sum());tn=int((known&(y==0)&~guess).sum());fp=int((known&(y==0)&guess).sum());fn=int((known&(y==1)&~guess).sum())
                    lookup=dict(zip(ids.tolist(),labels.tolist()));per=[0,0];mixed=0;unknown_grouped=0
                    for group in hypotheses.groups:
                        kinds={lookup[r] for r in group if lookup[r]>=0};mixed+=int(len(kinds)>1)
                        for k in kinds:per[k]+=1
                        unknown_grouped+=sum(lookup[r]<0 for r in group)
                    row=dict(observation=i,tp=tp,tn=tn,fp=fp,fn=fn,groups=len(hypotheses.groups),groups_per_core=per,cross_core_merges=mixed,labeled_unresolved=sum(lookup[r]>=0 for r in hypotheses.unresolved_ray_ids),unknown_grouped_rays=unknown_grouped,unresolved_rays=len(hypotheses.unresolved_ray_ids),unknown_confident_pairs=int(((~known)&(signed!=0)).sum()),unknown_pairs=int((~known).sum()))
                    rows.append(row);print(json.dumps(dict(evaluation=n,**row)),flush=True)
            total={k:sum(r[k] for r in rows) for k in ('tp','tn','fp','fn')};tp,tn,fp,fn=[total[k] for k in ('tp','tn','fp','fn')]
            macro=.5*(2*tp/max(1,2*tp+fp+fn)+2*tn/max(1,2*tn+fp+fn))
            good=sum(r['groups_per_core']==[1,1] and not r['cross_core_merges'] and not r['labeled_unresolved'] for r in rows)
            report=dict(step=n,macro_f1=macro,positive_recall=tp/max(1,tp+fn),negative_recall=tn/max(1,tn+fp),complete_core_cases=good,observations=rows,**total)
            write(out/f'metrics/evaluation_{n:04d}.json',report);results[str(n)]=report;cap();return report
        checkpoint(0);evaluate(0);model.train()
        with (out/'logs/training.jsonl').open('x') as log:
            for step in range(1,501):
                optimizer.zero_grad(set_to_none=True);loss_sum=0.
                for micro in range(4):
                    i=((step-1)*4+micro)%12;evidence,pairs,targets,*_=population[i]
                    pred=model(evidence,pairs);loss,_=branch_relation_loss(pred,targets)
                    if not torch.isfinite(loss):raise ValueError('nonfinite loss')
                    (loss/4).backward();loss_sum+=float(loss.detach())/4
                if not all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):raise ValueError('nonfinite gradients')
                optimizer.step();log.write(json.dumps(dict(step=step,loss=loss_sum,elapsed_s=time.monotonic()-start))+'\n');log.flush()
                if step%25==0:
                    print(json.dumps(dict(step=step,loss=loss_sum,elapsed_s=time.monotonic()-start)),flush=True);cap();write(out/'RUN_STATE.json',dict(state='RUNNING',stage='training',updates=step),'w')
                if step%100==0:checkpoint(step)
        evaluate(500)
    except BaseException:error=traceback.format_exc();print(error,flush=True)
    finally:signal.alarm(0)
    final=results.get('500',{});passed=not error and final.get('macro_f1',0)>=.9 and final.get('complete_core_cases',0)==12
    summary=dict(status='GATE_FAIL' if error or not passed else 'GATE_MIXED',diagnostic_fit_pass=bool(passed),error=error,updates=step,partial_reference_only=True,full_branch_qualified=False,method_advantage_proven=False,elapsed_s=time.monotonic()-start,
        initial_macro_f1=results.get('0',{}).get('macro_f1'),final_macro_f1=final.get('macro_f1'),final_complete_core_cases=final.get('complete_core_cases'),peak_gpu_reserved_bytes=torch.cuda.max_memory_reserved(),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
    write(out/'metrics/summary.json',summary);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',updates=step),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary),flush=True);return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
