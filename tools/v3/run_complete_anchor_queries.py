"""Frozen checkpoint: same anchors, complete direct queries, no optimizer."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import write,sha
from run_nonexclusive_support import FIT,CACHE
import json,hashlib,sys,argparse,time,resource,traceback,zipfile
NAME='gse_complete_anchor_queries_v1';CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
OLD='results/gate3_semantics/gate3_20260913_gse_nonexclusive_support_v1_seed0'
CHECKPOINT=f'{FIT}/artifacts/step_0500.pt'

def freeze():
    entries=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_branch_core_fit_v1.json').read_text())['scope']['entries']
    s=dict(entries=entries,observations=12,parents=3,raw_frames=24,rays=135699,complete_pairs=4336032,maximum_anchors=32,training_steps=0,thresholds=[.1,.9],teacher_in_inference=False,
        split='Original C01/C06 fit only,3fragments,4overlapping5frame windows each; not independent testing',sampling='Exactly same saved32anchors each case, all valid rays, no score/GT selection',teacher='No labels read, no new task qualification',
        method='Frozen C final500; reproduce original full pair forward, then use unchanged ray tokens and relation head in65536-pair chunks for all anchor-ray pairs',
        numerical_contract='Original logits max abs difference<=1e-5 and .1/.9 masks identical; common dense probabilities<=1e-6 and masks identical; mismatch fails, no overwrite',
        limits=dict(gpu_bytes=28*1024**3,host_bytes=8*1024**3,output_bytes=512*1024**2,wall_seconds=600))
    a=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',authorized_operations=['audit'],authorized_gates=[3],confirmation_reference='User 继续 following explicit need to complete anchor-observation queries under frozen model; no retraining',scope='One original12 full anchor-query frozen inference diagnostic',scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version=NAME,scope=s,approval=a)
    from mtare_topo.governance_complete_anchor import validate_card
    assert validate_card(card).passed
    write(ROOT/CARD,card);pins={CARD:sha(ROOT/CARD),CHECKPOINT:sha(ROOT/CHECKPOINT)}
    for i in range(12):
        for p in [f'{FIT}/artifacts/prediction_0500_{i:02d}.npz',f'{CACHE}/artifacts/common_{i:02d}.npz',f'{OLD}/artifacts/support_{i:02d}.json']:pins[p]=sha(ROOT/p)
    code={str(p.relative_to(ROOT)):sha(p) for folder in ['src/mtare_topo','tools/v3'] for p in (ROOT/folder).rglob('*.py')}
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='audit',data_card=CARD,user_authorization=a,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','CUBLAS_WORKSPACE_CONFIG=:4096:8',sys.executable,'tools/v3/run_complete_anchor_queries.py','--execute'],question='Does missing direct-query coverage explain sparse support, while unchanged scorer remains semantically unqualified?',method=s['method'],baseline='Same32anchors and old sparse predictions',fallback='Numerical/input/resource mismatch stops; no training, threshold or anchor change',estimated_cost=dict(compute='5090 frozen12 token forwards plus complete relation queries',host_ram_gb=8,gpu_vram_gb=28,disk_gb=.5,wall_time_hours=1/6),acceptance_criteria=['Exact same anchors and rays','Common-query numerical checks','All4336032 unordered queries saved','Weight state unchanged, zero optimizer','Report unsupported/shared/all-positive ambiguity; no branch success from coverage alone'],expected_evidence=['Full probabilities/anchors/rays','Numerical equivalence and output counts','Environment,source snapshot,logs,seal'],input_sha256=pins,execution_source_sha256=code)
    write(ROOT/SPEC,spec);print(SPEC)

def execute():
    import numpy as np,torch
    from dataclasses import fields
    from mtare_topo.representation.branch_relation_learning import RayBranchRelationModel,LocalBranchEvidence
    from mtare_topo.representation.gse_surface_patches_v1 import SurfacePatches
    from mtare_topo.representation.gse_structural_representation import bind_structural_patches
    from mtare_topo.representation.gse_structure_context import CausalFrameOrderContext
    from mtare_topo.representation.gse_surface_training_v1 import module_state_sha256
    from mtare_topo.representation.anchor_relation_queries import complete_anchor_pairs
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());s=json.loads((ROOT/CARD).read_text())['scope']
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    for p,h in {**spec['input_sha256'],**spec['execution_source_sha256']}.items():assert sha(ROOT/p)==h,p
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w');start=time.monotonic();rows=[];error=None
    try:
        torch.set_num_threads(1);torch.manual_seed(0);torch.use_deterministic_algorithms(True);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
        model=RayBranchRelationModel('C').cuda().eval();model.load_state_dict(torch.load(ROOT/CHECKPOINT,map_location='cpu',weights_only=False)['model']);before=module_state_sha256(model)
        write(out/'config/inference_environment.json',dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,gpu=torch.cuda.get_device_name(0),initial_model_sha256=before))
        with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p in spec['execution_source_sha256']:z.write(ROOT/p,p)
        for i,e in enumerate(s['entries']):
            with np.load(ROOT/FIT/f'artifacts/prediction_0500_{i:02d}.npz',allow_pickle=False) as d:ids=d['ray_ids'];oldpairs=d['pairs'];oldlogits=d['logits'];oldprob=d['probabilities']
            anchors=json.loads((ROOT/OLD/f'artifacts/support_{i:02d}.json').read_text())['anchors'];pairs=complete_anchor_pairs(ids,anchors)
            assert len(pairs)==32*len(ids)-528
            with np.load(ROOT/CACHE/f'artifacts/common_{i:02d}.npz',allow_pickle=False) as d:
                kwargs={f.name:d['patch_'+f.name].copy() for f in fields(SurfacePatches) if 'patch_'+f.name in d};kwargs.update(voxel_size_m=.5,roi_radius_m=10.);patch=SurfacePatches(**kwargs)
                def tensor(x):return torch.from_numpy(np.array(x,copy=True)).float().cuda()
                frames=tuple(e['frame_rows']);evidence=LocalBranchEvidence(torch.from_numpy(ids).long().cuda(),tensor(d['registered_returns_xyz_m'][4*11520+ids]),tensor(d['full_sensor_context'][4*180+(ids%720)//4]),tensor(d['patch_observed_features']),bind_structural_patches([patch],device='cuda'),CausalFrameOrderContext('sensor_current',frames,frames[-1],e['task']))
            with torch.no_grad():
                prediction=model(evidence,torch.from_numpy(oldpairs).long().cuda());reproduced=prediction.logits.cpu().numpy();checkprob=prediction.logits.sigmoid().cpu().numpy()
                olderror=float(np.max(np.abs(reproduced-oldlogits)));assert olderror<=1e-5
                for threshold in [.1,.9]:assert np.array_equal(checkprob>=np.float32(threshold),oldprob>=np.float32(threshold))
                tokens=prediction.ray_tokens;chunks=[]
                for offset in range(0,len(pairs),65536):
                    p=torch.from_numpy(pairs[offset:offset+65536]).long().cuda();a,b=tokens[p[:,0]],tokens[p[:,1]]
                    chunks.append(model.relation(torch.cat((a+b,(a-b).abs()),-1)).squeeze(-1).sigmoid().cpu().numpy())
                probabilities=np.concatenate(chunks)
            keys=pairs[:,0]*len(ids)+pairs[:,1];oldkeys=oldpairs[:,0]*len(ids)+oldpairs[:,1];positions=np.searchsorted(keys,oldkeys);inrange=positions<len(keys);common=np.flatnonzero(inrange);common=common[keys[positions[common]]==oldkeys[common]]
            error_common=float(np.max(np.abs(probabilities[positions[common]]-oldprob[common])));assert error_common<=1e-6
            assert np.array_equal(probabilities[positions[common]]>=np.float32(.9),oldprob[common]>=np.float32(.9))
            assert np.array_equal(probabilities[positions[common]]<=np.float32(.1),oldprob[common]<=np.float32(.1))
            count=np.zeros(len(ids),dtype=int);anchors_set=set(anchors)
            for (a,b),p in zip(pairs,probabilities):
                if p>=np.float32(.9):
                    if int(ids[a]) in anchors_set:count[b]+=1
                    if int(ids[b]) in anchors_set:count[a]+=1
            np.savez_compressed(out/f'artifacts/complete_{i:02d}.npz',ray_ids=ids,anchor_ids=np.array(anchors),pairs=pairs,probabilities=probabilities,support_count=count)
            row=dict(observation=i,rays=len(ids),pairs=len(pairs),supported=int((count>0).sum()),shared=int((count>1).sum()),unsupported=int((count==0).sum()),median_supporting_anchors=float(np.median(count)),original_logit_max_error=olderror,common_probability_max_error=error_common,common_pairs=len(common));rows.append(row)
            with (out/'logs/inference.jsonl').open('a') as log:log.write(json.dumps(row)+'\n')
            print(json.dumps(row),flush=True)
            assert time.monotonic()-start<s['limits']['wall_seconds'];assert torch.cuda.max_memory_reserved()<s['limits']['gpu_bytes'];assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<s['limits']['host_bytes']
            del evidence,prediction,tokens
        assert sum(r['pairs'] for r in rows)==4336032 and module_state_sha256(model)==before
    except Exception:
        error=traceback.format_exc();(out/'logs/failure.txt').write_text(error)
    write(out/'metrics/summary.json',dict(status='GATE_FAIL' if error else 'GATE_MIXED',observations=rows,error=error,training_steps=0,elapsed_s=time.monotonic()-start,weight_unchanged=not bool(error),branch_identity_verified=False,method_advantage_proven=False))
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    if error:raise RuntimeError(error)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args();freeze() if a.freeze else execute()
