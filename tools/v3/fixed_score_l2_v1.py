"""At most three frozen L2 heads and full original CUDA inference checks."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import io
import json
from pathlib import Path
from types import SimpleNamespace
import resource
import shlex
import signal
import time
import traceback
import zipfile
import numpy as np
import torch
from surface_features_v1 import PYTHON,sha,environment
from development_grids_v1 import write
from mtare_topo.governance_score_l2_v1 import SCHEMA,SLUG,PARENT,NUMERIC,POLICY,AUTH,compile_scope,validate_card
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.fixed_score_l2_v1 import LAMBDAS,solve,stable_scores
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.grouping_center_training_v1 import forward
from mtare_topo.evaluation.grouping_center_scoring_v1 import center_score,score_prediction
from grouping_center_fit_v1 import SelectedReader
from fixed_score_numerics_v1 import replay,aggregate,label_scores,json_value_equal

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed0'

def fixed_region_replay(rows,logits,reference):
    records=[];offset=0
    for row,old in zip(rows,reference['observations']):
        assert row['source']==old['source'];n=len(row['feature'])
        keep=(torch.from_numpy(logits[offset:offset+n].copy()).sigmoid()>=.5).numpy();offset+=n
        oldscore=old['scores']['4.0'];assert oldscore['query_indices']==list(range(n))
        cov=oldscore['coverage'];scoreable=np.asarray(cov['query_scoreable_mask'],bool);allowed=~np.asarray(cov['possible_unconfirmed_reference_mask'],bool)
        scores={}
        for r in (.5,1.,2.,4.):
            result=center_score(row['position_m'][keep],row['targets'],scoreable[keep],allowed[keep],radius=r)
            result.update(query_indices=np.flatnonzero(keep).tolist(),threshold=.5);scores[str(r)]=result
        records.append(dict(source=row['source'],scores=scores,fixed_scoreable_mask=scoreable.tolist(),fixed_allowed_mask=allowed.tolist()))
    return dict(observations=records,summary={str(r):aggregate([o['scores'][str(r)] for o in records]) for r in (.5,1.,2.,4.)},
        inference_filter='only sigmoid>=.5, no GT or coverage mask filters predictions',fixed_coverage_radius=4.)

def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    scope=compile_scope(ROOT);a=dict(status='APPROVED',approved_by='user-exact-latest-task',approved_at='2026-09-10',authorized_operations=['training'],authorized_gates=[3],
        scope_sha256=digest(scope),confirmation_reference=AUTH,scope='same16 full frozen inference/512features; three prescribed L2 affine solves, no protected data; conditional dev baseline freeze')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=a)
    assert validate_card(card).passed;write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,'tests/v3/unit/test_fixed_score_l2_v1.py'})
    cmd=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0','CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,
        'tools/v3/fixed_score_l2_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=0,operation='training',data_card=CARD,config_path=CARD,user_authorization=a,
        question='Can bounded L2 stabilize the same head and pass complete frozen forward without GT output filtering?',method=POLICY['loss']+'; '+POLICY['solver'],
        baseline='section14/15 sealed results reused; no repeated separability or unregularized fitting',fallback=POLICY['failure'],
        command=cmd,wall_time_cap_s=1800,estimated_cost=dict(compute='three tiny CPU fullbatch solves;48 CUDA no-gradient full frozen forwards',host_ram_gb=32,gpu_vram_gb=28,disk_gb=2,wall_time_hours=.5),
        acceptance_criteria=[POLICY['acceptance'],POLICY['selection'],'Unknown not background; full512queries retained; no more than3solves; old metrics unchanged and fixed4mregion sensitivity added.'],
        expected_evidence=['three solver logs/heads;512scores and full forward equivalence for each; frozen regions; both evaluation tables; baseline decision or stop; seal'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['counts'],lambdas=LAMBDAS)))

def execute(spec,run):
    started=time.monotonic();error=None;results=[];reader=None;solves=0
    assert run.resolve()==ROOT/RUN and json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def expire(*_):raise TimeoutError('1800s cap')
    signal.signal(signal.SIGALRM,expire);signal.alarm(1800)
    try:
        card=json.loads((ROOT/CARD).read_text());assert validate_card(card).passed
        assert environment()==spec['environment'] and (run/'config/command.txt').read_text()==shlex.join(spec['command'])+'\n'
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/environment.json',spec['environment'])
        assert torch.cuda.is_available();torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(POLICY['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        idx=card['scope']['bound_sha256'];opened={}
        def raw(p):opened[p]=idx[p];return read_pinned(ROOT,p,idx[p])
        def load(p):return json.loads(raw(p))
        def npz(p):
            with np.load(io.BytesIO(raw(p))) as z:return {k:z[k].copy() for k in z.files}
        design=npz(NUMERIC+'/artifacts/fixed_design.npz');initial=npz(NUMERIC+'/artifacts/solved_affine.npz')['initial_theta']
        reduction=load(NUMERIC+'/metrics/loss_equivalence.json');assert reduction['unknown_weight_sum']==0
        write(run/'metrics/reused_reduction.json',dict(parent=NUMERIC,values=reduction,recomputed=False),'x')
        fixed=load(PARENT+'/metrics/candidate_scores_0000.json');ref=load(PARENT+'/metrics/evaluation_0000.json');rows=[]
        for o in fixed['observations']:
            k=digest(o['source']);r=npz(PARENT+'/artifacts/fixed_candidates_'+k+'.npz');r['source']=o['source']
            r['targets']=npz(PARENT+'/artifacts/input_'+k+'.npz')['target_positions_m'];rows.append(r)
        X=design['X'];w=design['weights'];y=design['y'];known=w>0
        assert np.array_equal(X[:,:128],np.concatenate([r['feature'] for r in rows])) and X.shape==(512,129) and known.sum()==174
        assert np.array_equal(design['positive'],np.concatenate([r['positive'] for r in rows])) and np.array_equal(design['negative'],np.concatenate([r['negative'] for r in rows]))
        state=torch.load(io.BytesIO(raw(PARENT+'/checkpoints/step_1000.pt')),map_location='cpu',weights_only=False)['model']
        assert np.array_equal(initial,np.r_[state['head.head.anchor.weight'][3].numpy(),state['head.head.anchor.bias'][3].numpy()])
        reader=SelectedReader(card['scope']['full_forward_scope']);examples=[]
        for o,_ in reader.selected():examples.append(o);print(json.dumps(dict(stage='full_forward_inputs',loaded=len(examples))),flush=True)
        examples.sort(key=lambda o:digest(o.source));assert [o.source for o in examples]==[r['source'] for r in rows]
        allowed=card['scope']['full_forward_allowed_reads']
        for p,h in reader.opened.items():assert allowed[p]==h
        model=build_model(device='cuda');model.load_state_dict(state,strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        (run/'checkpoints').mkdir(exist_ok=True)
        for lam in LAMBDAS:
            tag=format(lam,'.0e');print('solving lambda '+tag,flush=True)
            with (run/'logs'/('solve_'+tag+'.jsonl')).open('x') as log:
                def progress(r):log.write(json.dumps(r)+'\n');log.flush()
                solves+=1;theta,opt=solve(X[known],y[known],w[known],initial,lam,callback=progress)
            write(run/'metrics'/('optimization_'+tag+'.json'),opt,'x')
            cpu={k:v.clone() for k,v in state.items()};cpu['head.head.anchor.weight'][3]=torch.tensor(theta[:-1],dtype=torch.float32);cpu['head.head.anchor.bias'][3]=torch.tensor(theta[-1],dtype=torch.float32)
            model.load_state_dict(cpu,strict=True)
            for k,v in model.state_dict().items():
                original=state[k];value=v.cpu()
                if k in ('head.head.anchor.weight','head.head.anchor.bias'):assert torch.equal(value[:3],original[:3])
                else:assert torch.equal(value,original)
            cpu_logits=[];gpu_logits=[];full_logits=[];checks=[];actual=[]
            with torch.no_grad():
                for row,o in zip(rows,examples):
                    feature=torch.tensor(row['feature']);head=model['head'].head.anchor
                    c=torch.nn.functional.linear(feature,cpu['head.head.anchor.weight'],cpu['head.head.anchor.bias'])[:,3]
                    g=head(feature.cuda())[:,3];captured={}
                    def hook(_,args,output):captured['feature']=args[0].detach().cpu()
                    h=head.register_forward_hook(hook)
                    # No labels/source IDs are even available on the forward input object.
                    try:out=forward(model,SimpleNamespace(student_representations={'PRIMITIVE':o.student_representations['PRIMITIVE']}),'PRIMITIVE')
                    finally:h.remove()
                    assert torch.equal(captured['feature'],feature)
                    p=out.prediction;assert np.array_equal(p.position_m.cpu().numpy(),row['position_m'])
                    orig=npz(PARENT+'/artifacts/prediction_0000_'+digest(row['source'])+'.npz')
                    assert np.array_equal(out.query_source_indices.cpu().numpy(),orig['query_indices']) and np.array_equal(out.query_positions_m.cpu().numpy(),orig['query_positions_m'])
                    assert torch.equal(g,p.presence_logits)
                    cpu_logits.extend(c.numpy());gpu_logits.extend(g.cpu().numpy());full_logits.extend(p.presence_logits.cpu().numpy())
                    scores={str(r):score_prediction(p,o,radius=r) for r in (.5,1.,2.,4.)};actual.append(dict(source=o.source,scores=scores))
                    np.savez_compressed(run/'artifacts'/('prediction_'+tag+'_'+digest(o.source)+'.npz'),position_m=p.position_m.cpu().numpy(),presence_logits=p.presence_logits.cpu().numpy(),query_indices=out.query_source_indices.cpu().numpy(),query_positions_m=out.query_positions_m.cpu().numpy())
                    checks.append(dict(source=o.source,queries=len(feature),feature_bitwise_equal=True,position_bitwise_equal=True,cache_cuda_full_bitwise_equal=True,forward_label_fields_absent=True))
            z64=X@theta;zc=np.array(cpu_logits,dtype=np.float32);zg=np.array(gpu_logits,dtype=np.float32);zf=np.array(full_logits,dtype=np.float32)
            stability=stable_scores(z64,[zc,zg,zf],known);labels=label_scores(zf,y,design['positive'],design['negative'])
            old=replay(rows,zf,ref);new=fixed_region_replay(rows,zf,ref)
            for a,b in zip(old['observations'],actual):
                for radius in a['scores']:
                    for k,v in a['scores'][radius].items():assert json_value_equal(v,b['scores'][radius][k]),(tag,radius,k)
            write(run/'metrics'/('old_evaluation_'+tag+'.json'),old,'x');write(run/'metrics'/('fixed_region_'+tag+'.json'),new,'x')
            write(run/'metrics'/('full_forward_'+tag+'.json'),dict(checks=checks,all_queries=512,stability=stability,old_actual_scorer_equivalent=True),'x')
            np.savez_compressed(run/'artifacts'/('head_'+tag+'.npz'),theta_float64=theta,logits_float64=z64,logits_cpu=zc,logits_cuda=zg,logits_full=zf)
            torch.save(dict(model=cpu,lambda_l2=lam,parent=PARENT),run/'checkpoints'/('head_'+tag+'.pt'))
            a=old['summary']['1.0'];b=new['summary']['1.0']
            passed=bool(opt['converged'] and labels['all_known_correct'] and stability['pass_stability'] and min(a['precision'],a['recall'],b['precision'],b['recall'])>=.9)
            result=dict(lambda_l2=lam,passed=passed,converged=opt['converged'],labels=labels,stability=stability,old=a,fixed_region=b,weight_norm=opt['weight_norm'],bias=opt['bias'],iterations=opt['iterations'])
            results.append(result);print(json.dumps(result),flush=True)
            assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<POLICY['limits']['host_bytes'] and torch.cuda.max_memory_reserved()<POLICY['limits']['gpu_bytes']
        chosen=next((r for r in results if r['passed']),None)
        if chosen:
            tag=format(chosen['lambda_l2'],'.0e');p=run/'checkpoints'/('head_'+tag+'.pt')
            write(run/'artifacts/development_baseline.json',dict(lambda_l2=chosen['lambda_l2'],checkpoint=str(p.relative_to(ROOT)),sha256=sha(p),scope='fit-only frozen development baseline; independent structure holdout required',selection=POLICY['selection'],no_gt_filter=True),'x')
        summary=dict(status='GATE_PASS' if chosen else 'GATE_FAIL',scope='bounded fixed-feature development-baseline check, not paper method validation',results=results,numerical_solves=solves,
            selected_lambda=chosen['lambda_l2'] if chosen else None,baseline_frozen=bool(chosen),separability_reruns=0,unregularized_reruns=0,adam_batches=0,
            next='independent structure holdout inventory and frozen evaluation scope; no more fitting this16' if chosen else 'stop rescuing this fixed-feature scoring implementation; no model search or fourth strength',
            phase3_research_pass=False,graph_started=False)
        write(run/'artifacts/source_reads_sha256.json',dict(cache=opened,full_forward=reader.opened),'x')
        for p,h in {**opened,**reader.opened,**spec['source_sha256']}.items():assert sha(ROOT/p)==h
        assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<POLICY['limits']['output_bytes']
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error);summary=dict(status='GATE_FAIL',error=error,results=results,numerical_solves=solves,system_failure=True,next='stop affected work; do not repeat completed solves')
    finally:
        signal.alarm(0);summary.update(elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_reserved())
        write(run/'metrics/summary.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt';seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute(json.loads(a.spec.read_text()),a.run_dir))
