"""Single sealed-cache linear-feasibility/fullbatch affine scoring diagnostic."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import io
import json
from pathlib import Path
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
from mtare_topo.governance_score_numerics_v1 import SCHEMA,SLUG,PARENT,SEAL,AUTH,POLICY,compile_scope,validate_card
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.fixed_score_numerics_v1 import equivalent_weights,objective,separability,solve_once
from mtare_topo.representation.geometry_match_presence_v1 import restore_presence
from mtare_topo.evaluation.grouping_center_scoring_v1 import center_score

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed0'

def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('no overwrite')
    scope=compile_scope(ROOT)
    a=dict(status='APPROVED',approved_by='user-exact-latest-task',approved_at='2026-09-10',authorized_operations=['training'],authorized_gates=[3],
        scope_sha256=digest(scope),confirmation_reference=AUTH,scope='section14 fixed512 cache,174known/338unknown; one affine numerical solve; no raw data/teacher/encoder/position/A/B runs')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=a)
    assert validate_card(card).passed;write(ROOT/CARD,card,'x')
    files=sorted({str(p.relative_to(ROOT)) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')}|{CARD,'tests/v3/unit/test_fixed_score_numerics_v1.py'})
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','MPLCONFIGDIR=/tmp/gse_numerics_mpl',PYTHON,
        'tools/v3/fixed_score_numerics_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=0,operation='training',data_card=CARD,config_path=CARD,user_authorization=a,
        question=POLICY['question'],method=POLICY['solver'],baseline='sealed section14 final affine head; no retraining',fallback='report unresolved vs nonseparable vs insufficient optimization vs label/detection gap; no expansion',
        command=command,wall_time_cap_s=600,estimated_cost=dict(compute='CPU only,174x129 linear program and one fullbatch numerical solve',host_ram_gb=8,gpu_vram_gb=0,disk_gb=1,wall_time_hours=1/6),
        acceptance_criteria=['Exact frozen512 and174labels; unknown zero weight; original schedule/group loss value and gradient equivalent.',
            'Separability needs verified primal/dual certificate; solver failure is undetermined, not nonseparable.',
            'Report solver termination/residuals separately from174label-fit and original512candidate detector score. No new threshold or loss.',
            'All nonexistence weights and positions fixed; only one numerical minimization; no graph or model search.'],
        expected_evidence=['LP certificate/status,exact weights,loss gradient equivalence,full solver log,129head and original float32 checkpoint,512predictions,original scorer replay,summary,seal'],
        source_sha256={p:sha(ROOT/p) for p in files},environment=environment())
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,counts=scope['candidate_counts'],parents=scope['counts']['parent_maps'])))

def aggregate(rows):
    c={k:sum(r[k] for r in rows) for k in ('tp','fp','fn','ignored','output_count')};tp,fp,fn=c['tp'],c['fp'],c['fn']
    return dict(c,precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,f1=2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.)

def json_value_equal(a,b):
    return json.loads(json.dumps(a))==json.loads(json.dumps(b))

def replay(rows,logits,reference_evaluation):
    records=[];offset=0
    for row,old in zip(rows,reference_evaluation['observations']):
        assert row['source']==old['source'];n=len(row['feature']);scores={}
        keep=(torch.from_numpy(logits[offset:offset+n].copy()).sigmoid()>=.5).numpy();offset+=n
        for radius in (.5,1.,2.,4.):
            original=old['scores'][str(radius)]
            assert original['query_indices']==list(range(n))
            cov=original['coverage']
            assert np.array_equal(row['position_m'],np.asarray(cov['query_xyz_m']))
            score=center_score(row['position_m'][keep],row['targets'],np.array(cov['query_scoreable_mask'],bool)[keep],
                ~np.array(cov['possible_unconfirmed_reference_mask'],bool)[keep],radius=radius)
            score.update(query_indices=np.flatnonzero(keep).tolist(),threshold=.5)
            scores[str(radius)]=score
        records.append(dict(source=row['source'],scores=scores))
    assert offset==512
    return dict(observations=records,summary={str(r):aggregate([o['scores'][str(r)] for o in records]) for r in (.5,1.,2.,4.)},
        coverage='original sealed all-query pointwise coverage reused; original center_score/threshold/matching, no new teacher')

def label_scores(logits,y,positive,negative):
    keep=torch.from_numpy(logits.copy()).sigmoid().numpy()>=.5;unknown=~(positive|negative)
    return dict(known=174,positive=12,negative=162,unknown=338,positive_selected=int(keep[positive].sum()),negative_selected=int(keep[negative].sum()),
        known_correct=int(keep[positive].sum()+(~keep[negative]).sum()),all_known_correct=bool(keep[positive].all() and not keep[negative].any()),
        unknown_selected=int(keep[unknown].sum()),minimum_known_signed_margin=float(np.min(y[positive|negative]*logits[positive|negative])))

def execute(spec,run):
    started=time.monotonic();summary={};error=None
    assert run.resolve()==ROOT/RUN
    assert json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def timeout(*_):raise TimeoutError('600s diagnostic limit')
    signal.signal(signal.SIGALRM,timeout);signal.alarm(600)
    try:
        torch.set_num_threads(1)
        card=json.loads((ROOT/CARD).read_text());assert validate_card(card).passed
        assert environment()==spec['environment'] and (run/'config/command.txt').read_text()==shlex.join(spec['command'])+'\n'
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/environment.json',spec['environment'])
        idx=card['scope']['bound_sha256'];opened={}
        def raw(rel):
            p=PARENT+'/'+rel;opened[p]=idx[p];return read_pinned(ROOT,p,idx[p])
        def load(rel):return json.loads(raw(rel))
        def npz(rel):
            with np.load(io.BytesIO(raw(rel))) as z:return {k:z[k].copy() for k in z.files}
        parent_spec=load('config/run_spec.json')
        for p in ('src/mtare_topo/representation/geometry_match_presence_v1.py','src/mtare_topo/representation/frozen_candidate_scoring_v1.py',
            'src/mtare_topo/evaluation/grouping_center_scoring_v1.py','src/mtare_topo/teacher/gse_reference_query_coverage_v1.py','src/mtare_topo/teacher/gse_reference_query_coverage_v2.py'):
            read_pinned(ROOT,p,parent_spec['source_sha256'][p])
        fixed=load('metrics/candidate_scores_0000.json');ref=load('metrics/evaluation_0000.json');oldfinal=load('metrics/evaluation_1000.json');rows=[]
        for obs in fixed['observations']:
            key=digest(obs['source']);r=npz('artifacts/fixed_candidates_'+key+'.npz');r['source']=obs['source']
            for k in ('positive','negative','unknown','assignment'):assert np.array_equal(r[k],obs[k])
            original=npz('artifacts/prediction_0000_'+key+'.npz');assert np.array_equal(r['position_m'],original['position_m'])
            r['targets']=npz('artifacts/input_'+key+'.npz')['target_positions_m'];rows.append(r)
        assert [digest(r['source']) for r in rows]==sorted(digest(r['source']) for r in rows)
        schedule=load('config/schedule.json');assert len(schedule)==1000
        weights,frequency=equivalent_weights([(r['positive'],r['negative']) for r in rows],schedule)
        X=np.column_stack([np.concatenate([r['feature'] for r in rows]).astype(np.float64),np.ones(512)])
        pos=np.concatenate([r['positive'] for r in rows]);neg=np.concatenate([r['negative'] for r in rows]);known=pos|neg;y=np.where(pos,1.,-1.)
        assert X.shape==(512,129) and pos.sum()==12 and neg.sum()==162 and known.sum()==174 and np.all(weights[~known]==0)
        ckpt=torch.load(io.BytesIO(raw('checkpoints/step_1000.pt')),map_location='cpu',weights_only=False);state=ckpt['model']
        theta0=np.r_[state['head.head.anchor.weight'][3].numpy(),state['head.head.anchor.bias'][3].numpy()].astype(float)
        # Establish equivalence to original differentiable group loss, not a pooled BCE.
        t=torch.tensor(theta0,requires_grad=True);z=torch.tensor(X)@t;total=0.;offset=0
        for row,f in zip(rows,frequency):
            r=restore_presence(dict(assignment=torch.tensor(row['assignment']),negative_mask=torch.tensor(row['negative']),position=torch.tensor(0.)),z[offset:offset+len(row['feature'])])
            total=total+r['presence']*int(f)/np.asarray(schedule).size;offset+=len(row['feature'])
        total.backward();v,g=objective(theta0,X,y,weights)
        eq=dict(value_difference=abs(v-float(total.detach())),gradient_max_difference=float(np.max(np.abs(g-t.grad.numpy()))),
            observation_frequency=frequency.tolist(),weight_sum=float(weights.sum()),unknown_weight_sum=float(weights[~known].sum()),regularization=0.,normalization='sum over original1000 batches of per-observation original loss/4, then /1000')
        assert eq['value_difference']<1e-10 and eq['gradient_max_difference']<1e-10
        write(run/'metrics/loss_equivalence.json',eq,'x')
        np.savez_compressed(run/'artifacts/fixed_design.npz',X=X,y=y,weights=weights,positive=pos,negative=neg)
        # Reuse already certified pointwise masks; test original initial AND final evaluator outputs exactly.
        for step,expected in [('0000',ref),('1000',oldfinal)]:
            oldlogits=np.concatenate([npz('artifacts/prediction_'+step+'_'+digest(r['source'])+'.npz')['presence_logits'] for r in rows])
            reproduced=replay(rows,oldlogits,ref)
            for a,b in zip(reproduced['observations'],expected['observations']):
                for radius in a['scores']:
                    for k,v0 in a['scores'][radius].items():assert json_value_equal(v0,b['scores'][radius][k]),(step,radius,k)
        write(run/'metrics/replay_equivalence.json',dict(parent_initial_final_all16_all4radii_exact=True,positions_fixed=True,unknown_not_background=True),'x')
        print('loss and original cached evaluator equivalence PASS; checking separability',flush=True)
        sep=separability(X[known],y[known]);write(run/'metrics/separability.json',sep,'x');print(json.dumps({k:v for k,v in sep.items() if k in ('status','message','min_signed_margin')}),flush=True)
        with (run/'logs/numerical_iterations.jsonl').open('x') as log:
            def progress(row):
                log.write(json.dumps(row)+'\n');log.flush()
                if row['iteration']%100==0:print(json.dumps(row),flush=True)
            theta,opt=solve_once(X[known],y[known],weights[known],theta0,callback=progress)
        write(run/'metrics/optimization.json',opt,'x')
        for k,value in state.items():
            if k=='head.head.anchor.weight':value[3]=torch.tensor(theta[:128],dtype=value.dtype)
            elif k=='head.head.anchor.bias':value[3]=torch.tensor(theta[128],dtype=value.dtype)
        fresh=torch.load(io.BytesIO(raw('checkpoints/step_1000.pt')),map_location='cpu',weights_only=False)['model']
        for k in state:
            assert torch.equal(state[k][:3],fresh[k][:3]) if k in ('head.head.anchor.weight','head.head.anchor.bias') else torch.equal(state[k],fresh[k])
        (run/'checkpoints').mkdir(exist_ok=True);torch.save(dict(model=state,parent=PARENT,numerical_head_only=True),run/'checkpoints/solved_head_model.pt')
        np.savez_compressed(run/'artifacts/solved_affine.npz',theta_float64=theta,initial_theta=theta0)
        # Original float32 affine head execution, not optimized float64 scores masquerading as model output.
        logits32=np.concatenate([torch.nn.functional.linear(torch.tensor(r['feature']),state['head.head.anchor.weight'],state['head.head.anchor.bias'])[:,3].numpy() for r in rows])
        logits64=X@theta;label64=label_scores(logits64,y,pos,neg);label32=label_scores(logits32,y,pos,neg)
        detection=replay(rows,logits32,ref);write(run/'metrics/evaluation_solved.json',detection,'x')
        offset=0
        for row in rows:
            n=len(row['feature']);np.savez_compressed(run/'artifacts'/('solved_prediction_'+digest(row['source'])+'.npz'),position_m=row['position_m'],presence_logits=logits32[offset:offset+n],presence_logits_float64=logits64[offset:offset+n],positive=row['positive'],negative=row['negative'],unknown=row['unknown']);offset+=n
        d=detection['summary']['1.0'];passed=d['precision']>=.9 and d['recall']>=.9
        summary=dict(status='GATE_PASS' if label32['all_known_correct'] and passed else 'GATE_MIXED',scope='numerical scoring diagnostic only, not research or generalization pass',
            separability=sep['status'],optimization_status=opt['optimization_status'],solver_success=opt['solver_success'],weighted_loss=opt['loss'],
            labels_float64=label64,labels_original_float32=label32,full_detection=d,detection_pass=passed,
            label_pass_but_detection_fail=label32['all_known_correct'] and not passed,
            numerically_unresolved=sep['status']=='NUMERIC_UNDETERMINED',known_labels_not_strictly_separable=sep['status']=='NOT_STRICTLY_SEPARABLE_CERTIFIED',
            optimization_insufficient=not opt['first_order_tolerance_met'],
            separable_logistic_infimum_zero=sep['status']=='STRICTLY_SEPARABLE_CERTIFIED',loss_gap_to_zero=opt['loss'],
            same129head=True,all512_positions_unchanged=True,nonexistence_parameters_unchanged=True,unknown_count=338,
            numerical_solve_count=1,adam_batches=0,encoder_runs=0,new_network=False,graph_started=False,next='report only; no automatic expansion')
        write(run/'artifacts/source_reads_sha256.json',opened,'x')
        for p,h in opened.items():assert sha(ROOT/p)==h
        for p,h in spec['source_sha256'].items():assert sha(ROOT/p)==h
        assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<POLICY['limits']['host_ram_bytes']
        assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<POLICY['limits']['output_bytes']
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error);summary.update(status='GATE_FAIL',error=error,numeric_result_not_research_failure=True)
    finally:
        signal.alarm(0);summary.update(elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        write(run/'metrics/summary.json',summary);write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt';seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute(json.loads(a.spec.read_text()),a.run_dir))
