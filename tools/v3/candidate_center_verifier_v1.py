"""One bounded center-validity training, plus zero-training old-head NMS controls."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter
import csv
import io
import json
import math
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
from types import SimpleNamespace
import zipfile
import numpy as np
import torch
from surface_features_v1 import PYTHON,sha,environment
from development_grids_v1 import write
from mtare_topo.governance_center_verifier_v1 import SCHEMA,SLUG,SCORE,L2,POLICY,AUTH,compile_scope,validate_card
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.candidate_center_verifier_v1 import CandidateCenterVerifierV1,balanced_validity_loss,capture_observation_memory
from mtare_topo.representation.candidate_validity_v1 import candidate_validity
from mtare_topo.evaluation.candidate_center_selection_v1 import select_candidates
from mtare_topo.evaluation.grouping_center_scoring_v1 import center_score
from grouping_center_fit_v1 import SelectedReader
from fixed_score_numerics_v1 import aggregate

CARD='configs/v3/gate3/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate3/'+SLUG+'.json'
RUN='results/gate3_semantics/gate3_20260910_'+SLUG+'_seed0'
SOURCES=[
 'tools/v3/candidate_center_verifier_v1.py','src/mtare_topo/governance_center_verifier_v1.py','src/mtare_topo/governance.py',
 'src/mtare_topo/representation/candidate_center_verifier_v1.py','src/mtare_topo/representation/candidate_validity_v1.py',
 'src/mtare_topo/evaluation/candidate_center_selection_v1.py','tests/v3/unit/test_candidate_center_verifier_v1.py',
 'src/mtare_topo/representation/observed_anchor_detector_v1.py','src/mtare_topo/representation/gse_block_point_encoder.py',
 'src/mtare_topo/representation/grouping_center_training_v1.py','src/mtare_topo/representation/grouping_zero_residual_init_v1.py',
 'src/mtare_topo/evaluation/grouping_center_scoring_v1.py','tools/v3/grouping_center_fit_v1.py','tools/v3/fixed_score_numerics_v1.py']

def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise FileExistsError('immutable single run')
    scope=compile_scope(ROOT);a=dict(status='APPROVED',approved_by='user-taskbook-exact-scope',approved_at='2026-09-10',authorized_operations=['training'],authorized_gates=[3],
        scope_sha256=digest(scope),confirmation_reference=AUTH,scope='same16/all512;candidate_validity_v1 derived once from sealed evidence;cache observation tokens;one1000full16update training;old3L2zero-training same2mselection;no graph/search')
    card=dict(schema_version=SCHEMA,card_id=SLUG,operation='training',scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=a)
    assert validate_card(card).passed;write(ROOT/CARD,card,'x')
    cmd=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0','CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,
        'tools/v3/candidate_center_verifier_v1.py','--spec',str(ROOT/SPEC),'--run-dir',str(ROOT/RUN)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=SLUG,seed=0,operation='training',data_card=CARD,config_path=CARD,user_authorization=a,
        question='Can observation-block geometry reread at frozen predicted centers fit position validity with separate score-only3Dduplicate selection?',method=POLICY['architecture'],
        baseline='three sealed section16L2 logits, each identical2mNMS,zero training;supervision and capacity differ,not pure geometry ablation',
        fallback='illegal/empty-class labels stop training;final failed fit stop this configuration,no reset or second network',command=cmd,wall_time_cap_s=43200,
        estimated_cost=dict(compute='one1000update full16 new21185parameter verifier;GPU parent frozen extraction and final checks',host_ram_gb=32,gpu_vram_gb=28,disk_gb=3,wall_time_hours=12),
        acceptance_criteria=[POLICY['evaluation'],POLICY['numeric'],'Source-bound new position labels; preserve old counts,all512raw outputs and every suppression;no GT runtime filtering.',
            'No LP/LBFGS/L2/position/A-B training repeated;one run,1000actualupdates16000observation exposures,not equal to old4000exposures.'],
        expected_evidence=['parent final fingerprint,memory cache+geometry,validity transition512reasons,oldNMScontrols,new checkpoints/predictions,NMS traces,unifiedCSV,all16views/curve,seal'],
        source_sha256={p:sha(ROOT/p) for p in SOURCES+[CARD]},environment=environment())
    write(ROOT/SPEC,spec,'x');print(json.dumps(dict(spec=SPEC,observations=16,candidates=512,parameters=21185,new_labels='inventory before training',parent_checkpoint=scope['parent_checkpoint'])))

def score_outputs(rows,logits,reference):
    records=[]
    for row,z,ref in zip(rows,logits,reference['observations']):
        assert row['source']==ref['source'];selection=select_candidates(row['position_m'],z);scores={}
        for region,key in [('old','1.0'),('fixed','4.0')]:
            allref=ref['scores'][key];assert allref['query_indices']==list(range(32));cov=allref['coverage']
            assert np.array_equal(row['position_m'],np.asarray(cov['query_xyz_m']))
            s=np.asarray(cov['query_scoreable_mask'],bool);a=~np.asarray(cov['possible_unconfirmed_reference_mask'],bool)
            scores[region]={}
            for stage in ('before','after'):
                chosen=np.array(selection[stage],dtype=int)
                result=center_score(row['position_m'][chosen],row['targets'],s[chosen],a[chosen],radius=1.)
                result['original_query_slots']=chosen.tolist();scores[region][stage]=result
        records.append(dict(source=row['source'],logits=np.asarray(z).tolist(),selection=selection,scores=scores))
    return dict(observations=records,summary={r:{s:aggregate([o['scores'][r][s] for o in records]) for s in ('before','after')} for r in ('old','fixed')},
        raw_candidates=sum(len(z) for z in logits),inference_uses_gt=False)

def passed_metrics(e):
    return all(e['summary'][r]['after']['precision']>=.9 and e['summary'][r]['after']['recall']>=.9 for r in ('old','fixed'))

def execute(spec,run):
    started=time.monotonic();error=None;updates=0;exposures=0;reader=None;model=None;verifier=None;log=None;curve=[];baselines={};legal=False;runtime=None
    assert run.resolve()==ROOT/RUN and json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def expire(*_):raise TimeoutError('bounded run time exceeded')
    signal.signal(signal.SIGALRM,expire);signal.alarm(POLICY['limits']['wall_s'])
    try:
        card=json.loads((ROOT/CARD).read_text());assert validate_card(card).passed and environment()==spec['environment']
        assert (run/'config/command.txt').read_text()==shlex.join(spec['command'])+'\n'
        with zipfile.ZipFile(run/'artifacts/changed_source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as archive:
            for p,h in spec['source_sha256'].items():archive.writestr(p,read_pinned(ROOT,p,h))
        write(run/'config/environment.json',spec['environment'])
        idx=card['scope']['bound_sha256'];opened={}
        def raw(p):opened[p]=idx[p];return read_pinned(ROOT,p,idx[p])
        def load(p):return json.loads(raw(p))
        def npz(p):
            with np.load(io.BytesIO(raw(p))) as z:return {k:z[k].copy() for k in z.files}
        oldlabels=load(SCORE+'/metrics/candidate_scores_0000.json');reference=load(SCORE+'/metrics/evaluation_0000.json');rows=[];validity=[];counts=Counter();transition=Counter()
        for old,ref in zip(oldlabels['observations'],reference['observations']):
            key=digest(old['source']);cache=npz(SCORE+'/artifacts/fixed_candidates_'+key+'.npz');inputs=npz(SCORE+'/artifacts/input_'+key+'.npz')
            r=dict(source=old['source'],position_m=cache['position_m'],targets=inputs['target_positions_m']);rows.append(r)
            ev=load(SCORE+'/artifacts/fixed_supervision_evidence_'+key+'.json')
            label=candidate_validity(r['position_m'],r['targets'],ev,ref['scores']['1.0']['coverage'],r['source'],old)
            validity.append(label);counts.update(label['counts']);transition.update(label['transition'])
            write(run/'artifacts'/('candidate_validity_'+key+'.json'),label,'x')
            np.savez_compressed(run/'artifacts'/('input_'+key+'.npz'),xyz_m=inputs['xyz_m'],position_m=r['position_m'],target_positions_m=r['targets'])
        label_inventory=dict(schema='candidate_validity_v1',counts=dict(counts),transition=dict(transition),observations=validity,old_counts=dict(positive=12,negative=162,unknown=338))
        write(run/'metrics/validity_inventory.json',label_inventory,'x');print(json.dumps(dict(stage='validity_inventory',counts=dict(counts),transition=dict(transition))),flush=True)
        # Fair zero-training NMS controls, even if the new-label training gate fails.
        fp_audit=[]
        for tag in ('1e-02','1e-04','1e-06'):
            oldhead=npz(L2+'/artifacts/head_'+tag+'.npz');zs=np.split(oldhead['logits_full'],16)
            e=score_outputs(rows,zs,reference);baselines[tag]=e;write(run/'metrics'/('old_l2_'+tag+'.json'),e,'x')
            previous=load(L2+'/metrics/old_evaluation_'+tag+'.json')
            for i,(a,b) in enumerate(zip(e['observations'],previous['observations'])):
                for name in ('tp','fp','fn','ignored'):assert a['scores']['old']['before'][name]==b['scores']['1.0'][name]
                s=a['scores']['old']['before'];matched={s['original_query_slots'][j] for j,_ in s['pairs']};cov=reference['observations'][i]['scores']['1.0']['coverage']['query_scoreable_mask']
                for q in s['original_query_slots']:
                    if q not in matched and cov[q]:fp_audit.append(dict(old_lambda=tag,source=rows[i]['source'],**validity[i]['candidates'][q]))
            print(json.dumps(dict(old_lambda=tag,zero_training=True,summary=e['summary'])),flush=True)
        write(run/'metrics/old_fp_position_evidence.json',fp_audit,'x')
        legal=counts['positive']>0 and counts['negative']>0
        if not legal:raise ValueError('POSITION_SUPERVISION_CLASS_EMPTY; no training allowed')
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
        torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
        torch.cuda.set_per_process_memory_fraction(POLICY['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
        parent=card['scope']['parent_checkpoint'];state=torch.load(io.BytesIO(raw(parent)),map_location='cpu',weights_only=False)['model']
        model=build_model(device='cuda');model.load_state_dict(state,strict=True);model.eval()
        for p in model.parameters():p.requires_grad_(False)
        reader=SelectedReader(card['scope']['full_forward_scope']);examples=[]
        for o,_ in reader.selected():examples.append(o)
        examples.sort(key=lambda o:digest(o.source));assert [o.source for o in examples]==[r['source'] for r in rows]
        for p,h in reader.opened.items():assert card['scope']['full_forward_allowed_reads'][p]==h
        packets=[];memory_records=[];masks=[]
        for r,o,v in zip(rows,examples,validity):
            student=SimpleNamespace(student_representations={'PRIMITIVE':o.student_representations['PRIMITIVE']})
            out,packet=capture_observation_memory(model,student)
            assert np.array_equal(out.prediction.position_m.cpu().numpy(),r['position_m']) and len(r['position_m'])==32
            if not packet['block_valid'].any():raise ValueError('REAL_EMPTY_OBSERVATION; no guessed negative region')
            packets.append(packet);masks.append((torch.tensor(v['positive'],device='cuda'),torch.tensor(v['negative'],device='cuda')))
            key=digest(r['source']);np.savez_compressed(run/'artifacts'/('observation_cache_'+key+'.npz'),**{k:t.cpu().numpy() for k,t in packet.items()})
            memory_records.append(dict(source=r['source'],variable='memory=ObservedAnchorDetectorV1.adapter(...)',shape=list(packet['block_features'].shape),
                includes_robot_xyz=True,includes_extra_position_encoding=False,all_observation_blocks_valid=True,parameters_frozen=True,final_query_features_used=False))
        write(run/'metrics/token_provenance.json',memory_records,'x')
        torch.manual_seed(0);verifier=CandidateCenterVerifierV1().cuda();assert sum(p.numel() for p in verifier.parameters())==21185
        optimizer=torch.optim.AdamW([dict(params=[p for n,p in verifier.named_parameters() if n.endswith('weight')],weight_decay=1e-4),
            dict(params=[p for n,p in verifier.named_parameters() if n.endswith('bias')],weight_decay=0.)],lr=1e-3)
        (run/'checkpoints').mkdir(exist_ok=True)
        @torch.no_grad()
        def evaluate(step):
            verifier.eval();zs=[verifier(**p).cpu().numpy() for p in packets]
            e=score_outputs(rows,zs,reference);labelcounts=Counter()
            for z,v,r in zip(zs,validity,rows):
                pred=torch.tensor(z).sigmoid().numpy()>=.5;positive=np.asarray(v['positive']);negative=np.asarray(v['negative']);unknown=np.asarray(v['unknown'])
                labelcounts.update(dict(positive_correct=int(pred[positive].sum()),positive_total=int(positive.sum()),negative_correct=int((~pred[negative]).sum()),negative_total=int(negative.sum()),unknown_selected=int(pred[unknown].sum())))
                np.savez_compressed(run/'artifacts'/f'prediction_{step:04d}_{digest(r["source"])}.npz',position_m=r['position_m'],logits=z)
            e.update(step=step,validity_labels=dict(labelcounts));write(run/'metrics'/f'evaluation_{step:04d}.json',e,'x')
            torch.save(dict(model=verifier.state_dict(),optimizer=optimizer.state_dict(),updates=updates,observation_exposures=exposures,parent_checkpoint=parent),run/'checkpoints'/f'step_{step:04d}.pt')
            curve.append(dict(step=step,summary=e['summary'],validity_labels=dict(labelcounts)))
            print(json.dumps(curve[-1]),flush=True);return e
        final=evaluate(0);log=(run/'logs/updates.jsonl').open('x')
        for step in range(1,1001):
            verifier.train();optimizer.zero_grad(set_to_none=True);lr=1e-5+.5*(1e-3-1e-5)*(1+math.cos(math.pi*(step-1)/999))
            for group in optimizer.param_groups:group['lr']=lr
            loss_value=0.
            for packet,(positive,negative) in zip(packets,masks):
                z=verifier(**packet);loss=balanced_validity_loss(z,positive,negative)/16.
                if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                loss.backward();loss_value+=float(loss.detach());exposures+=1
            norm=torch.nn.utils.clip_grad_norm_(verifier.parameters(),1.,error_if_nonfinite=True)
            assert all(p.grad is None for p in model.parameters())
            optimizer.step();updates+=1
            log.write(json.dumps(dict(update=updates,observation_exposures=exposures,lr=lr,loss=loss_value,gradient_norm_before_clip=float(norm)))+'\n');log.flush()
            if step%100==0:final=evaluate(step)
            if step%20==0:
                print(json.dumps(dict(stage='training',updates=updates,exposures=exposures,elapsed_s=time.monotonic()-started)),flush=True)
                assert resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024<POLICY['limits']['host_bytes'] and torch.cuda.max_memory_reserved()<POLICY['limits']['gpu_bytes']
        log.close();log=None
        # Final same-device full frozen inference and CPU execution, never a new fit.
        cpu=CandidateCenterVerifierV1();cpu.load_state_dict({k:v.cpu() for k,v in verifier.state_dict().items()});cpu.eval();verifier.eval();checks=[]
        with torch.no_grad():
            for r,o,p,(positive,negative) in zip(rows,examples,packets,masks):
                _,fresh=capture_observation_memory(model,SimpleNamespace(student_representations={'PRIMITIVE':o.student_representations['PRIMITIVE']}))
                for k in p:assert torch.equal(p[k],fresh[k])
                a=verifier(**p);b=verifier(**fresh);c=cpu(**{k:v.cpu() for k,v in fresh.items()})
                equal=torch.allclose(a,b,atol=1e-5,rtol=1e-5);difference=float((a.cpu()-c).abs().max())
                changed=torch.nonzero((a.cpu().sigmoid()>=.5)!=(c.sigmoid()>=.5)).flatten().tolist();known=positive|negative
                margin=float(a[known].abs().min()) if known.any() else None
                gpu_selection=select_candidates(r['position_m'],a.cpu().numpy())['after'];cpu_selection=select_candidates(r['position_m'],c.numpy())['after']
                nms_changed=sorted(set(gpu_selection)^set(cpu_selection))
                okay=equal and not changed and not nms_changed and (margin is None or margin>max(1e-4,10*difference))
                checks.append(dict(source=r['source'],cache_full_equal=equal,cpu_cuda_max_error=difference,cpu_cuda_selection_changed_slots=changed,cpu_cuda_nms_changed_slots=nms_changed,known_min_abs_logit=margin,passed=okay))
        for k,v in model.state_dict().items():assert torch.equal(v.cpu(),state[k])
        runtime=dict(checks=checks,passed=all(c['passed'] for c in checks),parent_all_parameters_equal=True,positions512_equal=True,parent_gradients_none=True)
        write(run/'metrics/final_execution_checks.json',runtime,'x')
        old_pass=[tag for tag,e in baselines.items() if passed_metrics(e)];new_pass=passed_metrics(final) and runtime['passed']
        if new_pass or old_pass:
            chosen='old_l2_'+old_pass[0] if old_pass else 'candidate_center_verifier_v1'
            write(run/'artifacts/fit_version.json',dict(chosen=chosen,old_pass=old_pass,new_pass=new_pass,scope='16 observation known-domain fit only; unknown outputs retained; not deployable or geometry advantage',next='independent development structure split specification only,no training'), 'x')
        summary=dict(status='GATE_PASS' if new_pass else 'GATE_MIXED' if old_pass else 'GATE_FAIL',new_fit_pass=bool(new_pass),old_nms_pass=old_pass,
            legal_supervision=legal,label_counts=dict(counts),transition=dict(transition),parameters=21185,updates=updates,observation_exposures=exposures,
            old_training_exposures=4000,not_equal_training_budget=True,final=final['summary'],final_label_fit=final['validity_labels'],runtime_checks_pass=runtime['passed'],
            decision='keep simpler oldNMS fit version;new necessity unproven' if old_pass else 'keep new local fit version,not geometric advantage' if new_pass else 'stop this configuration,no second verifier or model search',
            graph_started=False,geometry_advantage_proven=False,holdout_training_started=False)
        write(run/'artifacts/source_reads_sha256.json',dict(cache=opened,full_forward=reader.opened),'x')
        for p,h in {**opened,**reader.opened,**spec['source_sha256']}.items():assert sha(ROOT/p)==h
        assert sum(p.stat().st_size for p in run.rglob('*') if p.is_file())<POLICY['limits']['output_bytes']
    except Exception:
        error=traceback.format_exc();(run/'logs/error.log').write_text(error)
        if verifier is not None:
            (run/'checkpoints').mkdir(exist_ok=True);torch.save(dict(model=verifier.state_dict(),updates=updates,exposures=exposures),run/'checkpoints/interrupted.pt')
        summary=dict(status='GATE_FAIL',error=error,updates=updates,observation_exposures=exposures,legal_supervision=legal,decision='stop affected task;preserve consumed budget,no restart')
    finally:
        if log:log.close()
        signal.alarm(0);summary.update(elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,peak_cuda_bytes=torch.cuda.max_memory_reserved())
        write(run/'metrics/curve.json',curve);write(run/'metrics/summary.json',summary)
        with (run/'metrics/unified_results.csv').open('x') as f:
            writer=csv.DictWriter(f,fieldnames=['method','step','region','selection','tp','fp','fn','ignored','output_count','precision','recall','f1']);writer.writeheader()
            items=[('old_l2_'+tag,'reused',e['summary']) for tag,e in baselines.items()]+[('candidate_center_verifier_v1',str(c['step']),c['summary']) for c in curve]
            for method,step,entry in items:
                for region,stages in entry.items():
                    for selection,v in stages.items():writer.writerow(dict(method=method,step=step,region=region,selection=selection,**v))
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED',run_id=run.name,error=error))
        seal=run/'artifacts/evidence_sha256.txt';seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(error is not None)

if __name__=='__main__':
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute(json.loads(a.spec.read_text()),a.run_dir))
