"""Source-based zero-training qualification, then exactly one same-head fit."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter
import csv
import gzip
import io
import json
import math
from pathlib import Path
import resource
import shlex
import signal
import time
import traceback
import zipfile
import numpy as np
import torch
from mtare_topo.governance_localization_quality_v1 import SCHEMA,PARENT,SCORE,AUTH,POLICY,paths,compile_scope,validate_card
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.center_localization_quality_v1 import localization_quality
from mtare_topo.representation.candidate_center_verifier_v1 import CandidateCenterVerifierV1,balanced_validity_loss
from candidate_center_verifier_v1 import score_outputs,passed_metrics
from development_grids_v1 import write
from surface_features_v1 import PYTHON,sha,environment

SOURCES=['tools/v3/localization_quality_v1.py','src/mtare_topo/governance_localization_quality_v1.py',
 'src/mtare_topo/representation/center_localization_quality_v1.py','tests/v3/unit/test_center_localization_quality_v1.py',
 'src/mtare_topo/representation/candidate_center_verifier_v1.py','src/mtare_topo/evaluation/candidate_center_selection_v1.py',
 'src/mtare_topo/evaluation/grouping_center_scoring_v1.py','tools/v3/candidate_center_verifier_v1.py',
 'src/mtare_topo/teacher/gse_reference_query_coverage_v1.py','src/mtare_topo/teacher/gse_reference_query_coverage_v2.py',
 'src/mtare_topo/teacher/gse_reference_exclusion_v1.py']

def freeze(mode):
    slug,card_path,spec_path,run_path=paths(mode)
    if any((ROOT/p).exists() for p in (card_path,spec_path,run_path)):raise FileExistsError('single non-overwriting run')
    scope=compile_scope(ROOT,mode)
    a=dict(status='APPROVED',approved_by='user-exact-implementation-plan',approved_at='2026-09-10',scope='same16/512 cached source-based localization labels; audit first then single same-head1000updates; no graph or new model',
        scope_sha256=digest(scope),authorized_operations=[mode],authorized_gates=[3],confirmation_reference=AUTH)
    card=dict(schema_version=SCHEMA,card_id=slug,operation=mode,scope=scope,scope_sha256=digest(scope),policy=POLICY,approval=a)
    assert validate_card(card).passed
    write(ROOT/card_path,card,'x')
    command=['env','OMP_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','MKL_NUM_THREADS=1','PYTHONHASHSEED=0','CUBLAS_WORKSPACE_CONFIG=:4096:8',PYTHON,
        'tools/v3/localization_quality_v1.py','--mode',mode,'--spec',str(ROOT/spec_path),'--run-dir',str(ROOT/run_path)]
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260910',slug=slug,seed=0,operation=mode,data_card=card_path,config_path=card_path,user_authorization=a,
        question='Does independently supported localization-quality supervision align with unchanged full detection, and can the same frozen-candidate verifier fit it?',
        method='zero-training source-based labels and unknown-high oracle' if mode=='audit' else 'identical21185head from sealed step0; only labels changed;one1000full16updates',
        baseline='immutable section17 verifier and section18 evidence; no prior model retrained',command=command,
        estimated_cost=dict(compute='CPU cached-source audit, no network' if mode=='audit' else 'one GPU cached21185head1000updates,16000exposures; no parent reencoding',disk_gb=3,wall_time_hours=12 if mode=='training' else .25,host_ram_gb=32,gpu_vram_gb=28 if mode=='training' else 0),
        acceptance_criteria=[POLICY['compatibility'],POLICY['evaluation'],'No GT forward filtering; retain all original4mnegatives; unknowns zero training weight; no best checkpoint selection.'],
        expected_evidence=['source hashes,all512labels/transition/reasons,oracle predictions incl unknown,same initial checkpoint,all curve/predictions,old3/21/7 comparison,logs,seal'],
        source_sha256={p:sha(ROOT/p) for p in SOURCES+[card_path]},environment=environment())
    write(ROOT/spec_path,spec,'x');print(json.dumps(dict(mode=mode,spec=spec_path,counts=scope['counts'],queries=512,conditional_training=mode=='training')))

def execute(mode,spec,run):
    started=time.monotonic();err=None;curve=[];updates=0;exposures=0;net=None;log=None;opened={};summary={}
    assert run.resolve()==ROOT/paths(mode)[3]
    assert json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    assert (run/'config/command.txt').read_text()==shlex.join(spec['command'])+'\n'
    write(run/'RUN_STATE.json',dict(state='RUNNING',run_id=run.name))
    def alarm(*_):raise TimeoutError('frozen resource cap')
    signal.signal(signal.SIGALRM,alarm);signal.alarm(43200 if mode=='training' else 900)
    try:
        card=json.loads((ROOT/spec['data_card']).read_text());assert validate_card(card).passed and environment()==spec['environment']
        with zipfile.ZipFile(run/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
            for p,h in spec['source_sha256'].items():z.writestr(p,read_pinned(ROOT,p,h))
        # create_run owns environment.json; preserve it and store detailed sidecar separately.
        write(run/'config/sidecar_environment.json',spec['environment'],'x');bound=card['scope']['bound_sha256']
        def raw(p):opened[p]=bound[p];return read_pinned(ROOT,p,bound[p])
        def load(p):return json.loads(raw(p))
        def npz(p):
            with np.load(io.BytesIO(raw(p))) as z:return {k:z[k].copy() for k in z.files}
        old=load(PARENT+'/metrics/validity_inventory.json');reference=load(SCORE+'/metrics/evaluation_0000.json')
        oldfinal=load(PARENT+'/metrics/evaluation_1000.json');rows=[];labels=[];counts=Counter();transition=Counter();added=Counter()
        targets={digest(r['source']):r for r in card['scope']['target_rows']}
        for v,o in zip(old['observations'],reference['observations']):
            source=v['source'];key=digest(source);assert source==o['source']
            data=npz(PARENT+'/artifacts/input_'+key+'.npz');row=dict(source=source,position_m=data['position_m'],targets=data['target_positions_m']);rows.append(row)
            if mode=='audit':
                ev=load(SCORE+'/artifacts/fixed_supervision_evidence_'+key+'.json');tr=targets[key]
                produced=json.loads(gzip.decompress(raw(tr['target_path'])))['produced_targets'];record=produced['record']
                assert produced['source_binding']==tr['source_binding']==v['source_binding']
                assert canonical_sha(record)==o['scores']['1.0']['coverage']['target_record_sha256']==produced['target_record_sha256']
                assert record['score_region']['center_m']==[0.,0.,0.] and record['score_region']['radius_m']==10.
                assert record['source_frame_indices']==source['frame_rows']
                # Parent labels are stored as float32; compare the exact original
                # conversion, not impossible float64-to-float32 bit identity.
                assert np.array_equal(np.asarray([r['position_m'] for r in record['anchors']],dtype=row['targets'].dtype).reshape(-1,3),row['targets'])
                lab=localization_quality(row['position_m'],row['targets'],ev,o['scores']['1.0']['coverage'],source)
            else:lab=load(paths('audit')[3]+'/artifacts/quality_'+key+'.json')
            labels.append(lab);counts.update(lab['counts'])
            for a,b in zip(v['candidates'],lab['candidates']):
                transition[a['new_label']+'->'+b['label']]+=1
                if a['new_label']=='unknown' and b['label']=='negative':added[b['support_type']]+=1
            assert not any(a and not b for a,b in zip(v['negative'],lab['negative']))
            write(run/'artifacts'/('quality_'+key+'.json'),lab,'x')
        inventory=dict(counts=dict(counts),transition=dict(transition),added_support=dict(added),observations=labels,whole_region_complete=False)
        write(run/'metrics/label_inventory.json',inventory,'x')
        if mode=='audit':
            oracle=[]
            for unknown_logit in (-5.,10.):
                logits=[np.where(v['positive'],5.,np.where(v['negative'],-5.,unknown_logit)).astype(np.float32) for v in labels]
                e=score_outputs(rows,logits,reference);oracle.append(e)
                write(run/'metrics'/('ideal_unknown_'+('high' if unknown_logit>0 else 'low')+'.json'),e,'x')
            compatible=all(all(e['summary'][r][s]['tp']==12 and e['summary'][r][s]['fp']==0 and e['summary'][r][s]['fn']==0 for r in ('old','fixed') for s in ('before','after')) for e in oracle)
            assert oracle[1]['summary']['old']['before']['output_count']==counts['positive']+counts['unknown']
            summary=dict(status='GATE_PASS' if compatible else 'GATE_FAIL',compatibility_pass=compatible,counts=dict(counts),transition=dict(transition),added_support=dict(added),
                unknown_high=oracle[1]['summary'],training_updates=0,oracle_is_model_result=False,decision='allow one frozen same-head fit spec' if compatible else 'stop interface, no training')
        else:
            assert counts['positive']>0 and counts['negative']>0
            torch.set_num_threads(1);torch.use_deterministic_algorithms(True);assert torch.cuda.is_available()
            torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.backends.cudnn.benchmark=False
            torch.cuda.set_per_process_memory_fraction(POLICY['limits']['gpu_bytes']/torch.cuda.get_device_properties(0).total_memory)
            torch.manual_seed(0);net=CandidateCenterVerifierV1().cuda();initial=torch.load(io.BytesIO(raw(card['scope']['parent_initial_checkpoint'])),map_location='cpu',weights_only=False)
            assert initial['updates']==0 and initial['observation_exposures']==0
            net.load_state_dict(initial['model'],strict=True);assert sum(p.numel() for p in net.parameters())==21185
            packets=[];masks=[]
            for row,lab in zip(rows,labels):
                packet={k:torch.as_tensor(v,device='cuda') for k,v in npz(PARENT+'/artifacts/observation_cache_'+digest(row['source'])+'.npz').items()}
                assert np.array_equal(packet['candidate_xyz'].cpu().numpy(),row['position_m']) and all(not x.requires_grad for x in packet.values())
                packets.append(packet);masks.append((torch.tensor(lab['positive'],device='cuda'),torch.tensor(lab['negative'],device='cuda')))
            optimizer=torch.optim.AdamW([dict(params=[p for n,p in net.named_parameters() if n.endswith('weight')],weight_decay=1e-4),dict(params=[p for n,p in net.named_parameters() if n.endswith('bias')],weight_decay=0.)],lr=1e-3)
            (run/'checkpoints').mkdir(exist_ok=True)
            @torch.no_grad()
            def evaluate(step):
                net.eval();zs=[net(**p).cpu().numpy() for p in packets];e=score_outputs(rows,zs,reference);lc=Counter()
                for row,v,z in zip(rows,labels,zs):
                    chosen=torch.tensor(z).sigmoid().numpy()>=.5;pos=np.array(v['positive']);neg=np.array(v['negative']);unk=np.array(v['unknown'])
                    lc.update(dict(positive_correct=int(chosen[pos].sum()),positive_total=int(pos.sum()),negative_correct=int((~chosen[neg]).sum()),negative_total=int(neg.sum()),unknown_selected=int(chosen[unk].sum())))
                    np.savez_compressed(run/'artifacts'/f'prediction_{step:04d}_{digest(row["source"])}.npz',position_m=row['position_m'],logits=z)
                e.update(step=step,validity_labels=dict(lc));write(run/'metrics'/f'evaluation_{step:04d}.json',e,'x')
                torch.save(dict(model=net.state_dict(),optimizer=optimizer.state_dict(),updates=updates,observation_exposures=exposures,initial_checkpoint=card['scope']['parent_initial_checkpoint']),run/'checkpoints'/f'step_{step:04d}.pt')
                curve.append(dict(step=step,summary=e['summary'],validity_labels=dict(lc)));print(json.dumps(curve[-1]),flush=True);return e
            final=evaluate(0);log=(run/'logs/updates.jsonl').open('x')
            for step in range(1,1001):
                net.train();optimizer.zero_grad(set_to_none=True);lr=1e-5+.5*(1e-3-1e-5)*(1+math.cos(math.pi*(step-1)/999))
                for group in optimizer.param_groups:group['lr']=lr
                value=0.
                for packet,(pos,neg) in zip(packets,masks):
                    loss=balanced_validity_loss(net(**packet),pos,neg)/16.
                    if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
                    loss.backward();value+=float(loss.detach());exposures+=1
                norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);optimizer.step();updates+=1
                log.write(json.dumps(dict(update=updates,observation_exposures=exposures,lr=lr,loss=value,gradient_norm_before_clip=float(norm)))+'\n');log.flush()
                if step%100==0:final=evaluate(step)
                if resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024>POLICY['limits']['host_bytes'] or torch.cuda.max_memory_reserved()>POLICY['limits']['gpu_bytes']:raise MemoryError('resource cap')
            log.close();log=None
            assert all(t.grad is None and not t.requires_grad for p in packets for t in p.values())
            for row,p in zip(rows,packets):
                frozen=npz(PARENT+'/artifacts/observation_cache_'+digest(row['source'])+'.npz')
                assert all(np.array_equal(t.cpu().numpy(),frozen[k]) for k,t in p.items())
            success=passed_metrics(final)
            summary=dict(status='GATE_PASS' if success else 'GATE_FAIL',new_fit_pass=success,counts=dict(counts),transition=dict(transition),added_support=dict(added),updates=updates,observation_exposures=exposures,
                final=final['summary'],label_fit=final['validity_labels'],frozen_inputs_unchanged=True,initial_checkpoint_exact=True,parameters=21185,
                independent_generalization=False,geometry_advantage_proven=False,graph_started=False,decision='retain local fit; next independent development specification only' if success else 'stop this verifier configuration; no retry/newhead/graph')
            # Each originally selected FP is followed by source and slot, not just totals.
            failures=[]
            for prev,new,ref,v in zip(oldfinal['observations'],final['observations'],reference['observations'],old['observations']):
                for region,covkey in [('old','1.0'),('fixed','4.0')]:
                    def fps(e):
                        s=e['scores'][region]['after'];slots=s['original_query_slots'];matched={slots[j] for j,_ in s['pairs']};cov=ref['scores'][covkey]['coverage']['query_scoreable_mask']
                        return {q for q in slots if q not in matched and cov[q]}
                    before=fps(prev);after=fps(new)
                    for q in sorted(before|after):failures.append(dict(source=new['source'],query_slot=q,region=region,was_fp=q in before,is_fp=q in after,new_fp=q not in before,
                        old_label=v['candidates'][q]['new_label'],old_duplicate_source=v['candidates'][q]['old_duplicate'],final_logit=new['logits'][q],selected=q in new['selection']['after']))
            write(run/'metrics/failure_followup.json',failures,'x')
        write(run/'artifacts/source_reads_sha256.json',opened,'x')
        for p,h in {**opened,**spec['source_sha256']}.items():assert sha(ROOT/p)==h
        if sum(p.stat().st_size for p in run.rglob('*') if p.is_file())>POLICY['limits']['output_bytes']:raise MemoryError('disk cap')
    except Exception:
        err=traceback.format_exc();(run/'logs/error.log').write_text(err);summary=dict(status='GATE_FAIL',error=err,updates=updates,observation_exposures=exposures,decision='stop affected work, no restart')
        if net is not None:
            (run/'checkpoints').mkdir(exist_ok=True)
            torch.save(dict(model=net.state_dict(),updates=updates,exposures=exposures),run/'checkpoints/interrupted.pt')
    finally:
        if log:log.close()
        signal.alarm(0);summary.update(elapsed_s=time.monotonic()-started,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024)
        if mode=='training':summary['peak_cuda_bytes']=torch.cuda.max_memory_reserved()
        write(run/'metrics/curve.json',curve);write(run/'metrics/summary.json',summary)
        write(run/'RUN_STATE.json',dict(state='FAILED' if err else 'COMPLETED',run_id=run.name,error=err))
        seal=run/'artifacts/evidence_sha256.txt';seal.write_text(''.join(sha(p)+'  '+str(p.relative_to(ROOT))+'\n' for p in sorted(run.rglob('*')) if p.is_file() and p!=seal))
    print(json.dumps(summary),flush=True);return int(err is not None)

if __name__=='__main__':
    parser=argparse.ArgumentParser(__doc__);parser.add_argument('--freeze',action='store_true');parser.add_argument('--mode',choices=['audit','training'],required=True);parser.add_argument('--spec',type=Path);parser.add_argument('--run-dir',type=Path);args=parser.parse_args()
    if args.freeze:freeze(args.mode)
    else:raise SystemExit(execute(args.mode,json.loads(args.spec.read_text()),args.run_dir))
