"""One fixed CPU local registration pass over existing native retrieval pairs."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
from collections import Counter
from dataclasses import asdict
import json
import os
import platform
import resource
import time
import traceback
import zipfile

from run_native_token_retrieval import SOURCE,SEAL,sha,write
from mtare_topo.governance_native_structure_capture import scope_digest,validate_registration_card

RETRIEVAL='results/gate6_single_robot/gate6_20260913_gse_native_token_retrieval_v1_seed11'
NAME='gse_native_token_registration_v1'
RUN='results/gate6_single_robot/gate6_20260914_'+NAME+'_seed11'
CARD='configs/v3/gate6/data_cards/'+NAME+'.json';SPEC='configs/v3/gate6/'+NAME+'.json'
CONFIG='configs/v3/gate6/native_sensor_registration_v1.json'
PYTHON='/tmp/mtare_gate4_meshing_sidecar_v1/bin/python'


def freeze():
    if any((ROOT/p).exists() for p in (RUN,CARD,SPEC)):raise ValueError('fresh paths required')
    scope=dict(source_run=SOURCE,retrieval_run=RETRIEVAL,world='tunnel',seed=11,trajectories=1,
        raw_frames=617,selected_raw_frames=575,windows=115,candidate_pairs=560,
        independent_units='one120s historically used development trajectory, not560independent revisits',
        sampling='all saved native epochs and original top5 past candidates, no model/GT selection',
        temporal_spacing='native planning cadence with exact raw stamps in sealed source; no spatial resampling',
        split='development integration only, no strict test, no calibration',teacher=None,protected_worlds_read=[],
        points='original_current_raw_finite_returns_within_existing10m',
        pose_source='original_commanded_sensor_poses_not_physical_link_measurements',
        training_steps=0,model_forwards=0,new_simulations=0,control_changes=False,threshold_selection=False)
    approval=dict(status='APPROVED',approved_by='user',approved_at='2026-09-14',authorized_operations=['audit'],authorized_gates=[6],
        scope_sha256=scope_digest(scope),scope='Same recorded native candidates to fixed-config geometric verification; no training/control',
        confirmation_reference='User 继续推进 and active objective explicitly connecting retrieval/geometric verification/task states/native advice; standing interface execution authority')
    card=dict(schema_version='gse_native_registration_card_v1',card_id=NAME,scope=scope,approval=approval)
    if not validate_registration_card(card).passed:raise ValueError('card mismatch')
    write(ROOT/CARD,card)
    sources=['tools/v3/run_native_token_registration.py','tools/v3/run_native_token_retrieval.py','tools/v3/_bootstrap.py',
        'src/mtare_topo/integration/native_token_registration.py','src/mtare_topo/integration/native_token_archive.py',
        'src/mtare_topo/integration/structural_token_retrieval.py','src/mtare_topo/topology/gse_registration.py',
        'src/mtare_topo/governance.py','src/mtare_topo/governance_native_structure_capture.py',CONFIG,CARD]
    inputs=[SOURCE+'/artifacts/evidence_sha256.txt',RETRIEVAL+'/artifacts/retrieval.jsonl',RETRIEVAL+'/artifacts/evidence_sha256.txt']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260914',slug=NAME,seed=11,
        operation='audit',data_card=CARD,user_authorization=approval,
        command=[PYTHON,'tools/v3/run_native_token_registration.py','--execute'],
        question='Which of the560 already retrieved native historical candidates have usable local geometric alignment?',
        method='Fixed original-sensor-frame point-to-plane ICP with recorded fullSE3 initialization and bidirectional overlap/degeneracy evidence',
        baseline='Prior unverified560 token candidates; this is not a learned/nonlearned accuracy comparison',
        fallback='Retain rejection, ambiguity and tasks; no ICP retries, score/parameter selection or task completion from ICP',
        estimated_cost=dict(wall_time_hours=.5,disk_gb=.25,compute='CPU single-thread,8GiB/1800s cap; no GPU'),
        acceptance_criteria=['all115epochs and560candidates preserved', 'original source points and sensor poses remain bound',
            'strictly past same-segment matching, accepted/rejected reasons and runtime saved',
            'no identity/graph/control inferred from local ICP; no result-dependent parameter change'],
        expected_evidence=['all560pair transforms, bidirectional overlap/RMSE/normal/information diagnostics; source-point indices; epoch ambiguity; logs/config/source snapshot/seal'],
        source_sha256={p:sha(ROOT/p) for p in sources},input_sha256={p:sha(ROOT/p) for p in inputs}))
    print(SPEC)


def execute():
    # Set before importing numerical libraries, including the same source loader.
    for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
    os.environ['CUDA_VISIBLE_DEVICES']=''
    import numpy as np
    import open3d
    from mtare_topo.integration.native_token_archive import NativeTokenArchive
    from mtare_topo.integration.native_token_registration import original_sensor_cloud,verify_native_candidate
    from mtare_topo.topology.gse_registration import RegistrationConfig
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((out/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('run already used')
    if json.loads((out/'config/run_spec.json').read_text())!=spec:raise ValueError('spec drift')
    if not validate_registration_card(json.loads((ROOT/CARD).read_text())).passed:raise ValueError('card drift')
    for path,digest in {**spec['source_sha256'],**spec['input_sha256']}.items():
        if sha(ROOT/path)!=digest:raise ValueError('source drift: '+path)
    if (open3d.__version__,open3d.__DEVICE_API__,np.__version__,platform.python_version())!=('0.19.0','cpu','1.26.4','3.12.3'):
        raise ValueError('frozen CPU registration environment mismatch')
    config=RegistrationConfig(**json.loads((ROOT/CONFIG).read_text())['registration'])
    resource.setrlimit(resource.RLIMIT_AS,(8*1024**3,8*1024**3));resource.setrlimit(resource.RLIMIT_CPU,(1800,1800))
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in spec['source_sha256']:z.write(ROOT/p,p)
    write(out/'config/execution_environment.json',dict(python=platform.python_version(),numpy=np.__version__,
        open3d=open3d.__version__,device=open3d.__DEVICE_API__,threads=1,registration=asdict(config)))
    write(out/'RUN_STATE.json',dict(state='RUNNING'),'w')
    start=time.monotonic();error=None;history={};epochs=[];counts=Counter();pairs=0;accepted=0
    try:
        retrieval=[json.loads(line) for line in (ROOT/RETRIEVAL/'artifacts/retrieval.jsonl').read_text().splitlines()]
        if len(retrieval)!=115 or sum(len(r['candidates']) for r in retrieval)!=560:raise ValueError('population mismatch')
        archive=NativeTokenArchive(ROOT,SOURCE,seal_sha256=SEAL)
        with (out/'artifacts/registration.jsonl').open('x') as output, (out/'logs/registration.log').open('x') as log, (out/'artifacts/clouds.jsonl').open('x') as clouds:
            for record,meta in archive.records():
                row=retrieval[record.order]
                if row['epoch']!=record.record_id or row['source_refs']!=list(record.source_refs):raise ValueError('retrieval/source mismatch')
                current=original_sensor_cloud(archive,record)
                clouds.write(json.dumps(dict(epoch=current.epoch,points_sha256=current.points_sha256,
                    original_return_indices=current.raw_return_indices,input_sha256=current.input_sha256,
                    world_from_sensor=current.world_from_sensor.tolist(),source_key=current.source_key))+'\n');clouds.flush()
                eligible=[]
                for rank,candidate in enumerate(row['candidates']):
                    if time.monotonic()-start>1800:raise RuntimeError('wall budget exceeded')
                    if output.tell()>250*1024**2:raise RuntimeError('evidence size cap exceeded')
                    target=history[candidate['record_id']]
                    if candidate['current_source_refs']!=list(record.source_refs):raise ValueError('candidate source mismatch')
                    pair_start=time.monotonic();result=verify_native_candidate(current,target,config)
                    result.update(rank=rank,retrieval_score=candidate['score'],elapsed_s=time.monotonic()-pair_start)
                    output.write(json.dumps(result,allow_nan=False)+'\n');output.flush()
                    pairs+=1;accepted+=int(result['accepted']);counts.update(result['rejection_reasons'])
                    if result['accepted']:eligible.append(target.epoch)
                    log.write(f"pair={pairs}/560 epoch={current.epoch} rank={rank} local_fit={result['accepted']} reasons={result['rejection_reasons']}\n");log.flush()
                epochs.append(dict(epoch=current.epoch,candidates=len(row['candidates']),local_fit_candidates=eligible,
                    multiple_local_fits=len(eligible)>1,confirmed_task_identity=False))
                history[current.epoch]=current
        if len(epochs)!=115 or pairs!=560:raise ValueError('incomplete population')
    except Exception:error=traceback.format_exc()
    write(out/'artifacts/epoch_outcomes.json',epochs)
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,epochs=len(epochs),candidate_pairs=pairs,
        local_fit_accepted=accepted,local_fit_rejected=pairs-accepted,rejection_counts=dict(counts),
        epochs_with_local_fit=sum(bool(e['local_fit_candidates']) for e in epochs),
        epochs_with_multiple_local_fits=sum(e['multiple_local_fits'] for e in epochs),
        confirmed_associations=0,direction_tasks_created=0,control_changes=0,training_steps=0,model_forwards=0,
        elapsed_s=time.monotonic()-start,peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        limitation='Local geometric fit under commanded poses is not task identity, direction agreement or physical pose verification',
        method_advantage_proven=False)
    write(out/'metrics/summary.json',result);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');args=p.parse_args()
    if args.freeze:freeze()
    else:raise SystemExit(execute())
