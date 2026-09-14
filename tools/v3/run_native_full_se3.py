"""Same120 saved-input full-SE3 production replay; no simulator or optimizer."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import json
import os
from pathlib import Path
import subprocess
import time
import traceback
import zipfile
from run_native_structure_capture import sha,write,MODEL_PY,CHECKPOINTS
from finalize_native_structure_capture import SOURCE,MANIFEST
from mtare_topo.governance_native_structure_capture import validate_se3_card,scope_digest

NAME='gse_native_full_se3_v1'
CARD=f'configs/v3/gate6/data_cards/{NAME}.json';SPEC=f'configs/v3/gate6/{NAME}.json'
RUN=f'results/gate6_single_robot/gate6_20260913_{NAME}_seed11'
PREVIOUS='results/gate6_single_robot/gate6_20260913_gse_native_structure_finalize_v1_seed11'


def verify_seals():
    counts={}
    for base in (SOURCE,PREVIOUS):
        count=0
        for line in (ROOT/base/'artifacts/evidence_sha256.txt').read_text().splitlines():
            expected,path=line.split('  ',1);target=(ROOT/path).resolve()
            if not target.is_relative_to((ROOT/base).resolve()) or sha(target)!=expected:
                raise ValueError('old evidence drift: '+path)
            count+=1
        if not count:raise ValueError('empty source seal')
        counts[base]=count
    return counts


def scope():
    return dict(source_run=SOURCE,world='tunnel',seed=11,trajectories=1,windows=120,
        original_simulation_sec=120,original_raw_frames=610,original_synchronized_frames=600,
        original_distance_m=235.55039253774507,independent_units='one previously used development world/trajectory, not120independent structures',
        spacing='same preregistered first eligible raw window each1simsec; five raw frames span0.8sec; no spatial resampling',
        split='development integration only; no training, calibration or held-out evaluation',teacher=None,
        preprocessing_version='full_relative_se3_v1',raw_and_world_sensor_pose_bytes_unchanged=True,
        network_xyz_preprocessing_changed=True,model_architecture_changed=False,checkpoint_selection=False,
        new_simulations=0,training_steps=0,resampled_windows=0,old_results_preserved=True,protected_worlds_read=[],
        semantic_claim='all-ray frozen local features only; no direction probability, confirmed association or native-control benefit',
        limits=dict(host_ram_gib=8,gpu_vram_gib=28,wall_sec=600,output_bytes=2*1024**3))


def freeze():
    if any((ROOT/x).exists() for x in (CARD,SPEC,RUN)):raise ValueError('new non-overwriting run paths required')
    verify_seals();s=scope()
    windows=json.loads((ROOT/MANIFEST).read_text())['windows']
    if len(windows)!=120:raise ValueError('fixed population drift')
    a=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',authorized_operations=['closed_loop_single'],
        authorized_gates=[6],scope_sha256=scope_digest(s),
        scope='Same120 saved-input full-SE3 preprocessing repair; unchanged weights/raw/poses, zero training or simulation',
        confirmation_reference='User 现在解决 ... 设立一个目标一直跑 after explanation of full-rotation wiring and native-window binding; implements previously approved interface-repair branch with same recorded inputs, no new model or training')
    card=dict(schema_version='gse_native_full_se3_card_v1',card_id=NAME,scope=s,approval=a)
    if not validate_se3_card(card).passed:raise ValueError('invalid same120 SE3 card')
    write(ROOT/CARD,card)
    inputs={**CHECKPOINTS,MANIFEST:sha(ROOT/MANIFEST)}
    inputs.update({w['input']['path']:w['input']['sha256'] for w in windows})
    for base in (SOURCE,PREVIOUS):
        path=base+'/artifacts/evidence_sha256.txt';inputs[path]=sha(ROOT/path)
    for path in (ROOT/PREVIOUS/'artifacts/frozen_tokens').glob('record_*.json'):
        inputs[str(path.relative_to(ROOT))]=sha(path)
    paths=list((ROOT/'src/mtare_topo').rglob('*.py'))+list((ROOT/'tools/v3').glob('*.py'))+[ROOT/CARD]
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260913',slug=NAME,seed=11,
        operation='closed_loop_single',data_card=CARD,user_authorization=a,
        command=['python3','tools/v3/run_native_full_se3.py','--execute'],
        question='Does full-SE3 passthrough remove the recorded pose-interface rejection without changing raw inputs or model weights?',
        method='Same frozen encoder/C500, full relative R for both registered XYZ and encoder memory, all120windows',
        baseline='Sealed same120 yaw-only execution14processed/106rejected, not a semantic accuracy comparison',
        fallback='Preserve each failure and unknown; no sample substitution, tilt flattening, training or threshold modification',
        estimated_cost=dict(compute='one GPU worker,8GiB host/28GiB GPU caps',wall_time_hours=1/6,disk_gb=2),
        acceptance_criteria=['120requested/recorded/processed with preserved input and source fields',
            'All retained local tokens finite and full R passed through encoder and geometry',
            'Old160+154sealed evidence unchanged; no model or control change',
            'Record per-window old/new outcome, actual resource use, terminal and seal'],
        expected_evidence=['all120token outcomes, raw source references, per-window comparison, resources, command/source snapshot/seal'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},input_sha256=inputs))
    print(SPEC)


def execute():
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((out/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('run already executed')
    if json.loads((out/'config/run_spec.json').read_text())!=spec:raise ValueError('run spec drift')
    if not validate_se3_card(json.loads((ROOT/CARD).read_text())).passed:raise ValueError('scope drift')
    for path,digest in {**spec['source_sha256'],**spec['input_sha256']}.items():
        if sha(ROOT/path)!=digest:raise ValueError('source/input drift: '+path)
    write(out/'metrics/old_seal_before.json',verify_seals())
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in spec['source_sha256']:z.write(ROOT/path,path)
    start=time.monotonic();error=None;model=None;proc=None;peak=0
    write(out/'RUN_STATE.json',dict(state='RUNNING',stage='FULL_SE3_SAME120_FORWARD'),'w')
    try:
        command=[MODEL_PY,'tools/v3/native_structure_token_process.py','--manifest',MANIFEST,
            '--manifest-sha256',spec['input_sha256'][MANIFEST],'--output',str(out/'artifacts/frozen_tokens'),
            '--resource-output',str(out/'metrics/model_resources.json'),'--preprocessing','full_relative_se3_v1']
        write(out/'logs/model_command.json',command)
        with (out/'logs/model.log').open('xb') as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            while proc.poll() is None:
                try:peak=max(peak,next(int(x.split()[1])*1024 for x in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if x.startswith('VmHWM:')))
                except (FileNotFoundError,ProcessLookupError,StopIteration):pass
                if peak>8*1024**3 or time.monotonic()-start>600 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>2*1024**3:
                    raise RuntimeError('fixed resource limit exceeded')
                time.sleep(.2)
        if (out/'artifacts/frozen_tokens/summary.json').exists():model=json.loads((out/'artifacts/frozen_tokens/summary.json').read_text())
        if proc.returncode:raise RuntimeError('frozen worker exit '+str(proc.returncode)+'; see saved model log/records')
        rows=[]
        for i in range(120):
            name=f'record_{i:06d}.json'
            old=json.loads((ROOT/PREVIOUS/'artifacts/frozen_tokens'/name).read_text())
            new=json.loads((out/'artifacts/frozen_tokens'/name).read_text())
            if old['source']!=new['source']:raise ValueError('source window changed')
            rows.append(dict(index=i,window_id=new['window_id'],old_status=old['status'],new_status=new['status'],
                old_reason=old.get('reason'),new_reason=new.get('reason'),ray_count=new.get('ray_count'),
                raw_input_sha256=new['source']['input']['sha256']))
        write(out/'metrics/window_comparison.json',rows)
        if not model or model['processed_windows']!=120:raise RuntimeError('not all120windows completed full-SE3 token production')
    except Exception:error=traceback.format_exc()
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
        try:write(out/'metrics/old_seal_after.json',verify_seals())
        except Exception:error=(error or '')+'\n'+traceback.format_exc()
    if error:write(out/'logs/failure.json',dict(error=error))
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',input_path_complete=not bool(error),error=error,model=model,
        training_steps=0,new_simulations=0,resampled_windows=0,old_results_preserved=True,
        learning_changed_live_control=False,method_advantage_proven=False,
        peak_observed_worker_rss_bytes=peak,elapsed_s=time.monotonic()-start)
    write(out/'metrics/summary.json',result);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result),flush=True);return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
