"""Resume saved120 windows only; never launch or recapture the simulator."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import time
import traceback
import zipfile
from run_native_structure_capture import sha,write,IMAGE,MODEL_PY,CHECKPOINTS
from mtare_topo.governance_native_structure_capture import validate_finalize_card,scope_digest

NAME='gse_native_structure_finalize_v1'
CARD=f'configs/v3/gate6/data_cards/{NAME}.json';SPEC=f'configs/v3/gate6/{NAME}.json'
RUN=f'results/gate6_single_robot/gate6_20260913_{NAME}_seed11'
SOURCE='results/gate6_single_robot/gate6_20260913_gse_native_structure_capture_v1_seed11'
CASE=SOURCE+'/artifacts/cases/tunnel_seed11_native_shadow_capture'
MANIFEST=SOURCE+'/artifacts/model_inputs/windows_manifest.json'
CONTAINER='gse-native-structure-finalize-20260913-v1'


def verify_source_seal():
    base=(ROOT/SOURCE).resolve();count=0
    for line in (base/'artifacts/evidence_sha256.txt').read_text().splitlines():
        digest,path=line.split('  ',1);target=(ROOT/path).resolve()
        if not target.is_relative_to(base) or sha(target)!=digest:
            raise ValueError('original capture seal drift: '+path)
        count+=1
    if not count:raise ValueError('empty source seal')
    return dict(verified_files=count,source_run=SOURCE,all_passed=True)


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):raise ValueError('new finalization paths required')
    verify_source_seal()
    manifest=json.loads((ROOT/MANIFEST).read_text())
    assert len(manifest['windows'])==120
    s=dict(source_run=SOURCE,world='tunnel',seed=11,raw_frames=610,windows=120,
        independent_units='same one captured development trajectory; no new population',
        new_simulations=0,training_steps=0,resampled_windows=0,old_failure_preserved=True,
        repair='decode_ros_callerid_bytes_only_no_input_or_model_change',protected_worlds_read=[],
        pose='original saved full commanded poses; reject non-yaw relative motion',
        sampling='all existing120window manifest entries in saved order; no reexport or selection',
        teacher=None,split='same development integration, no calibration/training/test use',
        limits=dict(host_ram_gib=8,gpu_vram_gib=28,wall_sec=600,output_bytes=2*1024**3))
    a=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',
        authorized_operations=['closed_loop_single'],authorized_gates=[6],scope_sha256=scope_digest(s),
        scope='Finalize same authorized capture after an IO-only bug; no new simulation, model or input change',
        confirmation_reference='User 继续执行 authorized same-scene capture and accepted plan permits interface-error repair without retraining; failed v1 retained')
    card=dict(schema_version='gse_native_structure_finalize_card_v1',card_id=NAME,scope=s,approval=a)
    if not validate_finalize_card(card).passed:raise ValueError('invalid exact repair card')
    write(ROOT/CARD,card)
    paths=list((ROOT/'src/mtare_topo').rglob('*.py'))+list((ROOT/'tools/v3').glob('*.py'))+[ROOT/CARD]
    inputs={**CHECKPOINTS,MANIFEST:sha(ROOT/MANIFEST)}
    for entry in manifest['windows']:inputs[entry['input']['path']]=entry['input']['sha256']
    for path in [SOURCE+'/artifacts/model_inputs/pose_evidence.json',SOURCE+'/artifacts/model_inputs/frame_index.json',
                 SOURCE+'/artifacts/model_inputs/rejected_windows.json',SOURCE+'/artifacts/evidence_sha256.txt',
                 CASE+'/raw.bag',CASE+'/summary.json',CASE+'/planner/advice_sidecar/summary.json',
                 CASE+'/planner/advice_sidecar/native_region_tasks.json']:
        inputs[path]=sha(ROOT/path)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260913',slug=NAME,seed=11,
        operation='closed_loop_single',data_card=CARD,user_authorization=a,
        command=['python3','tools/v3/finalize_native_structure_capture.py','--execute'],
        question='Can the already exported120raw windows run the frozen token path after IO-only summary recovery?',
        method='Exactly same frozen tokens/inputs; bag connection-header UTF8 decode only',
        baseline='Previously captured native identity routes, not a new performance comparison',
        fallback='Keep all window rejection and fatal causes; no extra recording, training or input repair',
        estimated_cost=dict(compute='one metadata-only Docker then frozen GPU worker28GiB/host8GiB',disk_gb=2,wall_time_hours=1/6),
        acceptance_criteria=['Original capture seal and all120saved inputs unchanged',
            'Callerid decoded explicitly and sole native publisher verified',
            'All120frozen window outcomes preserved, no failed-window substitution',
            'Actual frozen model attempts/resources recorded; no advantage or live-control claim'],
        expected_evidence=['corrected metadata, unchanged source references, all model outputs/rejections, resources, terminal and seal'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths))},input_sha256=inputs))
    print(SPEC)


def execute():
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((out/'config/run_spec.json').read_text())==spec
    assert validate_finalize_card(json.loads((ROOT/CARD).read_text())).passed
    before=verify_source_seal()
    for path,digest in {**spec['source_sha256'],**spec['input_sha256']}.items():
        if sha(ROOT/path)!=digest:raise ValueError('source/input drift: '+path)
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in spec['source_sha256']:z.write(ROOT/path,path)
    start=time.monotonic();error=None;model=None;proc=None;peak=0
    write(out/'metrics/source_seal_before.json',before)
    write(out/'RUN_STATE.json',dict(state='RUNNING',stage='METADATA_ONLY_FINALIZATION'),'w')
    try:
        args=['python3','/workspace/tools/v3/inspect_native_capture_metadata.py','--bag','/workspace/'+CASE+'/raw.bag',
            '--output','/evidence/capture_metadata.json']
        shell='source /opt/ros/noetic/setup.bash && export PYTHONPATH=/workspace/src:$PYTHONPATH && '+shlex.join(args)
        command=['docker','run','--rm','--name',CONTAINER,'--network','none','--memory','2g','--cpus','2',
            '--user',str(os.getuid())+':'+str(os.getgid()),'-e','PYTHONDONTWRITEBYTECODE=1',
            '-v',str(ROOT)+':/workspace:ro','-v',str(out/'artifacts')+':/evidence:rw',
            '--entrypoint','/bin/bash',IMAGE,'-lc',shell]
        write(out/'logs/metadata_command.json',command)
        # No pre-existing container may be claimed by this run's cleanup.
        existing=subprocess.run(['docker','inspect',CONTAINER],capture_output=True)
        if existing.returncode==0:raise ValueError('named finalizer container already exists')
        try:
            with (out/'logs/metadata.log').open('xb') as f:
                subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,timeout=90,check=True)
        finally:
            subprocess.run(['docker','stop','--time','5',CONTAINER],capture_output=True,timeout=30)
            remaining=subprocess.run(['docker','inspect','-f','{{.State.Running}}',CONTAINER],capture_output=True,text=True)
            if remaining.returncode==0 and remaining.stdout.strip()=='true':
                raise RuntimeError('metadata container did not stop; do not run model')
        command=[MODEL_PY,'tools/v3/native_structure_token_process.py','--manifest',MANIFEST,
            '--manifest-sha256',spec['input_sha256'][MANIFEST],'--output',str(out/'artifacts/frozen_tokens'),
            '--resource-output',str(out/'metrics/model_resources.json')]
        write(out/'logs/model_command.json',command)
        write(out/'RUN_STATE.json',dict(state='RUNNING',stage='FROZEN_SAVED_INPUT_TOKEN_FORWARD'),'w')
        with (out/'logs/model.log').open('xb') as f:
            proc=subprocess.Popen(command,cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',
                    CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            while proc.poll() is None:
                try:peak=max(peak,next(int(x.split()[1])*1024 for x in Path(f'/proc/{proc.pid}/status').read_text().splitlines() if x.startswith('VmHWM:')))
                except (FileNotFoundError,ProcessLookupError,StopIteration):pass
                if peak>8*1024**3 or time.monotonic()-start>600 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>2*1024**3:
                    raise RuntimeError('fixed finalization resource limit exceeded')
                time.sleep(.2)
        path=out/'artifacts/frozen_tokens/summary.json'
        if path.exists():model=json.loads(path.read_text())
        if proc.returncode:raise RuntimeError('frozen worker exit '+str(proc.returncode)+'; original reason retained in model.log and records')
        write(out/'metrics/source_seal_after.json',verify_source_seal())
    except Exception:error=traceback.format_exc()
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    if error:write(out/'logs/failure.json',dict(error=error))
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,model=model,
        old_capture_failure_preserved=True,new_simulations=0,training_steps=0,
        resampled_windows=0,method_advantage_proven=False,learning_changed_live_control=False,
        peak_observed_worker_rss_bytes=peak,elapsed_s=time.monotonic()-start)
    write(out/'metrics/summary.json',result);write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for path in sorted(out.rglob('*')):
            if path.is_file() and path.name!='evidence_sha256.txt':f.write(sha(path)+'  '+str(path.relative_to(ROOT))+'\n')
    print(json.dumps(result),flush=True)
    return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    else:raise SystemExit(execute())
