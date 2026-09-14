"""One bounded same-scene source supplement and native-bound frozen forward."""
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
import run_native_structure_capture as prior
from run_native_structure_capture import sha,write,CHECKPOINTS,MODEL_PY,IMAGE
from mtare_topo.governance_native_structure_capture import scope_digest,validate_explicit_source_card

NAME='gse_explicit_source_capture_v1r1'
RUN=f'results/gate6_single_robot/gate6_20260913_{NAME}_seed11'
CARD=f'configs/v3/gate6/data_cards/{NAME}.json';SPEC=f'configs/v3/gate6/{NAME}.json'
TOPICS='configs/v3/gate6/closed_loop_recording_topics_gse_explicit_source_v1.json'
CASE='tunnel_seed11_native_shadow_capture'
CONTAINER='gse-explicit-source-20260913-v1'
EXECUTION_ENVIRONMENT='config/execution_environment.json'


def scope():
    return dict(world='tunnel',seed=11,robots=1,episodes=1,runtime_sec=120,start_xyz_yaw=[0,0,0,0],
        source='one explicit callback-source supplement; old120cannot establish exact native sources',
        sampling=dict(select='all_recorded_native_epochs_exact_callback_bound_five_frames',
            max_windows=120,max_raw_frames=700,actual_raw_frames=None,actual_native_windows=None,
            score_based_selection=False,spacing='native planning cadence, typically1simsec; source frame times saved exactly'),
        independent_units='one previously used development world and one continuous episode; not120independent structures',
        trajectories='single original controller trajectory; actual length/frame counts frozen after recording',
        split='development engineering transfer only; no train/test/threshold/checkpoint selection',
        source_pairing='explicit_same_callback_headers_no_time_or_index_inference',
        teacher=None,training_steps=0,protected_worlds_read=[],model_weights_changed=False,
        preprocessing='full_relative_se3_v1',control='original_mtare_only',bridge_mode='shadow',
        actual_link_pose_measured=False,old_results_preserved=True,
        output_claim='exact native-window frozen features and task/route records, not live learned control or advantage',
        limits=dict(wall_sec=1800,container_ram_gib=8,model_ram_gib=8,gpu_vram_gib=28,cpus=4,output_bytes=10*1024**3))


def container_command(out):
    cmd=prior.container_command(out)
    cmd[cmd.index('--name')+1]=CONTAINER
    shell=cmd[-1].split('python3 /workspace/tools/v3/native_structure_capture_postprocess.py',1)[0]
    shell=shell.replace('native_structure_capture_case.py','native_explicit_source_capture_case.py')
    shell=shell.replace(prior.TOPICS,TOPICS)
    insertion='''python3 /workspace/integration/native_structure_bridge/prepare_scan_source_overlay.py --source /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/src/vehicleSimulator.cpp --output /evidence/source_overlay
cp /evidence/source_overlay/vehicleSimulator.cpp /home/docker-user/mtare/autonomous_exploration_development_environment/src/vehicle_simulator/src/vehicleSimulator.cpp
cmake --build /home/docker-user/mtare/autonomous_exploration_development_environment/build --target vehicleSimulator -- -j2
sha256sum /home/docker-user/mtare/autonomous_exploration_development_environment/devel/lib/vehicle_simulator/vehicleSimulator
'''
    shell=shell.replace('cmake --build /home/docker-user/mtare/tare_system/build',insertion+'cmake --build /home/docker-user/mtare/tare_system/build',1)
    command=['python3','/workspace/tools/v3/export_explicit_native_windows.py',
        '--bag',f'/evidence/cases/{CASE}/raw.bag','--output','/evidence/model_inputs',
        '--logical',RUN+'/artifacts/model_inputs','--source-manifest','/evidence/source_overlay/manifest.json',
        '--source-manifest-logical',RUN+'/artifacts/source_overlay/manifest.json']
    cmd[-1]=shell+shlex.join(command)+'\n'
    return cmd


def freeze():
    if any((ROOT/x).exists() for x in (CARD,SPEC,RUN)):raise ValueError('fresh nonoverwriting paths required')
    s=scope();a=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',
        authorized_operations=['closed_loop_single'],authorized_gates=[6],scope_sha256=scope_digest(s),
        scope='Bounded same-scene explicit-source repair and native-bound frozen forward; original control, no training',
        confirmation_reference='User continued 继续推进 and persistent goal explicitly repairing native five-frame sources within approved tunnel/seed11 full integration; standing execute interface-repair authorization, no added world/method/weights or long comparison')
    card=dict(schema_version='gse_explicit_source_capture_card_v1',card_id=NAME,scope=s,approval=a)
    if not validate_explicit_source_card(card).passed:raise ValueError('data card mismatch')
    write(ROOT/CARD,card)
    paths=list((ROOT/'src/mtare_topo').rglob('*.py'))+list((ROOT/'tools/v3').glob('*.py'))
    paths+=list((ROOT/'integration/native_structure_bridge').glob('*'))
    paths+=list((ROOT/'configs/v3/gate5/roslaunch').glob('*.launch'))
    paths+=[ROOT/'configs/v3/gate6/roslaunch/system_explicit_scan_source.launch',ROOT/TOPICS,ROOT/CARD]
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260913',slug=NAME,seed=11,
        operation='closed_loop_single',data_card=CARD,user_authorization=a,
        command=['python3','tools/v3/run_explicit_source_capture.py','--execute'],
        question='Do explicitly traced native five-frame windows reach the frozen full-SE3 model with unambiguous provenance?',
        method='original native shadow plus same-callback scan sources, all native windows, unchanged epoch2/C500 features',
        baseline='old120fullSE3 processed but not exact-native source-bound; old113identity native decisions retained',
        fallback='reject missing sources/poses and keep original control; no timestamp guesses, new network, retries or threshold changes',
        estimated_cost=dict(wall_time_hours=.5,disk_gb=10,compute='4CPU/8GiB capture then same GPU model28GiB cap'),
        acceptance_criteria=['source echo matches actual raw and registered headers',
            'all recorded native snapshots retain exact5raw sources or explicit rejection; no dropped population',
            'all exported windows complete frozen fullSE3 forward with unchanged weights',
            'native tokens indexed by actual epoch and source proof; no live availability or control claims',
            'sole original waypoint publisher, original history immutable, no optimizer'],
        expected_evidence=['raw bag; explicit source records; native traces; all-window manifests/poses/tokens; task state; trajectories/coverage proxy; source/command/environment/seal'],
        image=IMAGE,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sorted(set(paths)) if p.is_file()},input_sha256=CHECKPOINTS))
    print(SPEC)


def execute():
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((out/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('run already executed')
    if json.loads((out/'config/run_spec.json').read_text())!=spec:raise ValueError('spec drift')
    if not validate_explicit_source_card(json.loads((ROOT/CARD).read_text())).passed:raise ValueError('scope drift')
    for path,h in {**spec['source_sha256'],**spec['input_sha256']}.items():
        if sha(ROOT/path)!=h:raise ValueError('frozen input/source drift '+path)
    if subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()!=IMAGE:raise ValueError('image drift')
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in spec['source_sha256']:z.write(ROOT/p,p)
    (out/'artifacts/cases').mkdir()
    command=container_command(out);write(out/'logs/container_command.json',command)
    write(out/EXECUTION_ENVIRONMENT,dict(image=IMAGE,model_python=MODEL_PY,control='original_mtare_only',training_steps=0))
    start=time.monotonic();error=None;proc=None;model=None;exported=None
    write(out/'RUN_STATE.json',dict(state='RUNNING',stage='EXPLICIT_SOURCE_CAPTURE'),'w')
    try:
        with (out/'logs/container.log').open('xb') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            while proc.poll() is None:
                if time.monotonic()-start>1500 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>10*1024**3:
                    subprocess.run(['docker','stop','--time','20',CONTAINER],timeout=40,check=False)
                    raise RuntimeError('capture resource bound exceeded')
                time.sleep(1)
            if proc.returncode:raise RuntimeError('capture/export exit '+str(proc.returncode))
        exported=json.loads((out/'artifacts/model_inputs/summary.json').read_text())
        manifest=RUN+'/artifacts/model_inputs/windows_manifest.json'
        write(out/'RUN_STATE.json',dict(state='RUNNING',stage='EXACT_NATIVE_FROZEN_FORWARD'),'w')
        mc=[MODEL_PY,'tools/v3/native_structure_token_process.py','--manifest',manifest,
            '--manifest-sha256',sha(ROOT/manifest),'--output',str(out/'artifacts/frozen_tokens'),
            '--resource-output',str(out/'metrics/model_resources.json'),'--preprocessing','full_relative_se3_v1']
        write(out/'logs/model_command.json',mc)
        with (out/'logs/model.log').open('xb') as log:
            proc=subprocess.Popen(mc,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUBLAS_WORKSPACE_CONFIG=':4096:8'))
            proc.wait(timeout=max(1,1800-(time.monotonic()-start)))
            if proc.returncode:raise RuntimeError('model exit '+str(proc.returncode))
        model=json.loads((out/'artifacts/frozen_tokens/summary.json').read_text())
        bindings=json.loads((out/'artifacts/model_inputs/source_bindings.json').read_text());index=[]
        for i,b in enumerate(bindings):
            row=json.loads((out/'artifacts/frozen_tokens'/f'record_{i:06d}.json').read_text())
            if row['status']!='PROCESSED' or [f['source_key'] for f in row['source']['frames']]!=b['raw_source_keys']:
                raise ValueError('native window/token source mismatch')
            index.append(dict(epoch=b['epoch'],window_id=b['window_id'],native_source_frame_keys=b['registered_source_keys'],
                raw_source_frame_keys=b['raw_source_keys'],source_binding_ref=RUN+'/artifacts/model_inputs/source_bindings.json#'+b['window_id'],
                token_ref=row['token_ref'],input_ref=row['source']['input'],live_token_available=False,
                learning_changed_control=False))
        write(out/'artifacts/native_token_index.json',index)
        if len(index)!=exported['native_snapshots'] or model['processed_windows']!=len(index):raise ValueError('incomplete native population')
    except Exception:error=traceback.format_exc()
    finally:
        if proc is not None and proc.poll() is None:
            proc.terminate()
            try:proc.wait(timeout=10)
            except subprocess.TimeoutExpired:proc.kill();proc.wait()
    result=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,export=exported,model=model,
        elapsed_s=time.monotonic()-start,training_steps=0,learning_changed_control=False,method_advantage_proven=False,
        exact_native_model_input_complete=not bool(error))
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
