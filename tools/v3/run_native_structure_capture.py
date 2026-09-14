"""Single frozen missing-signal capture, followed by bounded read-only tokens."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time
import traceback
import zipfile
from mtare_topo.governance_native_structure_capture import IMAGE, SCHEMA, scope_digest, validate_card

NAME='gse_native_structure_capture_v1'
CARD=f'configs/v3/gate6/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate6/{NAME}.json'
RUN=f'results/gate6_single_robot/gate6_20260913_{NAME}_seed11'
CASE='tunnel_seed11_native_shadow_capture'
CONTAINER='gse-native-structure-capture-20260913-v1'
TOPICS='configs/v3/gate6/closed_loop_recording_topics_gse_native_replay_v1.json'
MODEL_PY='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'
CHECKPOINTS={
 'results/gate3_semantics/gate3_20260902_primitive_relation_observable_three_seed_training_v1_seed0/artifacts/models/seed0/selected.pt':'8d5d2e0ec9779c28b068d0c7db80aeed38ebdb22dff5b781c26234d9001287fb',
 'results/gate3_semantics/gate3_20260913_gse_branch_core_fit_v1_seed0/artifacts/step_0500.pt':'ac1badf396804debb2b7daa505375bb0885a9d79bd918832a9f66e16697f6a76'}


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024**2),b''):h.update(block)
    return h.hexdigest()


def write(path,obj,mode='x'):
    with path.open(mode) as f:json.dump(obj,f,indent=2,allow_nan=False)


def scope():
    return dict(world='tunnel',seed=11,robots=1,episodes=1,start_xyz_yaw=[0,0,0,0],
        runtime_sec=120,training_steps=0,control='original_mtare_only',bridge_mode='shadow',
        source='new_missing_signal_supplement_not_reconstructed_old_bag',
        protected_worlds_read=[],teacher=None,image=IMAGE,
        sampling=dict(max_windows=120,history_frames=5,period_sim_ns=1000000000,
            select='first_eligible_raw_per_second',score_dependent=False,
            history='immediately_previous_four_raw_frames_plus_current',actual_raw_frames=None,actual_effective_windows=None),
        independent_units='One previously used development world and one continuous episode; adjacent windows not independent places',
        split='development_integration_only; no train/validation/test selection, no C08-C10',
        spatial_spacing='Actual executed distance retained; no spatial resampling or structure-score selection',
        pose_binding='original_aee_commanded_odometry_full_rotation',actual_link_pose_measured=False,
        full_rotation_policy='reject_non_yaw_compatible_window_keep_evidence',
        claim='shadow_integration_only_not_method_advantage',
        limits=dict(container_ram_gib=8,host_model_ram_gib=8,gpu_vram_gib=28,cpus=4,wall_sec=1800,output_bytes=10*1024**3),
        checkpoint_sha256=CHECKPOINTS,topic_contract=TOPICS,storage='retain original raw bag, no cleanup')


def freeze():
    if any((ROOT/p).exists() for p in (CARD,SPEC,RUN)):
        raise ValueError('new non-overwriting card, spec and run required')
    s=scope()
    auth=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',
        authorized_operations=['closed_loop_single'],authorized_gates=[6],scope_sha256=scope_digest(s),
        scope='Same tunnel seed11 missing-signal supplement and frozen token shadow; no training or changed control',
        confirmation_reference='User 继续执行 after explicit request to permit same-scene missing raw/native signal recording; existing full structural-assisted integration plan retained')
    card=dict(schema_version=SCHEMA,card_id=NAME,scope=s,approval=auth)
    if not validate_card(card).passed:raise ValueError('invalid card')
    write(ROOT/CARD,card)
    paths=list((ROOT/'src/mtare_topo').rglob('*.py'))+list((ROOT/'tools/v3').glob('*.py'))
    paths+=list((ROOT/'integration/native_structure_bridge').glob('*'))
    paths+=list((ROOT/'configs/v3/gate5/roslaunch').glob('*.launch'))+[ROOT/TOPICS,ROOT/CARD]
    paths=sorted({p for p in paths if p.is_file()})
    sources={str(p.relative_to(ROOT)):sha(p) for p in paths}
    for path,digest in CHECKPOINTS.items():
        if sha(ROOT/path)!=digest:raise ValueError('frozen checkpoint drift: '+path)
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260913',slug=NAME,seed=11,
        operation='closed_loop_single',data_card=CARD,user_authorization=auth,
        command=['python3','tools/v3/run_native_structure_capture.py','--execute'],
        question='Can exact raw observations and native route/feedback traces support the frozen-feature shadow on one continuous development record?',
        method='Original M-TARE control plus instrumented native shadow, same frozen epoch2/C500 local tokens; no same-group direction relabeling',
        baseline='Original route preserved at every epoch; not a paired performance experiment',
        fallback='Missing cache/unknown identity yields exact native route; incompatible windows retained rejected, no replacement or retraining',
        estimated_cost=dict(compute='one4CPU8GiB container then one frozen CUDA worker <=28GiB VRAM',host_ram_gb=16,gpu_vram_gb=28,disk_gb=10,wall_time_hours=.5),
        acceptance_criteria=['120 simulation seconds bounded capture with raw LiDAR, pose and native traces',
            'Sole original waypoint publisher and exact native routes in shadow',
            'Each selected window exported or explicitly rejected with cause',
            'Frozen real-model outputs and all rejection records retained; no claim of learned control from identity fallback',
            'No input drift, training, test read, overwrite or synthetic raw reconstruction'],
        expected_evidence=['raw bag, poses/trajectory/coverage proxy, native route candidates/advice/decisions/feedback',
            'model source windows, token outputs/rejections, task registry trace, runtime and terminal/seal'],
        image=IMAGE,source_sha256=sources,input_sha256=CHECKPOINTS))
    print(SPEC)


def container_command(out):
    case_path=f'/evidence/cases/{CASE}'
    inner=['python3','/workspace/tools/v3/native_structure_capture_case.py','--case-dir',case_path,
        '--case-id',CASE,'--world','tunnel','--environment-seed','11','--method-id','original_mtare_native_shadow_instrumented',
        '--method-family','original_mtare','--execution-repeat','0','--block-id','development_native_capture',
        '--runtime-sec','120','--archive-mode','host','--host-uid',str(os.getuid()),'--host-gid',str(os.getgid()),
        '--topic-contract','/workspace/'+TOPICS]
    post=['python3','/workspace/tools/v3/native_structure_capture_postprocess.py','--case-dir',case_path,
        '--logical-output-relative',RUN+'/artifacts/model_inputs','--host-uid',str(os.getuid()),'--host-gid',str(os.getgid())]
    build='''set -e
python3 /workspace/integration/native_structure_bridge/prepare_native_overlay.py --source-root /home/docker-user/mtare/tare_system/src/tare_planner --output-root /tmp/native-overlay
cp /tmp/native-overlay/src/grid_world/grid_world.cpp /home/docker-user/mtare/tare_system/src/tare_planner/src/grid_world/grid_world.cpp
cp /tmp/native-overlay/src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp /home/docker-user/mtare/tare_system/src/tare_planner/src/sensor_coverage_planner/sensor_coverage_planner_ground.cpp
cp -r /tmp/native-overlay/include/native_structure_bridge /home/docker-user/mtare/tare_system/src/tare_planner/include/
source /opt/ros/noetic/setup.bash
source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash
source /home/docker-user/mtare/tare_system/devel/setup.bash --extend
export PYTHONPATH=/workspace/src:$PYTHONPATH
cmake --build /home/docker-user/mtare/tare_system/build --target tare_planner_node -- -j2
sha256sum /home/docker-user/mtare/tare_system/devel/lib/tare_planner/tare_planner_node
'''
    shell=build+shlex.join(inner)+'\n'+shlex.join(post)
    return ['docker','run','--rm','--network','none','--hostname','localhost','--name',CONTAINER,
        '--memory','8g','--cpus','4','-e','ROS_HOME=/tmp/ros','-e','GAZEBO_MODEL_DATABASE_URI=',
        '-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1',
        '-v',str(ROOT)+':/workspace:ro','-v',str(out/'artifacts')+':/evidence:rw',
        '--entrypoint','/bin/bash',IMAGE,'-lc',shell]


def execute():
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((out/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':raise ValueError('fresh run only')
    if spec!=json.loads((out/'config/run_spec.json').read_text()):raise ValueError('created spec differs')
    if not validate_card(json.loads((ROOT/CARD).read_text())).passed:raise ValueError('invalid frozen card')
    for path,digest in {**spec['source_sha256'],**spec['input_sha256']}.items():
        if sha(ROOT/path)!=digest:raise ValueError('frozen source/input drift:'+path)
    if shutil.disk_usage(out).free<12*1024**3:raise RuntimeError('insufficient disk reserve')
    if subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()!=IMAGE:raise ValueError('image drift')
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for path in spec['source_sha256']:z.write(ROOT/path,path)
    (out/'artifacts/cases').mkdir()
    command=container_command(out);write(out/'logs/actual_command.json',command)
    write(out/'config/execution_environment.json',dict(image=IMAGE,host_python=sys.version,model_python=MODEL_PY,
        control='native_only',inference='post_capture_frozen_shadow_not_live_control',training_steps=0))
    write(out/'RUN_STATE.json',dict(state='RUNNING',stage='NATIVE_CAPTURE',container=CONTAINER),'w')
    started=time.monotonic();error=None;case=None;model=None
    try:
        with (out/'logs/docker.log').open('xb') as log:
            proc=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            write(out/'logs/process.json',dict(pid=proc.pid,container=CONTAINER))
            while proc.poll() is None:
                if time.monotonic()-started>1500 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>10*1024**3:
                    subprocess.run(['docker','stop','--time','20',CONTAINER],check=False,timeout=40)
                    proc.wait(timeout=60);raise RuntimeError('capture resource cap exceeded')
                time.sleep(2)
            if proc.returncode:raise RuntimeError('capture/export process exit '+str(proc.returncode))
        case=json.loads((out/f'artifacts/cases/{CASE}/summary.json').read_text())
        export_summary=json.loads((out/'artifacts/model_inputs/summary.json').read_text())
        if export_summary['exported_windows']==0:
            model=dict(no_valid_transfer=True,model_forwards=0,reason='ALL_SELECTED_WINDOWS_REJECTED_DURING_EXPORT',
                selected_windows=export_summary['selected_windows'],rejected_windows=export_summary['rejected_windows'])
            raise RuntimeError('all selected raw windows were rejected before model input; see model_inputs/rejected_windows.json')
        manifest=out/'artifacts/model_inputs/windows_manifest.json'
        model_command=[MODEL_PY,'tools/v3/native_structure_token_process.py',
            '--manifest',str(manifest.relative_to(ROOT)),
            '--manifest-sha256',sha(manifest),'--output',str(out/'artifacts/frozen_tokens'),
            '--resource-output',str(out/'metrics/model_resources.json')]
        write(out/'logs/model_command.json',model_command)
        write(out/'RUN_STATE.json',dict(state='RUNNING',stage='FROZEN_TOKEN_SHADOW'),'w')
        with (out/'logs/model.log').open('xb') as log:
            process=subprocess.Popen(model_command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            model_start=time.monotonic()
            while process.poll() is None:
                status_path=Path(f'/proc/{process.pid}/status')
                try:
                    hwm=next(int(line.split()[1])*1024 for line in status_path.read_text().splitlines() if line.startswith('VmHWM:'))
                except (FileNotFoundError,ProcessLookupError,StopIteration):hwm=0
                if hwm>8*1024**3 or time.monotonic()-model_start>300 or sum(p.stat().st_size for p in out.rglob('*') if p.is_file())>10*1024**3:
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError('model RAM/wall/output resource cap exceeded; partial records retained')
                time.sleep(.2)
        model_path=out/'artifacts/frozen_tokens/summary.json'
        if model_path.exists():model=json.loads(model_path.read_text())
        if process.returncode:raise RuntimeError('frozen token worker exit '+str(process.returncode)+'; original cause in logs/model.log and worker records')
    except Exception:
        error=traceback.format_exc()
        with (out/'logs/failure.txt').open('x') as f:f.write(error)
    summary=dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,case=case,model=model,
        elapsed_s=time.monotonic()-started,training_steps=0,method_advantage_proven=False,
        learning_changed_live_control=False,complete_exploration_verified=False,
        raw_bag_retained=True,formal_comparison_started=False)
    write(out/'metrics/summary.json',summary)
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(summary),flush=True)
    return int(error is not None)


if __name__=='__main__':
    parser=argparse.ArgumentParser();g=parser.add_mutually_exclusive_group(required=True)
    g.add_argument('--freeze',action='store_true');g.add_argument('--execute',action='store_true')
    args=parser.parse_args()
    if args.freeze:freeze()
    else:raise SystemExit(execute())
