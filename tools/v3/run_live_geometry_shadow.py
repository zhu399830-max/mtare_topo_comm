"""Freeze and execute exactly one development live geometry shadow episode."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import tempfile
from mtare_topo.governance_live_geometry_shadow import IMAGE,SCHEMA,SLUG,SCOPE,validate_card

CARD='configs/v3/gate5/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate5/'+SLUG+'.json'
RUN='results/gate5_shadow/gate5_20260910_'+SLUG+'_seed11'
NAME='gse-live-geometry-shadow-v1'
VERSION=1
GATE=5

def configure_version(version):
    global VERSION,SCHEMA,SLUG,SCOPE,CARD,SPEC,RUN,NAME,GATE
    VERSION=version
    if version>=2:
        SCHEMA='v3_live_geometry_shadow_card_v2';SLUG='gse_live_geometry_shadow_v2'
        SCOPE=dict(SCOPE,pose_interface='aee-commanded-odometry',past_pose_max_age_s=.1,
                   actual_link_pose_measured=False)
        CARD='configs/v3/gate5/data_cards/'+SLUG+'.json'
        SPEC='configs/v3/gate5/'+SLUG+'.json'
        RUN='results/gate5_shadow/gate5_20260910_'+SLUG+'_seed11'
        NAME='gse-live-geometry-shadow-v2'
    if version>=3:
        GATE=6;SCHEMA='v3_live_geometry_execution_card_v1';SLUG='gse_live_geometry_execution_v1'
        SCOPE=dict(SCOPE,control_published=True,execution_policy=dict(arrival_radius_m=.5,target_timeout_s=30,revisit_radius_m=1))
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11'
        NAME='gse-live-geometry-execution-v1'
    if version>=4:
        SCHEMA='v3_live_geometry_execution_card_v2';SLUG='gse_live_geometry_execution_v2'
        SCOPE=dict(SCOPE,rigid_motion=True)
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-live-geometry-execution-v2'
    if version==5:
        SLUG='gse_live_geometry_execution_v3'
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-live-geometry-execution-v3'
    if version>=6:
        SCHEMA='v3_live_geometry_execution_card_v3';SLUG='gse_branch_geometry_execution_v1'
        SCOPE=dict(SCOPE,branch_policy=dict(direction_tolerance_deg=15,target_match_m=2,place_radius_m=4,support_observations=3))
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-branch-geometry-execution-v1'
    if version>=7:
        SCHEMA='v3_live_geometry_execution_card_v4';SLUG='gse_branch_geometry_return_v1'
        SCOPE=dict(SCOPE,remote_returns=True)
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-branch-geometry-return-v1'
    if version==8:
        SLUG='gse_branch_geometry_return_binding_v1'
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-branch-geometry-return-binding-v1'
    if version==9:
        SLUG='gse_branch_geometry_direct_return_v1'
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-branch-geometry-direct-return-v1'
    if version>=10:
        from mtare_topo.governance_live_geometry_shadow import LEARNED_SCOPE
        SCHEMA='v3_live_geometry_execution_card_v5';SLUG='gse_learned_geometry_execution_v1'
        SCOPE=dict(SCOPE,learned=LEARNED_SCOPE)
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-learned-geometry-execution-v1'
    if version==11:
        SLUG='gse_learned_geometry_execution_v1r1'
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-learned-geometry-execution-v1r1'
    if version==12:
        SLUG='gse_learned_geometry_execution_v1r2'
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-learned-geometry-execution-v1r2'
    if version>=13:
        SCHEMA='v3_live_geometry_execution_card_v6';SLUG='gse_learned_constrained_execution_v1'
        SCOPE=dict(SCOPE,learned=dict(SCOPE['learned'],task_interface='observed_coordinates_required_learned_segments_v1',direction_tolerance_deg=15))
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-learned-constrained-execution-v1'
    if version==14:
        SCHEMA='v3_live_geometry_execution_card_v7';SLUG='gse_learned_constrained_memory_v1'
        SCOPE=dict(SCOPE,branch_policy=dict(SCOPE['branch_policy'],retain_matching_partial_observations=True))
        CARD='configs/v3/gate6/data_cards/'+SLUG+'.json';SPEC='configs/v3/gate6/'+SLUG+'.json'
        RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed11';NAME='gse-learned-constrained-memory-v1'

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,obj,mode='x'):
    with p.open(mode) as f:json.dump(obj,f,indent=2,sort_keys=True,allow_nan=False)

def freeze():
    auth=dict(status='APPROVED',approved_by='user-standing-development-authorization',approved_at='2026-09-10',
        scope='One tunnel stationary live ROS geometry diagnostic; no control/training/protected data',
        authorized_operations=['shadow'],authorized_gates=[5],
        confirmation_reference='User requests full development system, authorizes autonomous in-scope implementation; live integration is a prerequisite, not research Gate PASS.')
    card=dict(schema_version=SCHEMA,card_id=SLUG,scope=SCOPE,approval=auth)
    if VERSION>=3:
        auth.update(scope='One tunnel geometry goal execution diagnostic through unchanged local planner; no training/protected data',authorized_operations=['closed_loop_single'],authorized_gates=[6])
    if VERSION>=13:
        auth.update(scope='User-confirmed observed target coordinates with required frozen learned geometry constraints; one tunnel episode, no training',
            confirmation_reference='User supports retained geometry-to-structure-to-topology direction and explicitly asks to continue after proposed responsibility split')
    assert validate_card(card).passed
    (ROOT/CARD).parent.mkdir(parents=True,exist_ok=True);write(ROOT/CARD,card)
    sources=list((ROOT/'src/mtare_topo').rglob('*.py'))
    sources += [ROOT/p for p in [CARD,'tools/v3/run_live_geometry_shadow.py',
        'tools/v3/live_geometry_shadow_trial.py','tools/v3/live_geometry_shadow_node.py',
        'configs/v3/gate5/roslaunch/system_seeded.launch',
        'configs/v3/gate5/roslaunch/vehicle_simulator_seeded.launch']]
    if VERSION>=10:sources += [ROOT/'tools/v3/serve_live_learned_geometry.py',ROOT/'tools/v3/_bootstrap.py']
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=GATE,date='20260910',slug=SLUG,
        seed=11,operation='closed_loop_single' if VERSION>=3 else 'shadow',data_card=CARD,config_path=CARD,user_authorization=auth,
        command=['python3','tools/v3/run_live_geometry_shadow.py','--execute','--version',str(VERSION)],
        question='Can actual tunnel ROS scans produce causal map-frame geometry and target diagnostics?',
        method=('Observed target coordinates with mandatory frozen learned segment direction support; ' if VERSION>=13 else ('Frozen learned geometry, same independent current sectors and original backend; ' if VERSION>=10 else 'Existing organized sensor operator and finite geometry frontend; '))+('tentative XY requests through original local planner' if VERSION>=3 else 'no waypoint publisher'),
        baseline='Integration only, not learned superiority, semantic graph or closed-loop qualification.',
        estimated_cost=dict(compute='CPU only, one episode, memory 8GiB, 240s wall timeout',disk_gb=2,wall_time_hours=1/15),
        acceptance_criteria=['At least five accepted live scans with source/TF and bag evidence; no process failure.',
            'At least one geometry-place task XY arrival' if VERSION>=6 else 'No geometry-place task acceptance requested',
            'At least one recorded return and current-geometry revalidation' if VERSION>=7 else 'No remote return required',
            'Measured travel >=1m and >=1 XY arrival; no inferred semantic graph success.' if VERSION>=3 else 'No motion targets published by new frontend; unknown geometry not declared traversable.'],
        expected_evidence=['raw bag, geometry records, snapshot, process logs, summary, seal'],
        image=IMAGE,input_sha256={SCOPE['learned']['checkpoint']:SCOPE['learned']['checkpoint_sha256']} if VERSION>=10 else {},
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources}))
    print(SPEC)

def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text())
    if json.loads((run/'RUN_STATE.json').read_text())['state']!='CREATED_NOT_EXECUTED':
        raise ValueError('fresh run required')
    if json.loads((run/'config/run_spec.json').read_text())!=spec:raise ValueError('spec drift')
    for p,h in dict(spec['source_sha256'],**spec.get('input_sha256',{})).items():
        if sha(ROOT/p)!=h:raise ValueError('source drift: '+p)
    if not validate_card(json.loads((ROOT/CARD).read_text())).passed:raise ValueError('card drift')
    evidence=run/'artifacts'
    write(evidence/'policy.json',dict(composition_policy=dict(max_residual_m=.01,
        min_crossing_sine=.1,endpoint_tolerance_m=1e-8,maximum_candidates=32),
        anchor_spacing_m=4.,lookahead_m=4.,rigid_motion=VERSION>=4,**({'constrained_observed_tasks':True} if VERSION>=13 else {})))
    if VERSION>=3:write(evidence/'execution_policy.json',SCOPE['execution_policy'])
    if VERSION>=6:write(evidence/'branch_policy.json',SCOPE['branch_policy'])
    command=['docker','run','--rm','--name',NAME,'--network','none','--memory','8g','--cpus','4',
        '--user',str(os.getuid())+':'+str(os.getgid()),
        '--env','ROS_HOME=/tmp/ros','--env','ROS_HOSTNAME=localhost',
        '--env','ROS_MASTER_URI=http://localhost:11311','--env','GAZEBO_MODEL_DATABASE_URI=',
        '--env','PYTHONDONTWRITEBYTECODE=1','--env','OMP_NUM_THREADS=1','--env','OPENBLAS_NUM_THREADS=1']
    command+=['--env','GSE_POSE_INTERFACE='+('aee-commanded-odometry' if VERSION>=2 else 'tf')]
    if VERSION>=3:command+=['--env','GSE_EXECUTION=1']
    if VERSION>=6:command+=['--env','GSE_BRANCHES=1']
    if VERSION>=7:command+=['--env','GSE_RETURNS=1']
    for path in ['src','tools/v3','configs/v3/gate5/roslaunch']:
        command+=['--mount',f'type=bind,src={ROOT/path},dst=/workspace/{path},readonly']
    command+=['--mount',f'type=bind,src={evidence},dst=/evidence',IMAGE,'bash','-c',
        'source /opt/ros/noetic/setup.bash && source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && export PYTHONPATH=/workspace/src:$PYTHONPATH && python3 /workspace/tools/v3/live_geometry_shadow_trial.py']
    worker=None;worker_log=None;ipc=None
    if VERSION>=10:
        ipc=tempfile.TemporaryDirectory(prefix='gse-learned-ipc-')
        # Inject options before Docker image; no network namespace sharing.
        at=command.index(IMAGE)
        command[at:at]=['--mount',f'type=bind,src={ipc.name},dst=/inference,readonly',
            '--env','GSE_LEARNED_SOCKET=/inference/model.sock','--env','GSE_CHECKPOINT_SHA256='+SCOPE['learned']['checkpoint_sha256']]
    write(run/'logs/actual_command.json',command)
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    start=time.monotonic();error=None;code=None
    try:
        if VERSION>=10:
            py='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'
            worker_command=[py,'tools/v3/serve_live_learned_geometry.py','--checkpoint',str(ROOT/SCOPE['learned']['checkpoint']),
                '--sha256',SCOPE['learned']['checkpoint_sha256'],'--socket',ipc.name+'/model.sock','--output',str(evidence/'model')]
            write(run/'logs/model_command.json',worker_command)
            worker_log=(run/'logs/model.log').open('xb')
            worker=subprocess.Popen(worker_command,cwd=ROOT,stdout=worker_log,stderr=subprocess.STDOUT,
                env=dict(os.environ,PYTHONPATH=str(ROOT/'src'),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1'))
            ready_deadline=time.monotonic()+60
            while not (evidence/'model/ready.json').exists():
                if worker.poll() is not None:raise RuntimeError('model worker startup failed')
                if time.monotonic()>ready_deadline:raise TimeoutError('model startup timeout')
                time.sleep(.1)
        with (run/'logs/docker.log').open('xb') as log:
            code=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=240).returncode
        if code:raise RuntimeError('container exit '+str(code))
    except Exception as exc:
        error=repr(exc)
        if isinstance(exc,subprocess.TimeoutExpired):
            subprocess.run(['docker','stop','--time','20',NAME],capture_output=True,timeout=30)
    finally:
        if worker is not None:
            if worker.poll() is None:worker.terminate()
            try:worker.wait(timeout=5)
            except subprocess.TimeoutExpired:worker.kill();worker.wait()
        if worker_log is not None:worker_log.close()
        if ipc is not None:ipc.cleanup()
        trial=evidence/'trial_summary.json'
        detail=json.loads(trial.read_text()) if trial.exists() else {}
        result=dict(status='GATE_FAIL' if error or detail.get('error') else 'GATE_MIXED',
            error=error,trial=detail,exit_code=code,elapsed_s=time.monotonic()-start,
            closed_loop=VERSION>=3,complete_exploration_verified=False,method_advantage_proven=False)
        write(run/'metrics/summary.json',result)
        write(run/'RUN_STATE.json',dict(state='FAILED' if result['status']=='GATE_FAIL' else 'COMPLETED'),'w')
        with (evidence/'evidence_sha256.txt').open('x') as f:
            for p in sorted(run.rglob('*')):
                if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print(json.dumps(result));return int(result['status']=='GATE_FAIL')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');p.add_argument('--version',type=int,choices=[1,2,3,4,5,6,7,8,9,10,11,12,13,14],default=1);a=p.parse_args()
    configure_version(a.version)
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('select action')
