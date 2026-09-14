"""Single existing-development-world original M-TARE episode, not model success."""
from _bootstrap import PROJECT_ROOT as ROOT
from run_short_observation_chain import write
import hashlib,json,sys,subprocess,os,time,shlex,shutil,traceback,zipfile
NAME='gse_baseline_restart_v1r1';CARD=f'configs/v3/gate6/data_cards/{NAME}.json';SPEC=f'configs/v3/gate6/{NAME}.json';RUN=f'results/gate6_single_robot/gate6_20260913_{NAME}_seed11'
IMAGE='sha256:5cbede29e4229e92d85ebaf919be2fb9ebc4ad9ed97029b7639997779062748c';CONTAINER='gse-baseline-restart-20260913-v1r1';CASE='tunnel_seed11_original_mtare'
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024**2),b''):h.update(b)
    return h.hexdigest()
def freeze():
    assert not (ROOT/RUN).exists(), 'Cannot refreeze after run creation'
    s=dict(world='tunnel',robots=1,episodes=1,runtime_sec=600,seed=11,method='original_mtare',training_steps=0,test_claim=False,teacher_map_to_planner=False,
        population='One previously used development scene, one start (0,0,0,0), one600-second trajectory; raw scan count measured after recording, not known in advance',split='Development integration only; no procedural C08-C10 or protected test reads',teacher='None; scan-derived exploration-volume proxy only, not true90% coverage',
        image=IMAGE,limits=dict(memory_gb=8,cpus=4,wall_seconds=1800,output_bytes=10*1024**3),comparison='Historical32 records/30conditions retained with duplicates, not selected by score; new episode is current integration baseline')
    a=dict(status='APPROVED',approved_by='user',approved_at='2026-09-13',authorized_operations=['closed_loop_single'],authorized_gates=[6],scope='Original-system development baseline and subsequent explicitly qualified integration, no new training/test use',confirmation_reference='User 那调整方案继续做啊 after explicit baseline-first complete-loop proposal',scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    card=dict(schema_version='gse_baseline_restart_v1',scope=s,approval=a)
    from mtare_topo.governance_baseline_restart import validate_card
    assert validate_card(card).passed
    write(ROOT/CARD,card,'w')
    files=list((ROOT/'src/mtare_topo').rglob('*.py'))+list((ROOT/'tools/v3').glob('*.py'))+list((ROOT/'configs/v3/gate5/roslaunch').glob('*.launch'))+[ROOT/'configs/v3/gate5/closed_loop_recording_topics_v1.json',ROOT/CARD]
    spec=dict(schema_version='v3_run_spec_v1',gate=6,date='20260913',slug=NAME,seed=11,operation='closed_loop_single',data_card=CARD,user_authorization=a,
        command=['python3','tools/v3/run_gse_baseline_restart.py','--execute'],question='Can unchanged original M-TARE complete the current600-second development baseline with recorded sensor/motion/coverage evidence?',method='Existing seeded launch and case runner, no learned control',baseline='Original M-TARE; not the new learned method',fallback='Preserve failure and processes/logs, no automatic retry or outcome selection',estimated_cost=dict(compute='One docker CPU simulation4cores8GiB, no model GPU',host_ram_gb=8,gpu_vram_gb=0,disk_gb=10,wall_time_hours=.5),acceptance_criteria=['600sim seconds recorded','Existing case recording audit passed','Movement and waypoints measured; completion of exploration not assumed','Raw bag lossless archive verified','No model or GT planner injection'],expected_evidence=['Case contract,trajectory,waypoints,coverage-time curve,runtime,raw bag,logs','Source snapshot,environment,summary,seal'],image=IMAGE,source_sha256={str(p.relative_to(ROOT)):sha(p) for p in files})
    write(ROOT/SPEC,spec,'w');print(SPEC)
def execute():
    out=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    from mtare_topo.governance_baseline_restart import validate_card
    assert validate_card(json.loads((ROOT/CARD).read_text())).passed
    for p,h in spec['source_sha256'].items():assert sha(ROOT/p)==h,p
    assert shutil.disk_usage(out).free>12*1024**3
    assert subprocess.check_output(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],text=True).strip()==IMAGE
    (out/'artifacts/cases').mkdir()
    with zipfile.ZipFile(out/'artifacts/source_snapshot.zip','x',compression=zipfile.ZIP_DEFLATED) as z:
        for p in spec['source_sha256']:z.write(ROOT/p,p)
    inner=['python3','/workspace/tools/v3/run_mtare_single_robot_case_v1.py','--case-dir',f'/evidence/cases/{CASE}','--case-id',CASE,'--world','tunnel','--environment-seed','11','--method-id','original_mtare','--method-family','original_mtare','--execution-repeat','0','--block-id','development_tunnel11','--runtime-sec','600','--archive-mode','host','--host-uid',str(os.getuid()),'--host-gid',str(os.getgid())]
    shell='source /opt/ros/noetic/setup.bash && source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash && source /home/docker-user/mtare/tare_system/devel/setup.bash --extend && export PYTHONPATH=/workspace/src:$PYTHONPATH && '+shlex.join(inner)
    command=['docker','run','--rm','--network','none','--hostname','localhost','--name',CONTAINER,'--memory','8g','--cpus','4','-e','ROS_HOME=/tmp/ros','-e','GAZEBO_MODEL_DATABASE_URI=','-e','OPENBLAS_NUM_THREADS=1','-e','OMP_NUM_THREADS=1','-e','PYTHONDONTWRITEBYTECODE=1','-v',f'{ROOT}:/workspace:ro','-v',f'{out}/artifacts:/evidence:rw','--entrypoint','/bin/bash',IMAGE,'-lc',shell]
    write(out/'logs/actual_command.json',command);write(out/'config/execution_environment.json',dict(image=IMAGE,python=sys.version,host_uid=os.getuid(),host_gid=os.getgid(),runtime='Docker ROS Noetic, host has no ROS installation'))
    write(out/'RUN_STATE.json',dict(state='RUNNING',container=CONTAINER),'w');start=time.monotonic();error=None;case=None
    try:
        with (out/'logs/docker.log').open('xb') as log:
            p=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            write(out/'logs/process.json',dict(pid=p.pid,container=CONTAINER))
            while p.poll() is None:
                if time.monotonic()-start>1800 or sum(x.stat().st_size for x in out.rglob('*') if x.is_file())>10*1024**3:
                    subprocess.run(['docker','stop','--time','20',CONTAINER],check=False);p.wait(timeout=60);raise RuntimeError('wall or output resource cap')
                time.sleep(2)
            assert p.returncode==0,f'docker exit{p.returncode}'
        case_dir=out/f'artifacts/cases/{CASE}';case=json.loads((case_dir/'summary.json').read_text())
        assert case['status']=='PASS_SINGLE_ROBOT_CASE_V2_PENDING_HOST_ARCHIVE'
        from mtare_topo.evaluation.host_bag_archive import finalize_handed_off_bag
        storage=finalize_handed_off_bag(case_dir,expected_uid=os.getuid(),expected_gid=os.getgid(),expected_raw_sha256=case['storage']['original_bag_sha256'])
        write(out/'metrics/archive.json',storage)
    except Exception:error=traceback.format_exc();(out/'logs/failure.txt').write_text(error)
    write(out/'metrics/summary.json',dict(status='GATE_FAIL' if error else 'GATE_MIXED',baseline_execution_complete=error is None,learned_method_evaluated=False,complete_exploration_verified=False,case=case,error=error,elapsed_s=time.monotonic()-start))
    write(out/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
    with (out/'artifacts/evidence_sha256.txt').open('x') as f:
        for p in sorted(out.rglob('*')):
            if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print('FAILED' if error else 'COMPLETED',flush=True)
    if error:raise RuntimeError(error)
if __name__=='__main__':
    assert len(sys.argv)==2 and sys.argv[1] in ['--freeze','--execute']
    freeze() if sys.argv[1]=='--freeze' else execute()
