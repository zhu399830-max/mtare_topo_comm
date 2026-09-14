"""One bounded frozen-input inference audit; never launches simulator/training."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from mtare_topo.governance_learned_pair import validate_card

SLUG='gse_learned_geometry_pair_v1'
CARD='configs/v3/gate6/data_cards/'+SLUG+'.json'
SPEC='configs/v3/gate6/'+SLUG+'.json'
RUN='results/gate6_single_robot/gate6_20260910_'+SLUG+'_seed0'
PY='/home/zeng-workstation/.local/share/mtare_topo_comm/envs/phase3_torch290_cu129_zarr2187_v1/bin/python'
IMAGE='sha256:5cbede29e4229e92d85ebaf919be2fb9ebc4ad9ed97029b7639997779062748c'


def sha(path):
    d=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(8388608),b''):d.update(b)
    return d.hexdigest()


def write(path,value,mode='x'):
    with path.open(mode) as f:json.dump(value,f,indent=2,allow_nan=False)


def freeze():
    old=ROOT/'configs/v3/gate6/data_cards/gse_learned_geometry_development_inputs_v1.json'
    card=json.loads(old.read_text());card['schema_version']='gse_learned_geometry_pair_audit_v1'
    card['approval'].update(authorized_operations=['audit'],authorized_gates=[6],
        scope='Bounded existing-input decode, frozen seed0 inference and paired given-pose replay; no training/control')
    assert validate_card(card).passed
    write(ROOT/CARD,card)
    sources=list((ROOT/'src/mtare_topo').rglob('*.py'))+[ROOT/p for p in [CARD,str(old.relative_to(ROOT)),
        'tools/v3/run_learned_geometry_pair.py','tools/v3/decode_learned_geometry_bag.py','tools/v3/infer_learned_geometry_pair.py']]
    write(ROOT/SPEC,dict(schema_version='v3_run_spec_v1',gate=6,date='20260910',slug=SLUG,seed=0,
        operation='audit',data_card=CARD,config_path=CARD,command=['python3','tools/v3/run_learned_geometry_pair.py','--execute'],
        question='Can frozen predicted geometry enter the same given-pose graph pipeline on all 290 existing windows?',
        method='Frozen Observable seed0, original 0.5 existence threshold, full rigid coordinates, independent current sectors',
        baseline='Same-input nonlearning fitter and shared recorded-pose graph; no semantic GT scores',
        estimated_cost=dict(compute='CPU-only 1 inference thread; 4GiB decoder container; 20min timeout',disk_gb=8,wall_time_hours=1/3),
        acceptance_criteria=['Exactly 294 decoded source frames and 290 windows per method','No input or checkpoint drift, training, teacher or control',
            'Preserve raw predictions, paired records, graph states and failures; task completion is not method superiority'],
        expected_evidence=['command, environment, raw outputs, paired graphs, logs, summary, seal'],
        source_sha256={str(p.relative_to(ROOT)):sha(p) for p in sources},input_sha256=card['scope']['source_sha256'],image=IMAGE))
    print(SPEC)


def execute():
    run=ROOT/RUN;spec=json.loads((ROOT/SPEC).read_text());card=json.loads((ROOT/CARD).read_text())
    assert json.loads((run/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    assert json.loads((run/'config/run_spec.json').read_text())==spec
    assert validate_card(card).passed
    for p,h in dict(spec['source_sha256'],**spec['input_sha256']).items():
        if sha(ROOT/p)!=h:raise ValueError('source/input drift: '+p)
    out=run/'artifacts';env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONPATH=str(ROOT/'src'))
    commands=[];start=time.monotonic();error=None
    write(run/'RUN_STATE.json',dict(state='RUNNING'),'w')
    try:
        check=[PY,'-c',"import torch,numpy,json; assert torch.__version__=='2.9.0+cu129'; print(json.dumps(dict(torch=torch.__version__,numpy=numpy.__version__,cuda=torch.cuda.is_available())))"]
        environment=subprocess.check_output(check,env=env,text=True);write(out/'runtime_environment.json',json.loads(environment))
        decode=['docker','run','--rm','--name','gse-learned-pair-decode-v1','--network','none','--memory','4g','--cpus','2',
            '--user',str(os.getuid())+':'+str(os.getgid()),'--env','PYTHONDONTWRITEBYTECODE=1','--env','OPENBLAS_NUM_THREADS=1',
            '--mount',f'type=bind,src={ROOT},dst=/workspace,readonly',
            '--mount',f'type=bind,src={out},dst=/output',IMAGE,'bash','-c',
            'source /opt/ros/noetic/setup.bash && export PYTHONPATH=/workspace/src:$PYTHONPATH && /usr/bin/python3 /workspace/tools/v3/decode_learned_geometry_bag.py --root /workspace --output /output/inputs.npz']
        inference=[PY,'tools/v3/infer_learned_geometry_pair.py','--input',str(out/'inputs.npz'),'--output',str(out/'pair'),'--device','cpu']
        for name,command,timeout in [('decode',decode,180),('inference',inference,1020)]:
            commands.append(command);write(out/'commands.json',commands,'w')
            with (run/'logs'/f'{name}.log').open('x') as log:
                subprocess.run(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,check=True)
        summary=json.loads((out/'pair/summary.json').read_text())
        if any(x['observations']!=290 for x in summary['methods'].values()):raise ValueError('incomplete population')
    except Exception as exc:error=repr(exc)
    finally:
        write(run/'metrics/summary.json',dict(status='GATE_FAIL' if error else 'GATE_MIXED',error=error,elapsed_s=time.monotonic()-start,advantage_proven=False))
        write(run/'RUN_STATE.json',dict(state='FAILED' if error else 'COMPLETED'),'w')
        with (out/'evidence_sha256.txt').open('x') as f:
            for p in sorted(run.rglob('*')):
                if p.is_file() and p.name!='evidence_sha256.txt':f.write(sha(p)+'  '+str(p.relative_to(ROOT))+'\n')
    print((run/'metrics/summary.json').read_text());return int(error is not None)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--execute',action='store_true');a=p.parse_args()
    if a.freeze:freeze()
    elif a.execute:raise SystemExit(execute())
    else:p.error('action required')
