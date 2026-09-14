"""Single authorized same16/same-model/same1000 supervision repair/retest."""
import argparse
from pathlib import Path
import os
import subprocess
import numpy as np
import torch
import grouping_center_fit_v1 as executor
from grouping_supervision_audit_v2 import audit_step,case_outcomes
from mtare_topo.governance_grouping_supervision_v2 import SCHEMA,SLUG,POLICY,PARENT,PARENT_SEAL,compile_scope,validate_card
from mtare_topo.representation.grouping_zero_residual_init_v1 import build_model
from mtare_topo.representation.grouping_supervision_v2 import center_objective

original_evaluate=executor.evaluate
old_final=None


def verify_parent():
    root=executor.ROOT;parent=root/PARENT;seal=parent/'artifacts/evidence_sha256.txt'
    if executor.sha(seal)!=PARENT_SEAL:raise ValueError('old seal drift')
    files=set()
    for line in seal.read_text().splitlines():
        h,p=line.split('  ',1)
        if executor.sha(root/p)!=h:raise ValueError('old evidence drift: '+p)
        files.add((root/p).resolve())
    if files!={p.resolve() for p in parent.rglob('*') if p.is_file() and p!=seal}:raise ValueError('old evidence file set drift')
    return len(files)


def evaluate(model,examples,run,step):
    global old_final
    if step==0:
        tests=subprocess.run([executor.PYTHON,'-m','pytest','-q','tests/v3/unit/test_grouping_supervision_v2.py'],
            cwd=executor.ROOT,env=dict(os.environ,PYTHONPATH='src:tools/v3'),capture_output=True,text=True)
        (run/'logs/supervision_gradient_tests.log').write_text(tests.stdout+tests.stderr)
        if tests.returncode:raise RuntimeError('supervision gradient tests failed before training')
        count=verify_parent()
        checkpoint=torch.load(executor.ROOT/PARENT/'checkpoints/step_0000.pt',map_location='cpu',weights_only=False)
        if any(not torch.equal(v.cpu(),checkpoint['model'][k]) for k,v in model.state_dict().items()):
            raise ValueError('same zero-residual initialization required')
        del checkpoint
        old=[]
        for s in range(0,1001,100):
            audit=audit_step(examples,executor.ROOT/PARENT,s);old.append(audit)
            executor.write(run/'metrics'/f'parent_supervision_audit_{s:04d}.json',audit,'x')
            print('Parent loss/gradient audit step '+str(s),flush=True)
        old_final=old[-1]
        misses=[t for o in old_final['observations'] for t in o['targets'] if not t['matched_1m']]
        extras=[q for o in old_final['observations'] for q in o['queries']
            if q['metric_state_1m']=='FP' and not q['training_positive'] and not q['old_negative']]
        if len(misses)!=4 or len(extras)!=5:raise ValueError('registered4FN/5extras audit not reproduced')
        if any(not q['new_negative'] or q['old_presence_gradient']!=0 or q['new_presence_gradient']<=0
               or not q['observed_inside'] or q['unconfirmed_conflict'] for q in extras):
            raise ValueError('targeted duplicate correction not justified by all five original errors')
        for o in old_final['observations']:
            for q in o['queries']:
                if q['old_position_gradient']!=q['new_position_gradient']:
                    raise ValueError('position objective must stay unchanged')
        executor.write(run/'metrics/pretraining_audit_gate.json',dict(status='AUDIT_PASS',old_sealed_files=count,
            four_misses=misses,five_extras=extras,initial_state_equal=True,
            background_radius_unchanged_m=4.,main_radius_unchanged_m=1.,unknown_to_background=False),'x')
    result=original_evaluate(model,examples,run,step)
    audit=audit_step(examples,run,step)
    executor.write(run/'metrics'/f'supervision_audit_{step:04d}.json',audit,'x')
    if step==1000:
        cases=case_outcomes(old_final,audit)
        executor.write(run/'metrics/fixed_nine_error_outcomes.json',dict(cases=cases,
            all_final_errors=[dict(source=o['source'],missed=[t for t in o['targets'] if not t['matched_1m']],
                false_predictions=[q for q in o['queries'] if q['metric_state_1m']=='FP'],
                unknown_predictions=[q for q in o['queries'] if q['metric_state_1m']=='IGNORED']) for o in audit['observations']],
            comparison_paused=True,representation_no_go=False),'x')
    return result


def configure():
    executor.SCHEMA=SCHEMA;executor.SLUG=SLUG;executor.POLICY=POLICY
    executor.compile_scope=compile_scope;executor.validate_card=validate_card
    executor.build_model=build_model;executor.center_objective=center_objective;executor.evaluate=evaluate
    executor.CARD='configs/v3/gate3/data_cards/'+SLUG+'.json'
    executor.SPEC='configs/v3/gate3/'+SLUG+'.json'
    executor.RUN='results/gate3_semantics/gate3_20260909_'+SLUG+'_seed0'
    executor.SCRIPT='tools/v3/grouping_center_supervision_v2.py'


if __name__=='__main__':
    configure()
    p=argparse.ArgumentParser(__doc__);p.add_argument('--freeze',action='store_true');p.add_argument('--spec',type=Path);p.add_argument('--run-dir',type=Path);a=p.parse_args()
    if a.freeze:executor.freeze()
    elif a.spec and a.run_dir:raise SystemExit(executor.execute(executor.load_json(a.spec),a.run_dir))
    else:p.error('--freeze or --spec/--run-dir required')
