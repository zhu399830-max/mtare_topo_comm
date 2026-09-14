"""Independent corrective proposal; original failed run is never restarted."""
from _bootstrap import PROJECT_ROOT as ROOT
import argparse,hashlib,json
from ai_junction_pilot import sha,write
from export_conditional_development_inputs import PYTHON
from mtare_topo.governance_conditional_roi_corrective import OLD,REUSE,validate_card
NAME='gse_conditional_fit_roi_corrective_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json';SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260911_{NAME}_seed0'

def prepare():
    old=json.loads((ROOT/OLD/'config/run_spec.json').read_text())
    c=json.loads((ROOT/OLD/'config/data_card.json').read_text())
    reuse=json.loads((ROOT/REUSE).read_text())
    s=dict(c['scope'],roi_policy='validated_frozen_student_roi_v1',reuse_cases=reuse['completed_cases'])
    a=dict(status='PENDING',authorized_gates=[3],authorized_operations=['data_export'],
           scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest(),
           scope='One corrected export,28complete cases reused and2852computed; same2880population and24hcap; no training',
           confirmation_reference='Specific corrective execution decision required under current PLAN; original no-retry run stays failed')
    write(ROOT/CARD,dict(schema_version='gse_conditional_roi_corrective_card_v1',scope=s,approval=a))
    pins=dict(old['input_sha256']);pins.pop(old['data_card'],None)
    pins.update({CARD:sha(ROOT/CARD),REUSE:sha(ROOT/REUSE),OLD+'/config/data_card.json':sha(ROOT/OLD/'config/data_card.json')})
    for case in reuse['completed_cases']:pins.update(case['artifacts'])
    spec=dict(old,slug=NAME,data_card=CARD,user_authorization=a,input_sha256=pins,
        command=['env','OPENBLAS_NUM_THREADS=1','OMP_NUM_THREADS=1','PYTHONPATH=src',PYTHON,'tools/v3/export_conditional_roi_corrective.py','--execute'],
        question='Does the unchanged conditional reference pipeline complete when input ROI comes from the frozen student observation?',
        method='Original matcher and compiler;validate frozen student ROI;copy28complete hash-bound cases with explicit origins;compute2852remaining',
        fallback='Seal new failure;no further automatic corrections or retries',
        source_sha256={str(p.relative_to(ROOT)):sha(p) for folder in ('src/mtare_topo','tools/v3') for p in (ROOT/folder).rglob('*.py')})
    write(ROOT/SPEC,spec);print(SPEC)

if __name__=='__main__':
    p=argparse.ArgumentParser();g=p.add_mutually_exclusive_group(required=True);g.add_argument('--prepare',action='store_true');g.add_argument('--execute',action='store_true');g.add_argument('--worker',type=int);a=p.parse_args()
    if a.prepare:prepare()
    elif a.worker is not None:
        from export_development_geometry import worker
        worker(a.worker,spec_path=SPEC,card_path=CARD,run_path=RUN)
    else:
        from export_development_geometry import execute
        raise SystemExit(execute(run_path=RUN,spec_path=SPEC,card_path=CARD,validator=validate_card,worker_script=str(__file__)))
