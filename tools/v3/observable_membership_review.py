"""Authorized separate reference review; no training or old-label mutation."""
from _bootstrap import PROJECT_ROOT as ROOT
import ai_branch_review as base
from run_short_observation_chain import sha,write
import argparse,hashlib,json,sys,shutil

NAME='gse_observable_membership_review_v1'
CARD=f'configs/v3/gate3/data_cards/{NAME}.json'
SPEC=f'configs/v3/gate3/{NAME}.json'
RUN=f'results/gate3_semantics/gate3_20260913_{NAME}_seed0'
OLD='results/gate3_semantics/gate3_20260913_gse_ai_branch_review_v1r1_seed0'

def freeze():
    card=json.loads((ROOT/'configs/v3/gate3/data_cards/gse_ai_branch_review_v1r1.json').read_text())
    s=card['scope'];s.update(output='Independent observed membership version: visible directional support, potentially shared junction surfaces, unknown; old labels immutable',
        review_rule='Apply one observation-only rule to all12, not just failed predictions; never infer one surface one branch or hidden connectivity',
        prior_prediction_exposure=True,reference_independence='New version only; nonblind AI interpretation, not independent gold',
        training_authorized=False)
    a=card['approval'];a.update(confirmation_reference='2026-09-13 user 允许 in response to separate observed-membership reference on original12, preserve old labels/results/unknown, no retraining',scope='Independent observation-membership reference review on exact existing12; no training',scope_sha256=hashlib.sha256(json.dumps(s,sort_keys=True).encode()).hexdigest())
    card['annotation']['prompt_reference']=s['review_rule'];card['annotation']['output_schema']='Per-observation visible support, shared/unknown reasons, optional ray-addressable membership; no hard negative from missing membership'
    from mtare_topo.governance import validate_data_card,validate_annotation_plan
    assert validate_data_card(card).passed and validate_annotation_plan(card).passed
    write(ROOT/CARD,card)
    pins={CARD:sha(ROOT/CARD)}
    for i,e in enumerate(s['entries']):
        for p in [e['student_path'],f'{base.CACHE}/artifacts/common_{i:02d}.npz',f'{OLD}/previews/observation_{i:02d}.png',f'{OLD}/previews/full_{i:02d}.png']:pins[p]=sha(ROOT/p)
    code=['tools/v3/observable_membership_review.py','tools/v3/ai_branch_review.py','src/mtare_topo/governance_branch_review.py']
    spec=dict(schema_version='v3_run_spec_v1',gate=3,date='20260913',slug=NAME,seed=0,operation='ai_annotation',data_card=CARD,user_authorization=a,
        command=[sys.executable,'tools/v3/observable_membership_review.py','--begin'],question='What observable directional membership is supported, shared or unknown on the existing12?',
        method='All12 nonblind AI review of original causal point clouds; separate membership reference',baseline='Old partial core reference retained unchanged',fallback='Abstain; no training authorization or automatic hard grouping labels',
        acceptance_criteria=['All12 reviewed under one rule','Each assertion refers to visible evidence','Unknown and shared surfaces not forced into hard negatives','Old labels/results unchanged'],
        expected_evidence=['Reused hash-verified original views','Raw AI reasoning','Versioned per-observation reference','Summary, environment, seal'],
        estimated_cost=dict(compute='CPU asset reuse and in-session AI review, zero training',host_ram_gb=4,gpu_vram_gb=0,disk_gb=.1,wall_time_hours=.25),input_sha256=pins,source_sha256={p:sha(ROOT/p) for p in code})
    write(ROOT/SPEC,spec);print(SPEC)

def begin():
    out=ROOT/RUN;s=json.loads((ROOT/SPEC).read_text())
    assert json.loads((out/'RUN_STATE.json').read_text())['state']=='CREATED_NOT_EXECUTED'
    for p,h in {**s['input_sha256'],**s['source_sha256']}.items():assert sha(ROOT/p)==h,p
    for i in range(12):
        for name in (f'observation_{i:02d}.png',f'full_{i:02d}.png'):shutil.copy2(ROOT/OLD/'previews'/name,out/'previews'/name)
    write(out/'config/review_environment.json',dict(python=sys.version,mode='in-session AI review; reused identical observation-only images; no background job'))
    write(out/'RUN_STATE.json',dict(state='RUNNING',activity='in-session observation review; no training'),'w')
    print('24 original views verified and copied; review active, no labels yet')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');p.add_argument('--begin',action='store_true');p.add_argument('--seal',action='store_true');a=p.parse_args()
    assert sum([a.freeze,a.begin,a.seal])==1
    if a.freeze:freeze()
    elif a.begin:begin()
    else:
        base.CARD=CARD;base.SPEC=SPEC;base.RUN=RUN;base.execute('seal')
