"""Exact cached16 scope for the prospective localization interface and one fit."""
import json
from pathlib import Path
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_center_verifier_v1 import POLICY as PARENT_POLICY

SCHEMA='v3_localization_quality_card_v1'
PARENT='results/gate3_semantics/gate3_20260910_gse_candidate_center_verifier_v1_seed0'
SCORE='results/gate3_semantics/gate3_20260910_gse_fixed_candidate_scoring_v1_seed0'
SEALS={PARENT:'20fa4ccd4256b8f21376b7cb6baf227a91cb21b0943ed997ec48a678cc846eff',SCORE:'300b8ca66a275c40b40850f557b76ca709741f2ea324f0fe107d15f98a30af70'}
AUTH='20260910 user PLEASE IMPLEMENT plan: independent reference-relative localization quality; exact16/512; original4m background unchanged; source-based labels and unknown-high compatibility before one same21185head step0 1000full16update fit; no new model/graph.'
POLICY=dict(supervision='confirmed1m without conflict positive; retain original4m negative; observed local1m exclusion negative; confirmed4m context without competing reference and offcenter>1m negative; rest unknown; not physical background',
    limits=PARENT_POLICY['limits'],training=PARENT_POLICY['training'],architecture=PARENT_POLICY['architecture'],selection=PARENT_POLICY['selection'],
    evaluation='unchanged old1m and fixed4mcoverage1m,full512without GTselection;final1000afterNMS P/Reach>=.9; report original3/21/7 and newFPs',
    cache='sealed predecoder observation tokens and original512coordinates, no encoder/position refit or repeated float diagnosis',
    compatibility='ideal positive/negative scores; unknowns high enough to outrank positives; all outputs visible before/after original2mNMS; both evaluations P/R1 with no FP/FN',
    no_retry=True,no_graph=True)

def paths(mode):
    slug='gse_localization_quality_'+('audit_v1r2' if mode=='audit' else 'fit_v1')
    return slug,'configs/v3/gate3/data_cards/'+slug+'.json','configs/v3/gate3/'+slug+'.json','results/gate3_semantics/gate3_20260910_'+slug+'_seed0'

def compile_scope(root,mode):
    indexes={r:{p:h for h,p in (l.split('  ',1) for l in read_pinned(root,r+'/artifacts/evidence_sha256.txt',v).decode().splitlines())} for r,v in SEALS.items()}
    ix=indexes[PARENT];pc=PARENT+'/config/data_card.json';card=json.loads(read_pinned(root,pc,ix[pc]));bound={pc:ix[pc]}
    prefixes=('metrics/validity_inventory.json','metrics/evaluation_0000.json','metrics/evaluation_1000.json','artifacts/input_','artifacts/observation_cache_','checkpoints/step_0000.pt','metrics/summary.json')
    bound.update({p:h for p,h in ix.items() if any(p.startswith(PARENT+'/'+v) for v in prefixes)})
    bound.update({p:h for p,h in indexes[SCORE].items() if p.startswith(SCORE+'/artifacts/fixed_supervision_evidence_') or p==SCORE+'/metrics/evaluation_0000.json'})
    targets=card['scope']['full_forward_scope']['selected_rows']
    target_rows=[{k:r[k] for k in ('source','source_binding','target_path','target_sha256')} for r in targets]
    for r in target_rows:bound[r['target_path']]=r['target_sha256']
    if mode=='training':
        ar=paths('audit')[3];seal=(root/ar/'artifacts/evidence_sha256.txt').read_text()
        ai={p:h for h,p in (l.split('  ',1) for l in seal.splitlines())}
        summ=ar+'/metrics/summary.json';summary=json.loads(read_pinned(root,summ,ai[summ]))
        if summary['status']!='GATE_PASS' or not summary['compatibility_pass']:raise ValueError('audit prerequisite not passed')
        bound.update({p:h for p,h in ai.items() if p.startswith(ar+'/artifacts/quality_') or p==summ})
    return dict(counts=card['scope']['counts'],candidate_queries=512,spacing=card['scope']['spacing'],split='same16fit-only no C08-C10 or holdout',
        bound_sha256=bound,target_rows=target_rows,parent_seals=SEALS,parent_initial_checkpoint=PARENT+'/checkpoints/step_0000.pt',
        source='sealed exact observations, no wholeworld/grid regeneration',expected_labels=dict(positive=12,negative=180,unknown=320),
        expected_labels_are_not_forced=True,operation=mode)

def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        mode=card['operation'];assert mode in ('audit','training')
        if card['schema_version']!=SCHEMA or card['card_id']!=paths(mode)[0]:errors.append('wrong card')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2],mode) or digest(card['scope'])!=card['scope_sha256']:errors.append('scope drift')
        if card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if a['status']!='APPROVED' or a['scope_sha256']!=card['scope_sha256'] or a['confirmation_reference']!=AUTH:errors.append('approval drift')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
