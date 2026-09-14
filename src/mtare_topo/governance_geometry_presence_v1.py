from pathlib import Path
import json
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_grouping_supervision_v2 import POLICY as OLD_POLICY
from mtare_topo.governance_spatial_fit_v1 import PARENT,PARENT_SEAL,common_implementation

SCHEMA='v3_geometry_presence_card_v1';SLUG='gse_geometry_match_presence_a_v1'
DYNAMIC='results/gate3_semantics/gate3_20260909_gse_primitive_position_dynamic_v1r_seed0'
DYNAMIC_SEAL='32be39a02bc791d5386b4d081d873e2aa66d0a5a5c9f286a5d41c398beb6a63e'
AUTH='20260910 user: reuse passed dynamic position; original step0 A restores repaired existence loss with geometry-only matching; same1000sequence/weights/scoring; reuse equivalent original B; if fullfit passes retain protocol for fair validation, no more unique-root-cause blocker.'
POLICY=dict(OLD_POLICY,authorization=AUTH,matching_presence_cost=0,
    loss='unchanged position/10 plus equally balanced positive/negative presence group means; only matching confidence cost disabled',
    question='Can repaired existence loss with pure geometric assignment fit centers and detection on same16?',
    baseline='Reuse equivalent sealed original PRIMITIVE82022 as B (confidence matching cost1); zero B updates.',
    parent_run=PARENT,parent_seal_sha256=PARENT_SEAL,paused=True,
    extra_sources=['tests/v3/unit/test_geometry_match_presence_v1.py'])


def index(root,run,h):
    raw=read_pinned(root,run+'/artifacts/evidence_sha256.txt',h).decode()
    return {p:s for s,p in (r.split('  ',1) for r in raw.splitlines())}


def compile_scope(root):
    from grouping_fit_scope_v1 import compile_scope as original
    from surface_features_v1 import environment
    scope=original(root);idx=index(root,PARENT,PARENT_SEAL)
    def read(rel):return json.loads(read_pinned(root,PARENT+'/'+rel,idx[PARENT+'/'+rel]))
    old=read('config/data_card.json');spec=read('config/run_spec.json')
    for k in ('selection','selected_rows','counts','spacing','split'):
        if scope[k]!=old['scope'][k]:raise ValueError('B population mismatch')
    for k in ('seed','updates','evaluate_every','microbatch','accumulation','optimizer','learning_rate','weight_decay',
        'dropout','augmentation','relation_attributes','frozen_context','positive_negative','threshold','main_radius_m',
        'auxiliary_radii_m','initialization','background_radius_m','duplicate_radius_m'):
        if POLICY[k]!=old['policy'][k]:raise ValueError('B unequal configuration '+k)
    if spec['environment']!=environment():raise ValueError('B environment mismatch')
    dynamic=index(root,DYNAMIC,DYNAMIC_SEAL)
    summary=json.loads(read_pinned(root,DYNAMIC+'/metrics/summary.json',dynamic[DYNAMIC+'/metrics/summary.json']))
    if summary['status']!='POSITION_PASS':raise ValueError('pure position not passed')
    scope.update(b_reuse=dict(run=PARENT,seal_sha256=PARENT_SEAL,common_code=common_implementation(root,spec),
        source_sha256={p:h for p,h in idx.items() if p.startswith(PARENT+'/artifacts/input_') or p in {
            PARENT+'/checkpoints/step_0000.pt',PARENT+'/config/schedule.json',PARENT+'/metrics/summary.json'}},
        method_difference_only='matching -sigmoid(actual_logit) removed in A; loss weights and evidence masks unchanged',new_B_updates=0),
        pure_position_reused=dict(run=DYNAMIC,seal_sha256=DYNAMIC_SEAL,decoder_or_position_reruns=0))
    return scope


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('operation drift')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]) or card['scope_sha256']!=digest(card['scope']):errors.append('scope drift')
        if card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256'] or a.get('confirmation_reference')!=AUTH:errors.append('authority missing')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
