from pathlib import Path
import json
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_grouping_fit_v1 import POLICY as BASE

SCHEMA='v3_frozen_scoring_card_v1';SLUG='gse_fixed_candidate_scoring_v1'
PARENT='results/gate3_semantics/gate3_20260909_gse_primitive_position_dynamic_v1r_seed0'
SEAL='32be39a02bc791d5386b4d081d873e2aa66d0a5a5c9f286a5d41c398beb6a63e'
AUTH='20260910 user: reuse passed dynamic position final checkpoint; freeze all shared/position paths, retain all queries; only original existence head trained for original1000batches; fixed geometric assignments and repaired positive/negative/unknown rules; no A/B/decoder/position rerun; pass retain2stage protocol, fail stop fixed scoring without joint expansion/model search.'
POLICY=dict(BASE,authorization=AUTH,stage='fixed_candidate_existence_only',allow_empty_batches=True,
    initialization='sealed passed dynamic position step1000 model; score row inherited, no reset',
    loss='only original balanced repaired positive/confirmed-negative presence means; fixed geometry assignment; unknown no loss',
    positive_negative='unchanged4m background plus repaired unique observed duplicate candidates, frozen from fixed positions',
    effective_trainable_parameters=129,trainable='only original head.head.anchor row3 weight128 and bias1',
    feature_cache='detached original128D queries, all valid queries; full path bitwise verification every100',
    question='Can the original existence row correctly score frozen, well-localized candidates?',
    baseline='inherited pre-score checkpoint; A/B and pure-position results reused only, no retraining',paused=True)


def index(root):
    lines=read_pinned(root,PARENT+'/artifacts/evidence_sha256.txt',SEAL).decode().splitlines()
    return {p:h for h,p in (s.split('  ',1) for s in lines)}


def compile_scope(root):
    from grouping_fit_scope_v1 import compile_scope as original
    from surface_features_v1 import environment
    scope=original(root);idx=index(root)
    def read(rel):return json.loads(read_pinned(root,PARENT+'/'+rel,idx[PARENT+'/'+rel]))
    old=read('config/data_card.json');spec=read('config/run_spec.json')
    if read('metrics/summary.json')['status']!='POSITION_PASS':raise ValueError('parent position not passed')
    for k in ('selection','selected_rows','counts','spacing','split'):
        if old['scope'][k]!=scope[k]:raise ValueError('same16 mismatch')
    if environment()!=spec['environment']:raise ValueError('environment mismatch')
    common={p:h for p,h in spec['source_sha256'].items() if p.startswith((
        'src/mtare_topo/representation/','src/mtare_topo/data/','src/mtare_topo/teacher/'))}
    for p,h in common.items():read_pinned(root,p,h)
    bound={p:h for p,h in idx.items() if p.startswith(PARENT+'/artifacts/input_')
        or p.startswith(PARENT+'/artifacts/prediction_1000_') or p in {
            PARENT+'/checkpoints/step_1000.pt',PARENT+'/config/schedule.json',PARENT+'/metrics/summary.json'}}
    for p,h in bound.items():read_pinned(root,p,h)
    scope.update(parent=dict(run=PARENT,seal_sha256=SEAL,bound_sha256=bound),
        unchanged_model_teacher_source_sha256=common,expected_all_valid_queries=512,
        labels='Existing reference-derived fixed masks; no new teacher or GT in forward',
        candidate_counts='positive/negative/unknown inventory computed once from fixed512 before updates, never selected by score')
    return scope


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('wrong operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]) or card['scope_sha256']!=digest(card['scope']):errors.append('scope changed')
        if card['policy']!=POLICY:errors.append('policy changed')
        a=card['approval']
        if a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256'] or a.get('confirmation_reference')!=AUTH:errors.append('authority missing')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
