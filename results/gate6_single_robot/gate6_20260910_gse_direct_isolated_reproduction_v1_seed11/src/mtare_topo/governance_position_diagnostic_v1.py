"""Exact same16, decoder then fixed-position then conditional dynamic-position."""
from pathlib import Path
import json
import ast
import zipfile
from mtare_topo.governance_surface_material import read_pinned
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_grouping_fit_v1 import POLICY as BASE

SCHEMA='v3_position_diagnostic_card_v1'
PARENT='results/gate3_semantics/gate3_20260909_gse_grouping_center_supervision_v2_seed0'
SEAL='a62d0798e3bfe2ee15d88d2f336ecc854dbd8d5416af4b97feb7175d14f5acba'
AUTH='20260909 latest user: keep fixes; actual decoder range and independent regression tensors first; same16 PRIMITIVE fixed one-to-one position-only; only after restored localization restart dynamic position matching from same initial state; no existence loss/cost, branches/maps/seeds/search; append existing report.'
ZERO_UPDATE_RUN='results/gate3_semantics/gate3_20260909_gse_primitive_position_fixed_v1_seed0'


def slug(mode):
    if mode not in ('fixed','dynamic'):raise ValueError('unknown diagnostic')
    return 'gse_primitive_position_'+mode+'_v1r'


def policy(mode):
    return dict(BASE,method='PRIMITIVE',stage='position_only_diagnostic',pair_models=[],
        loss='matched euclidean position mean/10 only; no existence or branch loss',
        assignment=('frozen step0 geometry Hungarian' if mode=='fixed' else 'current geometry Hungarian only'),
        existence_matching_cost=0,existence_loss=0,initialization='exact sealed parent step0000 state',
        positive_negative='no negatives supervised in position-only diagnostic; existing4m repair code retained',
        authorization=AUTH,mode=mode,final_assigned_within1m_required=11,reference_count=12,
        tensor_inverse_tolerance_m=1e-5,tensor_final_tolerance_m=.1,tensor_updates=1000,
        known_detection_claim=False,method_advantage_claim=False)


def sealed_index(root,run,seal_sha):
    data=read_pinned(root,run+'/artifacts/evidence_sha256.txt',seal_sha).decode()
    return {p:h for h,p in (s.split('  ',1) for s in data.splitlines())}


def compile_scope(root,mode):
    from grouping_fit_scope_v1 import compile_scope as original
    from surface_features_v1 import sha
    scope=original(root);index=sealed_index(root,PARENT,SEAL)
    card=json.loads(read_pinned(root,PARENT+'/config/data_card.json',index[PARENT+'/config/data_card.json']))
    for k in ('selection','selected_rows','counts','spacing','split'):
        if card['scope'][k]!=scope[k]:raise ValueError('same16 scope drift')
    bound={p:h for p,h in index.items() if p.startswith(PARENT+'/artifacts/input_')
        or p.startswith(PARENT+'/artifacts/prediction_0000_')
        or p in {PARENT+'/checkpoints/step_0000.pt',PARENT+'/config/schedule.json',PARENT+'/config/run_spec.json'}}
    for p,h in bound.items():read_pinned(root,p,h)
    spec=json.loads(read_pinned(root,PARENT+'/config/run_spec.json',index[PARENT+'/config/run_spec.json']))
    # Decoder, forward, initialization, fixed reader and repaired supervision remain byte-identical.
    common={p:h for p,h in spec['source_sha256'].items() if p.startswith((
        'src/mtare_topo/representation/','src/mtare_topo/data/','src/mtare_topo/teacher/'))
        or p=='tools/v3/bidirectional_paired_reader_v1.py'}
    for p,h in common.items():read_pinned(root,p,h)
    # Only SelectedReader is reused from the old executor. The later SPATIAL
    # wrapper changed that file's other functions, already documented previously.
    with zipfile.ZipFile(root/PARENT/'artifacts/source_snapshot.zip') as z:
        old=ast.parse(z.read('tools/v3/grouping_center_fit_v1.py'))
    new=ast.parse((root/'tools/v3/grouping_center_fit_v1.py').read_text())
    def reader(tree):return ast.dump(next(n for n in tree.body if getattr(n,'name',None)=='SelectedReader'))
    if reader(old)!=reader(new):raise ValueError('actual reused SelectedReader drift')
    scope.update(parent_run=PARENT,parent_seal_sha256=SEAL,parent_bound_sha256=bound,
        unchanged_common_source_sha256=common,reused_selected_reader_ast_equal=True,
        teacher_boundary='Targets from original loss_only only; no new label, candidate or GT forward input.',
        sampling_unit='same5 independent physical structures/5 parent maps;16 observations,12 positive references,4 zero positives,80 frame occurrences68 distinct variant frames')
    zero_seal=sha(root/ZERO_UPDATE_RUN/'artifacts/evidence_sha256.txt')
    zi=sealed_index(root,ZERO_UPDATE_RUN,zero_seal)
    zero=json.loads(read_pinned(root,ZERO_UPDATE_RUN+'/metrics/summary.json',zi[ZERO_UPDATE_RUN+'/metrics/summary.json']))
    check=json.loads(read_pinned(root,ZERO_UPDATE_RUN+'/metrics/decoder_tensor_check.json',zi[ZERO_UPDATE_RUN+'/metrics/decoder_tensor_check.json']))
    if zero['optimizer_steps']!=0 or not check['passed'] or 'schedule drift' not in zero['error']:
        raise ValueError('only explicit zero-update list/tuple correction permitted')
    scope['decoder_prerequisite']=dict(run=ZERO_UPDATE_RUN,seal_sha256=zero_seal,
        tensor_sha256=zi[ZERO_UPDATE_RUN+'/metrics/decoder_tensor_check.json'],optimizer_updates=0,
        correction='compare JSON list schedule to list-normalized original tuple; no order or budget change')
    if mode=='dynamic':
        run='results/gate3_semantics/gate3_20260909_'+slug('fixed')+'_seed0'
        seal=sha(root/run/'artifacts/evidence_sha256.txt');idx=sealed_index(root,run,seal)
        summary=json.loads(read_pinned(root,run+'/metrics/summary.json',idx[run+'/metrics/summary.json']))
        if summary['status']!='POSITION_PASS' or summary['final']['within1m']<11:
            raise ValueError('fixed assignment has not restored localization; dynamic prohibited')
        scope['fixed_prerequisite']=dict(run=run,seal_sha256=seal,summary_sha256=idx[run+'/metrics/summary.json'])
    return scope


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        mode=card['policy']['mode'];root=Path(__file__).resolve().parents[2]
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,slug(mode),'training'):errors.append('wrong operation')
        if card['scope']!=compile_scope(root,mode) or card['scope_sha256']!=digest(card['scope']):errors.append('scope drift')
        if card['policy']!=policy(mode):errors.append('policy drift')
        a=card['approval']
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
            or a.get('authorized_operations')!=['training'] or a.get('authorized_gates')!=[3]
            or a.get('confirmation_reference')!=AUTH):errors.append('scope-specific authority missing')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
