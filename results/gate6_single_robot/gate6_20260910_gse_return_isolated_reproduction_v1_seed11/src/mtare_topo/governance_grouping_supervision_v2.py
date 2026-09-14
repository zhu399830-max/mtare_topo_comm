from pathlib import Path
from mtare_topo.governance_grouping_fit_v1 import POLICY as ORIGINAL_POLICY
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned

SCHEMA='v3_grouping_supervision_card_v2'
SLUG='gse_grouping_center_supervision_v2'
PARENT='results/gate3_semantics/gate3_20260909_gse_grouping_center_fit_v1r_seed0'
PARENT_SEAL='b08341236c3e52e1c70ed461c7c0c40fe950decdba3901cc2cea4cd1724d5e1d'
POLICY=dict(ORIGINAL_POLICY,initialization='only3center_residual_rows_zero',
    positive_negative='unchanged4m background PLUS observed unique confirmed training-instance duplicates; no unknown competitors',
    loss='same assignment,position/10,balanced presence; only duplicate negative mask extended',
    parent_run=PARENT,parent_seal_sha256=PARENT_SEAL,background_radius_m=4.,duplicate_radius_m=4.,
    authorization='20260909 latest user: audit4FN5unpenalizedFP; targeted supervision repair; same16/model/budget rerun; no4to1; unknown not background',
    paused=['main_comparison','branches','real_mapping'],representation_no_go=False,
    extra_sources=['tests/v3/unit/test_grouping_supervision_v2.py'])


def compile_scope(root):
    from grouping_fit_scope_v1 import compile_scope as original
    scope=original(root)
    read_pinned(root,PARENT+'/artifacts/evidence_sha256.txt',PARENT_SEAL)
    scope['parent_run']=dict(path=PARENT,seal_sha256=PARENT_SEAL)
    return scope


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    try:
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('wrong operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]):errors.append('source/selection drift')
        if card['scope_sha256']!=digest(card['scope']) or card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
                or a.get('authorized_operations')!=['training'] or a.get('authorized_gates')!=[3]
                or a.get('confirmation_reference')!=POLICY['authorization']):errors.append('exact new supervision authority required')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
