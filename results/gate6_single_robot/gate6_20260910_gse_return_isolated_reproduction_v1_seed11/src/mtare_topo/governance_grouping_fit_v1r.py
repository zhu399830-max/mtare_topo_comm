from pathlib import Path
from mtare_topo.governance_grouping_fit_v1 import POLICY as ORIGINAL_POLICY
from mtare_topo.governance_surface_selection import digest
from mtare_topo.governance_surface_material import read_pinned

SCHEMA='v3_grouping_fit_card_v1r'
SLUG='gse_grouping_center_fit_v1r'
PARENT='results/gate3_semantics/gate3_20260909_gse_grouping_center_fit_v1_seed0'
PARENT_SEAL='25c28be34aee9ed8bc65f69e1917ed052080d1412ebf070d306a6e50bae9a1eb'
POLICY=dict(ORIGINAL_POLICY,initialization='only3center_residual_rows_zero',correction_count=1,
    correction_evidence='initial452/512boundary;final409/512;allcandidate1mcoverage1/12;finite1000pointgradients;branchgradients0',
    parent_run=PARENT,parent_seal_sha256=PARENT_SEAL,no_second_correction=True)


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
        if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('wrong corrective operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]):errors.append('source or selection drift')
        if card['scope_sha256']!=digest(card['scope']) or card['policy']!=POLICY:errors.append('policy drift')
        a=card['approval']
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
                or a.get('authorized_operations')!=['training'] or a.get('authorized_gates')!=[3]
                or not a.get('confirmation_reference')):errors.append('exact corrective authority required')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
