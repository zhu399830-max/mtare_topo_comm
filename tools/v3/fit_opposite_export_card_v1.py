from pathlib import Path
from fit_opposite_input_scope_v1 import compile_scope
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_fit_opposite_input_export_card_v1'
SLUG='gse_fit_opposite_input_export_v1'


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','approval'}:
        return ValidationReport(False,('closed multiview missing input card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):
        errors.append('single six-field input export only')
    try:
        expected=digest(compile_scope(Path(__file__).resolve().parents[2]))
        if digest(card['scope'])!=expected or card['scope_sha256']!=expected:errors.append('exact scope drift')
    except Exception as e:errors.append(str(e))
    a=card['approval']
    if (not isinstance(a,dict) or a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
            or a.get('authorized_operations')!=['data_export'] or a.get('authorized_gates')!=[3]
            or not a.get('confirmation_reference')):errors.append('scope-bound standing authority required')
    return ValidationReport(not errors,tuple(errors))

