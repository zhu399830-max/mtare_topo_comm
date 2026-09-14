from pathlib import Path
from .data.multiview_teacher_scope_v1 import compile_scope
from .governance_surface_selection import digest

SCHEMA='v3_multiview_historical_teacher_card_v1'
SLUG='gse_multiview_historical_teacher_v1'


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','approval'}:
        return ValidationReport(False,('closed historical multiview teacher card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):
        errors.append('historical partial teacher export only')
    try:
        expected=digest(compile_scope(Path(__file__).resolve().parents[2]))
        if digest(card['scope'])!=expected or card['scope_sha256']!=expected:errors.append('exact teacher scope drift')
    except Exception as error:errors.append(str(error))
    approval=card['approval']
    if (not isinstance(approval,dict) or approval.get('status')!='APPROVED'
            or approval.get('scope_sha256')!=card['scope_sha256']
            or approval.get('authorized_operations')!=['data_export']
            or approval.get('authorized_gates')!=[3] or not approval.get('confirmation_reference')):
        errors.append('exact operation standing authority required')
    return ValidationReport(not errors,tuple(errors))
