from pathlib import Path
from mtare_topo.governance_surface_selection import digest
from mtare_topo.data.bidirectional_partition_scope_v1 import compile_scope

SCHEMA='v3_bidirectional_partitions_card_v1'
SLUG='gse_bidirectional_partitions_v1r'


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','approval'}:
        return ValidationReport(False,('closed partition export card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export'):
        errors.append('partition export only')
    try:
        expected=digest(compile_scope(Path(__file__).resolve().parents[2]))
        if digest(card['scope'])!=expected or card['scope_sha256']!=expected:errors.append('scope drift')
    except Exception as e:errors.append(str(e))
    a=card['approval']
    if (not isinstance(a,dict) or a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
            or a.get('authorized_operations')!=['data_export'] or a.get('authorized_gates')!=[3]
            or not a.get('confirmation_reference')):errors.append('exact standing authorization required')
    return ValidationReport(not errors,tuple(errors))
