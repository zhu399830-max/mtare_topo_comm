from pathlib import Path
from .data.continuous_prediction_scope_v1 import compile_scope as compile_feature_scope
POLICY=dict(wall_time_s=1800,host_ram_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=2*1024**3,optimizer_steps=0)
from .governance_surface_selection import digest

SCHEMA='v3_continuous_predictions_card_v1'
SLUG='gse_continuous_predictions_v1'


def validate_card(card):
    from .governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}:
        return ValidationReport(False,('closed continuous prediction card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'data_export') or card['policy']!=POLICY:
        errors.append('frozen final-model inference-only policy required')
    try:
        expected=digest(compile_feature_scope(Path(__file__).resolve().parents[2]))
        if digest(card['scope'])!=expected or card['scope_sha256']!=expected:errors.append('scope drift')
    except Exception as e:errors.append(str(e))
    a=card['approval']
    if (not isinstance(a,dict) or a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
            or a.get('authorized_operations')!=['data_export'] or a.get('authorized_gates')!=[3]
            or not a.get('confirmation_reference')):errors.append('exact authority required')
    return ValidationReport(not errors,tuple(errors))
