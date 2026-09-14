from pathlib import Path
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_grouping_fit_card_v1'
SLUG='gse_grouping_center_fit_v1'
POLICY=dict(seed=0,updates=1000,evaluate_every=100,microbatch=1,accumulation=4,
    optimizer='AdamW',learning_rate=.001,weight_decay=0.,dropout=0.,augmentation=False,
    stage='center_only',method='PRIMITIVE',pair_models=['SPATIAL','PRIMITIVE'],
    relation_attributes=False,frozen_context='same original sensor token pooling, both methods',
    positive_negative='original authenticated predicted-center exclusion, not source query position',
    loss='original center position/10 plus balanced presence; branch terms absent',
    threshold=.5,main_radius_m=1.,precision_required=.9,recall_required=.9,
    auxiliary_radii_m=[.5,2.,4.],host_ram_bytes=32*1024**3,gpu_bytes=28*1024**3,
    output_bytes=30*1024**3,wall_time_s=43200,no_automatic_pair_on_failure=True)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    from grouping_fit_scope_v1 import compile_scope
    errors=[]
    try:
        if card['schema_version']!=SCHEMA or card['card_id']!=SLUG or card['operation']!='training':errors.append('wrong operation')
        if card['scope']!=compile_scope(Path(__file__).resolve().parents[2]):errors.append('selection or source drift')
        if card['scope_sha256']!=digest(card['scope']) or card['policy']!=POLICY:errors.append('contract drift')
        a=card['approval']
        if (a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
                or a.get('authorized_operations')!=['training'] or a.get('authorized_gates')!=[3]
                or not a.get('confirmation_reference')):errors.append('missing exact authority')
    except Exception as e:errors.append(str(e))
    return ValidationReport(not errors,tuple(errors))
