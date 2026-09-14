"""Exact307 partial-reference paired diagnostic, not complete detector authority."""
from pathlib import Path
from mtare_topo.data.development_paired_scope import compile_scope
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_block_relation_train_card_v1'
SLUG='gse_block_relation_train_v1'
POLICY=dict(methods=['c0','c1'],seed=0,updates_per_method=2000,microbatch=1,accumulation=4,
            optimizer='AdamW',learning_rate=.001,weight_decay=.0001,initial_and_final_only=True,
            parameters=789256,initial_state_sha256='118f9f420b7675fea79ae2b43f58c0d7a98b4b404ec4516b205f94ffa7329ab9',
            loss_reduction='mean of four observation-wise valid-target-normalized losses',
            anchor_probability=.5,branch_probability=.5,position_radii_m=[4.,1.,2.],direction_angles_deg=[10.,5.,15.],
            host_ram_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=30*1024**3,wall_time_s=43200,
            no_retry=True,whole_detection_eligible=False,graph_claim=False,
            branch_training_policy='direction_euclidean_minus_probability_balanced_groups_v1',
            node_training_policy='original_conditional_geometry_only',scoring_policy='original_unchanged',representation='r2',
            advance_condition='C1 development observed-reference branch recall at least0.05 above C0; known branchFP and unresolved predictions not higher; nodeTP not lower and known nodeFP not higher; report all parents and no graph/fullF1 claim')


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}:
        return ValidationReport(False,('closed paired training card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):errors.append('paired training only')
    try:
        expected=digest(compile_scope(Path(__file__).resolve().parents[2]))
        if digest(card['scope'])!=expected or card['scope_sha256']!=expected:errors.append('scope drift')
    except Exception as e:errors.append(str(e))
    if card['policy']!=POLICY:errors.append('paired policy drift')
    a=card['approval']
    if (not isinstance(a,dict) or a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
            or a.get('authorized_operations')!=['training'] or a.get('authorized_gates')!=[3]
            or not a.get('confirmation_reference')):errors.append('exact standing training authorization required')
    return ValidationReport(not errors,tuple(errors))

