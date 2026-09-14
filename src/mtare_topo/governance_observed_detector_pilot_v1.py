"""Exact existing860 observation-query pilot, no pair-training authority."""
from pathlib import Path
from bidirectional_paired_scope_v1 import compile_scope
from mtare_topo.governance_surface_selection import digest

SCHEMA='v3_observed_detector_pilot_card_v1'
SLUG='gse_observed_detector_pilot_v1'
POLICY=dict(method='observed_query_no_explicit_relation',seed=0,scheduled_batches=500,
    microbatch=1,accumulation=4,optimizer='AdamW',learning_rate=.001,weight_decay=.0001,
    parameters=793992,initial_state_sha256='30ebe609e00b803fb9836a4bedbc2ef5d7b46fba145b9eabd4a79f8d38fde3d3',
    encoder='existing_frozen_context_cache',trainable='point_adapter_and_observed_detector',
    representation='r2',query_sampling='lexical_unique_observed_FPS32_no_padding',
    residual='20*tanh_per_axis_then_project10m',matching='distance_div10_minus_sigmoid',
    presence='equal_positive_negative_group_means_unknown_disconnected',branch='existing_balanced_selection_v1',
    position_radii_m=[4.,1.,2.],direction_angles_deg=[10.,5.,15.],anchor_probability=.5,branch_probability=.5,
    observations=860,fit_observations=282,calibration_observations=578,
    reference_anchors=562,reference_branches=1813,fit_anchors=186,fit_branches=598,
    initial_and_final_only=True,template='fit_prediction_slot_mean_actual_support',
    shuffle='seed0_canonical_source_sha_order_cyclic_derangement_fit_only',
    pilot_requirements=dict(anchor_recall=.9,branch_recall=.9,combined_known_fp_per_observation=.1,
                            template_advantage_m=.2,shuffle_recall_drop=.1),
    host_ram_bytes=32*1024**3,gpu_bytes=28*1024**3,output_bytes=30*1024**3,wall_time_s=43200,
    no_automatic_pair_training=True,no_retry=True,independent_evaluation=False,graph_claim=False)


def validate_card(card):
    from mtare_topo.governance import ValidationReport
    errors=[]
    if type(card) is not dict or set(card)!={'schema_version','card_id','operation','scope','scope_sha256','policy','approval'}:
        return ValidationReport(False,('closed pilot card required',))
    if (card['schema_version'],card['card_id'],card['operation'])!=(SCHEMA,SLUG,'training'):
        errors.append('observation-query500batch pilot only')
    try:
        expected=digest(compile_scope(Path(__file__).resolve().parents[2]))
        if digest(card['scope'])!=expected or card['scope_sha256']!=expected:errors.append('scope drift')
    except Exception as e:errors.append(str(e))
    if card['policy']!=POLICY:errors.append('pilot policy drift')
    a=card['approval']
    if (not isinstance(a,dict) or a.get('status')!='APPROVED' or a.get('scope_sha256')!=card['scope_sha256']
            or a.get('authorized_operations')!=['training'] or a.get('authorized_gates')!=[3]
            or not a.get('confirmation_reference')):errors.append('exact standing training authorization required')
    return ValidationReport(not errors,tuple(errors))
