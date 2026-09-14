"""Partial terminal reference positions from exact source-bound cap returns."""
from copy import deepcopy
from .gse_terminal_diagnostic_v2 import diagnose_observation


def produce_terminal_reference_targets(bundle):
    diagnostic = diagnose_observation(bundle, range_error_bound_m=0.)
    anchors, provenance, unknown = [], [], []
    for row in diagnostic['terminal_references']:
        witnesses = row['accepted_witnesses']
        if not witnesses:
            unknown.append(dict(node_id_teacher_only=row['node_id_teacher_only'],
                reason='NO_UNIQUE_SOURCE_CAP_RETURN'))
            continue
        anchors.append(dict(position_m=list(row['anchor_current_sensor_m']),
            evidence='Partial construction terminal reference with exact unique-source cap first returns; not complete local annotation.'))
        provenance.append(dict(node_id_teacher_only=row['node_id_teacher_only'],
            endpoint_key_teacher_only=deepcopy(row['endpoint_key_teacher_only']),
            witnesses=deepcopy(witnesses),
            position_source='construction_terminal_reference_not_observed_center_estimate'))
    if len(anchors) > 32:
        raise OverflowError('terminal anchor capacity exceeded; no target truncation')
    record = dict(schema='gse_surface_observed_targets_v1', coordinate_frame='current_sensor_m',
        source_frame_indices=list(bundle['source']['frame_rows']), anchors=anchors,
        openings=[], membership=[], score_region=dict(center_m=[0.,0.,0.], radius_m=10.,
            anchors_complete=False, openings_complete=False,
            evidence='No complete-background certificate; unobserved references remain unknown.'))
    return dict(record=record, teacher_provenance=provenance, unknown_candidates=unknown,
        full_training_gate_eligible=False, human_reviewed=False, scientific_gate_pass=False)
