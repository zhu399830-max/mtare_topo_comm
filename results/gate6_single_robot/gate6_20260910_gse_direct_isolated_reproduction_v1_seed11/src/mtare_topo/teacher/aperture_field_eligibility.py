"""Diagnostic evidence ledger, deliberately not a target or authorization adapter.

Sparse ray support establishes neither aperture centers nor complete instances.
Keep the loss interface's optional attributes separate from required detections.
"""


def field_eligibility(row):
    if row.get('score_pass') is not None:
        raise ValueError('unscored diagnostic row required')
    out = row.get('diagnostic_readout')
    if not row.get('observation_adapter_available') or not isinstance(out, dict):
        raise ValueError('source-bound sparse observation readout required')
    if out.get('training_labels_qualified') is not False or out.get('complete_instance_count') is not None:
        raise ValueError('diagnostic evidence must not claim qualified instances')
    missing_detection = 'NOT_ESTABLISHED: independent observable instance and position targets'
    fields = {
        'anchor_presence': missing_detection,
        'anchor_position': missing_detection,
        'opening_presence': missing_detection,
        'opening_position': missing_detection,
        'opening_direction': 'MASK_UNTIL_VALID_OPENING: component support is not a direction target',
        'opening_dimensions': 'MASK_UNTIL_MEASURED: no qualified width/height',
        'reachability': 'MASK: no bound physical root-reach reference',
        'membership': 'MASK: both endpoint instances and observed correspondence required',
    }
    return {
        'schema': 'aperture_field_eligibility_v1',
        'case_id': row['case_id'],
        'geometry_evidence': ('NUMERICAL_MESH_PROXY' if row['mesh_distance_is_numerical_proxy']
                              else 'FINITE_POLYGON_CELL_BOUNDS'),
        'supported_isolated_components': out['supported_isolated_components'],
        'supported_unresolved_sets': out['supported_unresolved_sets'],
        'loss_fields': fields,
        'complete_background_qualified': False,
        'full_detection_f1_qualified': False,
        'bounded_fit_qualified': False,
        'training_labels_created': 0,
        'next_required_evidence': 'Observable instance locations and independent scoring coverage; '
                                  'physical reachability is not required for masked structure-only loss.',
    }
