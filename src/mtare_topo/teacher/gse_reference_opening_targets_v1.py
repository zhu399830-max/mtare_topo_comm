"""Partial source-reference opening supervision, not complete aperture truth.

Only the source-bound diagnostic entry point is public. A window section is
never a persistent graph anchor. Reference regression is explicitly conditional
on the construction; it does not certify unique inversion from sparse returns.
"""
import numpy as np

from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from .gse_window_opening_diagnostic_v1 import diagnose_observation


def produce_opening_reference_targets(bundle):
    return _produce_opening_reference_targets(bundle,diagnose_observation)


def _produce_opening_reference_targets(bundle,diagnostic_producer,*,require_axis_ownership=False):
    diagnostic = diagnostic_producer(bundle)  # verifies original source/motion
    sensor = bundle['sensor_teacher_only']
    origin = np.asarray(sensor['sensor_xyz_m'][-1], dtype=np.float64)
    yaw = float(sensor['yaw_deg'][-1])
    openings, provenance, unknown = [], [], []
    positions = set()
    for index, row in enumerate(diagnostic['proposals']):
        if require_axis_ownership and row.get('axis_reference_owned') is not True:
            unknown.append(dict(proposal_index=index,reason='NO_UNIQUE_AXIS_REFERENCE_CONTOUR',
                source_proposal=row))
            continue
        crossing = row['exclusive_outward_crossing_ray_indices']
        surface = row['surface_return_ray_indices']
        if not crossing or not surface:
            unknown.append(dict(proposal_index=index,
                reason='NO_EXCLUSIVE_CROSSING_OR_SURFACE_SUPPORT'))
            continue
        position = _current_sensor_transform(row['reference_position_m'], origin, yaw)
        direction = _current_sensor_transform(row['reference_direction'], np.zeros(3), yaw)
        if (position.shape != (3,) or direction.shape != (3,)
                or not np.isfinite(position).all() or not np.isfinite(direction).all()
                or not np.isclose(np.linalg.norm(direction), 1., rtol=0., atol=1e-12)):
            raise ValueError('invalid reference geometry')
        key = tuple(position)
        if key in positions:
            raise ValueError('coincident opening references need disambiguation; no silent folding')
        positions.add(key)
        openings.append(dict(position_m=position.tolist(), direction=direction.tolist(),
            width_m=None, height_m=None,
            evidence='Partial construction-reference window section with exclusive finite crossing and surface support; not uniquely inferred aperture geometry.'))
        provenance.append(dict(proposal_index=index,
            primitive_id_teacher_only=row['primitive_id_teacher_only'],
            reference_arc_m=row['reference_arc_m'],
            crossing_ray_indices=list(crossing), surface_ray_indices=list(surface),
            position_source='construction_window_reference_not_observed_center_estimate'))
    if len(openings) > 64:
        raise OverflowError('opening capacity exceeded; no target truncation')
    record = dict(schema='gse_surface_observed_targets_v1',
        coordinate_frame='current_sensor_m',
        source_frame_indices=list(bundle['source']['frame_rows']),
        anchors=[], openings=openings, membership=[[] for _ in openings],
        score_region=dict(center_m=[0., 0., 0.], radius_m=10.,
            anchors_complete=False, openings_complete=False,
            evidence='No complete-background certificate; unmatched predictions remain unknown.'))
    return dict(record=record, teacher_provenance=provenance,
        unknown_candidates=unknown, supervision_status='PARTIAL_REFERENCE_GEOMETRY_ONLY',
        full_training_gate_eligible=False, human_reviewed=False, scientific_gate_pass=False,
        missing_tasks=['opening_membership', 'complete_scoring_regions', 'reference_target_quality'])
