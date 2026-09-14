"""Assemble corrected sensor frames without promoting diagnostic supervision."""
import numpy as np
from .primitive_relation_sequences import causal_relative_odometry


def assemble_five_frame_bundle(*, case, indexed_frames, codebook):
    """Caller must verify sealed ray/source bindings before assembling frames.

    The learner gets ranges, validity and relative motion only. Reported
    surface source sets and absolute poses stay in a separate diagnostic role.
    No structure labels or training-eligibility claim are manufactured.
    """
    frames = list(indexed_frames)
    if [index for index, _ in frames] != list(range(5)):
        raise ValueError('exact causal frames 0..4 required; no sorting or padding')
    ids = [e['id'] for e in case['program']['edges']]
    if list(codebook.primitive_ids) != ids:
        raise ValueError('case/codebook source order mismatch')
    xyz = np.asarray(case['poses_world_m'], dtype=np.float64)
    yaw = np.asarray(case['yaw_deg'], dtype=np.float64)
    motion = causal_relative_odometry(xyz, yaw)
    scans = [frame for _, frame in frames]
    codes = np.stack([s.primitive_membership_code for s in scans])
    if codes.shape != (5, 16, 720):
        raise ValueError('wrong scan shape')
    codebook.decode(codes)  # all referenced codes must be defined
    ranges = np.stack([s.range_m for s in scans])
    valid = np.stack([s.valid_mask for s in scans])
    if not np.array_equal(codes == 0, valid == 0):
        raise ValueError('membership/validity mismatch')
    student = dict(ranges_m=ranges, valid_mask=valid,
        relative_translation_current_sensor_m=motion.translation_current_sensor_m.astype(np.float32),
        relative_yaw_current_sensor_deg=motion.yaw_current_sensor_deg.astype(np.float32))
    return dict(schema_version='covered_five_frame_diagnostic_bundle_v1', student=student,
        diagnostic_only=dict(sensor_xyz_m=xyz.copy(), yaw_deg=yaw.copy(),
                             primitive_membership_code=codes, codebook=codebook.as_dict()),
        source=dict(case_id=case['case_id'], frame_indices=list(range(5)),
                    continuous_route_evidence=False),
        qualification=dict(distance_evidence='requires_external_seal_binding',
                           source_completeness_qualified=False, training_eligible=False,
                           structure_labels_present=False))
