"""Five-frame source-bound terminal reference diagnostic, no model labels."""
import numpy as np

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from mtare_topo.data.primitive_relation_materialization import load_p1a_realized_construction
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_portal_ray_evidence import CausalRaySegments
from .gse_terminal_reference_v2 import terminal_reference_evidence
from .source_witness_binding import source_permission


def diagnose_observation(bundle, *, range_error_bound_m):
    sensor, student = bundle['sensor_teacher_only'], bundle['student']
    document, book = bundle['construction_teacher_only'], bundle['codebook_teacher_only']
    source = bundle['source']
    verify_alignment(sensor, student, document, book, source)
    frames = np.asarray(source['frame_rows'])
    if (frames.shape != (5,) or frames.dtype.kind not in 'iu'
            or (frames < 0).any() or not (np.diff(frames) > 0).all()):
        raise ValueError('five increasing causal frame rows required')
    ranges, valid = student['ranges_m'], student['valid_mask']
    if ranges.shape != (5, 16, 720) or ranges.dtype != np.float32:
        raise ValueError('original five-frame float32 ranges required')
    local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
    directions = np.concatenate([world_directions(local, float(y)) for y in sensor['yaw_deg']])
    source_directions = directions.copy()
    directions /= np.linalg.norm(directions, axis=1, keepdims=True)
    rays = CausalRaySegments(np.repeat(sensor['sensor_xyz_m'], 11520, axis=0), directions,
                            ranges.reshape(-1), valid.reshape(-1).astype(bool),
                            np.repeat(frames, 11520), int(frames[-1]), range_error_bound_m)
    groups = construction_incident_paths(document)
    _, primitives = load_p1a_realized_construction(document)
    origin = sensor['sensor_xyz_m'][-1]
    result = terminal_reference_evidence(groups, primitives, rays=rays,
        source_permission=source_permission(bundle),
        membership_codes=sensor['primitive_membership_code'].reshape(-1),
        source_sets=book['source_sets'], local_center_m=origin, source_directions=source_directions)
    for row in result['terminal_references']:
        row['anchor_current_sensor_m'] = _current_sensor_transform(
            np.asarray(row['anchor_world_m']), origin, float(sensor['yaw_deg'][-1])).tolist()
    return dict(source=source, **result,
                local_terminal_count=len(result['terminal_references']),
                observed_terminal_count=sum(r['terminal_reference_observed'] for r in result['terminal_references']),
                labels_generated=0, teacher_complete=False, scientific_gate_pass=False)
