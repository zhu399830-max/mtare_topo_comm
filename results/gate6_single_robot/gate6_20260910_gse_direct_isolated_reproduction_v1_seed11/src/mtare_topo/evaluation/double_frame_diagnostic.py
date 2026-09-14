"""Fixed full-frame diagnostic input; no labels or dataset-export interface."""
import hashlib
import json
import numpy as np

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.gse_synthetic_matrix import matrix
from .gse_convex_ray_union import declared_union_exit


CASE_ID = 'double_junction__circle__view0'
FRAME = 0
RAYS = 11520
ATOL_M = 1e-5


def array_sha256(array):
    return hashlib.sha256(np.ascontiguousarray(array, dtype='<f8').tobytes()).hexdigest()


def frame_inputs():
    case = next(c for c in matrix() if c['case_id'] == CASE_ID)
    origin = np.asarray(case['poses_world_m'][FRAME], dtype=np.float64)
    # Match render_primitive_sensor_frame: float32 angular lattice, cast to
    # float64 BEFORE multiplying by the float32 yaw rotation.
    local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
    directions = world_directions(local, case['yaw_deg'][FRAME])
    if directions.shape != (RAYS, 3) or not np.isfinite(directions).all():
        raise ValueError('actual LiDAR angular interface changed')
    if np.unique(directions, axis=0).shape[0] != RAYS:
        raise ValueError('duplicate angular directions')
    # ExactEventDiagnostic normalizes its input once. The independent oracle
    # must measure metres along exactly the same unit direction.
    unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
    expected, supported, _ = declared_union_exit(case, np.tile(origin, (RAYS, 1)), unit)
    if not supported.all() or not np.isfinite(expected).all():
        raise ValueError('independent reference does not support entire frame')
    manifest = {
        'schema_version': 'gse_double_frame_input_v1',
        'scope': 'synthetic_scan_diagnostic_not_labels_or_training',
        'case_id': CASE_ID, 'frame_index': FRAME, 'frames': 1,
        'independent_declarations': 1, 'five_frame_observations': 0,
        'rays': RAYS, 'shape': [16, 720], 'order': 'elevation_then_azimuth',
        'origin_m': origin.tolist(), 'yaw_deg': case['yaw_deg'][FRAME],
        'direction_sha256_f64le': array_sha256(directions),
        'unit_direction_sha256_f64le': array_sha256(unit),
        'oracle_distance_sha256_f64le': array_sha256(expected),
        'case_sha256': hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':')).encode()).hexdigest(),
        'axial_spacing_m': .05, 'angular_segments': 64,
        'maximum_m': 50., 'near_m': .3, 'comparison_atol_m': ATOL_M,
        'real_world_payload_reads': 0, 'labels_exported': 0, 'optimizer_steps': 0,
    }
    return case, origin, directions, expected, manifest


def score_return(result, expected_m):
    """Unknown is failure, never a manufactured maximum-range observation."""
    status = result.get('status')
    if status == 'out_of_range':
        return {'passed': bool(expected_m > 50.), 'error_m': None,
                'sensor_valid': False, 'reason': 'out_of_range'}
    if status not in ('candidate', 'reference_return') or 'distance_m' not in result:
        return {'passed': False, 'error_m': None, 'sensor_valid': False, 'reason': 'unknown'}
    distance = float(result['distance_m'])
    error = abs(distance - float(expected_m))
    finite = bool(np.isfinite(distance))
    return {'passed': finite and error <= ATOL_M,
            'error_m': error if finite else None,
            'sensor_valid': finite and .3 <= distance <= 50.,
            'reason': 'distance_comparison' if finite else 'nonfinite_return'}
