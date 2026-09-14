"""The original twelve double declarations, all five actual angular frames."""
import hashlib
import json
import numpy as np
from mtare_topo.data.cano_sensor_smoke import lidar_local_directions, world_directions
from mtare_topo.data.gse_synthetic_matrix import matrix
from .double_frame_diagnostic import array_sha256
from .gse_convex_ray_union import declared_union_exit


def frames():
    cases = [c for c in matrix() if c['program']['type'] == 'double_junction']
    wanted = [f'double_junction__{s}__view{v}'
              for s in ('circle', 'ellipse', 'rounded_rectangle') for v in range(4)]
    if [c['case_id'] for c in cases] != wanted:
        raise ValueError('original declaration population/order changed')
    local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
    for case in cases:
        if len(case['poses_world_m']) != 5 or len(case['yaw_deg']) != 5:
            raise ValueError('original history population changed')
        case_sha = hashlib.sha256(json.dumps(case, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        for frame, position in enumerate(case['poses_world_m']):
            origin = np.asarray(position, dtype=np.float64)
            directions = world_directions(local, case['yaw_deg'][frame])
            if directions.shape != (11520, 3) or np.unique(directions, axis=0).shape[0] != 11520:
                raise ValueError('angular population changed')
            unit = directions / np.linalg.norm(directions, axis=1, keepdims=True)
            expected, supported, _ = declared_union_exit(case, np.tile(origin, (11520, 1)), unit)
            if not supported.all() or not np.isfinite(expected).all():
                raise ValueError('unsupported independent reference')
            manifest = dict(case_id=case['case_id'], frame_index=frame, origin_m=origin.tolist(),
                yaw_deg=case['yaw_deg'][frame], rays=11520, case_sha256=case_sha,
                direction_sha256_f64le=array_sha256(directions),
                unit_direction_sha256_f64le=array_sha256(unit),
                oracle_distance_sha256_f64le=array_sha256(expected))
            yield case, origin, directions, expected, manifest


def input_manifest():
    return dict(schema_version='gse_double_population_input_v1', declarations=12,
                frames=60, rays=691200, labels_exported=0, optimizer_steps=0,
                axial_spacing_m=.05, angular_segments=64, maximum_m=50.,
                near_m=.3, comparison_atol_m=1e-5,
                frame_inputs=[row[-1] for row in frames()])
