"""Diagnostic-to-sensor IO only; source completeness is not certified here."""
import numpy as np
from .cano_sensor_smoke import lidar_local_directions, world_directions
from .primitive_relation_sensor_export import PrimitiveSensorFrame
from mtare_topo.teacher.primitive_provenance_field import PrimitiveRayHit


def diagnostic_frame(results, *, sensor_xyz_m, directions_xyz, codebook):
    """Preserve fixed scan layout and complete *reported* source sets.

    Caller owns input/geometry binding and qualification. No teacher, oracle,
    true identity selection, or threshold adjustment enters this conversion.
    Unknown aborts before the shared codebook is mutated.
    """
    origin = np.asarray(sensor_xyz_m, dtype=np.float64)
    directions = np.asarray(directions_xyz, dtype=np.float64)
    if origin.shape != (3,) or not np.isfinite(origin).all():
        raise ValueError('finite sensor origin required')
    if directions.shape != (11520, 3) or not np.isfinite(directions).all():
        raise ValueError('exact 11520 finite directions required')
    lengths = np.linalg.norm(directions, axis=1)
    if (lengths == 0).any():
        raise ValueError('zero direction')
    directions = directions / lengths[:, None]
    ranges = np.full(11520, 50., dtype=np.float32)
    hits = []; ambiguous = 0
    identities = set(codebook.primitive_ids)
    for i, result in enumerate(results):
        if i >= 11520:
            raise ValueError('too many ray results')
        status = result.get('status')
        if status == 'out_of_range':
            hits.append(None)
            continue
        if status not in ('candidate', 'reference_return'):
            raise ValueError(f'unqualified ray {i}: {status}')
        distance = float(result['distance_m'])
        sources = tuple(result['sources'])
        if not np.isfinite(distance) or distance <= 0:
            raise ValueError('invalid reported distance')
        if not sources or len(sources) != len(set(sources)) or not set(sources) <= identities:
            raise ValueError('invalid reported source set')
        if not .3 <= distance <= 50.:
            hits.append(None)
            continue
        ranges[i] = np.float32(distance)
        hits.append(PrimitiveRayHit(distance, tuple(origin + distance * directions[i]),
                                    sources, len(sources) == 1))
        ambiguous += len(sources) > 1
    if len(hits) != 11520:
        raise ValueError('missing ray results')
    codes = codebook.encode(hits)
    return PrimitiveSensorFrame(ranges.reshape(16, 720), (codes > 0).astype(np.uint8).reshape(16, 720),
                                codes.reshape(16, 720), ambiguous)


def render_covered_frame(*, diagnostic, codebook, sensor_xyz_m, yaw_deg):
    """Versioned renderer entry, no legacy field-based initial-inside override."""
    if not np.isfinite(yaw_deg):
        raise ValueError('finite yaw required')
    local = lidar_local_directions().reshape(-1, 3).astype(np.float64)
    directions = world_directions(local, float(yaw_deg))
    results = (diagnostic.query(sensor_xyz_m, d, maximum_m=50.) for d in directions)
    return diagnostic_frame(results, sensor_xyz_m=sensor_xyz_m,
                            directions_xyz=directions, codebook=codebook)
