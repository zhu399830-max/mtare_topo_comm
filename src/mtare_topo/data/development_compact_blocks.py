"""Source-authenticated development compact cache; no teacher/model execution."""
from dataclasses import dataclass
import hashlib
import io
import numpy as np
from mtare_topo.representation.gse_candidate_context import ObservationBinding
from mtare_topo.representation.gse_block_points import bind_block_points
from mtare_topo.representation.gse_block_context import bind_stamped_block_context


@dataclass(frozen=True)
class DevelopmentCompactPoints:
    points_xyz_m: np.ndarray
    source_flat_ray_index: np.ndarray
    context: np.ndarray
    full_valid: np.ndarray
    binding: ObservationBinding
    cache_sha256: str


def read_compact_points(payload, *, manifest_row, expected_source, encoder_sha256):
    """Caller binds the manifest bytes separately; payload must match its hash."""
    if set(expected_source) != {'task', 'source_sequence_id', 'frame_rows'}:
        raise ValueError('closed observation identity required')
    source = manifest_row['source']
    if (any(source[k] != expected_source[k] for k in expected_source)
            or source['coordinate_frame'] != 'current_sensor'
            or manifest_row['frozen_encoder_state_sha256'] != encoder_sha256
            or hashlib.sha256(payload).hexdigest() != manifest_row['sha256']):
        raise ValueError('cache source, encoder or bytes mismatch')
    stamp = ObservationBinding(str(source['task']) + ':' + str(source['source_sequence_id']),
        tuple(source['frame_rows']), 'current_sensor', source['input_file_sha256'], encoder_sha256)
    stamp.validate()
    with np.load(io.BytesIO(payload), allow_pickle=False) as data:
        if set(data.files) != {'points_xyz_m', 'frozen_sensor_context', 'valid'}:
            raise ValueError('student-only compact cache keys required')
        points = data['points_xyz_m']; memory = data['frozen_sensor_context']; valid = data['valid']
    if (points.shape != (1,57600,3) or points.dtype != np.float32
            or memory.shape != (1,900,128) or memory.dtype != np.float32
            or valid.shape != (1,57600) or valid.dtype != np.bool_
            or not np.isfinite(points).all() or not np.isfinite(memory).all()):
        raise ValueError('finite original compact layout required')
    indices = np.flatnonzero(valid[0] & (np.linalg.norm(points[0].astype(float), axis=1) <= 10.))
    arrays = (points[0,indices].copy(), indices, memory[0].copy(), valid[0].copy())
    for value in arrays:
        value.flags.writeable = False
    return DevelopmentCompactPoints(*arrays, stamp, manifest_row['sha256'])


def bind_partition(compact, assignment):
    """Assignment is a geometry-only grouping, not primitive or node truth."""
    if type(compact) is not DevelopmentCompactPoints:
        raise ValueError('authenticated compact points required')
    indices = compact.source_flat_ray_index
    blocks = bind_block_points(compact.points_xyz_m, indices // 11520, assignment)
    pooled = bind_stamped_block_context(blocks, indices, compact.context, compact.full_valid,
                                       compact.binding, compact.binding)
    return dict(blocks=blocks, context=pooled['context'], binding=compact.binding)


def sensor_layout_partition(compact):
    indices = compact.source_flat_ray_index
    return bind_partition(compact, (indices // 11520) * 180 + (indices % 720) // 4)


def bind_partition_payload(compact, payload, *, expected_sha256, expected_source):
    """Join a sealed native packet to the actual authenticated compact cache."""
    from .development_partition_handoff import decode_partitions
    if type(compact) is not DevelopmentCompactPoints:
        raise ValueError('authenticated compact cache required')
    if (set(expected_source)!={'task','source_sequence_id','frame_rows'}
            or compact.binding.observation_id!=str(expected_source['task'])+':'+str(expected_source['source_sequence_id'])
            or compact.binding.frame_indices!=tuple(expected_source['frame_rows'])):
        raise ValueError('partition source differs from authenticated compact cache')
    assignments=decode_partitions(payload,expected_sha256=expected_sha256,expected_source=expected_source,
        expected_cache_sha256=compact.cache_sha256,points_xyz_m=compact.points_xyz_m,
        source_flat_ray_index=compact.source_flat_ray_index)
    return dict(r0=sensor_layout_partition(compact),
                **{name:bind_partition(compact,assignment) for name,assignment in assignments.items()})
