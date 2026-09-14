"""Exact saved-window reader. Student payload and reference loading stay separate.

No teacher generation, candidate nomination or unknown-to-background conversion.
Callers must freeze and authorize the binding before accessing real payloads.
"""
from dataclasses import dataclass
import gzip
import hashlib
import json
from pathlib import Path
import numpy as np


STUDENT_FIELDS = ('ranges_m', 'valid_mask', 'relative_translation_current_sensor_m',
                  'relative_yaw_current_sensor_deg')


@dataclass(frozen=True)
class MembershipStudentWindow:
    ranges_m: np.ndarray
    valid_mask: np.ndarray
    relative_translation_current_sensor_m: np.ndarray
    relative_yaw_current_sensor_deg: np.ndarray


def _checked(root, relative, expected):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError('input outside bound repository')
    data = path.read_bytes()
    if hashlib.sha256(data).hexdigest() != expected:
        raise ValueError('bound input hash drift: ' + str(relative))
    return data


def load_student_window(root, binding):
    """Return only four sensor fields; source identities are checked then discarded.

    Batched NPZ necessarily decompresses selected arrays for the whole task;
    the binding must declare that collateral population, not claim sparse IO.
    """
    import io
    data = _checked(root, binding['student_path'], binding['student_sha256'])
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        layout = binding['layout']
        if layout == 'saved_task_batch':
            if set(archive.files) != set(STUDENT_FIELDS) | {'frame_rows', 'source_sequence_ids'}:
                raise ValueError('unexpected batched input fields')
            row = binding['row_index']; count = binding['decoded_observations']
            if not 0 <= row < count:
                raise ValueError('bound row out of range')
            ids, frames = archive['source_sequence_ids'], archive['frame_rows']
            if (ids.shape != (count,) or frames.shape != (count, 5)
                    or int(ids[row]) != binding['source_sequence_id']
                    or frames[row].tolist() != binding['frame_rows']):
                raise ValueError('source identity or frame mismatch')
            arrays = {}
            for key in STUDENT_FIELDS:
                value = archive[key]
                if value.shape[0] != count:
                    raise ValueError('collateral population count drift')
                arrays[key] = value[row].copy()
        elif layout == 'saved_single_window':
            if set(archive.files) != set(STUDENT_FIELDS):
                raise ValueError('single student window contains extra fields')
            if binding['decoded_observations'] != 1:
                raise ValueError('single-window count drift')
            arrays = {key: archive[key].copy() for key in STUDENT_FIELDS}
        else:
            raise ValueError('unknown saved layout')
    shapes = ((5, 16, 720), (5, 16, 720), (5, 3), (5,))
    for key, shape in zip(STUDENT_FIELDS, shapes):
        value = arrays[key]
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError('sensor shape or numerical drift: ' + key)
        if key == 'valid_mask':
            if value.dtype != np.uint8 or not np.isin(value, [0, 1]).all():
                raise ValueError('binary uint8 mask required')
        elif value.dtype != np.float32:
            raise ValueError('float32 sensor payload required')
        value.setflags(write=False)
    return MembershipStudentWindow(**arrays)


def load_student_task_batch(root, bindings):
    """Decode one fully bound task once, returning student-only immutable windows.

    Require the complete collateral population, not an implicitly widened subset.
    This helper grants no payload access; callers still require an exact card.
    """
    import io
    if not bindings:
        raise ValueError('nonempty complete task bindings required')
    first = bindings[0]
    count = first['decoded_observations']
    common = ('task', 'student_path', 'student_sha256', 'decoded_observations', 'layout')
    if (first['layout'] != 'saved_task_batch' or len(bindings) != count
            or [b['row_index'] for b in bindings] != list(range(count))
            or any(any(b[k] != first[k] for k in common) for b in bindings)):
        raise ValueError('one complete ordered task population required')
    data = _checked(root, first['student_path'], first['student_sha256'])
    with np.load(io.BytesIO(data), allow_pickle=False) as archive:
        if set(archive.files) != set(STUDENT_FIELDS) | {'frame_rows', 'source_sequence_ids'}:
            raise ValueError('unexpected batched input fields')
        ids, frames = archive['source_sequence_ids'], archive['frame_rows']
        if (ids.shape != (count,) or frames.shape != (count, 5)
                or ids.tolist() != [b['source_sequence_id'] for b in bindings]
                or frames.tolist() != [b['frame_rows'] for b in bindings]):
            raise ValueError('source identity or frame mismatch')
        arrays = {key: archive[key] for key in STUDENT_FIELDS}
    shapes = ((5, 16, 720), (5, 16, 720), (5, 3), (5,))
    for key, shape in zip(STUDENT_FIELDS, shapes):
        value = arrays[key]
        if value.shape != (count,) + shape or not np.isfinite(value).all():
            raise ValueError('sensor shape or numerical drift: ' + key)
        if key == 'valid_mask':
            if value.dtype != np.uint8 or not np.isin(value, [0, 1]).all():
                raise ValueError('binary uint8 mask required')
        elif value.dtype != np.float32:
            raise ValueError('float32 sensor payload required')
        value.setflags(write=False)
    return tuple(MembershipStudentWindow(**{key: value[i] for key, value in arrays.items()})
                 for i in range(count))


def load_partial_reference(root, binding):
    """Return original targets, without selecting only the nominated relation."""
    data = _checked(root, binding['target_path'], binding['target_sha256'])
    target = json.loads(gzip.decompress(data))['produced_targets']
    record = target['record']
    if record['source_frame_indices'] != binding['frame_rows']:
        raise ValueError('reference source frame mismatch')
    if record['coordinate_frame'] != 'current_sensor_m':
        raise ValueError('reference coordinate frame mismatch')
    return target
