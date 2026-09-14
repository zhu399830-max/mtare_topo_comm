"""Opt-in teacher-only source policy. Legacy bundles keep legacy behavior."""
import numpy as np

from .source_witness_policy import source_witness_policy


def permitted_sources(bundle):
    book = bundle['codebook_teacher_only']
    codes = np.asarray(bundle['sensor_teacher_only']['primitive_membership_code']).reshape(-1)
    valid = np.asarray(bundle['student']['valid_mask']).reshape(-1).astype(bool)
    ids = book['primitive_ids']
    sets = book['source_sets']
    if codes.dtype != np.uint16 or codes.shape != valid.shape or (codes >= len(sets)).any():
        raise ValueError('source codes do not bind scan rows')
    decoded = [list(sets[int(code)]) for code in codes]
    if 'source_witness_teacher_only' not in bundle:
        if bundle.get('source_witness_required', False):
            raise ValueError('source witness policy missing; legacy fallback forbidden')
        return decoded
    sidecar = bundle['source_witness_teacher_only']
    if (sidecar['primitive_ids'] != ids or sidecar['frame_rows'] != bundle['source']['frame_rows']
            or sidecar['source_sequence_id'] != bundle['source']['source_sequence_id']):
        raise ValueError('source policy identity or row binding differs')
    reported = np.zeros((len(codes), len(ids)), dtype=bool)
    for row, members in enumerate(decoded):
        if any(type(i) is not int or not 0 <= i < len(ids) for i in members):
            raise ValueError('invalid source index')
        reported[row, members] = True
    if not np.array_equal(reported, sidecar['reported']):
        raise ValueError('source audit reported membership differs')
    policy = source_witness_policy(valid=valid, reported=reported,
        possible_surface=sidecar['possible_surface'],
        uncertain_operands=sidecar['uncertain_operands'],
        surface_uniqueness_certified=sidecar['surface_uniqueness_certified'])
    owners = policy.owners(mode=sidecar['mode'])
    return [[int(i)] if i >= 0 else [] for i in owners]


def named_sources(bundle):
    ids = bundle['codebook_teacher_only']['primitive_ids']
    return [[ids[i] for i in members] for members in permitted_sources(bundle)]


def source_permission(bundle):
    return np.array([len(v) == 1 for v in permitted_sources(bundle)], dtype=bool)


def validate_permission(permission, shape):
    if permission is None:
        return np.ones(shape, dtype=bool)  # Explicit legacy call, not new audit.
    value = np.asarray(permission)
    if value.dtype != bool or value.shape != shape:
        raise ValueError('boolean source permission must bind every ray')
    return value
