"""Student-observation ROI authority, independent of teacher/world rounding.

This validates an already frozen observation, not a new geometric tolerance.
Callers must separately verify artifact hashes and source frame identities.
No reference labels, model scores or world-coordinate mask are accepted.
"""
import numpy as np


def validated_frozen_roi(registered_xyz_m, valid_mask, surface_return_indices):
    xyz = np.asarray(registered_xyz_m)
    valid = np.asarray(valid_mask)
    indices = np.asarray(surface_return_indices)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or xyz.dtype not in (np.dtype('float32'), np.dtype('float64')):
        raise ValueError('frozen N,3 float registered coordinates required')
    if valid.shape != (len(xyz),) or valid.dtype not in (np.dtype('bool'), np.dtype('uint8')):
        raise ValueError('aligned binary validity required')
    if not np.isin(valid, [0, 1]).all() or not np.isfinite(xyz).all():
        raise ValueError('nonbinary validity or nonfinite frozen coordinates')
    if indices.ndim != 1 or indices.dtype.kind not in 'iu':
        raise ValueError('one-dimensional integer frozen indices required')
    # Same stored student coordinates and float64 norm as common_observation.
    # Do not compute this population from separately reconstructed world points.
    expected = np.flatnonzero(valid.astype(bool) & (np.linalg.norm(xyz.astype(np.float64), axis=1) <= 10.0))
    if not np.array_equal(indices, expected):
        raise ValueError('frozen ROI is not the exact student 10m population')
    result = indices.astype(np.int64, copy=True)
    result.setflags(write=False)
    return result
