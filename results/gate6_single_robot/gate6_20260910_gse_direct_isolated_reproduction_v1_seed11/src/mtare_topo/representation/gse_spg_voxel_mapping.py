"""Source-preserving map for SPG's minimum-origin, float32 voxel convention.

No ROI selection, labels or point removal. IDs follow first point occurrence,
as in the pinned upstream prune implementation; they are not place identities.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class SPGVoxelMapping:
    centers_m: np.ndarray
    point_to_voxel: np.ndarray
    counts: np.ndarray
    frame_counts: np.ndarray
    grid_origin_m: np.ndarray


def map_spg_voxels(points, frame_slots, *, voxel_size_m=.03):
    if (not isinstance(points, np.ndarray) or points.ndim != 2 or points.shape[1] != 3
            or points.dtype != np.float32 or not np.isfinite(points).all()):
        raise ValueError('finite float32 N,3 points required')
    n = len(points)
    if (not isinstance(frame_slots,np.ndarray) or frame_slots.shape != (n,)
            or frame_slots.dtype.kind not in 'iu' or np.any(frame_slots < 0) or np.any(frame_slots > 4)):
        raise ValueError('explicit source frame slots 0..4 required')
    if not np.isfinite(voxel_size_m) or voxel_size_m <= 0:
        raise ValueError('finite positive voxel size required')
    size = np.float32(voxel_size_m)
    if not np.isfinite(size) or size <= 0:
        raise ValueError('voxel size not representable')
    origin = points.min(axis=0) if n else np.zeros(3,dtype=np.float32)
    with np.errstate(over='ignore',invalid='ignore'):
        keys_float = np.floor((points-origin)/size)
    if not np.isfinite(keys_float).all() or np.any(keys_float >= 2**32):
        raise OverflowError('upstream uint32 grid capacity exceeded')
    keys = keys_float.astype(np.uint32)
    lookup = {}; inverse = np.empty(n,dtype=np.int64)
    for i,key in enumerate(keys):
        inverse[i] = lookup.setdefault(tuple(key),len(lookup))
    m = len(lookup)
    centers = np.zeros((m,3),dtype=np.float32)
    counts = np.zeros(m,dtype=np.int64); frames = np.zeros((m,5),dtype=np.int64)
    # np.add.at preserves repeated-index accumulation instead of dropping it.
    np.add.at(centers,inverse,points)
    np.add.at(counts,inverse,1)
    np.add.at(frames,(inverse,frame_slots),1)
    if m:
        centers /= counts.astype(np.float32)[:,None]
    if not np.isfinite(centers).all():
        raise OverflowError('nonfinite accumulated center')
    for a in (centers,inverse,counts,frames,origin):
        a.flags.writeable = False
    return SPGVoxelMapping(centers,inverse,counts,frames,origin)
