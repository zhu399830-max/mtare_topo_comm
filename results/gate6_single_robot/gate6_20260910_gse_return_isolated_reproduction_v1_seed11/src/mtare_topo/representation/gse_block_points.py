"""Shared geometry-only block input. Group identifiers are indexing, not labels."""
from dataclasses import dataclass
import numpy as np
from .gse_partition_attributes import summarize_partition


@dataclass(frozen=True)
class BlockPoints:
    xyz_m: np.ndarray
    frame_index: np.ndarray
    point_to_block: np.ndarray
    block_ids: np.ndarray
    centers_m: np.ndarray
    extent_m: np.ndarray
    degenerate: np.ndarray


def bind_block_points(points,frames,assignment):
    if not isinstance(points,np.ndarray) or points.ndim!=2 or points.shape[1]!=3 or len(points)>57600:
        raise ValueError('bounded single observation N,3 points required')
    if points.dtype!=np.float32 or np.any(np.linalg.norm(points.astype(float),axis=1)>10.):
        raise ValueError('same float32 10m ROI required')
    a=summarize_partition(points,frames,assignment)
    # No point subsampling, no minimum-size rejection, no artificial padding.
    arrays=(points.copy(),frames.copy(),a.point_to_group.copy(),a.group_ids.copy(),
        a.centers_m.astype(np.float32),(a.bounds_max_m-a.bounds_min_m).astype(np.float32),~a.normal_valid)
    for value in arrays:value.flags.writeable=False
    return BlockPoints(*arrays)
