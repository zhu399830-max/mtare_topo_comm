"""Bounded CPU SPG adapter; requires separately authenticated native backends.

Input points must already be causally aligned and ROI-selected by the shared
reader. No teacher inputs. Components describe geometry, not navigable space.
"""
import numpy as np
from .gse_spg_voxel_mapping import map_spg_voxels
from .gse_connected_partition import split_connected_partition


def extract_spg(points, frame_slots, *, geof_backend, partition_backend):
    if not isinstance(points,np.ndarray) or len(points)>57600:
        raise OverflowError('maximum five-frame input is 57600 points; never truncate')
    voxels = map_spg_voxels(points,frame_slots)
    xyz = np.ascontiguousarray(voxels.centers_m)
    n = len(xyz)
    result = dict(voxels=voxels, status='EMPTY' if n==0 else 'INSUFFICIENT_GEOMETRY_NEIGHBORS',
                  raw_assignment=None, connected=None, original_point_to_component=None)
    if n<46:
        return result
    # Bound covariance arithmetic before passing data to Eigen float32.
    if np.max(np.abs(xyz.astype(float)))>np.sqrt(np.finfo(np.float32).max/(4*46)):
        raise OverflowError('unsafe float32 covariance magnitude')
    from scipy.spatial import cKDTree
    tree = cKDTree(xyz)
    distances, indices = tree.query(xyz,k=46,workers=1)
    # Sorting the queried set gives fixed-input tie order; ties at the cutoff
    # do not imply permutation-invariant selection across arbitrary point order.
    neighbor = np.empty((n,45),dtype=np.uint32)
    neighbor_distance = np.empty((n,45),dtype=np.float64)
    for i in range(n):
        use = indices[i]!=i
        ids = indices[i][use]; ds = distances[i][use]
        order = np.lexsort((ids,ds))[:45]
        neighbor[i]=ids[order]; neighbor_distance[i]=ds[order]
    features = np.asarray(geof_backend(xyz,neighbor.ravel(),45))
    if features.shape!=(n,4) or features.dtype!=np.float32 or not np.isfinite(features).all():
        raise ValueError('native geometric features invalid; do not repair with guessed zeros')
    features = features.copy(); features[:,3]*=2
    source = np.repeat(np.arange(n,dtype=np.uint32),10)
    target = np.ascontiguousarray(neighbor[:,:10].ravel())
    length = neighbor_distance[:,:10].ravel()
    mean = length.mean()
    if not np.isfinite(mean) or mean<=0:
        raise ValueError('degenerate adjacency length')
    weights = np.ascontiguousarray(1/(1+length/mean),dtype=np.float32)
    components, raw = partition_backend(features,source,target,weights,.1)
    raw = np.asarray(raw)
    if raw.shape!=(n,) or raw.dtype.kind not in 'iu' or np.any(raw<0):
        raise ValueError('native assignment malformed')
    seen = np.zeros(n,dtype=np.int64)
    for k,group in enumerate(components):
        ids = np.asarray(group)
        if ids.ndim!=1 or ids.dtype.kind not in 'iu' or not len(ids) or np.any(ids<0) or np.any(ids>=n):
            raise ValueError('native component malformed')
        if np.any(raw[ids]!=k):
            raise ValueError('native component/assignment mismatch')
        np.add.at(seen,ids,1)
    if not np.all(seen==1):
        raise ValueError('native output lost or duplicated points')
    connected = split_connected_partition(raw,source,target)
    lifted = connected.point_to_component[voxels.point_to_voxel].copy()
    lifted.flags.writeable=False
    result.update(status='PARTITIONED_NOT_SEMANTICALLY_QUALIFIED',raw_assignment=connected.raw_assignment,
                  connected=connected,original_point_to_component=lifted)
    return result
