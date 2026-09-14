"""Source-free non-learning direction diagnostic; NOT a channel detector.

Both estimators consume exactly the same patch neighbourhood's observed points.
POINT uses pooled point covariance's major axis. NORMAL uses the minor axis
of the count-weighted local-normal scatter. The latter only has a unique
answer when multiple reliable surface orientations constrain that axis.
All eigenvalues and validity are returned; no GT-driven candidate filtering.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AxisEstimate:
    axes: np.ndarray
    valid: np.ndarray
    eigenvalues: np.ndarray
    support_points: np.ndarray


def _canonical(v):
    return -v if v[np.argmax(np.abs(v))] < 0 else v


def estimate_axes(points, point_patch_index, neighbors, patch_count):
    points = np.asarray(points, dtype=np.float64)
    mapping = np.asarray(point_patch_index)
    neighbors = np.asarray(neighbors)
    if (points.ndim != 2 or points.shape[1] != 3 or not np.isfinite(points).all()
            or mapping.shape != (len(points),) or mapping.dtype.kind not in 'iu'
            or neighbors.ndim != 2 or len(neighbors) != patch_count
            or neighbors.dtype.kind not in 'iu' or np.any(mapping < -1)
            or np.any(mapping >= patch_count) or np.any(neighbors < -1)
            or np.any(neighbors >= patch_count)):
        raise ValueError('invalid observed points/patch indices/neighbours')
    count = np.zeros(patch_count, dtype=np.int64)
    sums = np.zeros((patch_count, 3)); products = np.zeros((patch_count, 3, 3))
    normals = np.zeros((patch_count, 3)); reliable = np.zeros(patch_count, bool)
    for p in range(patch_count):
        x = points[mapping == p]
        # Stable summation order independent of source point permutation.
        if len(x):
            x = x[np.lexsort((x[:, 2], x[:, 1], x[:, 0]))]
        count[p] = len(x)
        if not len(x):
            continue
        sums[p] = x.sum(0); products[p] = x.T @ x
        centered = x - x.mean(0)
        w, v = np.linalg.eigh(centered.T @ centered / len(x))
        eps = 64*np.finfo(float).eps*max(1., float(w[-1]))
        reliable[p] = len(x) >= 3 and w[1] - w[0] > eps
        if reliable[p]:
            normals[p] = _canonical(v[:, 0])
    outputs = {}
    for method in ('POINT', 'NORMAL'):
        axes = np.zeros((patch_count, 3)); valid = np.zeros(patch_count, bool)
        values = np.zeros((patch_count, 3)); support = np.zeros(patch_count, np.int64)
        for p in range(patch_count):
            ids = np.unique(np.r_[p, neighbors[p][neighbors[p] >= 0]])
            support[p] = count[ids].sum()
            if support[p] < 3:
                continue
            if method == 'POINT':
                mean = sums[ids].sum(0)/support[p]
                matrix = products[ids].sum(0)/support[p] - np.outer(mean, mean)
            else:
                usable = ids[reliable[ids]]
                if not len(usable):
                    continue
                matrix = np.einsum('n,ni,nj->ij', count[usable], normals[usable], normals[usable])/count[usable].sum()
            w, v = np.linalg.eigh(matrix); values[p] = w
            eps = 64*np.finfo(float).eps*max(1., float(w[-1]))
            gap = w[2]-w[1] if method == 'POINT' else w[1]-w[0]
            if gap > eps:
                valid[p] = True
                axes[p] = _canonical(v[:, -1] if method == 'POINT' else v[:, 0])
        outputs[method] = AxisEstimate(axes, valid, values, support)
    return outputs


def pair_predictions(estimate, neighbors):
    neighbors = np.asarray(neighbors)
    if neighbors.ndim != 2 or len(neighbors) != len(estimate.axes):
        raise ValueError('pair shape mismatch')
    exists = (neighbors >= 0) & (neighbors < len(estimate.axes))
    safe = np.where(exists, neighbors, 0)
    valid = exists & estimate.valid[:, None] & estimate.valid[safe]
    values = np.abs(np.einsum('mi,mki->mk', estimate.axes, estimate.axes[safe]))
    values = np.where(valid, np.clip(values, 0, 1), np.nan)
    return values, valid
