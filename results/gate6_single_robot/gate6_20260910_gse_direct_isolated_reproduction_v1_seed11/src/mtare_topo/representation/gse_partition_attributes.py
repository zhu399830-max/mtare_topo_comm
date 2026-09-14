"""Shared raw-return attribute reduction for R1 and R2 partitions.

Inputs are the SAME ROI-selected valid returns, not voxel centroids. Bounds
describe observed support, never tunnel width/height or traversable clearance.
"""
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class PartitionAttributes:
    group_ids: np.ndarray
    point_to_group: np.ndarray
    centers_m: np.ndarray
    normals: np.ndarray
    normal_valid: np.ndarray
    normal_uncertainty: np.ndarray
    bounds_min_m: np.ndarray
    bounds_max_m: np.ndarray
    roughness_m: np.ndarray
    point_count: np.ndarray
    frame_support: np.ndarray


def summarize_partition(points, frames, assignment):
    if (not isinstance(points,np.ndarray) or points.ndim!=2 or points.shape[1]!=3
            or points.dtype.kind!='f' or not np.isfinite(points).all()):
        raise ValueError('finite ROI-selected N,3 returns required')
    for a in (frames,assignment):
        if not isinstance(a,np.ndarray) or a.shape!=(len(points),) or a.dtype.kind not in 'iu' or np.any(a<0):
            raise ValueError('all selected points require nonnegative frame and group indices')
    if np.any(frames>4):
        raise ValueError('five-frame support only')
    ids,inverse,counts=np.unique(assignment,return_inverse=True,return_counts=True)
    m=len(ids)
    if m>4096:
        raise OverflowError('shared4096 group capacity exceeded; never truncate')
    xyz=points.astype(np.float64)
    order=np.lexsort((frames,xyz[:,2],xyz[:,1],xyz[:,0],inverse))
    center=np.zeros((m,3));normals=np.zeros((m,3));known=np.zeros(m,bool)
    uncertainty=np.ones(m);low=np.zeros((m,3));high=np.zeros((m,3))
    rough=np.zeros(m);support=np.zeros((m,5),dtype=np.int64)
    start=0
    for g,count in enumerate(counts):
        index=order[start:start+count];start+=count;p=xyz[index]
        c=p.mean(0);center[g]=c;low[g]=p.min(0);high[g]=p.max(0)
        residual=p-c
        values,vectors=np.linalg.eigh(residual.T@residual/count)
        values=np.maximum(values,0.)
        tolerance=64*np.finfo(np.float64).eps*max(1.,values[-1])
        rough[g]=np.sqrt(values[0])
        if count>=3 and values[1]>tolerance and values[1]-values[0]>tolerance:
            normal=vectors[:,0];dot=float(normal@c)
            if dot>tolerance or (abs(dot)<=tolerance and normal[np.argmax(np.abs(normal))]<0):
                normal=-normal
            normals[g]=normal;known[g]=True
            uncertainty[g]=1.-(values[1]-values[0])/max(values[-1],tolerance)
        support[g]=np.bincount(frames[index].astype(np.int64),minlength=5)
    arrays=(ids,inverse,center,normals,known,uncertainty,low,high,rough,counts,support)
    for a in arrays:
        if a.dtype.kind=='f' and not np.isfinite(a).all():
            raise ValueError('nonfinite group attributes')
        a.flags.writeable=False
    return PartitionAttributes(*arrays)
