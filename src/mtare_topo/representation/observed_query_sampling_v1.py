"""Observation-only deterministic sampling; no labels or model scores.

FPS mechanism motivated by 3DETR; independent NumPy implementation without
the upstream CUDA extension. Returned indices address this input point array.
Coincident coordinates share one seed; no duplicate padding or fake origin.
"""
import numpy as np


def sample_observed_queries(points_m, *, maximum=32):
    if (not isinstance(points_m,np.ndarray) or points_m.ndim!=2
            or points_m.shape[1:]!=(3,) or points_m.dtype not in (np.float32,np.float64)
            or len(points_m)>57600 or not np.isfinite(points_m).all()
            or type(maximum) is not int or not 1<=maximum<=32):
        raise ValueError('finite bounded floating observation points required')
    points=points_m.astype(np.float64)
    if np.any(np.linalg.norm(points,axis=1)>10.):
        raise ValueError('queries must originate in the fixed10m observation domain')
    unique,source=np.unique(points,axis=0,return_index=True)
    count=min(maximum,len(unique))
    if not count:return np.empty(0,dtype=np.int64)
    # np.unique sorts XYZ lexicographically. argmax therefore resolves exact
    # equal-distance ties by coordinates, not labels or incoming point order.
    chosen=[];distance=np.full(len(unique),np.inf);available=np.ones(len(unique),bool)
    index=0
    for _ in range(count):
        chosen.append(index);available[index]=False
        distance=np.minimum(distance,np.sum((unique-unique[index])**2,axis=1))
        index=int(np.argmax(np.where(available,distance,-np.inf)))
    return source[np.asarray(chosen,dtype=np.int64)].astype(np.int64)
