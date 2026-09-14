"""Geometry-only finite-axis intersection kernel, not connectivity evidence.

The caller must derive axis groups from observations, never teacher identity.
All tolerances are explicit; this module defines no experiment thresholds.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class AxisIntersection:
    position_m: tuple | None
    reason: str
    max_residual_m: float | None
    extent_coordinates_m: tuple
    connectivity_verified: bool = False


def finite_axis_intersection(segments, *, max_residual_m, min_crossing_sine,
                             endpoint_tolerance_m):
    s=np.asarray(segments,dtype=np.float64)
    if s.ndim!=3 or s.shape[1:]!=(2,3) or len(s)>32 or not np.isfinite(s).all():
        raise ValueError('at most32 finite observation segments Nx2x3 required')
    params=(max_residual_m,min_crossing_sine,endpoint_tolerance_m)
    if not all(np.isfinite(v) for v in params) or max_residual_m<0 or not 0<min_crossing_sine<=1 or endpoint_tolerance_m<0:
        raise ValueError('explicit finite geometric tolerances required')
    def reject(reason,residual=None,coordinates=()):
        return AxisIntersection(None,reason,residual,coordinates)
    if len(s)<2:return reject('insufficient_axes')
    delta=s[:,1]-s[:,0];length=np.linalg.norm(delta,axis=1)
    if np.any(length<=np.finfo(float).eps):return reject('degenerate_axis')
    u=delta/length[:,None]
    crossing=np.linalg.norm(np.cross(u[:,None,:],u[None,:,:]),axis=-1)
    if crossing.max()<min_crossing_sine:return reject('parallel_or_nearly_parallel')
    # Shift origin for numerical stability; all coordinates remain metric3D.
    origin=s[:,0].mean(axis=0);a=s[:,0]-origin
    projectors=np.eye(3)[None]-u[:,:,None]*u[:,None,:]
    matrix=projectors.sum(axis=0);rhs=np.einsum('nij,nj->i',projectors,a)
    values=np.linalg.eigvalsh(matrix)
    if values[0]<=64*np.finfo(float).eps*max(1.,values[-1]):return reject('underdetermined_intersection')
    local=np.linalg.solve(matrix,rhs);offset=local-a
    residual=float(np.linalg.norm(np.einsum('nij,nj->ni',projectors,offset),axis=1).max())
    coordinates=tuple(float(v) for v in np.einsum('ni,ni->n',offset,u))
    if residual>max_residual_m:return reject('separated_axes',residual,coordinates)
    t=np.asarray(coordinates)
    if np.any(t < -endpoint_tolerance_m) or np.any(t > length+endpoint_tolerance_m):
        return reject('intersection_outside_observed_extent',residual,coordinates)
    return AxisIntersection(tuple(float(v) for v in local+origin),'geometric_candidate_only',residual,coordinates)
