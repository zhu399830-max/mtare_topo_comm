"""Autonomous geometric proposals from finite axes; no teacher grouping."""
from dataclasses import dataclass
from itertools import combinations
import numpy as np
from .finite_axis_junction import finite_axis_intersection


@dataclass(frozen=True)
class AxisGroupCandidate:
    position_m: tuple
    segment_indices: tuple
    max_residual_m: float
    connectivity_verified: bool = False


def axis_intersection_candidates(segments, *, max_residual_m, min_crossing_sine,
                                 endpoint_tolerance_m, maximum_candidates):
    s=np.asarray(segments,dtype=float)
    policy=dict(max_residual_m=max_residual_m,min_crossing_sine=min_crossing_sine,
                endpoint_tolerance_m=endpoint_tolerance_m)
    # Reuse all layout/parameter checks, but not a whole-window grouping.
    finite_axis_intersection(s,**policy)
    if type(maximum_candidates) is not int or not 1<=maximum_candidates<=32:
        raise ValueError('explicit bounded output capacity required')
    if len(s)<2:return ()
    lengths=np.linalg.norm(s[:,1]-s[:,0],axis=1)
    if np.any(lengths<=np.finfo(float).eps):
        raise ValueError('degenerate observation axis; do not silently drop')
    directions=(s[:,1]-s[:,0])/lengths[:,None]
    groups={}
    for i,j in combinations(range(len(s)),2):
        seed=finite_axis_intersection(s[[i,j]],**policy)
        if seed.position_m is None:continue
        offset=np.asarray(seed.position_m)-s[:,0]
        along=np.einsum('ni,ni->n',offset,directions)
        residual=np.linalg.norm(offset-along[:,None]*directions,axis=1)
        members=tuple(np.flatnonzero((residual<=max_residual_m)&(along>=-endpoint_tolerance_m)&
                      (along<=lengths+endpoint_tolerance_m)).tolist())
        if i not in members or j not in members:continue
        if members in groups:continue
        result=finite_axis_intersection(s[list(members)],**policy)
        if result.position_m is None:continue
        groups[members]=AxisGroupCandidate(result.position_m,members,result.max_residual_m)
    # Inclusion-based suppression only: no spatial merge radius invented.
    retained=[v for k,v in groups.items() if not any(set(k)<set(other) for other in groups)]
    if len(retained)>maximum_candidates:
        raise ValueError('candidate capacity exceeded; no score-based truncation')
    return tuple(sorted(retained,key=lambda c:c.position_m))
