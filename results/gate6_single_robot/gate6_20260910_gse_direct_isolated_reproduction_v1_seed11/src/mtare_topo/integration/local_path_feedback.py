"""Read the original CMU vehicle-frame /path without inventing execution.

The message has no goal identifier. Even a fresh nontrivial path is only a
local-planner proposal, not acknowledgement of a specific target or a ground
connectivity certificate. Odometry/traversal evidence is separate.
"""
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class LocalPathFeedback:
    status: str
    point_count: int
    path_length_m: float
    target_acknowledged: bool = False
    arrival_verified: bool = False
    edge_verified: bool = False


def inspect_local_path(points_vehicle_m, *, frame_id, stamp_sec,
                       target_sent_sec, now_sec, maximum_age_sec):
    times=(stamp_sec,target_sent_sec,now_sec,maximum_age_sec)
    if any(isinstance(t,bool) or not isinstance(t,(int,float)) or not math.isfinite(t) for t in times) or maximum_age_sec<=0:
        raise ValueError('finite clock values and explicit positive maximum age required')
    if frame_id!='vehicle':raise ValueError('original /path must be vehicle-frame')
    points=np.asarray(points_vehicle_m,dtype=float)
    if points.ndim!=2 or points.shape[1]!=3 or not np.isfinite(points).all():
        raise ValueError('finite Nx3 path required')
    if now_sec<target_sent_sec or stamp_sec>now_sec:
        return LocalPathFeedback('CLOCK_INCONSISTENT',len(points),0.)
    if stamp_sec<target_sent_sec or now_sec-stamp_sec>maximum_age_sec:
        return LocalPathFeedback('STALE_OR_PREVIOUS_TARGET',len(points),0.)
    length=float(np.linalg.norm(np.diff(points,axis=0),axis=1).sum())
    if len(points)<2 or length==0.:
        return LocalPathFeedback('NO_NONTRIVIAL_LOCAL_PATH',len(points),length)
    return LocalPathFeedback('LOCAL_PATH_AVAILABLE_NOT_EXECUTED',len(points),length)
