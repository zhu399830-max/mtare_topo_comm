"""Supported curved-axis continuation proposals, NOT detected openings.

The three existing cross-section centers are used as a piecewise-linear axis;
no claim of spline recovery, floor support or collision-free space is made.
Consumers must validate robot/sensor height conventions and use the retained
local planner before execution. A support boundary is never a dead end.
"""
import numpy as np


def corridor_continuations(axis_controls_m, robot_xyz_m, *, lookahead_m,
                           primitive_index, source_refs):
    axis=np.asarray(axis_controls_m,dtype=float)
    robot=np.asarray(robot_xyz_m,dtype=float)
    if (axis.shape!=(3,3) or robot.shape!=(3,) or not np.isfinite(axis).all()
            or not np.isfinite(robot).all() or isinstance(lookahead_m,bool)
            or not np.isfinite(lookahead_m) or lookahead_m<=0):
        raise ValueError('three finite section centers, robot xyz and positive lookahead required')
    if type(primitive_index) is not int or primitive_index<0:
        raise ValueError('local primitive index required')
    if not isinstance(source_refs,(tuple,list)) or not source_refs or any(not isinstance(s,str) or not s for s in source_refs):
        raise ValueError('observed primitive source references required')
    # Canonical orientation makes reversal independent of output identity.
    if tuple(axis[-1])<tuple(axis[0]):axis=axis[::-1].copy()
    delta=np.diff(axis,axis=0);length=np.linalg.norm(delta,axis=1)
    if np.any(length<=np.finfo(float).eps):
        raise ValueError('nondegenerate finite segments required')
    cumulative=np.r_[0.,np.cumsum(length)]
    fractions=np.clip(np.sum((robot-axis[:-1])*delta,axis=1)/(length**2),0.,1.)
    projected=axis[:-1]+fractions[:,None]*delta
    distances=np.linalg.norm(projected-robot,axis=1)
    nearest=int(np.argmin(distances))
    start=float(cumulative[nearest]+fractions[nearest]*length[nearest])
    # Distinct equally near branches are an ambiguous attachment, not a guess.
    tied=np.isclose(distances,distances[nearest],rtol=0.,atol=1e-10)
    if any(np.linalg.norm(p-projected[nearest])>1e-9 for p in projected[tied]):
        return dict(status='AMBIGUOUS_AXIS_ATTACHMENT',proposals=[],opening_count=None)
    proposals=[]
    for sign in (-1,1):
        end=float(np.clip(start+sign*lookahead_m,0.,cumulative[-1]))
        travel=abs(end-start)
        if travel<=np.finfo(float).eps:continue
        segment=min(int(np.searchsorted(cumulative,end,side='right')-1),1)
        point=axis[segment]+((end-cumulative[segment])/length[segment])*delta[segment]
        proposals.append(dict(kind='corridor_continuation_proposal',
            primitive_index=primitive_index,axis_target_xyz_m=point.tolist(),
            axis_start_xyz_m=projected[nearest].tolist(),axis_travel_m=travel,
            approach_offset_m=float(distances[nearest]),
            reaches_observation_boundary=bool(end in (0.,cumulative[-1])),
            source_refs=list(source_refs),physical_opening=False,
            traversability='unknown_requires_local_planner',creates_edge=False))
    proposals.sort(key=lambda p:tuple(p['axis_target_xyz_m']))
    return dict(status='PROPOSALS' if proposals else 'NO_SUPPORTED_CONTINUATION',
                proposals=proposals,opening_count=None)
