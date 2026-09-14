"""Explicit XY-only CMU local-planner boundary; no inferred layer reachability.

Original localPlanner.cpp goalHandler reads point.x/y, not point.z. A 3-D
candidate alone therefore cannot authorize a different floor. The caller's
terrain/execution verifier must supply a current source-bound decision. This
transport records that assertion; it is not itself a physical safety proof.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class GroundTargetDecision:
    accepted: bool
    waypoint: dict | None
    structural_target_xyz_m: tuple
    consumed_axes: tuple
    reason: str
    evidence_ref: str | None


def prepare_ground_target(target_xyz_m, *, coordinate_frame, stamp_sec,
                          connected_ground_verified, verification_ref):
    xyz=tuple(target_xyz_m)
    if len(xyz)!=3 or any(isinstance(x,bool) or not isinstance(x,(float,int)) or not math.isfinite(x) for x in xyz):
        raise ValueError('finite3D structural target required')
    # No implicit transform: existing handoff publishes map-frame points.
    if coordinate_frame!='map' or not math.isfinite(stamp_sec):
        raise ValueError('explicit map transform and finite timestamp required')
    if type(connected_ground_verified) is not bool:
        raise ValueError('explicit ground-connection decision required')
    if not connected_ground_verified:
        return GroundTargetDecision(False,None,xyz,('x','y'),
            'ground_connection_unknown_do_not_rely_on_target_z',None)
    if not isinstance(verification_ref,str) or not verification_ref:
        raise ValueError('source-bound terrain/execution verification reference required')
    return GroundTargetDecision(True,dict(message_type='geometry_msgs/PointStamped',
        frame_id='map',stamp_sec=float(stamp_sec),xyz_m=list(xyz)),xyz,('x','y'),
        'xy_goal_only_local_planner_checks_obstacles',verification_ref)
