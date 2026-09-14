"""Tentative geometry goals -> original local planner -> pose progress.

No physical ground certificate, inferred graph edges or semantic labels. The
local planner receives requests, not commands to bypass obstacle checking.
Arrival is an XY odometry observation, not a floor identity verification.
"""
from copy import deepcopy
import math
import numpy as np


class GeometryExecution:
    def __init__(self, *, arrival_radius_m, target_timeout_s, revisit_radius_m):
        if any(not math.isfinite(v) or v<=0 for v in (arrival_radius_m,target_timeout_s,revisit_radius_m)):
            raise ValueError('explicit finite positive execution policy required')
        self.arrival=arrival_radius_m;self.timeout=target_timeout_s;self.revisit=revisit_radius_m
        self.active=None;self.events=[];self.visited=[];self.last_stamp=None;self.stopped=False

    def update(self, proposals, *, body_xyz_m, body_forward_world, stamp_sec):
        xyz=np.asarray(body_xyz_m,float);heading=np.asarray(body_forward_world,float)
        if (xyz.shape!=(3,) or heading.shape!=(3,) or not np.isfinite(xyz).all()
                or not np.isfinite(heading).all() or not np.isclose(np.linalg.norm(heading),1.)
                or not math.isfinite(stamp_sec) or (self.last_stamp is not None and stamp_sec<=self.last_stamp)):
            raise ValueError('finite body pose and strictly causal time required')
        self.last_stamp=stamp_sec
        if self.active is not None:
            distance=np.linalg.norm(xyz[:2]-np.asarray(self.active['target_xyz_m'])[:2])
            if distance<=self.arrival:
                self.events.append(dict(kind='XY_ARRIVAL_FROM_ODOMETRY',stamp_sec=stamp_sec,
                    target=deepcopy(self.active),body_xyz_m=xyz.tolist(),edge_verified=False))
                self.visited.append(self.active['target_xyz_m']);self.active=None
            elif stamp_sec-self.active['sent_sec']>=self.timeout:
                self.events.append(dict(kind='TARGET_TIMEOUT',stamp_sec=stamp_sec,target=deepcopy(self.active)))
                self.active=None;self.stopped=True
        if self.stopped:return self._request('HOLD_AFTER_TIMEOUT',xyz,stamp_sec)
        if self.active is not None:return self._request('PENDING_LOCAL_EXECUTION',self.active['target_xyz_m'],stamp_sec)
        candidates=[]
        for p in proposals:
            target=np.asarray(p['axis_target_xyz_m'],float)
            if (target.shape!=(3,) or not np.isfinite(target).all() or not p.get('source_refs')):
                raise ValueError('observed geometric target and source references required')
            delta=target-xyz
            if np.linalg.norm(delta[:2])<=self.arrival:continue
            if any(np.linalg.norm(target[:2]-np.asarray(v)[:2])<=self.revisit for v in self.visited):continue
            candidates.append((float(delta@heading/np.linalg.norm(delta)),target,p))
        if not candidates:return self._request('HOLD_NO_UNVISITED_GEOMETRY_TARGET',xyz,stamp_sec)
        _,target,p=min(candidates,key=lambda item:(-item[0],tuple(item[1])))
        self.active=dict(target_xyz_m=target.tolist(),sent_sec=stamp_sec,
            source_refs=list(p['source_refs']),physical_opening_confirmed=False)
        if 'branch_task' in p:self.active['branch_task']=deepcopy(p['branch_task'])
        if 'deferred_task' in p:self.active['deferred_task']=p['deferred_task']
        self.events.append(dict(kind='TENTATIVE_GOAL_REQUEST',stamp_sec=stamp_sec,target=deepcopy(self.active)))
        return self._request('NEW_LOCAL_PLANNING_REQUEST',target,stamp_sec)

    def _request(self,status,xyz,stamp):
        return dict(status=status,frame_id='map',xyz_m=list(xyz),stamp_sec=stamp,
            consumed_axes=['x','y'],ground_connection_verified=False,executed=False,
            original_local_planner_required=True,active=deepcopy(self.active))
