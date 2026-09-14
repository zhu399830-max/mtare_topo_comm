"""Sequential execution of a frozen recorded return route through XY planner."""
from copy import deepcopy
import math
import numpy as np


class RecordedReturnExecution:
    def __init__(self,route,*,created_sec,arrival_radius_m,target_timeout_s):
        if route.get('kind')!='recorded_trace_return' or route.get('new_edge_inferred') is not False:
            raise ValueError('recorded return route required')
        self.samples=deepcopy(route['samples'])
        if not self.samples or any(s['order']>created_sec for s in self.samples):
            raise ValueError('return cannot use future observations')
        for s in self.samples:
            point=np.asarray(s['xyz_m'],float)
            if point.shape!=(3,) or not np.isfinite(point).all() or not math.isfinite(s['order']):
                raise ValueError('finite recorded samples required')
        if not all(math.isfinite(v) for v in (created_sec,arrival_radius_m,target_timeout_s)) or min(arrival_radius_m,target_timeout_s)<=0:
            raise ValueError('explicit finite execution policy required')
        self.arrival=arrival_radius_m;self.timeout=target_timeout_s;self.index=0
        self.sent=None;self.last=None;self.stopped=False;self.events=[]
        self.created=created_sec

    def update(self,body_xyz_m,stamp_sec):
        xyz=np.asarray(body_xyz_m,float)
        if xyz.shape!=(3,) or not np.isfinite(xyz).all() or not math.isfinite(stamp_sec) or stamp_sec<self.created or (self.last is not None and stamp_sec<=self.last):
            raise ValueError('causal finite return feedback required')
        self.last=stamp_sec
        if self.stopped:return self._message('RETURN_TIMEOUT_HOLD',xyz,stamp_sec)
        while self.index<len(self.samples):
            target=self.samples[self.index]
            if np.linalg.norm(xyz[:2]-np.asarray(target['xyz_m'])[:2])>self.arrival:break
            self.events.append(dict(kind='RECORDED_SAMPLE_REACHED_XY',source_order=target['order'],
                                    stamp_sec=stamp_sec,body_xyz_m=xyz.tolist()))
            self.index+=1;self.sent=None
        if self.index==len(self.samples):return self._message('RETURN_OBSERVATION_REACHED',xyz,stamp_sec)
        if self.sent is None:self.sent=stamp_sec
        elif stamp_sec-self.sent>=self.timeout:
            self.stopped=True
            self.events.append(dict(kind='RETURN_TIMEOUT',stamp_sec=stamp_sec,sample_index=self.index))
            return self._message('RETURN_TIMEOUT_HOLD',xyz,stamp_sec)
        return self._message('FOLLOW_RECORDED_RETURN',self.samples[self.index]['xyz_m'],stamp_sec)

    def _message(self,status,xyz,stamp):
        return dict(status=status,xyz_m=list(xyz),frame_id='map',stamp_sec=stamp,
                    sample_index=self.index,ground_connection_verified=False,
                    local_planner_revalidation_required=True,new_edge_inferred=False)
