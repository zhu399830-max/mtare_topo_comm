"""Geometry-driven observation places, not inferred physical junction centers.

Three distinct supported continuations propose a local decision place. Repeated
observations support that proposal; clipping endpoints never become portals.
Task state does not imply a graph edge. Returning to a stored place still needs
the navigation backend's recorded traversal route.
"""
from copy import deepcopy
import math
import numpy as np


class GeometryBranchPlaces:
    def __init__(self, *, direction_tolerance_deg, target_match_m, place_radius_m, support_observations):
        if (not 0<direction_tolerance_deg<90 or target_match_m<=0 or place_radius_m<=0
                or not all(math.isfinite(x) for x in (direction_tolerance_deg,target_match_m,place_radius_m))
                or type(support_observations) is not int or support_observations<2):
            raise ValueError('explicit bounded place policy required')
        self.cosine=math.cos(math.radians(direction_tolerance_deg))
        self.target_distance=target_match_m;self.radius=place_radius_m;self.minimum=support_observations
        self.places=[];self.active=None;self.last_order=None;self.frame=None

    def _geometry(self,p):
        a=np.asarray(p['axis_start_xyz_m'],float);b=np.asarray(p['axis_target_xyz_m'],float)
        if a.shape!=(3,) or b.shape!=(3,) or not np.isfinite(a).all() or not np.isfinite(b).all() or not p.get('source_refs'):
            raise ValueError('supported finite primitive continuation required')
        d=b-a;n=np.linalg.norm(d)
        if n<=0:raise ValueError('nonzero continuation required')
        return b,d/n

    def observe(self,record,proposals,*,navigation_node_id):
        order=record['timestamp'];frame=record['coordinate_frame']
        if not math.isfinite(order) or (self.last_order is not None and order<=self.last_order):
            raise ValueError('strict causal observation order required')
        if self.frame is not None and frame!=self.frame:raise ValueError('coordinate stream changed')
        xyz=np.asarray(record['sensor_to_local_odometry'],float)[:3,3]
        if xyz.shape!=(3,) or not np.isfinite(xyz).all():raise ValueError('finite observer position required')
        groups=[]
        for p in sorted(proposals,key=lambda p:tuple(p['axis_target_xyz_m'])):
            target,direction=self._geometry(p)
            compatible=[g for g in groups if all(direction@x[1]>=self.cosine and np.linalg.norm(target-x[0])<=self.target_distance for x in g)]
            if len(compatible)>1:
                self.last_order=order;self.frame=frame;self.active=None
                return dict(status='AMBIGUOUS_GEOMETRY_GROUPING',place_id=None)
            if compatible:compatible[0].append((target,direction,p))
            else:groups.append([(target,direction,p)])
        self.last_order=order;self.frame=frame
        if len(groups)<3:
            self.active=None
            return dict(status='NO_MULTI_CONTINUATION_EVENT',place_id=None)
        def same_task(target,direction,p,task):
            if direction@np.asarray(task['direction_world'])<self.cosine:return False
            shared=not set(p['source_refs']).isdisjoint(task['source_refs'])
            return shared or np.linalg.norm(target-np.asarray(task['target_xyz_m']))<=self.target_distance
        if self.active is not None and np.linalg.norm(xyz-np.asarray(self.places[self.active]['observer_xyz_m']))>self.radius:
            old=self.places[self.active]['frontiers']
            # Continuity beyond the initial observer radius requires unique
            # shared original returns for EVERY current direction. No spatial
            # threshold expansion, and no cross-visit place merge.
            identities=[]
            for group in groups:
                target,direction,p=group[0]
                hits=[f['id'] for f in old if direction@np.asarray(f['direction_world'])>=self.cosine
                      and not set(p['source_refs']).isdisjoint(f['source_refs'])]
                identities.append(hits)
            if not all(len(h)==1 for h in identities) or len({h[0] for h in identities if len(h)==1})!=len(groups):
                self.active=None
        if self.active is None:
            self.active=len(self.places)
            self.places.append(dict(id=self.active,kind='multi_continuation_observation_place',
                observer_xyz_m=xyz.tolist(),navigation_node_id=navigation_node_id,
                physical_junction_center=False,state='provisional',orders=[],frontiers=[]))
        place=self.places[self.active];place['orders'].append(order)
        if len(place['orders'])>=self.minimum:place['state']='supported_observation'
        matched=set()
        for group in groups:
            target,direction,p=group[0]
            candidates=[f for f in place['frontiers'] if same_task(target,direction,p,f)]
            if len(candidates)>1 or (candidates and candidates[0]['id'] in matched):
                continue  # preserve uncertain old tasks; do not force identity
            if candidates:
                task=candidates[0]
            else:
                task=dict(id=len(place['frontiers']),state='unattempted',first_order=order,
                          physical_opening=False,connection_verified=False)
                place['frontiers'].append(task)
            matched.add(task['id'])
            task.update(target_xyz_m=target.tolist(),direction_world=direction.tolist(),
                        last_order=order,source_refs=list(p['source_refs']),
                        observation_binding=dict(order=order,observer_xyz_m=xyz.tolist(),
                                                 navigation_node_id=navigation_node_id))
            task['current_direction_supported']=p.get('current_direction_supported')
            if p.get('current_direction_supported') is True:
                task['return_observation']=dict(order=order,observer_xyz_m=xyz.tolist(),
                    navigation_node_id=navigation_node_id,target_xyz_m=target.tolist(),
                    direction_world=direction.tolist(),source_refs=list(p['source_refs']))
        return dict(status=place['state'],place_id=place['id'],
                    unattempted=[f['id'] for f in place['frontiers'] if f['state']=='unattempted'])

    def record_attempt(self,place_id,frontier_id,*,outcome,stamp_sec):
        if outcome not in {'requested','xy_arrived','blocked'} or not math.isfinite(stamp_sec):
            raise ValueError('explicit execution outcome required')
        place=self.places[place_id]
        if place['state']!='supported_observation':raise ValueError('unsupported place cannot schedule tasks')
        task=place['frontiers'][frontier_id]
        allowed={'unattempted':{'requested'},'requested':{'xy_arrived','blocked'},'xy_arrived':set(),'blocked':set()}
        if outcome not in allowed[task['state']] or stamp_sec<max(task['last_order'],task.get('execution_stamp_sec',float('-inf'))):
            raise ValueError('causal request before terminal execution feedback required')
        task['state']=outcome;task['execution_stamp_sec']=stamp_sec
        task['connection_verified']=False

    def snapshot(self):return deepcopy(dict(coordinate_frame=self.frame,places=self.places,verified_edges=[]))
