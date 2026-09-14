"""Local geometry-place task selection over the existing execution handoff.

Remote return follows recorded samples and requires current reobservation.
Stored geometry and its return viewpoint must share one observation binding.
"""
from copy import deepcopy
import numpy as np
from mtare_topo.integration.geometry_execution import GeometryExecution
from mtare_topo.topology.geometry_branch_places import GeometryBranchPlaces
from mtare_topo.planning.recorded_return_route import recorded_return_route
from mtare_topo.integration.recorded_return_execution import RecordedReturnExecution


class BranchGeometryExecution:
    def __init__(self, *, execution_policy, place_policy, remote_returns=False):
        self.execution=GeometryExecution(**execution_policy)
        self.places=GeometryBranchPlaces(**place_policy)
        self.events=self.execution.events
        self.decisions=[]
        self.remote_returns=remote_returns;self.history=[];self.deferred=[]
        self.returning=None;self.return_events=[];self.unresolved={}

    def update_geometry(self,result,*,body_xyz_m,body_forward_world,stamp_sec,navigation_graph=None):
        g=result['geometry'];proposals=result['decision']['proposals']
        if g.get('frontend') in {'frozen_learned_geometry','frozen_learned_geometry_constrained'}:
            # Preserve every model proposal in the recorded graph, but require
            # the same independent current support for ALL execution branches,
            # including ordinary corridor fallback and return reobservation.
            allowed=[p for p in proposals if p.get('current_direction_supported') is True]
            result['execution_support_audit']=dict(total_proposals=len(proposals),
                supported_proposals=len(allowed),
                unsupported_proposals=sum(p.get('current_direction_supported') is False for p in proposals),
                unknown_proposals=sum(p.get('current_direction_supported') is None for p in proposals),
                raw_graph_proposals_preserved=True)
            proposals=allowed
        observation=self.places.observe(g,proposals,navigation_node_id=result['decision']['node'])
        self.history.append(list(body_xyz_m))
        forced=None
        if self.returning is not None:
            request=self.returning['executor'].update(body_xyz_m,stamp_sec)
            if request['status']!='RETURN_OBSERVATION_REACHED':
                self.return_events.append(dict(stamp_sec=stamp_sec,status=request['status'],sample_index=request['sample_index']))
                return request
            destination=self.returning['destination']
            if not self.returning.get('reached'):
                self.return_events.append(dict(stamp_sec=stamp_sec,status='RETURN_PLACE_REACHED',destination=deepcopy(destination)))
                self.returning['reached']=stamp_sec
            matches=[]
            for p in proposals:
                if p.get('current_direction_supported') is not True:continue
                target,direction=self.places._geometry(p)
                if direction@np.asarray(destination['direction_world'])>=self.places.cosine and np.linalg.norm(target-np.asarray(destination['target_xyz_m']))<=self.places.target_distance:
                    matches.append(p)
            if len(matches)!=1:
                if stamp_sec-self.returning['reached']>=self.execution.timeout:
                    self.unresolved[destination['key']]='not_uniquely_reobserved_after_return'
                    self.returning=None
                return dict(status='HOLD_RETURN_REOBSERVATION',xyz_m=list(body_xyz_m),frame_id='map',stamp_sec=stamp_sec,ground_connection_verified=False)
            forced=deepcopy(matches[0]);forced.update(destination['identity'])
            self.return_events.append(dict(stamp_sec=stamp_sec,status='RETURN_DIRECTION_REOBSERVED',destination_key=destination['key']))
            self.returning=None
        selected=None
        if forced is not None:proposals=[forced]
        elif observation.get('status')=='supported_observation':
            place=self.places.places[observation['place_id']]
            candidates=[f for f in place['frontiers'] if f['state']=='unattempted' and f['last_order']==g['timestamp'] and f.get('current_direction_supported') is True]
            # At a decision place, prioritize a lateral alternative over
            # straight/back continuations. This is a fixed engineering policy,
            # not a learned planner or a claim that the branch is traversable.
            if candidates:
                selected=min(candidates,key=lambda f:(abs(float(np.asarray(f['direction_world'])@np.asarray(body_forward_world))),f['id']))
                proposals=[dict(axis_target_xyz_m=list(selected['target_xyz_m']),
                    source_refs=list(selected['source_refs']),
                    branch_task=dict(place_id=place['id'],frontier_id=selected['id']))]
        active=self.execution.active
        free=active is None or np.linalg.norm(np.asarray(active['target_xyz_m'])[:2]-np.asarray(body_xyz_m)[:2])<=self.execution.arrival
        if self.remote_returns and forced is None and selected is None and free and not self.execution.stopped:
            if navigation_graph is None:raise ValueError('recorded navigation graph required for return')
            for destination in self._remote_candidates(body_xyz_m):
                try:
                    route=recorded_return_route(nodes=navigation_graph.nodes,edges=navigation_graph.edges,
                        pending=navigation_graph.pending,current_node=navigation_graph.current,
                        target_node=destination['navigation_node_id'],target_order=destination['order'])
                except ValueError as exc:
                    self.return_events.append(dict(stamp_sec=stamp_sec,status='NO_RECORDED_RETURN',reason=str(exc),destination_key=destination['key']))
                    continue
                # Settle the completed local goal without issuing another.
                before=len(self.events)
                self.execution.update([],body_xyz_m=body_xyz_m,body_forward_world=body_forward_world,stamp_sec=stamp_sec)
                self._feedback(before,stamp_sec)
                for p in proposals:
                    if p.get('current_direction_supported') is True and self._unvisited(p['axis_target_xyz_m']):
                        target,direction=self.places._geometry(p)
                        self.deferred.append(dict(state='unattempted',order=g['timestamp'],observer_xyz_m=list(body_xyz_m),
                            navigation_node_id=result['decision']['node'],target_xyz_m=target.tolist(),direction_world=direction.tolist(),source_refs=list(p['source_refs'])))
                self.returning=dict(destination=destination,executor=RecordedReturnExecution(route,created_sec=stamp_sec,
                    arrival_radius_m=self.execution.arrival,target_timeout_s=self.execution.timeout))
                self.return_events.append(dict(stamp_sec=stamp_sec,status='RETURN_STARTED',destination=deepcopy(destination),route=route))
                return self.returning['executor'].update(body_xyz_m,stamp_sec)
        before=len(self.events)
        request=self.execution.update(proposals,body_xyz_m=body_xyz_m,
            body_forward_world=body_forward_world,stamp_sec=stamp_sec)
        self._feedback(before,stamp_sec)
        pending=sum(f['state']=='unattempted' for p in self.places.places if p['state']=='supported_observation' for f in p['frontiers'])
        self.decisions.append(dict(stamp_sec=stamp_sec,observation=observation,
            request_status=request['status'],task=request.get('active',{}).get('branch_task') if request.get('active') else None,
            pending_tasks=pending))
        return request

    def _feedback(self,before,stamp_sec):
        for event in self.events[before:]:
            outcome={'TENTATIVE_GOAL_REQUEST':'requested','XY_ARRIVAL_FROM_ODOMETRY':'xy_arrived','TARGET_TIMEOUT':'blocked'}[event['kind']]
            if 'deferred_task' in event['target']:
                self.deferred[event['target']['deferred_task']]['state']=outcome
            identity=event['target'].get('branch_task')
            if identity is None:continue
            self.places.record_attempt(**identity,outcome=outcome,stamp_sec=stamp_sec)

    def _unvisited(self,target):
        return all(np.linalg.norm(np.asarray(target)-p)>self.execution.revisit for p in self.history)

    def _remote_candidates(self,xyz):
        tasks=[]
        for p in self.places.places:
            if p['state']!='supported_observation':continue
            for f in p['frontiers']:
                if f['state']!='unattempted':continue
                binding=f.get('return_observation')
                if binding is None:continue  # task retained, no proven return viewpoint
                if binding['order']>f['last_order']:
                    raise ValueError('return observation cannot follow task observation')
                candidate=deepcopy(binding)
                candidate.update(key='branch:%d:%d'%(p['id'],f['id']),
                    identity=dict(branch_task=dict(place_id=p['id'],frontier_id=f['id'])))
                tasks.append(candidate)
        for i,t in enumerate(self.deferred):
            if t['state']=='unattempted':tasks.append(dict(t,key='deferred:'+str(i),identity=dict(deferred_task=i)))
        return sorted([t for t in tasks if t['key'] not in self.unresolved and self._unvisited(t['target_xyz_m'])
                       and np.linalg.norm(np.asarray(t['observer_xyz_m'])-xyz)>self.places.radius],key=lambda t:(t['order'],t['key']))

    def snapshot(self):
        return dict(places=self.places.snapshot(),decisions=deepcopy(self.decisions),
                    remote_return_implemented=self.remote_returns,return_events=deepcopy(self.return_events),
                    deferred=deepcopy(self.deferred),unresolved=deepcopy(self.unresolved))
