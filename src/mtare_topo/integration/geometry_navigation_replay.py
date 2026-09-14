"""Given-pose geometry-to-place-graph integration, not autonomous exploration.

Nodes are explicitly metric anchors, not learned junction detections. Geometry
is attached as structural hypotheses and directly determines target proposals.
Only adjacent recorded poses supply edges. No loop merge or unseen connectivity
is inferred; this diagnostic must not be reported as a closed-loop result.
"""
from copy import deepcopy
import math
import numpy as np
from mtare_topo.planning.primitive_corridor import corridor_continuations


class GeometryNavigationReplay:
    def __init__(self, *, anchor_spacing_m, lookahead_m):
        if any(not math.isfinite(v) or v<=0 for v in (anchor_spacing_m,lookahead_m)):
            raise ValueError('explicit finite positive integration distances required')
        self.spacing=anchor_spacing_m;self.lookahead=lookahead_m
        self.nodes=[];self.edges=[];self.decisions=[];self.pending=[]
        self.frame=None;self.last_order=None;self.last_source_frames=None
        self.distance=0.;self.current=None

    def update(self, record, *, continuous=True, task_proposals=None):
        if record.get('schema_version')!='geometry_structure_trace_v1':
            raise ValueError('geometry trace required')
        frame=record['coordinate_frame'];order=record['timestamp']
        pose=np.asarray(record['sensor_to_local_odometry'],float)
        if (pose.shape!=(4,4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3],[0,0,0,1])
                or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3,:3]),1.)
                or not math.isfinite(order)):
            raise ValueError('proper finite pose and source order required')
        sources=record['source_frame_keys']
        if len(sources)!=5 or len(set(sources))!=5:raise ValueError('five unique source frames required')
        if self.frame is not None and frame!=self.frame:
            raise ValueError('one coordinate stream per runtime; never join variants')
        if self.last_order is not None and order<=self.last_order:raise ValueError('strict causal order required')
        if continuous and self.last_source_frames is not None and sources[:-1]!=self.last_source_frames[1:]:
            raise ValueError('discontinuous source windows require explicit segment boundary')
        xyz=pose[:3,3]
        proposals=[]
        for primitive in record['primitives']:
            axis=primitive['axis_controls_world_m']
            if axis is None:continue
            refs=[f"{p['frame_key']}/ray:{p['ray_index']}" for p in primitive['source_rays']]
            prediction=primitive.get('prediction_provenance')
            if prediction is not None:
                if (refs or primitive.get('fit_status')!='model_prediction_not_surface_fit'
                        or prediction['input_frame_keys']!=list(sources)
                        or prediction['observation_order']!=order):
                    raise ValueError('prediction provenance cannot masquerade as fitted ray evidence')
                refs=[f"prediction:{prediction['checkpoint_sha256']}:{order}:slot:{prediction['slot']}"]
            result=corridor_continuations(axis,xyz,lookahead_m=self.lookahead,
                primitive_index=primitive['index'],source_refs=refs)
            columns=primitive.get('current_sector_columns')
            for proposal in result['proposals']:
                # Test each continuation independently: the opposite end of a
                # bidirectional axis does not inherit visibility from this end.
                direction=pose[:3,:3].T@(np.asarray(proposal['axis_target_xyz_m'])-np.asarray(proposal['axis_start_xyz_m']))
                if columns is None or np.linalg.norm(direction[:2])<=np.finfo(float).eps:
                    supported=None
                else:
                    heading=np.degrees(np.arctan2(direction[1],direction[0]))%360
                    supported=int(np.rint(heading*2))%720 in columns
                proposal['current_direction_supported']=supported
                proposal['geometry_source_kind']='model_prediction' if prediction is not None else 'observed_surface_fit'
                if prediction is not None:proposal['prediction_provenance']=deepcopy(prediction)
            proposals.extend(result['proposals'])
        raw_axis_proposals=None
        if task_proposals is not None:
            if record.get('frontend')!='frozen_learned_geometry_constrained':
                raise ValueError('explicit hybrid record required for task proposals')
            for p in task_proposals:
                if (p.get('geometry_source_kind')!='learned_constrained_observation'
                        or not p.get('learned_geometry_support') or p.get('current_direction_supported') is not True
                        or not p.get('source_refs') or p.get('creates_edge') is not False):
                    raise ValueError('task must bind learned and observed evidence')
            raw_axis_proposals=deepcopy(proposals)
            proposals=deepcopy(task_proposals)
        # Prefer progress along the current observed heading; no future route.
        def rank(p):
            delta=np.asarray(p['axis_target_xyz_m'])-xyz
            norm=np.linalg.norm(delta)
            alignment=float(delta@pose[:3,0]/norm) if norm else -1.
            return (-alignment,p['approach_offset_m'],tuple(p['axis_target_xyz_m']))
        selected=min(proposals,key=rank) if proposals else None
        if not continuous:
            self.pending=[];self.distance=0.;self.current=None
        sample=dict(order=order,xyz_m=xyz.tolist())
        if self.pending:
            self.distance+=float(np.linalg.norm(xyz-np.asarray(self.pending[-1]['xyz_m'])))
        self.pending.append(sample)
        if self.current is None or self.distance>=self.spacing:
            previous=self.current;self.current=len(self.nodes)
            self.nodes.append(dict(id=self.current,kind='metric_anchor_not_semantic_node',
                xyz_m=xyz.tolist(),order=order,geometry_observation_orders=[]))
            if previous is not None:
                self.edges.append(dict(source=previous,target=self.current,
                    kind='recorded_pose_traversal_not_planner_execution',length_m=self.distance,
                    trace=deepcopy(self.pending)))
            self.pending=[sample];self.distance=0.
        self.nodes[self.current]['geometry_observation_orders'].append(order)
        decision=dict(order=order,node=self.current,source_frame_keys=list(sources),
            structural_hypotheses=deepcopy(record['structures']),proposals=proposals,
            selected=deepcopy(selected),status='LOCAL_PLANNER_VALIDATION_REQUIRED' if selected else 'NO_GEOMETRY_TARGET',
            executed=False,geometry_used=selected is not None)
        if raw_axis_proposals is not None:decision['raw_axis_proposals_not_executable']=raw_axis_proposals
        self.decisions.append(decision);self.frame=frame;self.last_order=order
        self.last_source_frames=list(sources)
        return deepcopy(decision)

    def snapshot(self):
        return deepcopy(dict(schema_version='geometry_navigation_replay_v1',
            coordinate_frame=self.frame,nodes=self.nodes,edges=self.edges,decisions=self.decisions,
            pending_traversal=self.pending,given_pose=True,closed_loop=False,
            semantic_nodes_verified=False,opening_detection_verified=False))
