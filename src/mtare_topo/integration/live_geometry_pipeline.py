"""Causal live scan composition; no teacher, waypoint publication or safety claim.

The reused fitter supports relative yaw only. Reject non-yaw motion instead of
silently flattening it. Poses must describe the actual scan frame, not a vehicle
frame substituted without an extrinsic transform.
"""
from collections import deque
import numpy as np
from mtare_topo.integration.geometry_structure_trace import geometry_structure_record
from mtare_topo.integration.geometry_navigation_replay import GeometryNavigationReplay
from mtare_topo.semantics.full_sensor_local_fit import full_sensor_local_fit
from mtare_topo.semantics.observed_axis_structure import structure_from_observed_primitives
from mtare_topo.semantics.primitive_relation_nonlearning import PrimitiveRelationBaselineConfig
from mtare_topo.semantics.range_exit_baseline import RangeExitBaselineConfig


class LiveGeometryPipeline:
    def __init__(self, *, composition_policy, anchor_spacing_m, lookahead_m, rigid_motion=False,
                 predictor=None, checkpoint_sha256=None, constrained_observed_tasks=False):
        self.policy = dict(composition_policy)
        self.frames = deque(maxlen=5)
        self.graph = GeometryNavigationReplay(anchor_spacing_m=anchor_spacing_m,
                                              lookahead_m=lookahead_m)
        self.frame = None
        self.rigid_motion=rigid_motion
        self.predictor=predictor
        self.checkpoint_sha256=checkpoint_sha256
        self.constrained_observed_tasks=constrained_observed_tasks
        if constrained_observed_tasks and predictor is None:raise ValueError('model required for constrained tasks')
        if predictor is not None and (not rigid_motion or not checkpoint_sha256):
            raise ValueError('learned live path requires full rigid motion and checkpoint binding')

    def push(self, ranges_m, valid_mask, *, sensor_to_map, stamp_sec,
             source_key, coordinate_frame):
        ranges = np.asarray(ranges_m, dtype=np.float32)
        valid = np.asarray(valid_mask)
        pose = np.asarray(sensor_to_map, dtype=float)
        if (ranges.shape != (16,720) or valid.shape != ranges.shape
                or not np.isfinite(ranges).all() or np.any(ranges < 0)
                or np.any(ranges > 50) or np.any((valid != 0) & (valid != 1))
                or np.any((valid == 1) & (ranges <= 0))):
            raise ValueError('original finite 16x720 ranges and binary mask required')
        if (pose.shape != (4,4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0,0,0,1])
                or not np.allclose(pose[:3,:3].T @ pose[:3,:3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(pose[:3,:3]), 1)):
            raise ValueError('proper scan-frame pose required')
        if (not np.isfinite(stamp_sec) or not source_key or not coordinate_frame
                or (self.frame is not None and self.frame != coordinate_frame)):
            raise ValueError('finite timestamp and stable named coordinate frame required')
        if self.frames and (stamp_sec <= self.frames[-1][0]
                            or source_key in [f[1] for f in self.frames]):
            raise ValueError('strictly causal distinct scans required')
        candidate = list(self.frames)[-4:] + [(float(stamp_sec),source_key,
            pose.copy(),np.stack((ranges/50., valid.astype(np.float32))))]
        rotation = pose[:3,:3]
        translation = np.array([rotation.T @ (f[2][:3,3]-pose[:3,3]) for f in candidate])
        relative = np.array([rotation.T @ f[2][:3,:3] for f in candidate])
        # Current -> itself is exactly identity by construction, not an
        # estimated transform to be accepted by loosening a tolerance.
        relative[-1]=np.eye(3)
        translation[-1]=0.
        if not self.rigid_motion and (not np.allclose(relative[:,:,2], [0,0,1], atol=1e-6, rtol=0)
                or not np.allclose(relative[:,2,:], [0,0,1], atol=1e-6, rtol=0)):
            raise ValueError('relative roll/pitch unsupported by existing yaw-only fitter')
        yaw = np.degrees(np.arctan2(relative[:,1,0],relative[:,0,0]))
        if len(candidate) < 5:
            self.frames.append(candidate[-1]); self.frame=coordinate_frame
            return None
        values = np.stack([f[3] for f in candidate])
        if self.predictor is not None:
            from mtare_topo.integration.learned_geometry_trace import learned_geometry_record
            from mtare_topo.integration.common_geometry_domain import common_domain_record
            from mtare_topo.semantics.range_exit_baseline import RangeExitBaseline
            from mtare_topo.semantics.primitive_relation_nonlearning import ELEVATION_DEG,_sector_mask
            keys=[f[1] for f in candidate]
            prediction=self.predictor(values=values,translation=translation,yaw=yaw,
                rotation=relative,source_frame_keys=keys,timestamp=float(stamp_sec))
            if (prediction['source_frame_keys']!=keys or prediction['timestamp']!=float(stamp_sec)
                    or prediction['checkpoint_sha256']!=self.checkpoint_sha256):
                raise ValueError('live prediction source mismatch')
            mask=np.zeros(720,bool)
            sectors=RangeExitBaseline().predict(ranges,valid,np.asarray(ELEVATION_DEG))['sectors']
            for sector in sectors:
                mask|=_sector_mask(sector['heading_robot_deg'],sector['angular_width_deg'])
            record=learned_geometry_record(axis_controls_sensor_m=prediction['axes'],
                existence_probabilities=prediction['probabilities'],existence_threshold=.5,
                checkpoint_sha256=self.checkpoint_sha256,source_frame_keys=keys,
                sensor_to_world=pose,coordinate_frame=coordinate_frame,timestamp=stamp_sec,
                current_sector_columns=np.flatnonzero(mask).tolist())
            record=common_domain_record(record,10.)
            record.update(timestamp_kind='scan_seconds',registration='full_rigid',live_observation=True,
                          frontend='frozen_learned_geometry',inference_evidence=prediction.get('evidence'))
            if self.constrained_observed_tasks:
                from mtare_topo.integration.geometry_constrained_tasks import constrained_tasks
                observed=[]
                for sector in sectors:
                    columns=_sector_mask(sector['heading_robot_deg'],sector['angular_width_deg'])
                    horizon=np.abs(np.asarray(ELEVATION_DEG))<=5
                    supported=valid.astype(bool)&(ranges>=self.graph.lookahead)&horizon[:,None]&columns[None,:]
                    refs=[keys[-1]+'/ray:'+str(int(i)) for i in np.flatnonzero(supported)]
                    if refs:observed.append(dict(heading_robot_deg=sector['heading_robot_deg'],source_refs=refs))
                proposals,audit=constrained_tasks(record,observed,lookahead_m=self.graph.lookahead,direction_tolerance_deg=15)
                record.update(frontend='frozen_learned_geometry_constrained',task_constraint_audit=audit)
                decision=self.graph.update(record,task_proposals=proposals)
            else:decision=self.graph.update(record)
            self.frames.append(candidate[-1]);self.frame=coordinate_frame
            return dict(geometry=record,decision=decision,control_published=False)
        primitives = full_sensor_local_fit(values,translation,yaw,
            fit_config=PrimitiveRelationBaselineConfig(), exit_config=RangeExitBaselineConfig(),
            rotations=relative if self.rigid_motion else None)
        structures = structure_from_observed_primitives(primitives,**self.policy)
        record = geometry_structure_record(structures,
            source_frame_keys=[f[1] for f in candidate],
            source_ray_indices=np.flatnonzero(values[:,1].reshape(-1)),
            sensor_to_world=pose,coordinate_frame=coordinate_frame,timestamp=stamp_sec)
        record.update(timestamp_kind='scan_seconds', sensor_to_local_odometry=pose.tolist(),
                      registration='full_rigid' if self.rigid_motion else 'yaw_only',
                      live_observation=True)
        decision = self.graph.update(record)
        self.frames.append(candidate[-1]); self.frame=coordinate_frame
        return dict(geometry=record,decision=decision,control_published=False)
