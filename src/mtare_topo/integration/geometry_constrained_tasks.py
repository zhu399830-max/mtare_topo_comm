"""Observed horizontal tasks constrained by learned local segment directions.

This is an explicit hybrid interface, not learned opening/center detection.
Predicted segments are unoriented local geometric evidence, never a path to
traverse in control-point order. Raw predictions remain unchanged.
"""
from copy import deepcopy
import math
import numpy as np


def constrained_tasks(record, sectors, *, lookahead_m, direction_tolerance_deg):
    if not 0<lookahead_m<=10 or not 0<direction_tolerance_deg<90:
        raise ValueError('explicit bounded task geometry required')
    pose=np.asarray(record['sensor_to_local_odometry'],float)
    if pose.shape!=(4,4) or not np.isfinite(pose).all():raise ValueError('finite pose required')
    primitives=record['primitives'];proposals=[];rejected=[]
    cosine=math.cos(math.radians(direction_tolerance_deg))
    for sector_index,sector in enumerate(sectors):
        heading=float(sector['heading_robot_deg']);refs=sector['source_refs']
        if not math.isfinite(heading) or not refs:raise ValueError('observed heading and ray provenance required')
        for ref in refs:
            if ref.rsplit('/ray:',1)[0]!=record['source_frame_keys'][-1]:
                raise ValueError('task support must come from current scan')
        angle=math.radians(heading)
        sensor_direction=np.array([math.cos(angle),math.sin(angle),0.])
        direction=pose[:3,:3]@sensor_direction
        supporters=[]
        for primitive in primitives:
            provenance=primitive.get('prediction_provenance')
            if provenance is None:raise ValueError('learned geometry required, no fitted fallback')
            if (provenance['input_frame_keys']!=record['source_frame_keys']
                    or provenance['observation_order']!=record['timestamp']):
                raise ValueError('model evidence not bound to current observation')
            axis=np.asarray(primitive['axis_controls_world_m'],float)
            for segment,v in enumerate(np.diff(axis,axis=0)):
                norm=np.linalg.norm(v)
                if norm>0 and abs(float(v@direction/norm))>=cosine:
                    supporters.append(dict(primitive_index=primitive['index'],segment_index=segment,
                        prediction_provenance=deepcopy(provenance)))
        if not supporters:
            rejected.append(dict(sector_index=sector_index,reason='NO_LEARNED_GEOMETRY_SUPPORT'));continue
        target=pose[:3,3]+lookahead_m*direction
        proposals.append(dict(kind='observed_task_with_learned_geometry_constraint',primitive_index=sector_index,
            axis_start_xyz_m=pose[:3,3].tolist(),axis_target_xyz_m=target.tolist(),axis_travel_m=float(lookahead_m),
            approach_offset_m=0.,reaches_observation_boundary=False,source_refs=list(refs),
            physical_opening=False,traversability='unknown_requires_local_planner',creates_edge=False,
            current_direction_supported=True,geometry_source_kind='learned_constrained_observation',
            target_coordinate_source='current_observed_sector_and_fixed_lookahead',
            learned_geometry_support=supporters,sector_heading_robot_deg=heading,
            vertical_opening_inferred=False))
    return proposals,dict(observed_candidates=len(sectors),accepted=len(proposals),rejected=rejected,
        learned_support_required=True,physical_openings_confirmed=False)
