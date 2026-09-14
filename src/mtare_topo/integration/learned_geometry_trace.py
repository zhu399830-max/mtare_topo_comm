"""Prediction-only geometry transport; never fabricates fitted-ray evidence."""
import math
import re
import numpy as np


def learned_geometry_record(*, axis_controls_sensor_m, existence_probabilities,
                            existence_threshold, checkpoint_sha256, source_frame_keys,
                            sensor_to_world, coordinate_frame, timestamp,
                            current_sector_columns=None):
    axis=np.asarray(axis_controls_sensor_m,float)
    probabilities=np.asarray(existence_probabilities,float)
    pose=np.asarray(sensor_to_world,float)
    if (axis.ndim!=3 or axis.shape[1:]!=(3,3) or len(axis)>32
            or probabilities.shape!=(len(axis),) or not np.isfinite(axis).all()
            or not np.isfinite(probabilities).all() or np.any((probabilities<0)|(probabilities>1))):
        raise ValueError('finite at-most-32 predicted axes and probabilities required')
    if not math.isfinite(existence_threshold) or not 0<=existence_threshold<=1:
        raise ValueError('explicit frozen existence threshold required')
    if not re.fullmatch('[0-9a-f]{64}',checkpoint_sha256):raise ValueError('checkpoint SHA-256 required')
    if (len(source_frame_keys)!=5 or len(set(source_frame_keys))!=5
            or any(not isinstance(k,str) or not k for k in source_frame_keys)):
        raise ValueError('five distinct input frame keys required')
    if (pose.shape!=(4,4) or not np.isfinite(pose).all()
            or not np.allclose(pose[3],[0,0,0,1])
            or not np.allclose(pose[:3,:3].T@pose[:3,:3],np.eye(3),atol=1e-6)
            or not np.isclose(np.linalg.det(pose[:3,:3]),1)):
        raise ValueError('proper sensor pose required')
    if not coordinate_frame or not isinstance(coordinate_frame,str) or not math.isfinite(timestamp):
        raise ValueError('frame and timestamp required')
    if current_sector_columns is not None:
        if any(type(c) is not int or not 0<=c<720 for c in current_sector_columns):
            raise ValueError('current observed sector columns required')
        current_sector_columns=sorted(set(current_sector_columns))
    primitives=[];rejected=[]
    for index,(controls,probability) in enumerate(zip(axis,probabilities)):
        if probability<existence_threshold:
            rejected.append([index,'below_frozen_existence_threshold']);continue
        if np.any(np.linalg.norm(np.diff(controls,axis=0),axis=1)<=np.finfo(float).eps):
            rejected.append([index,'degenerate_predicted_axis']);continue
        primitives.append(dict(index=index,fit_status='model_prediction_not_surface_fit',
            axis_controls_world_m=(controls@pose[:3,:3].T+pose[:3,3]).tolist(),
            source_rays=[],residual=None,existence_probability=float(probability),
            current_sector_columns=current_sector_columns,
            prediction_provenance=dict(checkpoint_sha256=checkpoint_sha256,
                input_frame_keys=list(source_frame_keys),slot=index,observation_order=float(timestamp))))
    return dict(schema_version='geometry_structure_trace_v1',timestamp=float(timestamp),
        coordinate_frame=coordinate_frame,source_frame_keys=list(source_frame_keys),
        sensor_to_local_odometry=pose.tolist(),primitives=primitives,structures=[],
        rejected_axes=rejected,physical_openings_confirmed=False,traversed_edges=[],
        evidence_scope='predicted_axis_with_independent_current_direction_evidence',
        raw_candidate_count=len(axis),existence_threshold=float(existence_threshold))
