"""Existing six-field scan packages -> finite geometry composition records.

No file access, labels, model selection, synthetic portal completion or graph
edges. Caller binds the exact package SHA and manifest before reading bytes.
The local odometry origin is reset for every package; timestamps below are
source frame order indices, not invented wall-clock seconds.
"""
import numpy as np
from mtare_topo.data.gse_supplement_feature_input_v1 import bind_feature_input
from mtare_topo.evaluation.continuous_anchor_motion_v1 import compose_current_frames
from mtare_topo.integration.geometry_structure_trace import geometry_structure_record
from mtare_topo.semantics.full_sensor_local_fit import full_sensor_local_fit
from mtare_topo.semantics.observed_axis_structure import structure_from_observed_primitives
from mtare_topo.semantics.primitive_relation_nonlearning import PrimitiveRelationBaselineConfig
from mtare_topo.semantics.range_exit_baseline import RangeExitBaselineConfig


def replay_package(payload, manifest, *, composition_policy,
                   fit_config=None, exit_config=None):
    """Compute each observation once, preserving prefix-only motion.

    No count is inferred from the payload: the independently pinned manifest
    supplies source IDs, source frames, package size and population size.
    """
    count=manifest['observations']
    if (type(count) is not int or not 1<=count<=64
            or len(manifest['source_sequence_ids'])!=count
            or len(manifest['source_frames'])!=count
            or manifest['bytes']!=len(payload)):
        raise ValueError('exact manifest population and byte count required')
    fit_config=fit_config or PrimitiveRelationBaselineConfig()
    exit_config=exit_config or RangeExitBaselineConfig()
    translations=[];yaws=[];frames=[];records=[]
    for i in range(count):
        bound=bind_feature_input(payload, expected_sha256=manifest['sha256'],
            task=manifest['task'],row=i,source_sequence_id=manifest['source_sequence_ids'][i],
            frame_rows=manifest['source_frames'][i],observation_count=count)
        translations.append(bound.translation_m);yaws.append(bound.yaw_deg)
        frames.append(manifest['source_frames'][i])
        rotations,positions=compose_current_frames(translations,yaws,frames)
        primitives=full_sensor_local_fit(bound.range_valid,bound.translation_m,bound.yaw_deg,
            fit_config=fit_config,exit_config=exit_config)
        structure=structure_from_observed_primitives(primitives,**composition_policy)
        matrix=np.eye(4);matrix[:3,:3]=rotations[-1];matrix[:3,3]=positions[-1]
        record=geometry_structure_record(structure,
            source_frame_keys=tuple(f"{manifest['task']}/frame:{f}" for f in frames[-1]),
            source_ray_indices=np.flatnonzero(bound.range_valid[:,1].reshape(-1)),
            sensor_to_world=matrix,coordinate_frame=manifest['task']+'/first_sensor',
            timestamp=float(frames[-1][-1]))
        record.update(timestamp_kind='source_frame_order_not_seconds',
            source_sequence_id=manifest['source_sequence_ids'][i],
            input_binding_sha256=bound.input_binding_sha256,
            sensor_to_local_odometry=matrix.tolist())
        records.append(record)
    return records
