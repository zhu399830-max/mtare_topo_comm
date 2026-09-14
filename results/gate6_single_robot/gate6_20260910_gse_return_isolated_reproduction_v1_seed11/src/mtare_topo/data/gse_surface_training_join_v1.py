"""One authenticated feature/scan/partial-target join; no IO or qualification."""
from copy import deepcopy
import numpy as np
from .gse_surface_feature_cache_v1 import bind_feature_cache
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion
from mtare_topo.representation.gse_surface_observed_inputs_v1 import build_observed_surface_inputs
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.representation.gse_surface_training_v1 import SurfaceTrainingExample
from mtare_topo.representation.gse_surface_training_context_v1 import SurfaceSourceLossContext,validate_context


def join_training_observation(*, feature_bytes, feature_entry, source_input,
                              encoder_state_sha256, produced_targets, target_manifest,
                              bundle, opening_matching_radius_m, feature_projection_device='cpu'):
    source=bundle['source'];provenance=source_input.provenance
    if any(source[k]!=provenance[k] for k in ('task','source_sequence_id','frame_rows')):
        raise ValueError('feature and teacher observation identity mismatch')
    if target_manifest['source_binding']['source']!={k:source[k] for k in ('task','source_sequence_id','frame_rows')}:
        raise ValueError('target manifest observation mismatch')
    student=bundle['student']
    expected_image=np.stack((student['ranges_m']/np.float32(50),student['valid_mask'].astype(np.float32)),axis=1)
    if any(not np.array_equal(a,b) for a,b in (
        (source_input.range_valid,expected_image),
        (source_input.translation_m,student['relative_translation_current_sensor_m']),
        (source_input.yaw_deg,student['relative_yaw_current_sensor_deg']))):
        raise ValueError('feature and teacher raw student arrays differ')
    compact=bind_feature_cache(feature_bytes,manifest_entry=feature_entry,
        source_input=source_input,encoder_state_sha256=encoder_state_sha256,projection_device=feature_projection_device)
    points=compact.points_xyz_m[0].numpy();valid=compact.valid[0].numpy()
    slots=np.repeat(np.arange(5,dtype=np.int64),11520)
    origins=np.broadcast_to(source_input.translation_m[:,None,None,:],(5,16,720,3)).reshape(-1,3)
    _,patches,student_grid,_=build_observed_surface_inputs(points,valid,slots,origins)
    from .gse_feature_projection_v1 import project_input
    teacher_points,teacher_valid=project_input(source_input.range_valid,source_input.translation_m,
        source_input.yaw_deg,device='cpu')
    grid=build_surface_ray_grid(origins,teacher_points[0].numpy(),teacher_valid[0].numpy(),slots)
    # Verify original construction/source and grid even before a model forward.
    bound_reference_exclusion(bundle,grid,np.zeros((1,3)),
        expected_binding=target_manifest['source_binding'],matching_radius_m=4.)
    targets=observed_targets([produced_targets['record']])
    header=dict(schema_version='gse_surface_cached_input_header_v1',
        observation_id=source['task']+':'+str(source['source_sequence_id']),
        frame_indices=list(source['frame_rows']),coordinate_frame='current_sensor',
        input_binding_sha256=source_input.input_binding_sha256,
        target_binding_sha256=target_manifest['target_record_sha256'],
        frozen_encoder_state_sha256=encoder_state_sha256,
        student_relation_builder='gse_surface_observed_inputs_v1',
        student_grid_source_geometry_sha256=student_grid.source_geometry_sha256,
        student_grid_content_sha256=student_grid.content_sha256)
    example=SurfaceTrainingExample(header,compact,patches,targets)
    context=SurfaceSourceLossContext(source_input.input_binding_sha256,
        deepcopy(produced_targets),deepcopy(target_manifest),bundle,grid,opening_matching_radius_m,feature_projection_device)
    validate_context(example,context)
    return example,context
