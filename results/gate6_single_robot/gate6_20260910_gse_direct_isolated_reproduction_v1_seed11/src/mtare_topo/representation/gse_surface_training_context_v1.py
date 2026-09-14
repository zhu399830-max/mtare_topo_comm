"""Loss-only frozen-source context; never an argument to model.forward_compact.

Input cache authenticity remains the runner's responsibility. This layer binds
its exact declared fingerprint, frame rows and loss target to the source context.
"""
from dataclasses import dataclass, fields
import torch
import numpy as np
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.teacher.gse_surface_target_adapter_v1 import observed_targets
from .gse_surface_partial_losses_v2 import bound_partial_surface_losses


@dataclass(frozen=True)
class SurfaceSourceLossContext:
    input_binding_sha256: str
    produced_targets: dict
    manifest_row: dict
    bundle: dict
    grid: object
    opening_matching_radius_m: float
    feature_projection_device: str = 'cpu'


def validate_context(example, context):
    if type(context) is not SurfaceSourceLossContext:
        raise ValueError('explicit loss-only source context required')
    manifest=context.manifest_row; produced=context.produced_targets
    if (set(manifest)!={'source_binding','target_record_sha256'}
            or context.input_binding_sha256!=example.header['input_binding_sha256']
            or manifest['target_record_sha256']!=example.header['target_binding_sha256']
            or manifest['target_record_sha256']!=canonical_sha(produced['record'])
            or produced['source_binding']!=manifest['source_binding']
            or produced['target_record_sha256']!=manifest['target_record_sha256']
            or manifest['source_binding']['source']['frame_rows']!=list(example.header['frame_indices'])):
        raise ValueError('training example/source context fingerprint mismatch')
    targets=observed_targets([produced['record']])
    if any(not torch.equal(getattr(targets,f.name),getattr(example.targets,f.name).detach().cpu()) for f in fields(targets)):
        raise ValueError('training tensors differ from frozen positive record')
    if not isinstance(context.opening_matching_radius_m,float) or not 0<context.opening_matching_radius_m< float('inf'):
        raise ValueError('explicit finite opening matching radius required')
    from mtare_topo.data.gse_feature_projection_v1 import project_input
    student=context.bundle['student']
    image=np.stack((student['ranges_m']/np.float32(50),student['valid_mask'].astype(np.float32)),axis=1)
    points,valid=project_input(image,student['relative_translation_current_sensor_m'],
        student['relative_yaw_current_sensor_deg'],device=context.feature_projection_device)
    actual_valid=example.compact.valid.detach().cpu()
    if not torch.equal(valid,actual_valid) or not torch.equal(points[valid],example.compact.points_xyz_m.detach().cpu()[valid]):
        raise ValueError('student point coordinates or mask differ from source scan')


def source_bound_loss(prediction, example, context):
    validate_context(example,context)
    loss,evidence=bound_partial_surface_losses(prediction,produced_targets=[context.produced_targets],
        manifest_rows=[context.manifest_row],bundles=[context.bundle],grids=[context.grid],
        opening_matching_radius_m=context.opening_matching_radius_m)
    summary={}
    for kind,row in [('anchor',evidence[0]),('opening',evidence[0]['opening_exclusion'])]:
        coverage=row['confirmed_structure_duplicate_evidence_v2']
        summary[kind]=dict(background_query_indices=row['used_unmatched_negative_query_indices'],
            duplicate_query_indices=coverage['added_duplicate_negative_query_indices'],
            covered_queries=sum(coverage['query_scoreable_mask']),
            target_record_sha256=coverage['target_record_sha256'])
    return loss,summary
