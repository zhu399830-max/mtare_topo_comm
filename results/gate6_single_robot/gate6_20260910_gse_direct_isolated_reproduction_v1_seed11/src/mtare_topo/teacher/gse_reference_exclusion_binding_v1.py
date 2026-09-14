"""Bind conditional reference negatives to a frozen observation and grid.

No IO or automatic scientific qualification. The expected binding must be
provided by the frozen data manifest, not inferred from selected positive labels.
"""
import numpy as np
import torch
from mtare_topo.data.gse_surface_teacher_reader_v1 import verify_alignment
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.data.primitive_relation_dataset import _current_sensor_transform
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points
from mtare_topo.representation.gse_surface_ray_evidence_v1 import _digest
from .gse_construction_paths_v3 import construction_incident_paths
from .gse_reference_exclusion_v1 import exclusion_from_ray_grid


def bound_reference_exclusion(bundle, grid, query_xyz_m, *, expected_binding, matching_radius_m):
    sensor,student,doc,book=(bundle[k] for k in ('sensor_teacher_only','student','construction_teacher_only','codebook_teacher_only'))
    source=bundle['source']
    binding=dict(source={k:source[k] for k in ('task','source_sequence_id','frame_rows')},
        construction_sha256=canonical_sha(doc))
    if binding!=expected_binding:
        raise ValueError('frozen observation or construction binding mismatch')
    frames=source['frame_rows']
    if len(frames)!=5 or any(type(i) is not int or i<0 for i in frames) or sorted(set(frames))!=frames:
        raise ValueError('five increasing causal source frames required')
    verify_alignment(sensor,student,doc,book,source)
    ranges=student['ranges_m']; mask=student['valid_mask']
    if (ranges.shape!=(5,16,720) or ranges.dtype!=np.float32 or not np.isfinite(ranges).all()
            or np.any((ranges<0)|(ranges>50)) or mask.dtype!=np.uint8 or np.any(mask>1)):
        raise ValueError('original finite range/mask contract required')
    translation=student['relative_translation_current_sensor_m']
    yaw=student['relative_yaw_current_sensor_deg']
    image=np.stack((ranges/np.float32(50),mask.astype(np.float32)),axis=1)
    with torch.no_grad():
        points,valid=register_causal_lidar_points(torch.from_numpy(image[None]),
            torch.from_numpy(translation.copy()[None]),torch.from_numpy(yaw.copy()[None]))
    points=points.numpy().reshape(-1,3);valid=valid.numpy().reshape(-1)
    origins=np.broadcast_to(translation[:,None,None,:],(5,16,720,3)).reshape(-1,3)
    slots=np.repeat(np.arange(5,dtype=np.int64),11520)
    digest=_digest(np.where(valid[:,None],origins,0.).astype('<f8'),
        np.where(valid[:,None],points,0.).astype('<f8'),valid,slots.astype('<i8'),
        np.asarray([origins.dtype.str,points.dtype.str],dtype='S3'))
    if digest!=grid.source_geometry_sha256:
        raise ValueError('ray grid belongs to different original observation geometry')
    groups=construction_incident_paths(doc)
    # Include unsupported and out-of-ROI references; a reference just outside
    # the ROI may still lie within the metric's matching radius of a query.
    references=[g for g in groups if len(g['paths'])==1 or len(g['paths'])>=3]
    positions=np.asarray([g['anchor_world_m'] for g in references],dtype=np.float64).reshape(-1,3)
    local=_current_sensor_transform(positions,sensor['sensor_xyz_m'][-1],float(sensor['yaw_deg'][-1]))
    result=exclusion_from_ray_grid(query_xyz_m=query_xyz_m,grid=grid,all_anchor_xyz_m=local,
        inventory_complete=True,matching_radius_m=matching_radius_m)
    result.update(inventory_completeness_supplied_not_verified=False,
        observation_identity_binding_verified=True,
        inventory_definition='all degree1/degree>=3 references in frozen construction, not all possible physical events',
        reference_count=len(references),binding=binding)
    return result
