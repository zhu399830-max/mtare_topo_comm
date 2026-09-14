from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest
import torch
from test_gse_construction_paths_v2 import document
from mtare_topo.data.gse_structure_review_v1 import canonical_sha
from mtare_topo.representation.primitive_relation_model import register_causal_lidar_points
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion


def fixture():
    doc=document();doc.update(parent_id='synthetic',geometry_realization='ellipse')
    mask=np.zeros((5,16,720),dtype=np.uint8);mask[0,8,1]=1
    ranges=np.full(mask.shape,8.,dtype=np.float32)
    trans=np.zeros((5,3),dtype=np.float32);yaw=np.zeros(5,dtype=np.float32)
    with torch.no_grad():
        points,valid=register_causal_lidar_points(torch.from_numpy(np.stack((ranges/50,mask.astype(np.float32)),axis=1)[None]),
            torch.from_numpy(trans[None]),torch.from_numpy(yaw[None]))
    points=points.numpy().reshape(-1,3);valid=valid.numpy().reshape(-1)
    grid=build_surface_ray_grid(np.zeros_like(points),points,valid,np.repeat(np.arange(5),11520))
    source=dict(task='synthetic__ellipse',source_sequence_id=1,frame_rows=[0,1,2,3,4])
    bundle=dict(source=source,construction_teacher_only=doc,
        codebook_teacher_only=dict(parent_id='synthetic',geometry_realization='ellipse',primitive_ids=['p'],source_sets=[[],[0]]),
        sensor_teacher_only=dict(sensor_xyz_m=np.zeros((5,3)),yaw_deg=np.zeros(5),primitive_membership_code=mask.astype(np.uint16)),
        student=dict(ranges_m=ranges,valid_mask=mask,relative_translation_current_sensor_m=trans,relative_yaw_current_sensor_deg=yaw))
    binding=dict(source=deepcopy(source),construction_sha256=canonical_sha(doc))
    return bundle,grid,points[valid].copy(),binding


def test_original_projection_grid_and_full_reference_inventory_bound():
    b,g,q,binding=fixture()
    result=bound_reference_exclusion(b,g,q,expected_binding=binding,matching_radius_m=4.)
    assert result['reference_count']==2
    assert result['reference_negative_mask']==[True]
    assert result['observation_identity_binding_verified']
    assert not result['training_eligible']


def test_deleted_reference_cannot_keep_original_freeze():
    b,g,q,binding=fixture()
    b['construction_teacher_only']['base_construction']['composition_operations'].pop()
    with pytest.raises(ValueError,match='binding'):
        bound_reference_exclusion(b,g,q,expected_binding=binding,matching_radius_m=4.)


def test_wrong_grid_source_rejected():
    b,g,q,binding=fixture()
    with pytest.raises(ValueError,match='different original'):
        bound_reference_exclusion(b,replace(g,source_geometry_sha256='wrong'),q,expected_binding=binding,matching_radius_m=4.)


def test_wrong_original_frame_rejected():
    b,g,q,binding=fixture();b['source']['frame_rows'][-1]=5
    with pytest.raises(ValueError,match='binding'):
        bound_reference_exclusion(b,g,q,expected_binding=binding,matching_radius_m=4.)
