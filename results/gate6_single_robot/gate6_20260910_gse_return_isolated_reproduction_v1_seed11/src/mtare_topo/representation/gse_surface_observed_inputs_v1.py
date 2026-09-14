"""Shared training/runtime patch and ray-relation construction.

Only aligned observed XYZ, validity, causal frame slots and ray origins enter.
No construction, label, global identity or teacher-grid argument is accepted.
"""
from dataclasses import replace
import numpy as np
import torch

from .gse_surface_patches_v1 import extract_surface_patches
from .gse_surface_relation_model_v1 import collate_surface_patches
from .gse_surface_ray_evidence_v1 import build_surface_ray_grid, query_patch_gaps


def build_observed_surface_inputs(points_xyz_m, valid, frame_index, ray_origins_m, *,
                                 voxel_size_m=.5, roi_radius_m=10., max_patches=4096,
                                 device='cpu', dtype=torch.float32):
    patches = extract_surface_patches(points_xyz_m, valid, frame_index,
        ray_origins_m=ray_origins_m, voxel_size_m=voxel_size_m,
        roi_radius_m=roi_radius_m, max_patches=max_patches)
    batch = collate_surface_patches([patches], device=device, dtype=dtype)
    grid = build_surface_ray_grid(ray_origins_m, points_xyz_m, valid, frame_index)
    m = len(patches.centers_m)
    gap = query_patch_gaps(grid, patches.centers_m,
        batch.neighbor_index[0, :m].cpu().numpy(), batch.neighbor_valid[0, :m].cpu().numpy())
    features = np.where(gap.gap_defined[..., None], gap.fractions, np.asarray([0., 0., 1.]))
    relation = batch.relation.clone()
    relation[..., -3:] = relation.new_tensor([0., 0., 1.])
    relation[0, :m, :, -3:] = torch.as_tensor(features, dtype=dtype, device=device)
    return patches, replace(batch, relation=relation), grid, gap
