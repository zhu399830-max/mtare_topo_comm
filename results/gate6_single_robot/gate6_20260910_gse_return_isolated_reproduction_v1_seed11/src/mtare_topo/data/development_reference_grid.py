"""Reuse the existing CPU observation grid for conditional reference queries.

No new visibility rule, teacher label, feature encoding, or file access.
The caller must bind/authorize the exact raw observation before execution.
"""
import numpy as np
from .gse_feature_projection_v1 import project_input
from mtare_topo.representation.gse_surface_ray_evidence_v1 import build_surface_ray_grid
from mtare_topo.teacher.gse_reference_exclusion_binding_v1 import bound_reference_exclusion


def build_bound_reference_grid(bundle, *, expected_binding):
    student=bundle['student']
    image=np.stack((student['ranges_m']/np.float32(50),student['valid_mask'].astype(np.float32)),axis=1)
    translation=student['relative_translation_current_sensor_m']
    points,valid=project_input(image,translation,student['relative_yaw_current_sensor_deg'],device='cpu')
    origins=np.broadcast_to(translation[:,None,None,:],(5,16,720,3)).reshape(-1,3)
    slots=np.repeat(np.arange(5,dtype=np.int64),11520)
    grid=build_surface_ray_grid(origins,points[0].numpy(),valid[0].numpy(),slots)
    # Reuse full source/pose/codebook and exact CPU projection authentication.
    bound_reference_exclusion(bundle,grid,np.zeros((1,3)),
        expected_binding=expected_binding,matching_radius_m=4.)
    return grid
