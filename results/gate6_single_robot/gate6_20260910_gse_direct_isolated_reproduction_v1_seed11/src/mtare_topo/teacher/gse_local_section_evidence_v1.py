"""Joint local evidence, deliberately not a semantic opening teacher.

Reference mesh and rays must already share one coordinate frame. Returns are
derived from the SAME rays used for interior witnesses; callers cannot supply
an unrelated point cloud to certify the reference contour. A straight tunnel
section can satisfy both tests, so neither anchors nor labels are emitted.
"""
import numpy as np

from .gse_local_contour_support_v1 import contour_surface_support
from .gse_mesh_sections_v1 import section_ray_witnesses


def local_section_evidence(vertices_m, triangles, *, center_m, normal, rays):
    rays.validate()
    if len(rays.valid) > 57600:
        raise ValueError("five-frame ray capacity exceeded")
    section, witnesses = section_ray_witnesses(
        vertices_m, triangles, center_m=center_m, normal=normal, rays=rays)
    # Invalid/no-return ranges can be NaN: never use them in surface evidence.
    points = np.zeros((len(rays.valid), 3), dtype=np.float64)
    valid = rays.valid
    points[valid] = (rays.origins_m[valid]
                     + rays.directions[valid] * rays.first_return_m[valid, None])
    records = []
    for loop, indices in zip(section.loops_m, witnesses):
        contour = contour_surface_support(loop, points, valid)
        surface_indices = sorted({i for row in contour['supported_cell_point_indices']
                                  for i in row['point_indices']})
        through = bool(indices)
        boundary = contour['contour_fully_surface_supported']
        records.append(dict(
            contour=contour,
            exclusive_crossing_ray_indices=list(indices),
            crossing_source_frames=sorted(set(map(int, rays.source_frame_index[list(indices)]))),
            surface_source_frames=sorted(set(map(int, rays.source_frame_index[surface_indices]))),
            joint_local_evidence=through and boundary,
            missing_evidence=[name for name, present in
                              [('finite_interior_crossing', through),
                               ('contour_surface_support', boundary)] if not present],
            semantic_opening_label=None, anchor_label=None,
            physical_reachable=None, training_eligible=False))
    return dict(schema_version='gse_local_section_evidence_v1', sections=records,
                limitation='Joint evidence also occurs at arbitrary straight-tunnel sections; '
                           'it is not evidence of a structural event or a persistent node.')
