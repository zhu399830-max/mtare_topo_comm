"""Prospective positive-only window-opening proposal rule, not gold labels.

The supplied mesh must be a source-bound original mesh. Source provenance is
teacher-only. A crossing and a same-source boundary return are evidence for a
proposal, not sufficient proof of its center, width, membership or safety.
No anchor, complete background or trainable label is emitted by this pilot.
"""
from dataclasses import replace
import numpy as np
from .gse_local_section_evidence_v1 import local_section_evidence
from .gse_mesh_sections_v1 import section_ray_witnesses


def propose_window_openings(vertices_m, triangles, *, section_center_m,
                            outward_direction, roi_center_m, rays,
                            unique_return_source_index, source_index):
    rays.validate()
    center = np.asarray(section_center_m, dtype=np.float64)
    origin = np.asarray(roi_center_m, dtype=np.float64)
    direction = np.asarray(outward_direction, dtype=np.float64)
    owners = np.asarray(unique_return_source_index)
    if (any(x.shape != (3,) or not np.isfinite(x).all() for x in (center, origin, direction))
            or owners.shape != rays.valid.shape or owners.dtype.kind not in 'iu'
            or np.any(owners < -1) or type(source_index) is not int or source_index < 0):
        raise ValueError('finite geometry and explicit unique source index (-1 unknown) required')
    eps = 64*np.finfo(np.float64).eps
    if (abs(np.linalg.norm(center-origin)-10.) > 10*eps
            or abs(np.linalg.norm(direction)-1.) > eps
            or np.dot(direction, center-origin) <= 0):
        raise ValueError('source axis crossing of fixed 10m ROI with outward unit direction required')
    # Unknown/multi-source returns cannot certify a convenient source's contour
    # or traversal. Preserve the original arrays and caller's validity mask.
    source_rays = replace(rays, valid=rays.valid & (owners == source_index))
    joint = local_section_evidence(vertices_m, triangles, center_m=center,
                                   normal=direction, rays=source_rays)
    _, all_crossings = section_ray_witnesses(vertices_m, triangles,
        center_m=center, normal=direction, rays=rays, exclusive=False)
    if len(all_crossings) != len(joint['sections']):
        raise ValueError('source filtering changed geometric section population')
    proposals = []
    for row, geometric_crossings in zip(joint['sections'], all_crossings):
        crossings = [i for i in row['exclusive_crossing_ray_indices']
                     if np.dot(rays.directions[i], direction) > 0]
        surface = sorted({i for cell in row['contour']['supported_cell_point_indices']
                          for i in cell['point_indices']})
        proposals.append(dict(
            proposal_supported=bool(crossings and surface),
            reference_position_m=center.tolist(), reference_direction=direction.tolist(),
            outward_crossing_ray_indices=crossings, surface_return_ray_indices=surface,
            geometric_outward_crossing_ray_indices=[i for i in geometric_crossings
                if np.dot(rays.directions[i], direction) > 0],
            witness_source_frames=sorted({int(rays.source_frame_index[i]) for i in crossings+surface}),
            observed_contour_cells=len(row['contour']['supported_cell_point_indices']),
            reference_contour_cells=len(row['contour']['reference_contour_cells']),
            width_m=None, height_m=None, anchor=None, membership=None,
            complete_region=False, training_eligible=False,
            missing_evidence=[name for name, present in (
                ('outward_same_source_crossing', bool(crossings)),
                ('same_source_boundary_return', bool(surface))) if not present]))
    return dict(schema='gse_window_opening_proposals_v1', proposals=proposals,
                qualified_labels=0, physical_reachable=None,
                limitation='A partial-source-supported window proposal is not a '
                           'validated aperture label or structural anchor; references '
                           'must not enter student forward/candidate selection.')
