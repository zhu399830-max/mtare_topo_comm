"""Construction-conditioned references, NOT independently observable labels.

The source-arc component is an indexing unit, not a semantic tunnel instance.
Its saved middle-slab reference is a target, not the local surface's normal or
a ground support height. This routine neither reconstructs nor requalifies it.
"""
import numpy as np


def conditional_geometry_targets(candidates, point_patch_index, roi_indices, neighbors):
    assignment = np.asarray(point_patch_index)
    roi = np.asarray(roi_indices)
    neighbors = np.asarray(neighbors)
    if assignment.ndim != 1 or assignment.dtype.kind not in 'iu' or roi.ndim != 1 or roi.dtype.kind not in 'iu':
        raise ValueError('integer original return indices required')
    if neighbors.ndim != 2 or neighbors.dtype.kind not in 'iu':
        raise ValueError('integer patch neighbor matrix required')
    m = len(neighbors)
    if np.any(roi < 0) or np.any(roi >= len(assignment)) or len(np.unique(roi)) != len(roi):
        raise ValueError('ROI identity mismatch')
    if np.any(assignment[roi] < 0) or np.any(assignment[roi] >= m) or np.any(neighbors < -1) or np.any(neighbors >= m):
        raise ValueError('patch index out of range')
    # Mapping uses ALL nominated source-arc components, including unresolved
    # ones, so dropping a bad component cannot make a mixed patch pure.
    owner = np.full(len(assignment), -1, dtype=np.int64)
    references = []
    identities = set()
    for index, c in enumerate(candidates):
        identity = (c['source_index'], c['component_index'])
        if identity in identities: raise ValueError('duplicate source component')
        identities.add(identity)
        ids = np.asarray(c['source_return_indices'], dtype=np.int64)
        if np.any(ids < 0) or np.any(ids >= len(owner)) or len(np.unique(ids)) != len(ids):
            raise ValueError('component return identity invalid')
        owner[ids] = np.where(owner[ids] == -1, index, -2)
        center = np.asarray(c.get('center_m', []), dtype=float)
        normal = np.asarray(c.get('normal', []), dtype=float)
        usable = (center.shape == (3,) and normal.shape == (3,) and np.isfinite(center).all()
                  and np.isfinite(normal).all() and np.linalg.norm(center) <= 10
                  and np.isclose(np.linalg.norm(normal), 1., atol=1e-6, rtol=0))
        references.append((center, normal) if usable else None)
    component = np.full(m, -1, dtype=np.int64)
    reasons = np.full(m, 'EMPTY_PATCH', dtype='<U40')
    for p in range(m):
        ids = roi[assignment[roi] == p]
        if not len(ids): continue
        unique = np.unique(owner[ids])
        if len(unique) != 1 or unique[0] < 0:
            reasons[p] = 'MIXED_AMBIGUOUS_OR_MISSING_REFERENCE'
        elif references[int(unique[0])] is None:
            reasons[p] = 'NO_FINITE_IN_DOMAIN_REFERENCE'
        else:
            component[p] = unique[0]; reasons[p] = 'CONSTRUCTION_CONDITIONED_ONLY'
    axis = np.full(neighbors.shape, np.nan)
    height = np.full(neighbors.shape, np.nan)
    known = np.zeros(neighbors.shape, dtype=bool)
    same = np.zeros(neighbors.shape, dtype=bool)
    for p, slot in zip(*np.nonzero(neighbors >= 0)):
        q = neighbors[p, slot]; a = component[p]; b = component[q]
        if a < 0 or b < 0: continue
        ca, na = references[a]; cb, nb = references[b]
        axis[p, slot] = np.clip(abs(na @ nb), 0., 1.)
        height[p, slot] = cb[2] - ca[2]
        known[p, slot] = True; same[p, slot] = a == b
    return dict(axis_abs_dot=axis, axis_known=known.copy(), height_difference_m=height,
                height_known=known.copy(), section_log_ratio=np.full((*neighbors.shape, 2), np.nan),
                section_known=np.zeros((*neighbors.shape, 2), dtype=bool),
                correspondence=np.full(neighbors.shape, np.nan), correspondence_known=np.zeros(neighbors.shape, dtype=bool),
                patch_reference_component=component, patch_reason=reasons, same_reference_component=same,
                neighbor_index=neighbors.copy(), schema_version=np.array('construction_conditioned_geometry_targets_v1'),
                observability_certified=np.array(False), connectivity_certified=np.array(False))
