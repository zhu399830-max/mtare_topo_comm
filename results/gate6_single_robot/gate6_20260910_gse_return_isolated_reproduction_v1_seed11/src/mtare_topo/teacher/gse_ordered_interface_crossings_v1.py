"""Order source-interface intersections on individual finite observed rays.

Teacher-only diagnostic. Interfaces are NOT apertures, and a free ray is NOT
robot clearance. Caller supplies original caster intersections and a local ROI
mask; no source identity or constructed adjacency becomes a student candidate.
"""
import numpy as np


def ordered_crossings(*, ray_ids, interface_ids, hit_t, first_return, valid,
                      inside_roi):
    ray_ids = np.asarray(ray_ids)
    interface_ids = np.asarray(interface_ids)
    hit_t = np.asarray(hit_t)
    first_return = np.asarray(first_return)
    valid = np.asarray(valid)
    inside_roi = np.asarray(inside_roi)
    if (ray_ids.ndim != 1 or ray_ids.dtype.kind not in 'iu'
            or interface_ids.shape != ray_ids.shape
            or interface_ids.dtype.kind not in 'iu'
            or hit_t.shape != ray_ids.shape or hit_t.dtype != np.float32
            or inside_roi.shape != ray_ids.shape or inside_roi.dtype != bool
            or first_return.ndim != 1 or first_return.dtype != np.float32
            or valid.shape != first_return.shape or valid.dtype != bool
            or np.any(ray_ids < 0) or np.any(ray_ids >= len(first_return))
            or np.any(interface_ids < 0) or not np.isfinite(hit_t).all()
            or not np.isfinite(first_return[valid]).all()
            or np.any(first_return[valid] <= 0)):
        raise ValueError('original float32 intersections and bounded ray indices required')
    keep = (valid[ray_ids] & inside_roi & (hit_t > 0)
            & (hit_t < first_return[ray_ids]))
    # Deduplicate triangle-edge hits only when interface and parameter agree.
    records = sorted(set(zip(ray_ids[keep].tolist(), hit_t[keep].tolist(),
                             interface_ids[keep].tolist())))
    grouped = {}
    for ray, parameter, interface in records:
        grouped.setdefault(ray, {}).setdefault(parameter, []).append(interface)
    result = []
    for ray, levels in grouped.items():
        groups = [dict(t=t, interface_ids=ids) for t, ids in levels.items()]
        # Preserve coincident interfaces as one unordered group; do not invent
        # an ordering or bridge across that ambiguity.
        adjacent = []
        for left, right in zip(groups, groups[1:]):
            if len(left['interface_ids']) == len(right['interface_ids']) == 1:
                a, b = left['interface_ids'][0], right['interface_ids'][0]
                if a != b:
                    adjacent.append(dict(from_interface=a, to_interface=b,
                                         from_t=left['t'], to_t=right['t']))
        result.append(dict(ray_index=ray, crossing_groups=groups,
                           ordered_adjacent_crossings=adjacent))
    return dict(rays=result, semantic_label=None, training_eligible=False,
                limitation='Finite line-of-sight interface evidence only; no '
                           'aperture, junction, traversability or graph-edge claim.')
