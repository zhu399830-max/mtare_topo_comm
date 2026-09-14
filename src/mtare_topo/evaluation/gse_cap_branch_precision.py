"""Numerical qualification only, never a target or physical branch detector.

Callers must separately bind source geometry, ROI, first-return ordering and
triangle inclusion. This module cannot upgrade missing observation evidence.
"""
import numpy as np
from .gse_source_plane_intervals import source_plane_sign_bounds


def cap_branch_precision(triangles, origins, directions, ray_indices, inward_direction, *, direction_kind='entering'):
    if direction_kind not in ('entering','leaving'):
        raise ValueError('entering or leaving direction required')
    rays = np.asarray(ray_indices)
    inward = np.asarray(inward_direction, dtype=np.float64)
    if (rays.ndim != 1 or not np.issubdtype(rays.dtype, np.integer)
            or np.any(rays < 0) or len(rays) != len(triangles)):
        raise ValueError('one nonnegative archived ray index per hit required')
    if inward.shape != (3,) or not np.isfinite(inward).all() or np.linalg.norm(inward) == 0:
        raise ValueError('finite nonzero source branch direction required')
    bounds = source_plane_sign_bounds(triangles, origins, directions)
    # A duplicated boundary-triangle hit must not let one favourable occurrence
    # hide another unstable occurrence of the same archived ray.
    unique = np.unique(rays)
    rejected = set(rays[~bounds[direction_kind]].tolist())
    stable = [int(r) for r in unique if r not in rejected]
    return dict(inward_direction=inward.tolist(), saved_ray_count=len(unique),
        **{'stable_'+direction_kind+'_ray_indices':stable},
        unqualified_ray_indices=sorted(rejected),
        unknown_hit_count=int(bounds['unknown'].sum()),
        **{'numerical_'+direction_kind+'_hit_count':int(bounds[direction_kind].sum())},
        label_or_physical_connectivity_certified=False)


def summarize_cap_directions(branch_reports):
    """Preserve the existing exact-direction distinction, not a merge radius."""
    directions = {tuple(b['inward_direction']) for b in branch_reports
                  if b['stable_entering_ray_indices']}
    return dict(stable_distinct_cap_directions=len(directions),
        three_cap_directions_numerically_supported=len(directions) >= 3,
        label_or_physical_connectivity_certified=False)
