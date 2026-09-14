"""Independent closed half-space intersections for source diagnostics.

This is a floating-point geometric reference, not an exact uniqueness proof.
Unlike the open-volume distance reference it retains tangent and coplanar
surface encounters. No production triangle IDs or reported owners are used.
"""
import numpy as np
from .gse_convex_ray_union import prism_planes
from .source_rounding_diagnostic import float32_cells, source_candidates


def closed_intervals(origins, directions, planes):
    o = np.asarray(origins, dtype=np.float64)
    d = np.asarray(directions, dtype=np.float64)
    n, b = (np.asarray(x, dtype=np.float64) for x in planes)
    if (o.ndim != 2 or o.shape[1] != 3 or d.shape != o.shape
        or not np.isfinite(o).all() or not np.isfinite(d).all()
        or (np.linalg.norm(d, axis=1) == 0).any()):
        raise ValueError('finite nonzero aligned Nx3 rays required')
    if (n.ndim != 2 or n.shape[1] != 3 or len(n) < 1 or b.shape != (len(n),)
        or not np.isfinite(n).all() or not np.isfinite(b).all()
        or (np.linalg.norm(n, axis=1) == 0).any()):
        raise ValueError('finite nonzero half-space normals required')
    with np.errstate(over='raise', invalid='raise', divide='raise'):
        slack = b[None] - o @ n.T
        den = d @ n.T
        parallel = den == 0
        ratio = np.zeros_like(den)
        np.divide(slack, den, out=ratio, where=~parallel)
        low = np.where(den < 0, ratio, -np.inf).max(axis=1)
        high = np.where(den > 0, ratio, np.inf).min(axis=1)
    valid = (low <= high) & ~((parallel) & (slack < 0)).any(axis=1)
    spans = np.column_stack((np.where(valid,low,np.inf),np.where(valid,high,-np.inf)))
    return dict(intervals=spans, tangent=valid & (low == high),
                coplanar=valid & (parallel & (slack == 0)).any(axis=1),
                nonempty=valid, numerical_geometry_certified=False)


def declared_surface_candidates(case, origins, directions, stored_ranges):
    cells = float32_cells(stored_ranges)
    results = [closed_intervals(origins,directions,prism_planes(
        edge['points'],case['half_axes_m'],case['shape_exponent'])) for edge in case['program']['edges']]
    spans = np.stack([r['intervals'] for r in results],axis=1)
    coplanar = np.stack([r['coplanar'] for r in results],axis=1)
    tangent = np.stack([r['tangent'] for r in results],axis=1)
    # A coplanar ray can lie on a side face between its interval endpoints.
    # It matters to this observation only if that segment meets the range cell.
    within_cell = (spans[...,0] <= cells[:,1,None]) & (spans[...,1] >= cells[:,0,None])
    result = source_candidates(stored_ranges,spans,uncertain_operands=coplanar & within_cell)
    result.update(intervals=spans,coplanar=coplanar,tangent=tangent,
                  numerical_geometry_certified=False)
    return result
