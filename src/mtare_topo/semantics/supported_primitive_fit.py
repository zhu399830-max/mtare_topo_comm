"""Strict observation-support adapter, not the historical baseline unchanged.

Reuse its numerical fit only after excluding its point/extent completion paths.
Support means observed surface returns used by the estimator, NOT verified free
space, continuous surface coverage, a physical opening, or traversability.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .primitive_relation_nonlearning import (
    PrimitiveRelationBaselineConfig, _Candidate, _circular_distance, _fit_candidate,
)


@dataclass(frozen=True)
class SupportedPrimitiveFit:
    reason: str
    selected_ray_indices: tuple[int, ...]
    section_ray_indices: tuple[tuple[int, ...], ...] = ()
    observed_longitudinal_bounds_m: tuple[float, float] | None = None
    fitted_longitudinal_bounds_m: tuple[float, float] | None = None
    section_observed_bounds_m: tuple[tuple[float, float], ...] = ()
    axis_controls_m: tuple[tuple[float, ...], ...] | None = None
    endpoint_half_axes_m: tuple[tuple[float, ...], ...] | None = None
    endpoint_exponents: tuple[float, ...] | None = None
    residual: float | None = None
    section_size_floor_applied: tuple[bool, ...] = ()
    connectivity_verified: bool = False


def fit_supported_primitive(
    points_xyz_m, source_flat_ray_index, *, heading_deg: float,
    angular_width_deg: float, bidirectional: bool,
    config: PrimitiveRelationBaselineConfig,
) -> SupportedPrimitiveFit:
    """Fit one observation-derived heading; never accept teacher group IDs.

    Input ray IDs belong to five 16x720 scans already aligned to the current
    sensor frame. Caller must authenticate source and apply its declared ROI.
    Heading proposal generation is outside this function. No ROI is silently
    applied here. The historical 0.1 m half-size floor is explicitly recorded;
    resulting dimensions cannot be interpreted as measured safe clearance.
    """
    points = np.asarray(points_xyz_m, dtype=np.float64)
    rays = np.asarray(source_flat_ray_index)
    if points.ndim != 2 or points.shape[1:] != (3,) or not np.isfinite(points).all():
        raise ValueError('finite Nx3 points required')
    if (rays.shape != (len(points),) or rays.dtype.kind not in 'iu'
            or np.any(rays < 0) or np.any(rays >= 57600)
            or len(np.unique(rays)) != len(rays)):
        raise ValueError('unique five-frame source ray indices required')
    if (not np.isfinite([heading_deg, angular_width_deg]).all()
            or not 0 < angular_width_deg <= 360
            or not isinstance(bidirectional, (bool, np.bool_))):
        raise ValueError('finite heading, width in (0,360], boolean direction required')
    if (not isinstance(config, PrimitiveRelationBaselineConfig)
            or not np.isfinite(list(config.to_dict().values())).all()
            or isinstance(config.minimum_candidate_points, bool)
            or not isinstance(config.minimum_candidate_points, (int, np.integer))):
        raise ValueError('finite validated fitting config required')
    # Canonical ray order makes fitting and provenance invariant to input order.
    order = np.argsort(rays)
    points, rays = points[order], rays[order]
    heading = float(heading_deg) % 360.
    angle = np.radians(heading)
    axis = np.array([np.cos(angle), np.sin(angle)])
    side = np.array([-axis[1], axis[0]])
    s_all = points[:, :2] @ axis
    azimuth = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360.
    mask = np.minimum(_circular_distance(azimuth, heading),
                      _circular_distance(azimuth, heading + 180.)) <= (
                          angular_width_deg / 2 + config.angular_surface_margin_deg)
    if not bidirectional:
        mask &= s_all > 0
    selected = tuple(int(x) for x in rays[mask])
    if len(selected) < config.minimum_candidate_points:
        return SupportedPrimitiveFit('insufficient_directional_returns', selected)
    s = s_all[mask]
    observed = (float(s.min()), float(s.max()))
    q = config.section_quantile
    low, high = np.quantile(s, [q, 1-q])
    if not bidirectional:
        low = max(0., float(low))
    fitted = (float(low), float(high))
    if high-low < config.minimum_axis_extent_m:
        return SupportedPrimitiveFit('insufficient_observed_extent', selected,
                                     observed_longitudinal_bounds_m=observed,
                                     fitted_longitudinal_bounds_m=fitted)
    centers = np.linspace(low, high, 3)
    span = max((high-low)/2, config.minimum_axis_extent_m)
    memberships = tuple(np.abs(s-c) <= span for c in centers)
    section_rays = tuple(tuple(int(x) for x in rays[mask][m]) for m in memberships)
    if any(len(ids) < 16 for ids in section_rays):
        return SupportedPrimitiveFit('insufficient_section_returns', selected,
                                     section_rays, observed, fitted)
    bounds = tuple((float(s[m].min()), float(s[m].max())) for m in memberships)
    yz = np.column_stack([points[mask, :2] @ side, points[mask, 2]])
    floor = tuple(bool(np.any(np.diff(np.quantile(yz[m], [q, 1-q], axis=0),
                                      axis=0)[0] / 2 < .1)) for m in memberships)
    candidate = _Candidate(heading, float(angular_width_deg), np.zeros(720, bool),
                           np.zeros(5, bool), bool(bidirectional))
    controls, sizes, exponents, residual = _fit_candidate(candidate, points, config)
    if not all(np.isfinite(a).all() for a in (controls, sizes, exponents, residual)):
        raise ValueError('nonfinite historical fit')
    return SupportedPrimitiveFit(
        'observed_surface_fit_only', selected, section_rays, observed, fitted, bounds,
        tuple(tuple(float(x) for x in row) for row in controls),
        tuple(tuple(float(x) for x in row) for row in sizes),
        tuple(float(x) for x in exponents), float(residual), floor,
    )
