import numpy as np
import pytest

from mtare_topo.semantics.supported_primitive_fit import fit_supported_primitive
from mtare_topo.semantics.primitive_relation_nonlearning import (
    PrimitiveRelationBaselineConfig, _Candidate, _fit_candidate,
)


def cloud():
    s, theta = np.meshgrid(np.linspace(2, 8, 20), np.linspace(0, 2*np.pi, 24, endpoint=False))
    return np.column_stack([s.ravel(), np.cos(theta).ravel(), np.sin(theta).ravel()])


def run(points, rays=None, **kwargs):
    options = dict(heading_deg=0., angular_width_deg=90., bidirectional=False,
                   config=PrimitiveRelationBaselineConfig())
    options.update(kwargs)
    return fit_supported_primitive(points, np.arange(len(points)) if rays is None else rays, **options)


def test_supported_matches_old_fit_and_exact_section_provenance():
    p = cloud(); result = run(p)
    old = _fit_candidate(_Candidate(0., 90., np.zeros(720, bool), np.zeros(5, bool), False),
                         p, PrimitiveRelationBaselineConfig())
    assert result.reason == 'observed_surface_fit_only'
    for actual, expected in zip((result.axis_controls_m, result.endpoint_half_axes_m,
                                 result.endpoint_exponents, result.residual), old):
        np.testing.assert_allclose(actual, expected, rtol=0, atol=0)
    low, high = result.fitted_longitudinal_bounds_m
    for center, ids, bounds in zip(np.linspace(low, high, 3), result.section_ray_indices,
                                   result.section_observed_bounds_m):
        expected = np.flatnonzero(abs(p[:, 0]-center) <= max((high-low)/2, 1))
        assert ids == tuple(expected)
        assert bounds == (p[expected, 0].min(), p[expected, 0].max())
    assert not result.connectivity_verified


def test_no_point_completion_or_axis_extension():
    assert run(cloud()[:10]).reason == 'insufficient_directional_returns'
    p = cloud(); p[:, 0] = 3
    result = run(p)
    assert result.reason == 'insufficient_observed_extent'
    assert result.axis_controls_m is None
    assert result.observed_longitudinal_bounds_m == (3, 3)


def test_reject_sparse_section_without_nearest_point_fill():
    p = np.zeros((100, 3)); p[:94, 0] = 2; p[94:, 0] = 8
    result = run(p)
    assert result.reason == 'insufficient_section_returns'
    assert len(result.section_ray_indices[-1]) == 6
    assert result.axis_controls_m is None


def test_input_permutation_and_floor_provenance():
    p = cloud(); ids = np.arange(len(p)); permutation = np.random.default_rng(0).permutation(len(p))
    assert run(p) == run(p[permutation], ids[permutation])
    p[:, 1:] *= .01
    assert run(p).section_size_floor_applied == (True, True, True)


def test_direction_excludes_opposite_returns():
    p = np.concatenate([cloud(), -cloud()])
    assert run(p).selected_ray_indices == tuple(range(len(p)//2))
    assert len(run(p, bidirectional=True).selected_ray_indices) == len(p)


def test_invalid_sources_and_empty_input():
    p = cloud()
    for ids in (np.zeros(len(p), int), np.arange(len(p), dtype=float), np.arange(len(p))+57600):
        with pytest.raises(ValueError): run(p, ids)
    with pytest.raises(ValueError): run(p, heading_deg=np.nan)
    with pytest.raises(ValueError): run(p*np.nan)
    assert run(np.empty((0, 3))).reason == 'insufficient_directional_returns'
