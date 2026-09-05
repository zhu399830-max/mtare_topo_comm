import numpy as np
import pytest

from mtare_topo.representation.gse_route_geometry_profile import (
    PROFILE_OFFSETS_M,
    RouteGeometryProfile,
    objective_profile_from_sequence,
    range_image_profile_support,
    reverse_consistency_error,
)


def test_objective_profile_never_crosses_missing_traversal_rows():
    geometry = np.arange(120, dtype=np.float64).reshape(30, 4)
    valid = np.ones_like(geometry, dtype=bool)
    profile = objective_profile_from_sequence(
        geometry,
        valid,
        current_sequence_index=7,
        row_by_sequence_index={index: index for index in range(18)},
    )
    np.testing.assert_allclose(profile.values[:3], geometry[[7, 12, 17]])
    assert profile.valid_mask[:3].all()
    assert not profile.valid_mask[3:].any()


def test_range_profile_requires_all_four_surface_sides():
    ranges = np.full((16, 720), 50.0, dtype=np.float64)
    valid = np.zeros_like(ranges, dtype=bool)
    # Rays immediately inside +/-90 degrees give both lateral surfaces while
    # remaining unambiguously in the forward half-space.
    valid[7, 179] = valid[7, 541] = True
    ranges[7, 179] = ranges[7, 541] = 5.0
    valid[15, 0] = valid[0, 0] = True
    ranges[15, 0] = ranges[0, 0] = 2.0
    support = range_image_profile_support(ranges, valid)
    assert support.valid_mask[0]
    assert np.all(support.side_counts[0] > 0)
    assert not support.valid_mask[1:].any()


def test_reverse_consistency_flips_only_slope():
    values = np.asarray([[8.0, 7.0, 4.0, 0.02]] * len(PROFILE_OFFSETS_M))
    valid = np.ones_like(values, dtype=bool)
    forward = RouteGeometryProfile(np.asarray(PROFILE_OFFSETS_M), values, valid)
    reverse_values = values.copy(); reverse_values[:, 2] *= -1.0
    reverse = RouteGeometryProfile(np.asarray(PROFILE_OFFSETS_M), reverse_values, valid)
    assert reverse_consistency_error(forward, reverse) == {
        "width_m": 0.0,
        "height_m": 0.0,
        "slope_deg": 0.0,
        "curvature_per_m": 0.0,
    }


def test_profile_rejects_nonfinite_valid_value():
    with pytest.raises(ValueError, match="finite"):
        RouteGeometryProfile(
            np.asarray(PROFILE_OFFSETS_M),
            np.full((5, 4), np.nan),
            np.ones((5, 4), dtype=bool),
        )
