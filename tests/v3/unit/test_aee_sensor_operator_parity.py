import numpy as np
import pytest

from mtare_topo.data.aee_sensor_operator_parity import (
    aee_gazebo_source_azimuth_rad,
    aee_gazebo_lidar_directions_sensor,
    canonicalize_aee_gazebo_azimuth,
    evenly_spaced_indices,
    fill_unmeasured_azimuth_gaps,
    lidar_directions_sensor,
    observation_parity_metrics,
    organized_aee_gazebo_points_to_ranges,
    quaternion_xyzw_to_matrix,
    resample_aee_gazebo_organized_azimuth,
    resample_organized_azimuth,
    world_rays,
)


def test_aee_gazebo_angles_preserve_closed_grid_and_duplicate_endpoint():
    angles = aee_gazebo_source_azimuth_rad()
    assert angles.shape == (350,)
    assert angles[0] == -np.pi and angles[-1] == np.pi
    np.testing.assert_allclose(np.diff(angles), 2.0 * np.pi / 349.0)
    assert np.isclose(np.mod(angles[0], 2.0 * np.pi), np.mod(angles[-1], 2.0 * np.pi))


def test_aee_gazebo_source_rays_match_closed_plugin_grid():
    directions = aee_gazebo_lidar_directions_sensor()
    assert directions.shape == (16, 350, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=2), 1.0, atol=1e-12)
    np.testing.assert_allclose(directions[:, 0], directions[:, -1], atol=1e-12)
    # With 349 intervals, zero azimuth falls strictly between indices 174/175.
    assert directions[7, 174, 1] < 0 < directions[7, 175, 1]


def test_aee_gazebo_duplicate_endpoint_uses_nearest_valid_return():
    ranges = np.full((1, 350), 50.0, dtype=np.float32)
    valid = np.zeros((1, 350), dtype=np.uint8)
    ranges[0, 0] = 4.0; valid[0, 0] = 1
    ranges[0, -1] = 3.0; valid[0, -1] = 1
    canonical_range, canonical_valid, angles = canonicalize_aee_gazebo_azimuth(ranges, valid)
    seam = int(np.argmin(np.abs(angles - np.pi)))
    assert canonical_range.shape == canonical_valid.shape == (1, 349)
    assert canonical_valid[0, seam] == 1 and canonical_range[0, seam] == 3.0
    assert np.count_nonzero(canonical_valid) == 1


def test_aee_gazebo_resampling_has_correct_forward_direction_and_accuracy():
    angles = aee_gazebo_source_azimuth_rad()
    ranges = (2.0 + np.cos(angles))[None, :].astype(np.float32)
    valid = np.ones_like(ranges, dtype=np.uint8)
    target_range, target_valid = resample_aee_gazebo_organized_azimuth(ranges, valid)
    target_angles = 2.0 * np.pi * np.arange(720) / 720.0
    expected = 2.0 + np.cos(target_angles)
    assert target_valid.shape == target_range.shape == (1, 720)
    assert np.all(target_valid)
    assert target_range[0, 0] > 2.9999
    np.testing.assert_allclose(target_range[0], expected, atol=5e-5)


def test_aee_gazebo_resampling_propagates_bracketing_no_return():
    ranges = np.full((1, 350), 2.0, dtype=np.float32)
    valid = np.ones((1, 350), dtype=np.uint8)
    # Zero radians lies between raw indices 174 and 175 on the 349-interval grid.
    valid[0, 174] = 0
    target_range, target_valid = resample_aee_gazebo_organized_azimuth(ranges, valid)
    assert target_valid[0, 0] == 0 and target_range[0, 0] == 50.0
    assert target_valid[0, 360] == 1


def test_aee_gazebo_organized_layout_preserves_raw_and_model_views():
    points = np.zeros((350, 16, 3), dtype=np.float32)
    points[..., 0] = 2.0
    rings = np.broadcast_to(np.arange(16, dtype=np.uint16)[None, :], (350, 16)).copy()
    points[3, 4] = np.nan
    points[5, 6] = [0.2, 0.0, 0.0]
    points[7, 8] = [60.0, 0.0, 0.0]
    raw_range, raw_valid, model_range, model_valid, audit = organized_aee_gazebo_points_to_ranges(
        points, rings
    )
    assert raw_range.shape == raw_valid.shape == model_range.shape == model_valid.shape == (16, 350)
    assert raw_valid[4, 3] == 0 and raw_range[4, 3] == 130.0
    assert raw_valid[6, 5] == 1 and model_valid[6, 5] == 0
    assert raw_valid[8, 7] == 1 and model_valid[8, 7] == 0
    assert model_range[6, 5] == model_range[8, 7] == 50.0
    assert audit.raw_records == 5600 and audit.raw_no_returns == 1
    assert audit.model_near_rejections == audit.model_far_rejections == 1


def test_aee_gazebo_organized_layout_rejects_ring_or_partial_nan_drift():
    points = np.ones((350, 16, 3), dtype=np.float32)
    rings = np.broadcast_to(np.arange(16, dtype=np.uint16)[None, :], (350, 16)).copy()
    rings[2, 3] = 4
    with pytest.raises(ValueError, match="ring order"):
        organized_aee_gazebo_points_to_ranges(points, rings)
    rings[2, 3] = 3
    points[0, 0] = [np.nan, np.nan, 1.0]
    with pytest.raises(ValueError, match="three nonfinite"):
        organized_aee_gazebo_points_to_ranges(points, rings)


def test_integer_even_selection_is_exact_and_covers_five_trajectories():
    selected = evenly_spaced_indices(3000, 32)
    assert selected.tolist() == [(i * 2999) // 31 for i in range(32)]
    assert np.bincount(selected // 600, minlength=5).tolist() == [7, 6, 6, 6, 7]


def test_sensor_directions_follow_forward_left_up_contract():
    directions = lidar_directions_sensor()
    assert directions.shape == (16, 720, 3)
    np.testing.assert_allclose(np.linalg.norm(directions, axis=2), 1.0, atol=1e-12)
    row = 7  # -1 degree
    assert directions[row, 0, 0] > 0 and abs(directions[row, 0, 1]) < 1e-12
    assert directions[row, 180, 1] > 0 and abs(directions[row, 180, 0]) < 1e-12


def test_quaternion_and_world_rays_apply_full_orientation():
    half = np.sqrt(0.5)
    rotation = quaternion_xyzw_to_matrix([0, 0, half, half])
    np.testing.assert_allclose(rotation @ [1, 0, 0], [0, 1, 0], atol=1e-12)
    rays = world_rays([1, 2, 3], [0, 0, half, half])
    assert rays.shape == (11520, 6)
    np.testing.assert_allclose(rays[0, :3], [1, 2, 3])
    assert rays[0, 4] > 0


def test_parity_metrics_preserve_missing_return_asymmetry():
    real_range = np.asarray([[2.0, 50.0, 4.0, 5.0]])
    ideal_range = np.asarray([[1.0, 3.0, 50.0, 5.0]])
    real_valid = np.asarray([[1, 0, 1, 1]])
    ideal_valid = np.asarray([[1, 1, 0, 1]])
    metrics = observation_parity_metrics(real_range, real_valid, ideal_range, ideal_valid)
    assert metrics.real_only == 1 and metrics.ideal_only == 1 and metrics.both_valid == 2
    assert metrics.valid_agreement == 0.5 and metrics.valid_iou == 0.5
    assert metrics.common_valid_mae_m == 0.5
    assert metrics.common_valid_bias_real_minus_ideal_m == 0.5


def test_parity_rejects_shape_or_nonfinite_drift():
    with pytest.raises(ValueError):
        observation_parity_metrics(np.zeros((2, 2)), np.zeros((2, 2)), np.zeros((4,)), np.zeros((4,)))
    with pytest.raises(ValueError):
        observation_parity_metrics(np.asarray([[np.nan]]), [[1]], [[1.0]], [[1]])


def test_operator_fill_changes_only_one_or_two_column_sampling_gaps():
    ranges = np.full((1, 720), 50.0, dtype=np.float32)
    valid = np.zeros((1, 720), dtype=np.uint8)
    ranges[0, [10, 13, 17]] = [2.0, 5.0, 9.0]
    valid[0, [10, 13, 17]] = 1
    filled_range, filled_valid = fill_unmeasured_azimuth_gaps(ranges, valid)
    assert np.flatnonzero(filled_valid[0]).tolist() == [10, 11, 12, 13, 17]
    np.testing.assert_allclose(filled_range[0, 10:14], [2.0, 3.0, 4.0, 5.0])
    assert filled_range[0, 17] == 9.0


def test_operator_fill_handles_circular_seam_and_preserves_measured_cells():
    ranges = np.full((2, 720), 50.0, dtype=np.float32)
    valid = np.zeros((2, 720), dtype=np.uint8)
    ranges[0, [719, 1]] = [2.0, 4.0]; valid[0, [719, 1]] = 1
    ranges[1, 20] = 7.0; valid[1, 20] = 1
    filled_range, filled_valid = fill_unmeasured_azimuth_gaps(ranges, valid)
    assert filled_valid[0, 0] == 1 and filled_range[0, 0] == 3.0
    assert filled_range[0, 719] == 2.0 and filled_range[0, 1] == 4.0
    assert int(filled_valid[1].sum()) == 1 and filled_range[1, 20] == 7.0


def test_organized_resampling_propagates_physical_no_return():
    ranges = np.asarray([[2.0, 4.0, 50.0, 8.0]], dtype=np.float32)
    valid = np.asarray([[1, 1, 0, 1]], dtype=np.uint8)
    target_range, target_valid = resample_organized_azimuth(
        ranges,
        valid,
        target_azimuth_columns=8,
    )
    assert target_valid.tolist() == [[1, 1, 1, 0, 0, 0, 1, 1]]
    np.testing.assert_allclose(target_range[0, :3], [2.0, 3.0, 4.0])
    np.testing.assert_allclose(target_range[0, 3:6], [50.0, 50.0, 50.0])
    np.testing.assert_allclose(target_range[0, 6:], [8.0, 5.0])


def test_organized_resampling_is_deterministic_and_does_not_mutate_source():
    ranges = np.linspace(1.0, 10.0, 350, dtype=np.float32)[None, None, :]
    valid = np.ones_like(ranges, dtype=np.uint8)
    ranges_before = ranges.copy()
    valid_before = valid.copy()
    first = resample_organized_azimuth(ranges, valid)
    second = resample_organized_azimuth(ranges, valid)
    np.testing.assert_array_equal(first[0], second[0])
    np.testing.assert_array_equal(first[1], second[1])
    np.testing.assert_array_equal(ranges, ranges_before)
    np.testing.assert_array_equal(valid, valid_before)
    assert first[0].shape == (1, 1, 720)
    assert int(first[1].sum()) == 720


def test_organized_resampling_rejects_nonfinite_or_shape_drift():
    with pytest.raises(ValueError):
        resample_organized_azimuth(np.zeros((2, 4)), np.zeros((2, 3)))
    with pytest.raises(ValueError):
        resample_organized_azimuth(np.asarray([[1.0, np.nan]]), np.ones((1, 2)))
