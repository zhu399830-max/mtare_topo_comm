import numpy as np
import pytest

from mtare_topo.data.cano_sensor_smoke import lidar_local_directions
from mtare_topo.representation.gse_registered_structural_skeleton import (
    EgoConnectedStructuralSkeleton,
    deterministic_voxel_subsample,
    register_causal_range_window,
    relative_pose_window,
)


def _one_return_window():
    ranges = np.ones((5, 16, 720), dtype=np.float64)
    valid = np.zeros_like(ranges, dtype=bool)
    valid[:, 7, 0] = True
    xyz = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    yaw = np.zeros(5)
    return ranges, valid, xyz, yaw


def test_relative_registration_is_invariant_to_global_se2_transform():
    ranges, valid, xyz, yaw = _one_return_window()
    first = register_causal_range_window(ranges, valid, xyz, yaw)
    angle = np.radians(73.0)
    rotation = np.asarray([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    transformed = xyz.copy()
    transformed[:, :2] = xyz[:, :2] @ rotation.T + np.asarray([91.0, -37.0])
    transformed[:, 2] += 12.0
    second = register_causal_range_window(ranges, valid, transformed, yaw + 73.0)
    np.testing.assert_allclose(first.xyz_current_m, second.xyz_current_m, atol=1e-8)
    np.testing.assert_allclose(first.ray_origin_current_m, second.ray_origin_current_m, atol=1e-8)


def test_registration_uses_all_and_only_causal_frames():
    ranges, valid, xyz, yaw = _one_return_window()
    registered = register_causal_range_window(ranges, valid, xyz, yaw)
    assert registered.xyz_current_m.shape == (5, 3)
    assert registered.source_frame_index.tolist() == [0, 1, 2, 3, 4]
    ray = lidar_local_directions()[7, 0].astype(np.float64)
    expected = xyz - xyz[-1] + ray
    np.testing.assert_allclose(registered.xyz_current_m, expected, atol=1e-7)
    with pytest.raises(ValueError, match="end at the current"):
        type(registered.relative_poses)(
            registered.relative_poses.translation_current_m,
            registered.relative_poses.yaw_current_rad,
            current_index=3,
        )


def test_yaw_registration_rotates_past_returns_into_current_frame():
    ranges, valid, xyz, yaw = _one_return_window()
    yaw[:] = [0.0, 0.0, 0.0, 0.0, 90.0]
    registered = register_causal_range_window(ranges, valid, xyz, yaw)
    ray = lidar_local_directions()[7, 0].astype(np.float64)
    # Current forward is world +Y; its own return remains current-frame +X.
    np.testing.assert_allclose(registered.xyz_current_m[-1], ray, atol=1e-7)
    # First return is at world +X and the first origin is four metres behind
    # current along current-frame +Y.
    np.testing.assert_allclose(
        registered.xyz_current_m[0], [ray[1], 4.0 - ray[0], ray[2]], atol=1e-7,
    )


def test_deterministic_voxel_subsample_prefers_latest_frame():
    ranges, valid, xyz, yaw = _one_return_window()
    xyz[:] = 0.0
    registered = register_causal_range_window(ranges, valid, xyz, yaw)
    reduced = deterministic_voxel_subsample(registered, voxel_size_m=0.25)
    assert len(reduced.xyz_current_m) == 1
    assert reduced.source_frame_index.tolist() == [4]


def test_skeleton_rejects_disconnected_or_duplicate_graph():
    nodes = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    geometry = np.full((1, 4), np.nan)
    valid = np.zeros((1, 4), dtype=bool)
    with pytest.raises(ValueError, match="ego-connected"):
        EgoConnectedStructuralSkeleton(
            nodes, np.asarray([[0, 1]]), geometry, valid, 0,
            np.arange(3), np.asarray([10]),
        )
    with pytest.raises(ValueError, match="duplicate"):
        EgoConnectedStructuralSkeleton(
            nodes[:2], np.asarray([[0, 1], [1, 0]]),
            np.full((2, 4), np.nan), np.zeros((2, 4), dtype=bool), 0,
            np.arange(2), np.asarray([10, 10]),
        )


def test_pose_window_rejects_nonfinite_odometry():
    xyz = np.zeros((5, 3)); xyz[2, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        relative_pose_window(xyz, np.zeros(5))


def test_representation_package_keeps_torch_model_public_api_lazy():
    import mtare_topo.representation as representation

    assert "GSEModelConfig" in representation.__all__
    # Importing the package and NumPy-only ERCSS submodule must not itself
    # require eager access to the Torch model symbol.
    assert "GSEModelConfig" not in representation.__dict__
