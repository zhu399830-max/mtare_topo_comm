"""Synthetic CPU tests; runnable with stdlib unittest in the Open3D sidecar."""
from dataclasses import replace
import importlib.util
import unittest
from unittest.mock import patch

import numpy as np

from mtare_topo.topology.gse_registration import (
    RegistrationConfig, point_to_plane_information, register_local_clouds,
)


def config():
    # Software fixtures only, not registered scientific/deployment thresholds.
    return RegistrationConfig(
        max_correspondence_distance_m=.35, normal_radius_m=.34, normal_max_nn=40,
        normal_min_neighbors=6, max_normal_surface_variation=.02,
        min_normal_second_eigenvalue_ratio=.001, min_normal_coverage=.8,
        max_iterations=80, relative_fitness=1e-10, relative_rmse=1e-10,
        point_to_point_bootstrap_iterations=0, min_correspondences=40,
        min_fitness=.8, max_inlier_rmse_m=.01,
        information_eigenvalue_ratio=1e-7, max_information_condition=1e5,
    )


def asymmetric_cloud():
    u, v = np.meshgrid(np.linspace(0., 1., 14), np.linspace(0., 1., 13))
    u, v = u.ravel(), v.ravel()
    return np.concatenate((np.column_stack((u * 0., -2. + u, .1 + v)),
                           np.column_stack((.3 + 1.5 * u, v * 0., 1.2 + v)),
                           np.column_stack((2. + u, 1. + v, u * 0.))))


def transformed_pair():
    # Ground truth creates a test pair and is used only to assert recovery.
    # ICP receives identity, not the known transform or point correspondences.
    axis = np.array([.5, -.3, .8])
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    angle = .04
    transform = np.eye(4)
    transform[:3, :3] += np.sin(angle) * skew + (1 - np.cos(angle)) * (skew @ skew)
    transform[:3, 3] = (.07, -.05, .04)
    source = asymmetric_cloud()
    target = source @ transform[:3, :3].T + transform[:3, 3]
    return source, target, transform


class RegistrationPureTests(unittest.TestCase):
    def test_config_requires_every_threshold(self):
        with self.assertRaises(TypeError):
            RegistrationConfig()

    def test_invalid_config(self):
        for change in ({"normal_radius_m": 0.}, {"min_fitness": 0.}, {"max_iterations": True},
                       {"information_eigenvalue_ratio": 1.}, {"max_normal_surface_variation": .5},
                       {"normal_min_neighbors": 100}, {"point_to_point_bootstrap_iterations": -1},
                       {"relative_fitness": float("nan")}, {"min_correspondences": 1}):
            with self.subTest(change=change), self.assertRaises(ValueError):
                replace(config(), **change)

    def test_invalid_clouds_fail_before_open3d_import(self):
        for cloud in (np.empty((0, 3)), np.ones((4, 2)), np.ones((2, 3)), np.full((10, 3), np.nan)):
            with self.subTest(shape=cloud.shape), self.assertRaises(ValueError):
                register_local_clouds(cloud, asymmetric_cloud(), np.eye(4), config())

    def test_invalid_transform_nan_reflection_scale(self):
        reflection, scale, bad_last = np.eye(4), np.eye(4), np.eye(4)
        reflection[0, 0], scale[0, 0], bad_last[3, 0] = -1., 2., 1.
        for transform in (reflection, scale, bad_last, np.eye(3), np.full((4, 4), np.nan)):
            with self.subTest(transform=transform), self.assertRaises(ValueError):
                register_local_clouds(asymmetric_cloud(), asymmetric_cloud(), transform, config())

    def test_information_planar_rank_three(self):
        points = asymmetric_cloud()[:182]
        normals = np.tile((1., 0., 0.), (len(points), 1))
        info = point_to_plane_information(points, normals, eigenvalue_ratio=1e-7)
        self.assertEqual(info.rank, 3)
        self.assertIsNone(info.condition)

    def test_information_unit_scale_and_origin_invariance(self):
        rng = np.random.default_rng(8)
        points, normals = rng.normal(size=(90, 3)), rng.normal(size=(90, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        first = point_to_plane_information(points, normals, eigenvalue_ratio=1e-7)
        second = point_to_plane_information(points * 1000. + (10000., -300., 100.),
                                            normals, eigenvalue_ratio=1e-7)
        self.assertEqual(first.rank, 6)
        np.testing.assert_allclose(first.eigenvalues, second.eigenvalues, atol=1e-14)
        self.assertAlmostEqual(second.scale_m / first.scale_m, 1000.)

    def test_information_normal_sign_does_not_change_rank(self):
        rng = np.random.default_rng(2)
        points, normals = rng.normal(size=(40, 3)), rng.normal(size=(40, 3))
        normals /= np.linalg.norm(normals, axis=1, keepdims=True)
        flipped = normals.copy()
        flipped[::2] *= -1
        self.assertEqual(point_to_plane_information(points, normals, eigenvalue_ratio=1e-7),
                         point_to_plane_information(points, flipped, eigenvalue_ratio=1e-7))

    def test_information_missing_correspondences_not_perfect(self):
        info = point_to_plane_information(np.empty((0, 3)), np.empty((0, 3)), eigenvalue_ratio=1e-7)
        self.assertEqual(info.rank, 0)
        self.assertIsNone(info.condition)

    def test_information_rejects_nonunit_normals(self):
        with self.assertRaises(ValueError):
            point_to_plane_information(np.ones((8, 3)), np.zeros((8, 3)), eigenvalue_ratio=1e-7)


@unittest.skipUnless(importlib.util.find_spec("open3d"), "requires designated Open3D 0.19.0 CPU sidecar")
class Open3DRegistrationTests(unittest.TestCase):
    def test_real_icp_recovers_nontrivial_transform_from_identity(self):
        source, target, truth = transformed_pair()
        self.assertGreater(np.max(np.abs(truth - np.eye(4))), .05)
        result = register_local_clouds(source, target, np.eye(4), config())
        self.assertTrue(result.accepted, result.rejection_reasons)
        np.testing.assert_allclose(result.transformation_target_from_source, truth, atol=1e-6)
        self.assertEqual(result.forward.information.rank, 6)
        self.assertEqual(result.reverse.information.rank, 6)
        self.assertGreater(result.forward.correspondences, 500)
        self.assertLess(result.forward.inlier_rmse_m, 1e-6)
        self.assertEqual(result.bootstrap_iterations_requested, 0)
        self.assertIn("cpu", result.backend)

    def test_actual_noisy_icp_uses_no_truth_initialization(self):
        source, target, truth = transformed_pair()
        target += np.random.default_rng(0).normal(scale=.0003, size=target.shape)
        result = register_local_clouds(source, target, np.eye(4), config())
        self.assertTrue(result.accepted, result.rejection_reasons)
        np.testing.assert_allclose(result.transformation_target_from_source, truth, atol=.001)
        self.assertGreater(result.forward.inlier_rmse_m, 0.)

    def test_repeat_is_exact_under_single_thread_contract(self):
        source, target, _ = transformed_pair()
        a = register_local_clouds(source, target, np.eye(4), config())
        b = register_local_clouds(source, target, np.eye(4), config())
        self.assertEqual(a, b)

    def test_inputs_not_mutated(self):
        source, target, _ = transformed_pair()
        a, b, initial = source.copy(), target.copy(), np.eye(4)
        register_local_clouds(source, target, initial, config())
        np.testing.assert_array_equal(source, a)
        np.testing.assert_array_equal(target, b)
        np.testing.assert_array_equal(initial, np.eye(4))

    def test_no_overlap_zero_open3d_rmse_does_not_pass(self):
        source = asymmetric_cloud()
        result = register_local_clouds(source, source + 100., np.eye(4), config())
        self.assertFalse(result.accepted)
        self.assertEqual(result.forward.correspondences, 0)
        self.assertEqual(result.reverse.correspondences, 0)
        self.assertIsNone(result.forward.inlier_rmse_m)
        self.assertIn("forward_rmse_undefined", result.rejection_reasons)
        self.assertIn("reverse_low_overlap", result.rejection_reasons)

    def test_single_plane_rejected_even_with_zero_residual(self):
        points = asymmetric_cloud()[:182]
        result = register_local_clouds(points, points, np.eye(4), config())
        self.assertFalse(result.accepted)
        self.assertEqual(result.forward.inlier_rmse_m, 0.)
        self.assertEqual(result.forward.information.rank, 3)
        self.assertIn("forward_degenerate_geometry", result.rejection_reasons)

    def test_straight_open_corridor_rejects_unobservable_longitudinal_motion(self):
        x, small = np.meshgrid(np.linspace(-4., 4., 65), np.linspace(-.65, .65, 13))
        x, small = x.ravel(), small.ravel()
        points = np.concatenate((np.column_stack((x, x * 0. + 2., small)),
                                 np.column_stack((x, x * 0. - 2., small)),
                                 np.column_stack((x, small, x * 0. + 1.)),
                                 np.column_stack((x, small, x * 0. - 1.))))
        result = register_local_clouds(points, points, np.eye(4), config())
        self.assertFalse(result.accepted)
        self.assertEqual(result.forward.information.rank, 5)
        self.assertIn("forward_degenerate_geometry", result.rejection_reasons)

    def test_asymmetric_overlap_checked_both_directions(self):
        target = asymmetric_cloud()
        source = target[:182]
        result = register_local_clouds(source, target, np.eye(4), config())
        self.assertFalse(result.accepted)
        self.assertEqual(result.forward.fitness, 1.)
        self.assertLess(result.reverse.fitness, .4)
        self.assertIn("reverse_low_overlap", result.rejection_reasons)

    def test_normal_support_defect_returns_reason_not_fabricated_normals(self):
        source = asymmetric_cloud()
        result = register_local_clouds(source, source, np.eye(4), replace(config(), normal_radius_m=1e-6))
        self.assertFalse(result.accepted)
        self.assertIsNone(result.forward)
        self.assertEqual(result.source_normal_indices, ())
        self.assertIn("source_insufficient_normal_coverage", result.rejection_reasons)

    def test_collinear_points_not_accepted_as_valid_surface(self):
        source = np.column_stack((np.linspace(0., 3., 100), np.zeros(100), np.zeros(100)))
        result = register_local_clouds(source, source, np.eye(4), config())
        self.assertFalse(result.accepted)
        self.assertIn("source_insufficient_normal_points", result.rejection_reasons)

    def test_explicit_condition_bound_cannot_be_ignored(self):
        source, target, _ = transformed_pair()
        result = register_local_clouds(source, target, np.eye(4), replace(config(), max_information_condition=1.))
        self.assertFalse(result.accepted)
        self.assertEqual(result.forward.information.rank, 6)
        self.assertIn("forward_ill_conditioned_geometry", result.rejection_reasons)

    def test_explicit_point_to_point_bootstrap_is_recorded(self):
        source, target, truth = transformed_pair()
        result = register_local_clouds(source, target, np.eye(4),
                                       replace(config(), point_to_point_bootstrap_iterations=5))
        self.assertTrue(result.accepted, result.rejection_reasons)
        self.assertEqual(result.bootstrap_iterations_requested, 5)
        np.testing.assert_allclose(result.transformation_target_from_source, truth, atol=1e-6)

    def test_backend_failure_is_rejection_not_bool_success(self):
        import open3d as cpu
        with patch.object(cpu.pipelines.registration, "registration_icp", side_effect=RuntimeError("synthetic backend error")):
            result = register_local_clouds(asymmetric_cloud(), asymmetric_cloud(), np.eye(4), config())
        self.assertFalse(result.accepted)
        self.assertEqual(result.rejection_reasons, ("icp_backend_failure",))
        self.assertIn("synthetic backend error", result.runtime_error)


if __name__ == "__main__":
    unittest.main()
