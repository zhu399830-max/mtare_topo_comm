"""No datasets: binding mocks plus actual synthetic Open3D integration."""
from dataclasses import fields, replace
import importlib.util
import inspect
import unittest
from unittest.mock import patch

import numpy as np

from mtare_topo.topology.gse_graph_frames import DeploymentPose6D
from mtare_topo.topology.gse_registration import RegistrationConfig, RegistrationEvidence
from mtare_topo.topology.gse_registration_binding import (
    BoundRegistrationEvidence, RegistrationCandidate, RobotFrameScan,
    register_candidate, register_candidates,
)


def config():
    return RegistrationConfig(.35, .34, 40, 6, .02, .001, .8, 80, 1e-10, 1e-10, 0, 40, .8, .01, 1e-7, 1e5)


def pose(t, translation=(0., 0., 0.), rotation=None):
    return DeploymentPose6D(t, "odom", np.eye(3) if rotation is None else rotation, translation, np.eye(6) * .01)


def scan(t):
    return RobotFrameScan(t, ((0., 0., 0.), (1., 0., 0.), (0., 1., 0.)))


def candidate(node=7, t=1.):
    return RegistrationCandidate(node, scan(t), pose(t))


def fake_evidence(accepted):
    return RegistrationEvidence(None, None, None, accepted, () if accepted else ("synthetic_rejection",),
                                "mock-binding-only", (), (), 0)


class RegistrationBindingTests(unittest.TestCase):
    def test_initial_is_derived_from_deployment_pose(self):
        r = np.array([[0., -1., 0.], [1., 0., 0.], [0., 0., 1.]])
        history = RegistrationCandidate(7, scan(1.), pose(1., (1., 2., 0.), r))
        with patch("mtare_topo.topology.gse_registration_binding.register_local_clouds", return_value=fake_evidence(True)) as backend:
            result = register_candidate(current_scan=scan(2.), current_pose=pose(2., (1., 4., 3.)),
                                         candidate=history, config=config())
        initial = backend.call_args.args[2]
        np.testing.assert_array_equal(initial[:3, :3], r.T)
        np.testing.assert_array_equal(initial[:3, 3], (2., 0., 3.))
        self.assertEqual(result.candidate_graph_local_node_id, 7)
        self.assertEqual(result.current_scan_timestamp_s, 2.)
        self.assertEqual(result.historical_scan_timestamp_s, 1.)
        self.assertEqual(result.current_scan_sha256, scan(2.).points_sha256)
        self.assertEqual(result.configuration, config())

    def test_two_passing_candidates_do_not_choose_winner(self):
        with patch("mtare_topo.topology.gse_registration_binding.register_local_clouds", return_value=fake_evidence(True)) as backend:
            results = register_candidates(current_scan=scan(2.), current_pose=pose(2.),
                                           candidates=(candidate(9), candidate(7)), config=config())
        self.assertEqual(backend.call_count, 2)
        self.assertEqual([r.candidate_graph_local_node_id for r in results], [9, 7])
        self.assertTrue(all(r.registration.accepted for r in results))
        self.assertFalse(hasattr(results[0], "merge_node"))

    def test_failed_candidate_is_retained(self):
        with patch("mtare_topo.topology.gse_registration_binding.register_local_clouds", return_value=fake_evidence(False)):
            result = register_candidate(current_scan=scan(2.), current_pose=pose(2.), candidate=candidate(), config=config())
        self.assertFalse(result.registration.accepted)
        self.assertEqual(result.registration.rejection_reasons, ("synthetic_rejection",))

    def test_future_late_candidate_prevents_all_icp(self):
        with patch("mtare_topo.topology.gse_registration_binding.register_local_clouds") as backend:
            with self.assertRaisesRegex(ValueError, "future"):
                register_candidates(current_scan=scan(2.), current_pose=pose(2.),
                                    candidates=(candidate(7), candidate(8, 3.)), config=config())
            backend.assert_not_called()

    def test_current_pose_timestamp_mismatch_rejected(self):
        with patch("mtare_topo.topology.gse_registration_binding.register_local_clouds") as backend:
            for t in (1., 3.):
                with self.subTest(t=t), self.assertRaises(ValueError):
                    register_candidate(current_scan=scan(2.), current_pose=pose(t), candidate=candidate(), config=config())
            backend.assert_not_called()

    def test_historical_pose_timestamp_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "historical"):
            RegistrationCandidate(7, scan(1.), pose(2.))

    def test_reference_frame_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference frame"):
            register_candidate(current_scan=scan(2.), current_pose=replace(pose(2.), reference_frame="map"),
                               candidate=candidate(), config=config())

    def test_duplicate_candidates_rejected(self):
        with self.assertRaises(ValueError):
            register_candidates(current_scan=scan(2.), current_pose=pose(2.),
                                candidates=(candidate(), candidate()), config=config())

    def test_empty_candidates_return_empty_no_backend(self):
        with patch("mtare_topo.topology.gse_registration_binding.register_local_clouds") as backend:
            self.assertEqual(register_candidates(current_scan=scan(2.), current_pose=pose(2.),
                                                  candidates=(), config=config()), ())
            backend.assert_not_called()

    def test_scan_input_copy_and_hash(self):
        points = np.eye(3)
        a = RobotFrameScan(2., points)
        initial_hash = a.points_sha256
        points[0, 0] = 12.
        self.assertEqual(a.points_sha256, initial_hash)
        self.assertNotEqual(a.points_sha256, RobotFrameScan(2., points).points_sha256)

    def test_invalid_cloud_or_timestamp_rejected(self):
        for points, t in ((np.eye(2), 1.), (np.full((3, 3), np.nan), 1.), (np.eye(3), float("nan"))):
            with self.subTest(t=t), self.assertRaises(ValueError):
                RobotFrameScan(t, points)

    def test_no_external_initial_or_teacher_argument(self):
        names = set(inspect.signature(register_candidate).parameters)
        self.assertEqual(names, {"current_scan", "current_pose", "candidate", "config"})
        result_fields = {f.name for f in fields(BoundRegistrationEvidence)}
        self.assertFalse(result_fields & {"winner", "merge_node", "teacher_node_id", "traversal_id"})


@unittest.skipUnless(importlib.util.find_spec("open3d"), "requires qualified Open3D CPU sidecar")
class RegistrationBindingIntegrationTests(unittest.TestCase):
    def test_actual_icp_from_inexact_deployment_guess(self):
        u, v = np.meshgrid(np.linspace(0., 1., 14), np.linspace(0., 1., 13))
        u, v = u.ravel(), v.ravel()
        source = np.concatenate((np.column_stack((u * 0., -2. + u, .1 + v)),
                                 np.column_stack((.3 + 1.5 * u, v * 0., 1.2 + v)),
                                 np.column_stack((2. + u, 1. + v, u * 0.))))
        theta = .03
        r = np.array([[np.cos(theta), -np.sin(theta), 0.], [np.sin(theta), np.cos(theta), 0.], [0., 0., 1.]])
        translation = np.array([.07, -.05, .04])
        target = source @ r.T + translation
        truth = np.eye(4)
        truth[:3, :3], truth[:3, 3] = r, translation
        # Deployment pose has an intentionally incomplete 2cm x estimate, no
        # rotation truth. Actual ICP must recover the rest from local points.
        result = register_candidate(current_scan=RobotFrameScan(2., source), current_pose=pose(2., (.02, 0., 0.)),
                                     candidate=RegistrationCandidate(3, RobotFrameScan(1., target), pose(1.)), config=config())
        self.assertTrue(result.registration.accepted, result.registration.rejection_reasons)
        self.assertGreater(np.max(np.abs(np.asarray(result.initial_target_from_source) - truth)), .04)
        np.testing.assert_allclose(result.registration.transformation_target_from_source, truth, atol=1e-6)
        self.assertEqual(result.registration.forward.correspondences, 546)


if __name__ == "__main__":
    unittest.main()
