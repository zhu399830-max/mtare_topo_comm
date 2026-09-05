from dataclasses import fields, replace
import unittest

import numpy as np

from mtare_topo.topology.gse_causal_graph import StructuralObservation, StructuralPort
from mtare_topo.topology.gse_graph_frames import DeploymentPose6D
from mtare_topo.topology.gse_registered_node_frame import (
    RegisteredNodeFrame, registered_node_frame, registered_ports_in_node_frame,
)
from mtare_topo.topology.gse_registration import RegistrationConfig, RegistrationEvidence
from mtare_topo.topology.gse_registration_binding import BoundRegistrationEvidence


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew


def pose(t, r=None, xyz=(0., 0., 0.)):
    return DeploymentPose6D(t, "odom", np.eye(3) if r is None else r, xyz, np.eye(6) * .01)


def transform(r=None, xyz=(0., 0., 0.)):
    matrix = np.eye(4)
    matrix[:3, :3] = np.eye(3) if r is None else r
    matrix[:3, 3] = xyz
    return tuple(tuple(float(v) for v in row) for row in matrix)


def binding(node=7):
    # Synthetic evidence consumer fixture, not a claim that ICP ran in this test.
    estimated = transform(rotation((1., 2., 3.), .35), (.4, -.3, .2))
    registration = RegistrationEvidence(estimated, None, None, True, (), "synthetic-frame-only", (), (), 0)
    config = RegistrationConfig(.35, .34, 40, 6, .02, .001, .8, 80, 1e-10, 1e-10, 0, 40, .8, .01, 1e-7, 1e5)
    return BoundRegistrationEvidence(node, 3., 2., "odom", "0" * 64, "1" * 64,
                                      pose(3.), pose(2., rotation((2., 1., 0.), .6), (3., 4., 5.)),
                                      transform(), config, registration)


def anchor():
    return pose(1., rotation((0., 0., 1.), .5), (1., -2., 3.))


class RegisteredNodeFrameTests(unittest.TestCase):
    def test_nonzero_roll_pitch_yaw_and_translation_compose_history_not_initial(self):
        b, a = binding(), anchor()
        result = registered_node_frame(b, graph_local_node_id=7, node_anchor_pose=a)
        ref_node = np.asarray(transform(a.rotation_reference_from_local, a.translation_reference_from_local_m))
        ref_history = np.asarray(transform(b.historical_pose.rotation_reference_from_local,
                                           b.historical_pose.translation_reference_from_local_m))
        estimated = np.asarray(b.registration.transformation_target_from_source)
        expected = np.linalg.inv(ref_node) @ ref_history @ estimated
        np.testing.assert_allclose(result.transformation_node_from_current_robot, expected, atol=1e-14)
        self.assertGreater(np.max(np.abs(expected - estimated)), 1.)
        self.assertGreater(np.max(np.abs(expected - np.linalg.inv(ref_node))), 1.)
        self.assertEqual(result.registration_binding, b)
        self.assertEqual(result.node_anchor_pose, a)

    def test_anchor_equal_history_is_registration_transform(self):
        b = binding()
        result = registered_node_frame(b, graph_local_node_id=7, node_anchor_pose=b.historical_pose)
        np.testing.assert_allclose(result.transformation_node_from_current_robot,
                                   b.registration.transformation_target_from_source, atol=1e-14)

    def test_anchor_after_historical_scan_but_not_future_is_valid(self):
        result = registered_node_frame(binding(), graph_local_node_id=7, node_anchor_pose=replace(anchor(), timestamp_s=2.5))
        self.assertEqual(result.timestamp_s, 3.)

    def test_future_anchor_rejected(self):
        with self.assertRaisesRegex(ValueError, "future"):
            registered_node_frame(binding(), graph_local_node_id=7, node_anchor_pose=replace(anchor(), timestamp_s=4.))

    def test_wrong_node_id_rejected(self):
        for node_id in (8, True, -1):
            with self.subTest(node_id=node_id), self.assertRaises(ValueError):
                registered_node_frame(binding(), graph_local_node_id=node_id, node_anchor_pose=anchor())

    def test_reference_frame_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "reference"):
            registered_node_frame(binding(), graph_local_node_id=7,
                                  node_anchor_pose=replace(anchor(), reference_frame="map"))

    def test_binding_scan_pose_mismatch_or_future_history_rejected(self):
        for kwargs in ({"current_scan_timestamp_s": 4.}, {"historical_scan_timestamp_s": 3.},
                       {"historical_scan_timestamp_s": 4., "historical_pose": pose(4.)},
                       {"current_scan_timestamp_s": float("nan")}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                registered_node_frame(replace(binding(), **kwargs), graph_local_node_id=7, node_anchor_pose=anchor())

    def test_failed_unknown_or_contradictory_registration_never_falls_back(self):
        for kwargs in ({"accepted": False}, {"accepted": "True"}, {"transformation_target_from_source": None},
                       {"rejection_reasons": ("degenerate",)}, {"runtime_error": "failure"}):
            b = binding()
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "no pose fallback"):
                registered_node_frame(replace(b, registration=replace(b.registration, **kwargs)),
                                      graph_local_node_id=7, node_anchor_pose=anchor())

    def test_nonse3_estimate_cannot_be_hidden_by_accepted_flag(self):
        b = binding()
        invalid = np.eye(4)
        invalid[0, 0] = -1
        with self.assertRaisesRegex(ValueError, "SE3"):
            registered_node_frame(replace(b, registration=replace(b.registration,
                                  transformation_target_from_source=tuple(map(tuple, invalid)))),
                                  graph_local_node_id=7, node_anchor_pose=anchor())

    def test_two_accepted_candidates_remain_independent_hypotheses(self):
        results = tuple(registered_node_frame(binding(i), graph_local_node_id=i, node_anchor_pose=anchor()) for i in (7, 9))
        self.assertEqual(tuple(r.graph_local_node_id for r in results), (7, 9))
        self.assertEqual(len(results), 2)
        self.assertFalse(hasattr(results[0], "merge_node"))

    def test_port_directions_rotate_only_no_position_fabrication(self):
        frame = registered_node_frame(binding(), graph_local_node_id=7, node_anchor_pose=anchor())
        observation = StructuralObservation(3., "junction", .9, .1, (StructuralPort(8, (1., 0., 0.), 3., 4., .6),))
        ports = registered_ports_in_node_frame(observation, frame=frame)
        expected = np.asarray(frame.transformation_node_from_current_robot)[:3, 0]
        np.testing.assert_allclose(ports[0].geometry.direction_node, expected, atol=1e-14)
        self.assertEqual((ports[0].observation_local_id, ports[0].geometry.width_m,
                          ports[0].geometry.height_m, ports[0].geometry.confidence), (8, 3., 4., .6))
        self.assertFalse(hasattr(ports[0].geometry, "position_node"))

    def test_port_timestamp_mismatch_rejected(self):
        frame = registered_node_frame(binding(), graph_local_node_id=7, node_anchor_pose=anchor())
        with self.assertRaisesRegex(ValueError, "timestamp"):
            registered_ports_in_node_frame(StructuralObservation(2., "unknown", .1, .9), frame=frame)

    def test_tampered_frame_is_not_trusted(self):
        frame = registered_node_frame(binding(), graph_local_node_id=7, node_anchor_pose=anchor())
        with self.assertRaisesRegex(ValueError, "altered"):
            registered_ports_in_node_frame(StructuralObservation(3., "unknown", .1, .9),
                                            frame=replace(frame, transformation_node_from_current_robot=transform()))

    def test_no_covariance_or_final_merge_output_and_determinism(self):
        names = {f.name for f in fields(RegisteredNodeFrame)}
        self.assertFalse(names & {"covariance", "covariance_local_tangent", "winner", "merge_node", "edge_id"})
        self.assertEqual(registered_node_frame(binding(), graph_local_node_id=7, node_anchor_pose=anchor()),
                         registered_node_frame(binding(), graph_local_node_id=7, node_anchor_pose=anchor()))


if __name__ == "__main__":
    unittest.main()
