from dataclasses import replace

import numpy as np
import pytest

from mtare_topo.topology.gse_causal_graph import PoseEstimate, StructuralObservation, StructuralPort
from mtare_topo.topology.gse_graph_frames import DeploymentPose6D, node_from_robot, ports_in_node_frame


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    skew = np.array([[0., -z, y], [z, 0., -x], [-y, x, 0.]])
    return np.eye(3) + np.sin(angle) * skew + (1. - np.cos(angle)) * (skew @ skew)


def pose(t=1., r=None, translation=(0., 0., 0.), covariance=None):
    return DeploymentPose6D(t, "odom", np.eye(3) if r is None else r, translation,
                            np.eye(6) * .01 if covariance is None else covariance)


def observation(t=1., direction=(1., 0., 0.)):
    return StructuralObservation(t, "junction", .9, .1,
                                 (StructuralPort(9, direction, 3., 4., .6),))


def test_same_pose_preserves_geometry_and_local_id():
    anchor = pose(r=rotation((1., 2., 3.), .7), translation=(2., -4., 8.))
    result = ports_in_node_frame(observation(), robot_pose=anchor, node_pose=anchor)
    np.testing.assert_allclose(result[0].geometry.direction_node, (1., 0., 0.), atol=1e-14)
    assert (result[0].observation_local_id, result[0].geometry.width_m,
            result[0].geometry.height_m, result[0].geometry.confidence) == (9, 3., 4., .6)


def test_revisit_rotated_robot_recovers_persistent_direction():
    # Physical +x port appears as -y in a robot rotated +90 degrees.
    result = ports_in_node_frame(observation(2., (0., -1., 0.)),
                                 robot_pose=pose(2., rotation((0., 0., 1.), np.pi / 2), (10., 20., 30.)),
                                 node_pose=pose())
    np.testing.assert_allclose(result[0].geometry.direction_node, (1., 0., 0.), atol=1e-14)


def test_six_dof_roll_pitch_and_translation_not_yaw_only():
    r = rotation((1., 2., 0.), .8)
    result = ports_in_node_frame(observation(2.), robot_pose=pose(2., r), node_pose=pose())
    np.testing.assert_allclose(result[0].geometry.direction_node, r[:, 0], atol=1e-14)
    assert abs(result[0].geometry.direction_node[2]) > .1


def test_arbitrary_common_se3_leaves_relative_transform_and_ports_invariant():
    node = pose(r=rotation((1., 2., 3.), .4), translation=(2., 5., -3.))
    robot = pose(3., rotation((2., -1., 1.), 1.3), (-2., 3., 7.))
    common_r, common_t = rotation((3., 2., -2.), 1.9), np.array([50., -20., 17.])
    def move(p):
        return replace(p, rotation_reference_from_local=common_r @ p.rotation_reference_from_local,
                       translation_reference_from_local_m=common_r @ p.translation_reference_from_local_m + common_t)
    before = node_from_robot(node_pose=node, robot_pose=robot)
    after = node_from_robot(node_pose=move(node), robot_pose=move(robot))
    np.testing.assert_allclose(after.rotation_node_from_robot, before.rotation_node_from_robot, atol=1e-14)
    np.testing.assert_allclose(after.translation_node_from_robot_m, before.translation_node_from_robot_m, atol=1e-14)
    a = ports_in_node_frame(observation(3.), node_pose=node, robot_pose=robot)
    b = ports_in_node_frame(observation(3.), node_pose=move(node), robot_pose=move(robot))
    np.testing.assert_allclose(a[0].geometry.direction_node, b[0].geometry.direction_node, atol=1e-14)


def test_relative_translation_is_in_node_axes():
    result = node_from_robot(node_pose=pose(r=rotation((0., 0., 1.), np.pi / 2), translation=(1., 2., 0.)),
                             robot_pose=pose(2., translation=(1., 4., 3.)))
    np.testing.assert_allclose(result.translation_node_from_robot_m, (2., 0., 3.), atol=1e-14)


@pytest.mark.parametrize("r", [np.diag([-1., 1., 1.]), np.eye(3) * 2., np.zeros((3, 3)),
                                 np.eye(4), np.full((3, 3), np.nan)])
def test_invalid_rotation_reflection_scale_rejected(r):
    with pytest.raises(ValueError):
        pose(r=r)


@pytest.mark.parametrize("covariance", [np.eye(3), np.diag([1., 1., 1., 1., 1., -1.]),
                                          np.ones((6, 6)) + np.triu(np.ones((6, 6)), 1),
                                          np.full((6, 6), np.nan)])
def test_invalid_covariance_rejected_without_repair(covariance):
    with pytest.raises(ValueError):
        pose(covariance=covariance)


def test_semidefinite_covariance_and_defensive_copy():
    r, t, covariance = np.eye(3), np.zeros(3), np.zeros((6, 6))
    p = pose(r=r, translation=t, covariance=covariance)
    r[0, 0], t[0], covariance[0, 0] = 8., 9., 10.
    assert p.rotation_reference_from_local[0][0] == 1.
    assert p.translation_reference_from_local_m[0] == 0.
    assert p.covariance_local_tangent[0][0] == 0.


@pytest.mark.parametrize("t", [0., 2., float("nan"), float("inf")])
def test_stale_future_nonfinite_robot_pose_rejected(t):
    with pytest.raises(ValueError):
        ports_in_node_frame(observation(1.), robot_pose=pose(t), node_pose=pose(0.))


def test_future_node_anchor_rejected_but_old_node_anchor_is_valid():
    with pytest.raises(ValueError, match="future"):
        node_from_robot(node_pose=pose(3.), robot_pose=pose(2.))
    ports_in_node_frame(observation(3.), robot_pose=pose(3.), node_pose=pose(1.))


def test_different_reference_frames_rejected():
    with pytest.raises(ValueError, match="same reference"):
        node_from_robot(node_pose=pose(), robot_pose=replace(pose(2.), reference_frame="map"))


def test_position_only_pose_cannot_be_silently_upgraded():
    old = PoseEstimate(1., (0., 0., 0.), (.01, .01, .01))
    with pytest.raises(ValueError, match="6-DoF"):
        node_from_robot(node_pose=old, robot_pose=pose())


def test_no_ports_is_empty_not_synthetic_port():
    empty = StructuralObservation(1., "unknown", .1, .9)
    assert ports_in_node_frame(empty, robot_pose=pose(), node_pose=pose()) == ()


def test_transform_output_is_deterministic():
    kwargs = dict(robot_pose=pose(2., rotation((1., 3., 2.), .2)), node_pose=pose())
    assert ports_in_node_frame(observation(2.), **kwargs) == ports_in_node_frame(observation(2.), **kwargs)
