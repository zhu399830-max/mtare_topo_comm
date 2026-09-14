"""Deployment 6-DoF frame adapter, independent of the frozen graph kernel.

Transforms are frame-to-reference: p_ref = R_ref_frame @ p_frame + t_ref_frame.
Covariance is a 6x6 local-tangent covariance ordered [tx,ty,tz,rx,ry,rz],
translation in metres and rotation in radians. It is validated and preserved,
not inferred from a position-only covariance, propagated or calibrated here.
"""
from dataclasses import dataclass
import math

import numpy as np

from mtare_topo.topology.gse_causal_graph import StructuralObservation
from mtare_topo.topology.gse_port_updates import ObservedPort, PortGeometry


def _matrix(value, shape, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise ValueError(f"invalid {name}") from error
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"invalid {name}")
    return array


def _rotation(value):
    rotation = _matrix(value, (3, 3), "rotation")
    if (not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0., atol=1e-8)
            or abs(float(np.linalg.det(rotation)) - 1.) > 1e-8):
        raise ValueError("rotation must be proper SO(3), not a reflection or scale")
    return rotation


def _tuples(array):
    return tuple(tuple(float(v) for v in row) for row in array)


@dataclass(frozen=True)
class DeploymentPose6D:
    timestamp_s: float
    reference_frame: str
    rotation_reference_from_local: tuple[tuple[float, ...], ...]
    translation_reference_from_local_m: tuple[float, float, float]
    covariance_local_tangent: tuple[tuple[float, ...], ...]

    def __post_init__(self):
        if (isinstance(self.timestamp_s, bool) or not math.isfinite(self.timestamp_s)
                or type(self.reference_frame) is not str or not self.reference_frame.strip()):
            raise ValueError("invalid timestamp or reference frame")
        rotation = _rotation(self.rotation_reference_from_local)
        translation = _matrix(self.translation_reference_from_local_m, (3,), "translation")
        covariance = _matrix(self.covariance_local_tangent, (6, 6), "6-DoF covariance")
        tolerance = 1e-10 * max(1., float(np.max(np.abs(covariance))))
        if (not np.allclose(covariance, covariance.T, rtol=0., atol=tolerance)
                or np.linalg.eigvalsh((covariance + covariance.T) * .5).min() < -tolerance):
            raise ValueError("6-DoF covariance must be symmetric positive semidefinite")
        # Defensive immutable copies: callers may supply arrays without retaining
        # a writable alias to the pose. No rotation/covariance repair is applied.
        object.__setattr__(self, "rotation_reference_from_local", _tuples(rotation))
        object.__setattr__(self, "translation_reference_from_local_m", tuple(float(v) for v in translation))
        object.__setattr__(self, "covariance_local_tangent", _tuples(covariance))


@dataclass(frozen=True)
class NodeFromRobotTransform:
    rotation_node_from_robot: tuple[tuple[float, ...], ...]
    translation_node_from_robot_m: tuple[float, float, float]


def node_from_robot(*, node_pose: DeploymentPose6D,
                    robot_pose: DeploymentPose6D) -> NodeFromRobotTransform:
    """Return T_node_robot = inverse(T_ref_node) @ T_ref_robot.

    The node pose is its persistent frame anchor, so an older anchor is valid;
    a future anchor is not. Both poses must share the same deployment reference.
    """
    if not isinstance(node_pose, DeploymentPose6D) or not isinstance(robot_pose, DeploymentPose6D):
        raise ValueError("explicit deployment 6-DoF poses required")
    if node_pose.reference_frame != robot_pose.reference_frame:
        raise ValueError("poses must use the same reference frame")
    if node_pose.timestamp_s > robot_pose.timestamp_s:
        raise ValueError("node anchor cannot come from the future")
    node_rotation = np.asarray(node_pose.rotation_reference_from_local)
    robot_rotation = np.asarray(robot_pose.rotation_reference_from_local)
    rotation = node_rotation.T @ robot_rotation
    translation = node_rotation.T @ (np.asarray(robot_pose.translation_reference_from_local_m)
                                    - np.asarray(node_pose.translation_reference_from_local_m))
    return NodeFromRobotTransform(_tuples(rotation), tuple(float(v) for v in translation))


def ports_in_node_frame(observation: StructuralObservation, *,
                        robot_pose: DeploymentPose6D,
                        node_pose: DeploymentPose6D) -> tuple[ObservedPort, ...]:
    """Convert direction vectors only; width/height/confidence and IDs persist.

    Translation affects port positions, not directions. The frozen StructuralPort
    has no port position: this adapter does not invent one from the robot origin.
    All ports, including low-confidence candidates, are retained unchanged in
    number. This adapter performs neither correspondence nor visibility gating.
    """
    if not isinstance(observation, StructuralObservation) or not isinstance(robot_pose, DeploymentPose6D):
        raise ValueError("typed observation and deployment pose required")
    if observation.timestamp_s != robot_pose.timestamp_s:
        raise ValueError("observation and robot pose timestamp mismatch (stale/future pose)")
    transform = node_from_robot(node_pose=node_pose, robot_pose=robot_pose)
    rotation = np.asarray(transform.rotation_node_from_robot)
    return tuple(ObservedPort(port.local_id, PortGeometry(
        tuple(float(v) for v in rotation @ np.asarray(port.direction_robot)),
        port.width_m, port.height_m, port.confidence,
    )) for port in observation.ports)
