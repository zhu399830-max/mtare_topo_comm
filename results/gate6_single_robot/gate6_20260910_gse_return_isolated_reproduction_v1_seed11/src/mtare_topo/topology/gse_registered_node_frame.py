"""Compose an accepted scan registration into a persistent node frame.

This is coordinate/evidence plumbing only. An ICP candidate is not a globally
unique place, so this module never merges nodes, updates graphs or covariances.
"""
from dataclasses import dataclass
import math

import numpy as np

from mtare_topo.topology.gse_causal_graph import StructuralObservation
from mtare_topo.topology.gse_graph_frames import DeploymentPose6D
from mtare_topo.topology.gse_port_updates import ObservedPort, PortGeometry
from mtare_topo.topology.gse_registration import RegistrationEvidence, _se3
from mtare_topo.topology.gse_registration_binding import BoundRegistrationEvidence


@dataclass(frozen=True)
class RegisteredNodeFrame:
    graph_local_node_id: int
    timestamp_s: float
    transformation_node_from_current_robot: tuple[tuple[float, ...], ...]
    node_anchor_pose: DeploymentPose6D
    registration_binding: BoundRegistrationEvidence


def _pose_transform(pose):
    transform = np.eye(4)
    transform[:3, :3] = pose.rotation_reference_from_local
    transform[:3, 3] = pose.translation_reference_from_local_m
    return transform


def registered_node_frame(binding: BoundRegistrationEvidence, *,
                          graph_local_node_id: int,
                          node_anchor_pose: DeploymentPose6D) -> RegisteredNodeFrame:
    """T_node_current = inv(T_ref_node) @ T_ref_history @ T_history_current.

    The last factor is the estimated scan registration, never its deployment
    initial guess. Rejected, unknown or inconsistent evidence raises ValueError;
    no fallback to original deployment pose is performed.

    A node anchor may have been established after the stored historical scan,
    provided it was already available at the current observation time. The
    coordinate product does not require pretending those timestamps coincide.
    """
    if (not isinstance(binding, BoundRegistrationEvidence)
            or not isinstance(node_anchor_pose, DeploymentPose6D)
            or type(graph_local_node_id) is not int or graph_local_node_id < 0):
        raise ValueError("typed registration binding, node anchor and graph-local ID required")
    if (type(binding.candidate_graph_local_node_id) is not int
            or binding.candidate_graph_local_node_id != graph_local_node_id):
        raise ValueError("registration candidate/node ID mismatch")
    if not isinstance(binding.current_pose, DeploymentPose6D) or not isinstance(binding.historical_pose, DeploymentPose6D):
        raise ValueError("binding requires explicit deployment poses")
    if (not math.isfinite(binding.current_scan_timestamp_s)
            or not math.isfinite(binding.historical_scan_timestamp_s)
            or binding.current_scan_timestamp_s != binding.current_pose.timestamp_s
            or binding.historical_scan_timestamp_s != binding.historical_pose.timestamp_s
            or binding.historical_scan_timestamp_s > binding.current_scan_timestamp_s
            or node_anchor_pose.timestamp_s > binding.current_scan_timestamp_s):
        raise ValueError("inconsistent scan/pose time or future node anchor")
    if not (binding.reference_frame == binding.current_pose.reference_frame
            == binding.historical_pose.reference_frame == node_anchor_pose.reference_frame):
        raise ValueError("binding and node anchor reference frames differ")
    registration = binding.registration
    if (not isinstance(registration, RegistrationEvidence) or registration.accepted is not True
            or registration.rejection_reasons or registration.runtime_error is not None
            or registration.transformation_target_from_source is None):
        raise ValueError("registration is rejected, unknown or inconsistent; no pose fallback")
    historical_from_current = _se3(registration.transformation_target_from_source)
    node_from_history = np.linalg.inv(_pose_transform(node_anchor_pose)) @ _pose_transform(binding.historical_pose)
    node_from_current = _se3(node_from_history @ historical_from_current)
    return RegisteredNodeFrame(graph_local_node_id, binding.current_scan_timestamp_s,
                               tuple(tuple(float(v) for v in row) for row in node_from_current),
                               node_anchor_pose, binding)


def registered_ports_in_node_frame(observation: StructuralObservation, *,
                                   frame: RegisteredNodeFrame) -> tuple[ObservedPort, ...]:
    """Transform directions without inventing absent port positions.

    All local IDs, confidence, width and height persist. No correspondence,
    candidate ranking or mutation occurs. The returned directions remain a
    hypothesis tied to this specific registration candidate.
    """
    if not isinstance(observation, StructuralObservation) or not isinstance(frame, RegisteredNodeFrame):
        raise ValueError("typed structural observation and registered frame required")
    if observation.timestamp_s != frame.timestamp_s:
        raise ValueError("structural observation/registered frame timestamp mismatch")
    # Revalidate binding rather than trust a hand-edited dataclass transform.
    expected = registered_node_frame(frame.registration_binding, graph_local_node_id=frame.graph_local_node_id,
                                     node_anchor_pose=frame.node_anchor_pose)
    if frame != expected:
        raise ValueError("registered frame was altered after evidence composition")
    rotation = np.asarray(frame.transformation_node_from_current_robot)[:3, :3]
    return tuple(ObservedPort(port.local_id, PortGeometry(
        tuple(float(v) for v in rotation @ np.asarray(port.direction_robot)),
        port.width_m, port.height_m, port.confidence,
    )) for port in observation.ports)
