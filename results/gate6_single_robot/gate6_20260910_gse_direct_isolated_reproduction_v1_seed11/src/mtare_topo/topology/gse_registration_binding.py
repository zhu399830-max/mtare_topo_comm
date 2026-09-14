"""Causal scan/pose/node binding for local registration, not graph merging.

Scans are expressed in their corresponding robot-local pose frame. Raw sensor
clouds must first use an explicit calibrated sensor-to-robot transform outside
this module. Candidate node IDs are graph-local records, never teacher IDs.
"""
from dataclasses import dataclass
import hashlib
import math

import numpy as np

from mtare_topo.topology.gse_graph_frames import DeploymentPose6D, node_from_robot
from mtare_topo.topology.gse_registration import RegistrationConfig, RegistrationEvidence, register_local_clouds


@dataclass(frozen=True)
class RobotFrameScan:
    timestamp_s: float
    points_robot_m: tuple[tuple[float, float, float], ...]

    def __post_init__(self):
        if isinstance(self.timestamp_s, bool) or not math.isfinite(self.timestamp_s):
            raise ValueError("invalid scan timestamp")
        try:
            points = np.asarray(self.points_robot_m, dtype=np.float64)
        except (TypeError, ValueError) as error:
            raise ValueError("invalid robot-frame point cloud") from error
        if points.ndim != 2 or points.shape[1:] != (3,) or len(points) < 3 or not np.isfinite(points).all():
            raise ValueError("robot-frame cloud requires at least three finite 3D points")
        object.__setattr__(self, "points_robot_m", tuple(tuple(float(v) for v in row) for row in points))

    @property
    def points_sha256(self):
        # Array bytes have canonical little-endian float64 shape Nx3. This binds
        # existing input, it is not dataset access or a generated teacher label.
        return hashlib.sha256(np.asarray(self.points_robot_m, dtype="<f8").tobytes(order="C")).hexdigest()


@dataclass(frozen=True)
class RegistrationCandidate:
    graph_local_node_id: int
    historical_scan: RobotFrameScan
    historical_pose: DeploymentPose6D

    def __post_init__(self):
        if (type(self.graph_local_node_id) is not int or self.graph_local_node_id < 0
                or not isinstance(self.historical_scan, RobotFrameScan)
                or not isinstance(self.historical_pose, DeploymentPose6D)):
            raise ValueError("invalid graph-local registration candidate")
        if self.historical_scan.timestamp_s != self.historical_pose.timestamp_s:
            raise ValueError("historical scan/pose timestamps differ")


@dataclass(frozen=True)
class BoundRegistrationEvidence:
    candidate_graph_local_node_id: int
    current_scan_timestamp_s: float
    historical_scan_timestamp_s: float
    reference_frame: str
    current_scan_sha256: str
    historical_scan_sha256: str
    current_pose: DeploymentPose6D
    historical_pose: DeploymentPose6D
    initial_target_from_source: tuple[tuple[float, ...], ...]
    configuration: RegistrationConfig
    registration: RegistrationEvidence


def _prepare(current_scan, current_pose, candidate, config):
    if (not isinstance(current_scan, RobotFrameScan) or not isinstance(current_pose, DeploymentPose6D)
            or not isinstance(candidate, RegistrationCandidate) or not isinstance(config, RegistrationConfig)):
        raise ValueError("typed scan, deployment poses, candidate and complete registration config required")
    if current_scan.timestamp_s != current_pose.timestamp_s:
        raise ValueError("current scan/pose timestamps differ")
    if candidate.historical_scan.timestamp_s > current_scan.timestamp_s:
        raise ValueError("historical scan cannot come from the future")
    relative = node_from_robot(node_pose=candidate.historical_pose, robot_pose=current_pose)
    initial = np.eye(4)
    initial[:3, :3] = relative.rotation_node_from_robot
    initial[:3, 3] = relative.translation_node_from_robot_m
    return initial


def register_candidate(*, current_scan: RobotFrameScan, current_pose: DeploymentPose6D,
                       candidate: RegistrationCandidate,
                       config: RegistrationConfig) -> BoundRegistrationEvidence:
    """Compute the only allowed initial guess from deployment poses, then ICP.

    The historical cloud pose is not automatically the persistent node anchor.
    The resulting transform maps current robot coordinates into the historical
    cloud's robot coordinates. An eventual graph adapter must compose that with
    its stored node-frame transform; it must not assume the two frames coincide.

    No arbitrary/GT initial transform parameter is exposed. This cannot prove a
    caller did not forge deployment poses; provenance is a deployment contract.
    """
    initial = _prepare(current_scan, current_pose, candidate, config)
    evidence = register_local_clouds(current_scan.points_robot_m, candidate.historical_scan.points_robot_m,
                                     initial, config)
    return BoundRegistrationEvidence(
        candidate.graph_local_node_id, current_scan.timestamp_s, candidate.historical_scan.timestamp_s,
        current_pose.reference_frame, current_scan.points_sha256, candidate.historical_scan.points_sha256,
        current_pose, candidate.historical_pose,
        tuple(tuple(float(v) for v in row) for row in initial), config, evidence,
    )


def register_candidates(*, current_scan: RobotFrameScan, current_pose: DeploymentPose6D,
                        candidates: tuple[RegistrationCandidate, ...],
                        config: RegistrationConfig) -> tuple[BoundRegistrationEvidence, ...]:
    """Return evidence for every candidate, preserving order and ambiguity.

    No ranking, first-success shortcut, winner selection or node mutation occurs.
    All bindings are checked before any ICP, including late invalid candidates.
    """
    if (type(candidates) is not tuple or any(not isinstance(c, RegistrationCandidate) for c in candidates)
            or len({c.graph_local_node_id for c in candidates}) != len(candidates)):
        raise ValueError("distinct typed graph-local candidates required")
    if (not isinstance(current_scan, RobotFrameScan) or not isinstance(current_pose, DeploymentPose6D)
            or not isinstance(config, RegistrationConfig) or current_scan.timestamp_s != current_pose.timestamp_s):
        raise ValueError("typed synchronous current scan/pose and complete config required")
    for candidate in candidates:
        _prepare(current_scan, current_pose, candidate, config)
    return tuple(register_candidate(current_scan=current_scan, current_pose=current_pose,
                                     candidate=candidate, config=config) for candidate in candidates)
