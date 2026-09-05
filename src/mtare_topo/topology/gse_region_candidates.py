"""Lossless, decision-free single-observation region candidate conversion.

No teacher interface, filtering, confidence calibration, tracking or graph IO.
Numerical support is not visibility. Equal support masks cannot prove that two
nondegenerate arrays originated in the same forward; callers must bind the
prediction and axes at production time, not infer identity from geometry.
"""
from dataclasses import dataclass
import math
from numbers import Real

import numpy as np
import torch

from mtare_topo.representation.gse_region_queries import RegionPrediction, tokens_from_axes
from mtare_topo.topology.gse_registration import _se3


def _frame(value):
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class ObservationStamp:
    timestamp_s: float
    sensor_frame: str
    robot_frame: str

    def __post_init__(self):
        if (isinstance(self.timestamp_s, bool) or not isinstance(self.timestamp_s, Real)
                or not math.isfinite(self.timestamp_s)
                or not _frame(self.sensor_frame) or not _frame(self.robot_frame)):
            raise ValueError("finite explicit observation time and frame names required")


@dataclass(frozen=True)
class SensorToRobotExtrinsic:
    sensor_frame: str
    robot_frame: str
    transformation_robot_from_sensor: tuple[tuple[float, ...], ...]
    calibration_reference: str

    def __post_init__(self):
        if not all(_frame(x) for x in (self.sensor_frame, self.robot_frame, self.calibration_reference)):
            raise ValueError("explicit extrinsic frame names and calibration reference required")
        matrix = _se3(self.transformation_robot_from_sensor)
        object.__setattr__(self, "transformation_robot_from_sensor", tuple(tuple(float(x) for x in row) for row in matrix))


@dataclass(frozen=True)
class DirectionCandidate:
    observation_local_token_id: int
    anchor_robot_m: tuple[float, float, float]
    direction_robot: tuple[float, float, float] | None
    numerical_direction_supported: bool
    opening_width_m: None = None
    opening_height_m: None = None


@dataclass(frozen=True)
class RegionCandidate:
    observation_local_query_id: int
    center_robot_m: tuple[float, float, float]
    event_logits: tuple[float, float, float]  # corridor/junction/terminal, NOT selected event
    presence_logit: float
    uncertainty_uncalibrated: float
    membership_logits: tuple[float, ...]
    numerical_query_supported: bool
    numerical_member_supported: tuple[bool, ...]


@dataclass(frozen=True)
class RegionCandidateBatch:
    stamp: ObservationStamp
    extrinsic: SensorToRobotExtrinsic
    candidates: tuple[RegionCandidate, ...]
    direction_tokens: tuple[DirectionCandidate, ...]
    decisions_made: bool = False


def region_candidates_from_prediction(prediction: RegionPrediction, axes_current_sensor_m: torch.Tensor,
                                      observation_stamp: ObservationStamp,
                                      robot_from_sensor: SensorToRobotExtrinsic) -> RegionCandidateBatch:
    """Convert ONE synchronized observation, preserving all 2S<=64 candidates.

Positions undergo R*p+t; directions undergo R*d only. Unsupported directions
are None, not invented unit vectors. All logits and uncertainty are preserved.
An unsupported query retains its numeric center with its support flag false;
that center is not a confirmed structure observation. Width/height remain None.
The output contains copied immutable CPU values, never writable tensor aliases.
"""
    if (not isinstance(prediction, RegionPrediction) or not isinstance(observation_stamp, ObservationStamp)
            or not isinstance(robot_from_sensor, SensorToRobotExtrinsic)):
        raise ValueError("typed prediction, stamp and calibrated extrinsic required")
    # Revalidate immutable descriptors so accidental object-level alteration
    # cannot bypass the coordinate boundary.
    observation_stamp.__post_init__()
    robot_from_sensor = SensorToRobotExtrinsic(robot_from_sensor.sensor_frame, robot_from_sensor.robot_frame,
        robot_from_sensor.transformation_robot_from_sensor, robot_from_sensor.calibration_reference)
    if (observation_stamp.sensor_frame != robot_from_sensor.sensor_frame
            or observation_stamp.robot_frame != robot_from_sensor.robot_frame):
        raise ValueError("stamp/extrinsic frame mismatch")
    if not isinstance(axes_current_sensor_m, torch.Tensor) or axes_current_sensor_m.ndim != 4 or axes_current_sensor_m.shape[0] != 1:
        raise ValueError("exactly one observation of B=1 axes required")
    tokens = tokens_from_axes(axes_current_sensor_m)
    n = tokens.positions_m.shape[1]
    shapes = {"centers_m": (1, n, 3), "event_logits": (1, n, 3),
              "presence_logits": (1, n), "membership_logits": (1, n, n), "uncertainty": (1, n)}
    for name, shape in shapes.items():
        value = getattr(prediction, name)
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != axes_current_sensor_m.dtype
                or value.device != axes_current_sensor_m.device or not torch.isfinite(value).all()):
            raise ValueError(f"finite prediction shape/dtype/device mismatch: {name}")
    for name, shape in (("query_supported", (1, n)), ("member_supported", (1, n, n))):
        value = getattr(prediction, name)
        if (not isinstance(value, torch.Tensor) or value.shape != shape or value.dtype != torch.bool
                or value.device != axes_current_sensor_m.device):
            raise ValueError("boolean prediction support shape/device mismatch")
    if (not torch.equal(prediction.query_supported, tokens.valid)
            or not torch.equal(prediction.member_supported, tokens.valid[:, :, None] & tokens.valid[:, None])):
        raise ValueError("prediction support disagrees with associated axes")
    if ((prediction.uncertainty < 0) | (prediction.uncertainty > 1)).any():
        raise ValueError("uncertainty must preserve original bounded output")
    matrix = np.asarray(robot_from_sensor.transformation_robot_from_sensor)
    rotation, translation = matrix[:3, :3], matrix[:3, 3]
    def cpu(value):
        return value.detach().cpu().numpy()
    def points(value):
        return cpu(value)[0].astype(np.float64) @ rotation.T + translation
    centers, anchors = points(prediction.centers_m), points(tokens.positions_m)
    directions = cpu(tokens.tangents)[0].astype(np.float64) @ rotation.T
    if not all(np.isfinite(x).all() for x in (centers, anchors, directions)):
        raise ValueError("nonfinite transformed coordinates")
    def vector(value):
        return tuple(float(x) for x in value)
    valid = cpu(tokens.valid)[0]
    direction_tokens = tuple(DirectionCandidate(i, vector(anchors[i]), vector(directions[i]) if valid[i] else None,
                                               bool(valid[i])) for i in range(n))
    event, presence = cpu(prediction.event_logits)[0], cpu(prediction.presence_logits)[0]
    uncertainty, membership = cpu(prediction.uncertainty)[0], cpu(prediction.membership_logits)[0]
    support = cpu(prediction.member_supported)[0]
    candidates = tuple(RegionCandidate(i, vector(centers[i]), vector(event[i]), float(presence[i]),
        float(uncertainty[i]), vector(membership[i]), bool(valid[i]), tuple(bool(x) for x in support[i])) for i in range(n))
    return RegionCandidateBatch(observation_stamp, robot_from_sensor, candidates, direction_tokens)
