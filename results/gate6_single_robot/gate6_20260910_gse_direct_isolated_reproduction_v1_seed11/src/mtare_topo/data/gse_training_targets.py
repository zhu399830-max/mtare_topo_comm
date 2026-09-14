"""Robot-frame and circular-augmentation contracts for GSE training targets."""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np


AZIMUTH_COLUMNS = 720
AZIMUTH_RESOLUTION_DEG = 0.5


def world_vector_to_robot(vector_world_xyz: np.ndarray, *, robot_yaw_deg: float) -> np.ndarray:
    """Rotate a finite 3-D direction into the yaw-only LiDAR robot frame."""

    vector = np.asarray(vector_world_xyz, dtype=np.float64)
    if vector.shape != (3,) or not np.all(np.isfinite(vector)) or not math.isfinite(float(robot_yaw_deg)):
        raise ValueError("world vector and robot yaw must be finite")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("world vector must be nonzero")
    vector = vector / norm
    yaw = math.radians(float(robot_yaw_deg))
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    robot = np.asarray(
        (
            cosine * vector[0] + sine * vector[1],
            -sine * vector[0] + cosine * vector[1],
            vector[2],
        ),
        dtype=np.float64,
    )
    robot /= np.linalg.norm(robot)
    return robot


def heading_deg_to_unit(heading_robot_deg: float) -> np.ndarray:
    """Match the model's frozen [sin(theta), cos(theta)] heading convention."""

    if not math.isfinite(float(heading_robot_deg)):
        raise ValueError("heading must be finite")
    radians = math.radians(float(heading_robot_deg) % 360.0)
    return np.asarray((math.sin(radians), math.cos(radians)), dtype=np.float32)


def circular_roll_training_example(
    student: np.ndarray,
    targets: Mapping[str, Any],
    *,
    shift_columns: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Roll train-only azimuth and rotate every robot-frame directional target.

    A positive array roll moves a return to a larger azimuth column, so target
    headings and horizontal axis components rotate by the same positive angle.
    Validation must call this with zero shift.
    """

    scan = np.asarray(student)
    if scan.ndim != 4 or scan.shape[-2:] != (16, AZIMUTH_COLUMNS):
        raise ValueError("student sequence must be [T,C,16,720]")
    if not isinstance(shift_columns, int) or isinstance(shift_columns, bool):
        raise ValueError("shift_columns must be an integer")
    shift = shift_columns % AZIMUTH_COLUMNS
    result = dict(targets)
    axis = np.asarray(targets["local_axis"], dtype=np.float64)
    heading = np.asarray(targets["exit_heading_unit"], dtype=np.float64)
    if axis.shape != (3,) or heading.ndim != 2 or heading.shape[1] != 2:
        raise ValueError("directional targets have invalid shapes")
    angle = math.radians(shift * AZIMUTH_RESOLUTION_DEG)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    rotated_axis = axis.copy()
    rotated_axis[0] = cosine * axis[0] - sine * axis[1]
    rotated_axis[1] = sine * axis[0] + cosine * axis[1]
    # Heading units store [sin(theta), cos(theta)].
    rotated_heading = heading.copy()
    rotated_heading[:, 0] = sine * heading[:, 1] + cosine * heading[:, 0]
    rotated_heading[:, 1] = cosine * heading[:, 1] - sine * heading[:, 0]
    result["local_axis"] = rotated_axis.astype(np.float32)
    result["exit_heading_unit"] = rotated_heading.astype(np.float32)
    return np.roll(scan, shift=shift, axis=-1), result


__all__ = [
    "AZIMUTH_COLUMNS",
    "AZIMUTH_RESOLUTION_DEG",
    "circular_roll_training_example",
    "heading_deg_to_unit",
    "world_vector_to_robot",
]
