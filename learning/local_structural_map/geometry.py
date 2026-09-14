import math
from typing import Tuple

import numpy as np

from .schema import Pose2D


def yaw_from_quaternion(x: float, y: float, z: float, w: float) -> float:
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return math.atan2(siny_cosp, cosy_cosp)


def rotation_matrix_z(yaw: float) -> np.ndarray:
    c = math.cos(yaw)
    s = math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float32)


def transform_points_local_to_world(points_local: np.ndarray, pose: Pose2D) -> np.ndarray:
    rot = rotation_matrix_z(pose.yaw)
    trans = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
    return np.asarray(points_local, dtype=np.float32) @ rot.T + trans


def transform_points_world_to_local(points_world: np.ndarray, pose: Pose2D) -> np.ndarray:
    rot = rotation_matrix_z(pose.yaw)
    trans = np.array([pose.x, pose.y, pose.z], dtype=np.float32)
    return (np.asarray(points_world, dtype=np.float32) - trans) @ rot


def grid_indices(local_xy: np.ndarray, size_m: float, resolution_m: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    half = size_m / 2.0
    x = local_xy[:, 0]
    y = local_xy[:, 1]
    cols = np.floor((x + half) / resolution_m).astype(np.int32)
    rows = np.floor((half - y) / resolution_m).astype(np.int32)
    n = int(round(size_m / resolution_m))
    valid = (cols >= 0) & (cols < n) & (rows >= 0) & (rows < n)
    return rows, cols, valid
