#!/usr/bin/env python3
"""Read-only numerical diagnosis of two same-seed M-TARE ROS bags."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import rosbag
from sensor_msgs import point_cloud2


def scans_and_poses(path: Path, count: int):
    scans = []
    poses = []
    with rosbag.Bag(str(path), "r") as bag:
        for topic, message, _ in bag.read_messages(topics=["/registered_scan", "/state_estimation_at_scan"]):
            if topic == "/registered_scan" and len(scans) < count:
                xyz = np.asarray(
                    list(point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=False)),
                    dtype=np.float64,
                )
                scans.append(xyz)
            elif topic == "/state_estimation_at_scan" and len(poses) < count:
                p = message.pose.pose.position
                q = message.pose.pose.orientation
                poses.append(np.asarray([p.x, p.y, p.z, q.x, q.y, q.z, q.w], dtype=np.float64))
            if len(scans) >= count and len(poses) >= count:
                break
    return scans, poses


def voxel_set(points: np.ndarray, resolution: float) -> set[tuple[int, int, int]]:
    finite = points[np.all(np.isfinite(points), axis=1)]
    return set(map(tuple, np.floor(finite / resolution).astype(np.int64)))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag-a", required=True, type=Path)
    parser.add_argument("--bag-b", required=True, type=Path)
    parser.add_argument("--frames", type=int, default=100)
    args = parser.parse_args()
    scans_a, poses_a = scans_and_poses(args.bag_a, args.frames)
    scans_b, poses_b = scans_and_poses(args.bag_b, args.frames)
    count = min(len(scans_a), len(scans_b), len(poses_a), len(poses_b), args.frames)
    if count != args.frames:
        raise RuntimeError(f"insufficient aligned ordinal frames: {count}")
    equal_shapes = 0
    pointwise_mean = []
    pointwise_max = []
    jaccard = {resolution: [] for resolution in (0.01, 0.05, 0.1)}
    for left, right in zip(scans_a, scans_b):
        if left.shape == right.shape:
            equal_shapes += 1
            delta = np.linalg.norm(left - right, axis=1)
            finite = delta[np.isfinite(delta)]
            pointwise_mean.append(float(np.mean(finite)))
            pointwise_max.append(float(np.max(finite)))
        for resolution in jaccard:
            a, b = voxel_set(left, resolution), voxel_set(right, resolution)
            jaccard[resolution].append(len(a & b) / len(a | b) if a or b else 1.0)
    pose_delta = np.stack(poses_a) - np.stack(poses_b)
    position_norm = np.linalg.norm(pose_delta[:, :3], axis=1)
    quaternion_norm = np.linalg.norm(pose_delta[:, 3:], axis=1)
    result = {
        "frames": count,
        "equal_scan_shapes": equal_shapes,
        "first_scan_point_counts": [int(scans_a[0].shape[0]), int(scans_b[0].shape[0])],
        "pointwise_xyz_mean_delta_m": {
            "first": pointwise_mean[0] if pointwise_mean else None,
            "median_over_frames": float(np.median(pointwise_mean)) if pointwise_mean else None,
            "max_over_frames": max(pointwise_max) if pointwise_max else None,
        },
        "voxel_jaccard": {
            str(resolution): {
                "first": values[0],
                "mean": float(np.mean(values)),
                "minimum": min(values),
            }
            for resolution, values in jaccard.items()
        },
        "pose_position_delta_m": {
            "first": float(position_norm[0]),
            "mean": float(np.mean(position_norm)),
            "maximum": float(np.max(position_norm)),
        },
        "pose_quaternion_l2_delta": {
            "first": float(quaternion_norm[0]),
            "mean": float(np.mean(quaternion_norm)),
            "maximum": float(np.max(quaternion_norm)),
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
