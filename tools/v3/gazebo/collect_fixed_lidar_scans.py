#!/usr/bin/env python3
"""Collect three organized PointCloud2 messages from every frozen parity topic."""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import rospy
from sensor_msgs.msg import PointCloud2

sys.path.insert(0, "/workspace/src")
from mtare_topo.data.cano_gazebo_parity import decode_organized_cloud


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--sensor-count", required=True, type=int)
    parser.add_argument("--scans", default=3, type=int)
    parser.add_argument("--warmup", default=2, type=int)
    parser.add_argument("--timeout", default=120.0, type=float)
    args = parser.parse_args()
    collected = {index: [] for index in range(args.sensor_count)}
    seen = {index: 0 for index in range(args.sensor_count)}
    stamps = {index: [] for index in range(args.sensor_count)}
    rospy.init_node("mtare_fixed_lidar_parity_collector", anonymous=True, disable_signals=True)

    def callback(message: PointCloud2, index: int) -> None:
        if len(collected[index]) >= args.scans:
            return
        seen[index] += 1
        if seen[index] <= args.warmup:
            return
        ranges, valid = decode_organized_cloud(message.data, message.width, message.height, message.point_step)
        collected[index].append((ranges, valid))
        stamps[index].append(message.header.stamp.to_sec())

    subscribers = [
        rospy.Subscriber(f"/parity/lidar_{index:02d}", PointCloud2, callback, callback_args=index, queue_size=1)
        for index in range(args.sensor_count)
    ]
    deadline = time.monotonic() + args.timeout
    while not rospy.is_shutdown() and time.monotonic() < deadline:
        if all(len(value) >= args.scans for value in collected.values()):
            break
        rospy.sleep(0.02)
    for subscriber in subscribers:
        subscriber.unregister()
    counts = {str(index): len(value) for index, value in collected.items()}
    if any(count != args.scans for count in counts.values()):
        print(json.dumps({"status": "FAIL", "counts": counts, "seen": seen, "warmup_discarded_per_sensor": args.warmup}, indent=2))
        return 2
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {}
    for index in range(args.sensor_count):
        payload[f"range_{index:02d}"] = np.stack([item[0] for item in collected[index]])
        payload[f"valid_{index:02d}"] = np.stack([item[1] for item in collected[index]])
        payload[f"stamp_{index:02d}"] = np.asarray(stamps[index], dtype=np.float64)
    np.savez_compressed(args.output, **payload)
    print(json.dumps({"status": "PASS", "counts": counts, "seen": seen, "warmup_discarded_per_sensor": args.warmup, "output": str(args.output)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
