#!/usr/bin/env python3
"""Summarize one closed-loop bag under the method-independent Gate-5 contract."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import statistics
from typing import Any

import numpy as np

from mtare_topo.evaluation.closed_loop_recording import (
    TopicObservation,
    audit_topic_inventory,
    load_topic_contract,
)
from mtare_topo.evaluation.mtare_coverage import MTAReCoverageAccumulator


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pose(message: Any) -> tuple[list[float], list[float]]:
    p = message.pose.pose.position
    q = message.pose.pose.orientation
    xyz = [float(p.x), float(p.y), float(p.z)]
    xyzw = [float(q.x), float(q.y), float(q.z), float(q.w)]
    if not all(math.isfinite(value) for value in xyz + xyzw):
        raise RuntimeError("non-finite odometry")
    return xyz, xyzw


def yaw_rad(xyzw: list[float]) -> float:
    x, y, z, w = xyzw
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def summarize(
    *,
    bag_path: Path,
    output_dir: Path,
    topic_contract_path: Path,
    start_sim_sec: float,
    end_sim_sec: float,
) -> dict[str, Any]:
    import rosbag
    from sensor_msgs import point_cloud2

    if not bag_path.is_file() or end_sim_sec <= start_sim_sec:
        raise ValueError("bag and positive simulation interval are required")
    output_dir.mkdir(parents=True, exist_ok=False)
    contract = load_topic_contract(topic_contract_path)
    accumulator = MTAReCoverageAccumulator()
    scans: dict[int, Any] = {}
    poses: dict[int, Any] = {}
    coverage_rows: list[dict[str, Any]] = []
    trajectory_rows: list[dict[str, Any]] = []
    waypoint_rows: list[dict[str, Any]] = []
    runtime_values: list[float] = []
    finish_values: list[bool] = []
    input_scan_points = 0

    def consume(stamp_ns: int) -> None:
        nonlocal input_scan_points
        scan = scans.pop(stamp_ns)
        odom = poses.pop(stamp_ns)
        stamp_sec = stamp_ns / 1e9
        if stamp_sec < start_sim_sec or stamp_sec > end_sim_sec:
            return
        xyz, xyzw = pose(odom)
        values = np.asarray(
            list(point_cloud2.read_points(scan, field_names=("x", "y", "z"), skip_nans=True)),
            dtype=np.float64,
        ).reshape(-1, 3)
        if len(values) < 20 or not np.all(np.isfinite(values)):
            raise RuntimeError("registered scan is empty or non-finite after filtering")
        cycle = accumulator.update(
            stamp_sec=stamp_sec,
            registered_scan_xyz_m=values,
            vehicle_xyz_m=xyz,
            vehicle_yaw_rad=yaw_rad(xyzw),
        )
        input_scan_points += len(values)
        elapsed = stamp_sec - start_sim_sec
        coverage_rows.append({**cycle.to_dict(), "elapsed_sec": elapsed})
        trajectory_rows.append({"stamp_sec": stamp_sec, "elapsed_sec": elapsed, "xyz_m": xyz, "orientation_xyzw": xyzw})

    with rosbag.Bag(str(bag_path), "r") as bag:
        info = bag.get_type_and_topic_info().topics
        observed = {
            name: TopicObservation(type=str(item.msg_type), messages=int(item.message_count))
            for name, item in info.items()
        }
        recording_audit = audit_topic_inventory(contract, observed)
        if not recording_audit.passed:
            raise RuntimeError(f"recording contract failed: {recording_audit.to_dict()}")
        topics = [
            "/registered_scan", "/state_estimation_at_scan", "/way_point", "/runtime",
            "/sensor_coverage_planner/exploration_finish",
        ]
        for topic, message, bag_stamp in bag.read_messages(topics=topics):
            if topic == "/registered_scan":
                stamp_ns = int(message.header.stamp.to_nsec())
                if stamp_ns in scans:
                    raise RuntimeError("duplicate scan timestamp")
                scans[stamp_ns] = message
                if stamp_ns in poses:
                    consume(stamp_ns)
            elif topic == "/state_estimation_at_scan":
                stamp_ns = int(message.header.stamp.to_nsec())
                if stamp_ns in poses:
                    raise RuntimeError("duplicate odometry timestamp")
                poses[stamp_ns] = message
                if stamp_ns in scans:
                    consume(stamp_ns)
            elif topic == "/way_point":
                stamp_sec = float(message.header.stamp.to_sec())
                if start_sim_sec <= stamp_sec <= end_sim_sec:
                    xyz = [float(message.point.x), float(message.point.y), float(message.point.z)]
                    if not all(math.isfinite(value) for value in xyz):
                        raise RuntimeError("non-finite waypoint")
                    waypoint_rows.append({"stamp_sec": stamp_sec, "elapsed_sec": stamp_sec - start_sim_sec, "xyz_m": xyz})
            elif start_sim_sec <= float(bag_stamp.to_sec()) <= end_sim_sec:
                if topic == "/runtime":
                    value = float(message.data)
                    if not math.isfinite(value) or value < 0.0:
                        raise RuntimeError("invalid planner runtime sample")
                    runtime_values.append(value)
                else:
                    finish_values.append(bool(message.data))
    if len(coverage_rows) < 100 or len(waypoint_rows) < 20 or not runtime_values or not finish_values:
        raise RuntimeError(
            f"insufficient case evidence: coverage={len(coverage_rows)} waypoints={len(waypoint_rows)} "
            f"runtime={len(runtime_values)} finish={len(finish_values)}"
        )
    duration = end_sim_sec - start_sim_sec
    times = np.asarray([0.0] + [row["elapsed_sec"] for row in coverage_rows] + [duration], dtype=np.float64)
    volumes = np.asarray(
        [0.0] + [row["explored_volume_m3"] for row in coverage_rows] + [coverage_rows[-1]["explored_volume_m3"]],
        dtype=np.float64,
    )
    if np.any(np.diff(times) < 0.0):
        raise RuntimeError("coverage time is not monotonic")
    auc = float(np.trapz(volumes, times))
    native = accumulator.summary(budget_sec=max(duration, accumulator.cycles[-1].elapsed_sec))
    runtime_sorted = sorted(runtime_values)
    p95_index = min(len(runtime_sorted) - 1, math.ceil(0.95 * len(runtime_sorted)) - 1)
    metrics = {
        "schema_version": "mtare_closed_loop_case_metrics_v1",
        "start_sim_sec": start_sim_sec,
        "end_sim_sec": end_sim_sec,
        "duration_sec": duration,
        "synchronized_frame_count": len(coverage_rows),
        "input_scan_points": input_scan_points,
        "waypoint_count": len(waypoint_rows),
        "runtime_sample_count": len(runtime_values),
        "finish_sample_count": len(finish_values),
        "finish_true_count": sum(finish_values),
        "final_explored_voxels": native["final_explored_voxels"],
        "final_explored_volume_m3": native["final_explored_volume_m3"],
        "coverage_time_auc_m3_s": auc,
        "mean_explored_volume_m3": auc / duration,
        "traveling_distance_m": native["traveling_distance_m"],
        "final_pose_xyz_m": trajectory_rows[-1]["xyz_m"],
        "final_waypoint_xyz_m": waypoint_rows[-1]["xyz_m"],
        "waypoint_path_length_m": float(
            sum(
                math.dist(left["xyz_m"], right["xyz_m"])
                for left, right in zip(waypoint_rows[:-1], waypoint_rows[1:])
            )
        ),
        "final_volume_per_travel_meter_m2": native["final_explored_volume_m3"] / max(native["traveling_distance_m"], 1e-12),
        "cumulative_point_redundancy": native["cumulative_point_redundancy"],
        "planner_runtime_mean_sec": statistics.fmean(runtime_values),
        "planner_runtime_p95_sec": runtime_sorted[p95_index],
        "planner_runtime_max_sec": max(runtime_values),
        "recording_audit": recording_audit.to_dict(),
    }
    for name, rows in (("coverage_curve.jsonl", coverage_rows), ("trajectory.jsonl", trajectory_rows), ("waypoints.jsonl", waypoint_rows)):
        with (output_dir / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
    write_json(output_dir / "metrics.json", metrics)
    write_json(
        output_dir / "exit_state.json",
        {"schema_version": "mtare_closed_loop_exit_state_v1", "finish_values": finish_values, "final_finish": finish_values[-1]},
    )
    return metrics


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bag", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--topic-contract", required=True, type=Path)
    parser.add_argument("--start-sim-sec", required=True, type=float)
    parser.add_argument("--end-sim-sec", required=True, type=float)
    args = parser.parse_args()
    result = summarize(
        bag_path=args.bag.resolve(), output_dir=args.output_dir.resolve(),
        topic_contract_path=args.topic_contract.resolve(), start_sim_sec=args.start_sim_sec,
        end_sim_sec=args.end_sim_sec,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
