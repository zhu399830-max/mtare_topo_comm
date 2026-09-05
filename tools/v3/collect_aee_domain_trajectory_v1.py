#!/usr/bin/env python3
"""Collect one exact original-M-TARE AEE adaptation trajectory and sensor shard."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import numpy as np

from mtare_topo.data.aee_domain_adaptation import (
    EFFECTIVE_FRAMES_PER_TRAJECTORY,
    MAXIMUM_COLLECTION_SIM_SECONDS,
    MINIMUM_TRAJECTORY_DISTANCE_M,
    RAW_FRAMES_PER_TRAJECTORY,
    audit_sensor_shard,
    effective_frame_indices,
    quaternion_yaw_deg,
    sha256,
    trajectory_distance_m,
)
from mtare_topo.evaluation.closed_loop_recording import (
    TopicObservation,
    audit_topic_inventory,
    load_topic_contract,
)
from mtare_topo.integration.range_image_adapter import registered_points_to_range_image


SOURCE_ENV = """source /opt/ros/noetic/setup.bash
source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash
source /home/docker-user/mtare/tare_system/devel/setup.bash --extend
export PYTHONPATH=/workspace/src:$PYTHONPATH
"""
TOPIC_CONTRACT = Path("/workspace/configs/v3/gate5/closed_loop_recording_topics_v1.json")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def start(command: str, log_path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    stream = log_path.open("xb")
    process = subprocess.Popen(
        ["/bin/bash", "-lc", SOURCE_ENV + command],
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, stream


def stop(process: subprocess.Popen[bytes] | None, timeout: float = 30.0) -> None:
    if process is None or process.poll() is not None:
        return
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGINT)
    try:
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=10.0)


def wait_for_master(process: subprocess.Popen[bytes], timeout_sec: float = 120.0) -> None:
    deadline = time.monotonic() + timeout_sec
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"system launch exited before ROS master: {process.returncode}")
        check = subprocess.run(
            ["/bin/bash", "-lc", SOURCE_ENV + "rostopic type /clock"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if check.returncode == 0 and check.stdout.strip() == b"rosgraph_msgs/Clock":
            return
        time.sleep(0.25)
    raise TimeoutError("ROS /clock did not become ready")


def extract_sensor_shard(
    bag_path: Path,
    output_path: Path,
    start_sim_sec: float,
    trajectory_id: str,
) -> dict[str, Any]:
    import rosbag
    from sensor_msgs import point_cloud2

    scans: dict[int, Any] = {}
    odometry: dict[int, Any] = {}
    selected = set(int(value) for value in effective_frame_indices())
    raw_xyz: list[tuple[float, float, float]] = []
    arrays: dict[str, list[Any]] = {
        "range_m": [],
        "valid_mask": [],
        "sensor_xyz_m": [],
        "sensor_orientation_xyzw": [],
        "yaw_deg": [],
        "stamp_sec": [],
        "raw_frame_index": [],
        "frame_id": [],
    }
    audit_totals = {
        "input_points": 0,
        "finite_points": 0,
        "accepted_points": 0,
        "unique_valid_cells": 0,
        "duplicate_cell_returns": 0,
        "out_of_range_points": 0,
        "out_of_ring_points": 0,
    }
    raw_index = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        info = bag.get_type_and_topic_info().topics
        contract = load_topic_contract(TOPIC_CONTRACT)
        observed = {
            name: TopicObservation(type=item.msg_type, messages=int(item.message_count))
            for name, item in info.items()
        }
        topic_audit = audit_topic_inventory(contract, observed)
        if not topic_audit.passed:
            raise RuntimeError(f"topic inventory failed: {topic_audit.to_dict()}")
        for topic, message, _ in bag.read_messages(
            topics=["/registered_scan", "/state_estimation_at_scan"]
        ):
            stamp = int(message.header.stamp.to_nsec())
            if message.header.stamp.to_sec() + 1e-9 < start_sim_sec:
                continue
            if topic == "/registered_scan":
                scans[stamp] = message
            else:
                odometry[stamp] = message
            if stamp not in scans or stamp not in odometry:
                continue
            scan = scans.pop(stamp)
            odom = odometry.pop(stamp)
            position = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            xyz = (float(position.x), float(position.y), float(position.z))
            xyzw = (
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            raw_xyz.append(xyz)
            points = np.asarray(
                list(point_cloud2.read_points(scan, field_names=("x", "y", "z"), skip_nans=False)),
                dtype=np.float64,
            ).reshape(-1, 3)
            range_m, valid, conversion = registered_points_to_range_image(
                points,
                sensor_origin_world=xyz,
                sensor_orientation_xyzw=xyzw,
            )
            for key, value in conversion.to_dict().items():
                audit_totals[key] += int(value)
            if conversion.out_of_ring_points != 0:
                raise RuntimeError(f"off-ring returns at raw frame {raw_index}: {conversion.out_of_ring_points}")
            if raw_index in selected:
                arrays["range_m"].append(range_m.astype(np.float32))
                arrays["valid_mask"].append(valid.astype(np.uint8))
                arrays["sensor_xyz_m"].append(xyz)
                arrays["sensor_orientation_xyzw"].append(xyzw)
                arrays["yaw_deg"].append(quaternion_yaw_deg(xyzw))
                arrays["stamp_sec"].append(float(scan.header.stamp.to_sec()))
                arrays["raw_frame_index"].append(raw_index)
                arrays["frame_id"].append(f"{trajectory_id}:{stamp}")
            raw_index += 1
            if raw_index == RAW_FRAMES_PER_TRAJECTORY:
                break
    if raw_index != RAW_FRAMES_PER_TRAJECTORY or len(raw_xyz) != RAW_FRAMES_PER_TRAJECTORY:
        raise RuntimeError(f"exact raw frame count failed: {raw_index}")
    payload = {
        "range_m": np.stack(arrays["range_m"]).astype(np.float32),
        "valid_mask": np.stack(arrays["valid_mask"]).astype(np.uint8),
        "sensor_xyz_m": np.asarray(arrays["sensor_xyz_m"], dtype=np.float64),
        "sensor_orientation_xyzw": np.asarray(arrays["sensor_orientation_xyzw"], dtype=np.float64),
        "yaw_deg": np.asarray(arrays["yaw_deg"], dtype=np.float64),
        "stamp_sec": np.asarray(arrays["stamp_sec"], dtype=np.float64),
        "raw_frame_index": np.asarray(arrays["raw_frame_index"], dtype=np.int64),
        "frame_id": np.asarray(arrays["frame_id"], dtype="U64"),
    }
    shard_audit = audit_sensor_shard(payload)
    if not shard_audit["passed"]:
        raise RuntimeError(f"sensor shard audit failed: {shard_audit}")
    distance = trajectory_distance_m(raw_xyz)
    if distance < MINIMUM_TRAJECTORY_DISTANCE_M:
        raise RuntimeError(f"trajectory distance {distance:.6f} m below 50 m")
    np.savez_compressed(output_path, **payload)
    return {
        "raw_frames": raw_index,
        "effective_frames": EFFECTIVE_FRAMES_PER_TRAJECTORY,
        "trajectory_distance_m": distance,
        "topic_audit": topic_audit.to_dict(),
        "conversion_totals": audit_totals,
        "sensor_shard_audit": shard_audit,
        "sensor_shard_bytes": output_path.stat().st_size,
        "sensor_shard_sha256": sha256(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--world", choices=("tunnel", "garage"), required=True)
    parser.add_argument("--split", choices=("train", "validation"), required=True)
    parser.add_argument("--environment-seed", required=True, type=int)
    args = parser.parse_args()
    if args.environment_seed not in (11, 23, 37, 53, 71):
        raise ValueError("environment seed outside approved set")
    if args.split != ("train" if args.world == "tunnel" else "validation"):
        raise ValueError("world/split contract drift")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    bag_path = output / "raw.bag"
    contract = {
        "schema_version": "aee_domain_trajectory_contract_v1",
        "trajectory_id": args.trajectory_id,
        "world": args.world,
        "split": args.split,
        "environment_seed": args.environment_seed,
        "planner_seed": args.environment_seed,
        "test_id": "0001",
        "raw_frames": RAW_FRAMES_PER_TRAJECTORY,
        "effective_stride": 5,
        "effective_frames": EFFECTIVE_FRAMES_PER_TRAJECTORY,
        "minimum_distance_m": MINIMUM_TRAJECTORY_DISTANCE_M,
        "maximum_collection_sim_seconds": MAXIMUM_COLLECTION_SIM_SECONDS,
    }
    write_json(output / "contract.json", contract)
    processes: list[subprocess.Popen[bytes] | None] = [None, None, None]
    streams: list[Any] = []
    start_sim: float | None = None
    matched_count = 0
    started = time.monotonic()
    try:
        processes[0], stream = start(
            "roslaunch /workspace/configs/v3/gate5/roslaunch/system_seeded.launch "
            f"world_name:={args.world} gazebo_seed:={args.environment_seed} "
            "vehicleX:=0 vehicleY:=0 terrainZ:=0 vehicleYaw:=0 rviz:=false vis_tools:=false gazebo_gui:=false",
            logs / "system.log",
        )
        streams.append(stream)
        wait_for_master(processes[0])
        topic_names = " ".join(item.name for item in load_topic_contract(TOPIC_CONTRACT))
        processes[1], stream = start(
            f"rosbag record --lz4 --buffsize=2048 -O {bag_path} {topic_names}",
            logs / "recorder.log",
        )
        streams.append(stream)
        processes[2], stream = start(
            "roslaunch /workspace/configs/v3/gate5/roslaunch/explore_seeded.launch "
            f"scenario:={args.world} planner_seed:={args.environment_seed} rviz:=false "
            "use_boundary:=false robot_num:=1 robot_id:=0 test_id:=0001",
            logs / "planner.log",
        )
        streams.append(stream)

        import rospy
        from geometry_msgs.msg import PointStamped
        from nav_msgs.msg import Odometry
        from rosgraph_msgs.msg import Clock
        from sensor_msgs.msg import PointCloud2

        state: dict[str, Any] = {"clock": None, "waypoints": 0, "scans": set(), "odometry": set()}
        rospy.init_node(f"aee_domain_collector_{args.world}_{args.environment_seed}", anonymous=True, disable_signals=True)
        subscribers = [
            rospy.Subscriber("/clock", Clock, lambda msg: state.__setitem__("clock", msg.clock.to_sec())),
            rospy.Subscriber("/way_point", PointStamped, lambda _: state.__setitem__("waypoints", state["waypoints"] + 1)),
        ]

        def observe(message: Any, key: str) -> None:
            nonlocal matched_count
            if start_sim is None or message.header.stamp.to_sec() + 1e-9 < start_sim:
                return
            stamp = int(message.header.stamp.to_nsec())
            state[key].add(stamp)
            other = "odometry" if key == "scans" else "scans"
            if stamp in state[other]:
                state[key].discard(stamp)
                state[other].discard(stamp)
                matched_count += 1

        subscribers.extend(
            [
                rospy.Subscriber("/registered_scan", PointCloud2, lambda msg: observe(msg, "scans"), queue_size=100),
                rospy.Subscriber("/state_estimation_at_scan", Odometry, lambda msg: observe(msg, "odometry"), queue_size=100),
            ]
        )
        ready_deadline = time.monotonic() + 180.0
        while (state["clock"] is None or state["waypoints"] < 1) and time.monotonic() < ready_deadline:
            if any(process is not None and process.poll() is not None for process in processes):
                raise RuntimeError("ROS process exited before collection readiness")
            time.sleep(0.05)
        if state["clock"] is None or state["waypoints"] < 1:
            raise TimeoutError("original M-TARE did not reach collection readiness")
        start_sim = float(state["clock"])
        deadline = start_sim + MAXIMUM_COLLECTION_SIM_SECONDS
        wall_deadline = time.monotonic() + 2400.0
        while matched_count < RAW_FRAMES_PER_TRAJECTORY:
            if any(process is not None and process.poll() is not None for process in processes):
                raise RuntimeError("ROS process exited during collection")
            if state["clock"] is not None and float(state["clock"]) > deadline:
                raise RuntimeError(f"only {matched_count} synchronized frames before 660 sim seconds")
            if time.monotonic() > wall_deadline:
                raise TimeoutError("collection exceeded 2400 wall seconds")
            time.sleep(0.05)
        for subscriber in subscribers:
            subscriber.unregister()
    finally:
        stop(processes[2])
        stop(processes[1])
        stop(processes[0])
        for stream in streams:
            stream.close()
    if start_sim is None or not bag_path.is_file() or bag_path.stat().st_size == 0:
        raise RuntimeError("collection did not produce a finalized bag")
    evidence = extract_sensor_shard(
        bag_path,
        output / "sensor_shard.npz",
        start_sim,
        args.trajectory_id,
    )
    summary = {
        "schema_version": "aee_domain_trajectory_summary_v1",
        "status": "PASS_AEE_DOMAIN_TRAJECTORY_V1",
        "contract": contract,
        "start_sim_sec": start_sim,
        "live_matched_frames": matched_count,
        "raw_bag_bytes": bag_path.stat().st_size,
        "raw_bag_sha256": sha256(bag_path),
        "evidence": evidence,
        "wall_duration_sec": time.monotonic() - started,
        "training_steps": 0,
        "teacher_queries": 0,
        "c09_reads": 0,
        "c10_reads": 0,
    }
    write_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
