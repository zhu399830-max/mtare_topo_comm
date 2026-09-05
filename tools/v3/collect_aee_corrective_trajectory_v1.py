#!/usr/bin/env python3
"""Collect one frozen AEE trajectory for the 11,000-frame corrective set."""

from __future__ import annotations

import argparse
from contextlib import suppress
import json
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

import numpy as np

from mtare_topo.data.aee_corrective_dataset import (
    AEE_FRAMES_PER_TRAJECTORY,
    aee_corrective_frame_indices,
    audit_corrective_sensor_shard,
    enumerate_aee_corrective_trajectories,
    monotonic_nearest_stamp_pairs,
)
from mtare_topo.data.aee_domain_adaptation import (
    MAXIMUM_COLLECTION_SIM_SECONDS,
    MINIMUM_TRAJECTORY_DISTANCE_M,
    RAW_FRAMES_PER_TRAJECTORY,
    quaternion_yaw_deg,
    sha256,
    trajectory_distance_m,
)
from mtare_topo.data.aee_sensor_operator_parity import (
    organized_aee_gazebo_points_to_ranges,
    resample_aee_gazebo_organized_azimuth,
)
from mtare_topo.evaluation.closed_loop_recording import (
    TopicObservation,
    audit_topic_inventory,
    load_topic_contract,
)


SOURCE_ENV = """source /opt/ros/noetic/setup.bash
source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash
source /home/docker-user/mtare/tare_system/devel/setup.bash --extend
export PYTHONPATH=/workspace/src:$PYTHONPATH
"""
TOPIC_CONTRACT = Path("/workspace/configs/v3/gate5/closed_loop_recording_topics_v1.json")
RAW_TOPIC = "/velodyne_points"
REGISTERED_TOPIC = "/registered_scan"
ODOMETRY_TOPIC = "/state_estimation_at_scan"
MAXIMUM_RAW_REGISTERED_STAMP_DELTA_SEC = 0.1


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


def validate_raw_message_layout(message: Any) -> None:
    fields = {field.name: field for field in message.fields}
    if int(message.width) != 16 or int(message.height) != 350:
        raise RuntimeError(f"raw organized layout drift: width={message.width}, height={message.height}")
    if int(message.point_step) != 22 or int(message.row_step) != 352:
        raise RuntimeError("raw organized PointCloud2 stride drift")
    required = {
        "x": (0, 7),
        "y": (4, 7),
        "z": (8, 7),
        "ring": (16, 4),
    }
    observed = {
        key: (int(fields[key].offset), int(fields[key].datatype))
        for key in required
        if key in fields
    }
    if observed != required or bool(message.is_bigendian) or bool(message.is_dense):
        raise RuntimeError(f"raw organized PointCloud2 field contract drift: {observed}")
    if len(message.data) != 350 * 16 * 22:
        raise RuntimeError("raw organized PointCloud2 byte count drift")


def decode_raw_message(message: Any, point_cloud2: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    validate_raw_message_layout(message)
    records = list(
        point_cloud2.read_points(
            message,
            field_names=("x", "y", "z", "ring"),
            skip_nans=False,
        )
    )
    if len(records) != 350 * 16:
        raise RuntimeError("raw organized PointCloud2 record count drift")
    points = np.asarray([row[:3] for row in records], dtype=np.float64).reshape(350, 16, 3)
    rings = np.asarray([row[3] for row in records], dtype=np.int64).reshape(350, 16)
    raw_range, raw_valid, source_range, source_valid, source_audit = organized_aee_gazebo_points_to_ranges(
        points, rings
    )
    model_range, model_valid = resample_aee_gazebo_organized_azimuth(source_range, source_valid)
    return raw_range, raw_valid, model_range, model_valid, source_audit.to_dict()


def extract_sensor_shard(
    bag_path: Path,
    output_path: Path,
    start_sim_sec: float,
    trajectory_id: str,
) -> dict[str, Any]:
    import rosbag
    from sensor_msgs import point_cloud2

    selected = set(int(value) for value in aee_corrective_frame_indices())
    raw_candidate_stamps: list[int] = []
    registered_candidate_stamps: list[int] = []
    odometry: dict[int, Any] = {}
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
        raw_info = info.get(RAW_TOPIC)
        if raw_info is None or raw_info.msg_type != "sensor_msgs/PointCloud2":
            raise RuntimeError("raw organized /velodyne_points topic missing or wrong type")
        for topic, message, _ in bag.read_messages(topics=[RAW_TOPIC, REGISTERED_TOPIC, ODOMETRY_TOPIC]):
            stamp_sec = float(message.header.stamp.to_sec())
            if stamp_sec + 1e-9 < start_sim_sec:
                continue
            stamp_ns = int(message.header.stamp.to_nsec())
            if topic == RAW_TOPIC:
                raw_candidate_stamps.append(stamp_ns)
            elif topic == REGISTERED_TOPIC:
                registered_candidate_stamps.append(stamp_ns)
            elif topic == ODOMETRY_TOPIC and stamp_ns not in odometry:
                odometry[stamp_ns] = message
    registered_with_odometry = [stamp for stamp in registered_candidate_stamps if stamp in odometry]
    pairs = monotonic_nearest_stamp_pairs(
        raw_candidate_stamps,
        registered_with_odometry,
        maximum_delta_ns=int(MAXIMUM_RAW_REGISTERED_STAMP_DELTA_SEC * 1e9),
        required_pairs=RAW_FRAMES_PER_TRAJECTORY,
    )
    raw_stamps = [item.raw_stamp_ns for item in pairs]
    registered_stamps = [item.registered_stamp_ns for item in pairs]
    deltas = np.asarray([item.absolute_delta_ns / 1e9 for item in pairs], dtype=np.float64)

    arrays: dict[str, list[Any]] = {
        "raw_range_m": [],
        "raw_valid_mask": [],
        "range_m": [],
        "valid_mask": [],
        "sensor_xyz_m": [],
        "sensor_orientation_xyzw": [],
        "yaw_deg": [],
        "raw_stamp_sec": [],
        "registered_stamp_sec": [],
        "raw_frame_index": [],
        "source_raw_message_index": [],
        "source_registered_message_index": [],
        "pair_delta_sec": [],
        "frame_id": [],
    }
    source_totals: dict[str, int] = {}
    selected_pairs = [pairs[index] for index in sorted(selected)]
    selected_row = 0
    with rosbag.Bag(str(bag_path), "r") as bag:
        for _, message, _ in bag.read_messages(topics=[RAW_TOPIC]):
            stamp_ns = int(message.header.stamp.to_nsec())
            if selected_row >= len(selected_pairs) or stamp_ns != selected_pairs[selected_row].raw_stamp_ns:
                continue
            pair = selected_pairs[selected_row]
            raw_range, raw_valid, model_range, model_valid, source_audit = decode_raw_message(
                message, point_cloud2
            )
            for key, value in source_audit.items():
                source_totals[key] = source_totals.get(key, 0) + int(value)
            odom = odometry[pair.registered_stamp_ns]
            position = odom.pose.pose.position
            orientation = odom.pose.pose.orientation
            xyz = (float(position.x), float(position.y), float(position.z))
            xyzw = (
                float(orientation.x),
                float(orientation.y),
                float(orientation.z),
                float(orientation.w),
            )
            arrays["raw_range_m"].append(raw_range)
            arrays["raw_valid_mask"].append(raw_valid)
            arrays["range_m"].append(model_range)
            arrays["valid_mask"].append(model_valid)
            arrays["sensor_xyz_m"].append(xyz)
            arrays["sensor_orientation_xyzw"].append(xyzw)
            arrays["yaw_deg"].append(quaternion_yaw_deg(xyzw))
            arrays["raw_stamp_sec"].append(stamp_ns / 1e9)
            arrays["registered_stamp_sec"].append(pair.registered_stamp_ns / 1e9)
            arrays["raw_frame_index"].append(pair.pair_index)
            arrays["source_raw_message_index"].append(pair.raw_message_index)
            arrays["source_registered_message_index"].append(pair.registered_message_index)
            arrays["pair_delta_sec"].append(pair.absolute_delta_ns / 1e9)
            arrays["frame_id"].append(f"{trajectory_id}:p{pair.pair_index:04d}:{stamp_ns}")
            selected_row += 1
            if selected_row == AEE_FRAMES_PER_TRAJECTORY:
                break
    if selected_row != AEE_FRAMES_PER_TRAJECTORY:
        raise RuntimeError("selected paired raw scans were not recovered exactly in the second pass")
    payload = {
        "raw_range_m": np.stack(arrays["raw_range_m"]).astype(np.float32),
        "raw_valid_mask": np.stack(arrays["raw_valid_mask"]).astype(np.uint8),
        "range_m": np.stack(arrays["range_m"]).astype(np.float32),
        "valid_mask": np.stack(arrays["valid_mask"]).astype(np.uint8),
        "sensor_xyz_m": np.asarray(arrays["sensor_xyz_m"], dtype=np.float64),
        "sensor_orientation_xyzw": np.asarray(arrays["sensor_orientation_xyzw"], dtype=np.float64),
        "yaw_deg": np.asarray(arrays["yaw_deg"], dtype=np.float64),
        "raw_stamp_sec": np.asarray(arrays["raw_stamp_sec"], dtype=np.float64),
        "registered_stamp_sec": np.asarray(arrays["registered_stamp_sec"], dtype=np.float64),
        "raw_frame_index": np.asarray(arrays["raw_frame_index"], dtype=np.int64),
        "source_raw_message_index": np.asarray(arrays["source_raw_message_index"], dtype=np.int64),
        "source_registered_message_index": np.asarray(arrays["source_registered_message_index"], dtype=np.int64),
        "pair_delta_sec": np.asarray(arrays["pair_delta_sec"], dtype=np.float64),
        "frame_id": np.asarray(arrays["frame_id"], dtype="U96"),
    }
    shard_audit = audit_corrective_sensor_shard(payload, AEE_FRAMES_PER_TRAJECTORY)
    index_exact = np.array_equal(payload["raw_frame_index"], aee_corrective_frame_indices())
    if not shard_audit["passed"] or not index_exact:
        raise RuntimeError(f"corrective sensor shard audit failed: {shard_audit}, index_exact={index_exact}")
    distance = trajectory_distance_m(
        [
            (
                float(odometry[stamp].pose.pose.position.x),
                float(odometry[stamp].pose.pose.position.y),
                float(odometry[stamp].pose.pose.position.z),
            )
            for stamp in registered_stamps
        ]
    )
    if distance < MINIMUM_TRAJECTORY_DISTANCE_M:
        raise RuntimeError(f"trajectory distance {distance:.6f} m below 50 m")
    np.savez_compressed(output_path, **payload)
    return {
        "raw_frames": RAW_FRAMES_PER_TRAJECTORY,
        "effective_frames": AEE_FRAMES_PER_TRAJECTORY,
        "trajectory_distance_m": distance,
        "maximum_raw_registered_stamp_delta_sec": float(np.max(deltas)),
        "raw_candidate_frames": len(raw_candidate_stamps),
        "registered_candidate_frames": len(registered_candidate_stamps),
        "registered_frames_with_exact_odometry": len(registered_with_odometry),
        "pairing_method": "strict_monotonic_nearest_timestamp_v1",
        "topic_audit": topic_audit.to_dict(),
        "organized_source_totals": source_totals,
        "sensor_shard_audit": shard_audit,
        "sensor_shard_bytes": output_path.stat().st_size,
        "sensor_shard_sha256": sha256(output_path),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--world", choices=("tunnel", "garage"), required=True)
    parser.add_argument("--split", choices=("corrective_train",), required=True)
    parser.add_argument("--environment-seed", required=True, type=int)
    args = parser.parse_args()
    expected = {
        item.trajectory_id: item for item in enumerate_aee_corrective_trajectories()
    }.get(args.trajectory_id)
    if expected is None or (args.world, args.split, args.environment_seed) != (
        expected.world,
        expected.split,
        expected.environment_seed,
    ):
        raise ValueError("AEE corrective trajectory identity drift")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=False)
    logs = output / "logs"
    logs.mkdir()
    bag_path = output / "raw.bag"
    contract = {
        "schema_version": "aee_corrective_trajectory_contract_v1",
        **expected.to_dict(),
        "planner_seed": args.environment_seed,
        "test_id": "0001",
        "raw_frames": RAW_FRAMES_PER_TRAJECTORY,
        "effective_frames": AEE_FRAMES_PER_TRAJECTORY,
        "effective_raw_indices": aee_corrective_frame_indices().tolist(),
        "pairing_method": "strict_monotonic_nearest_timestamp_v1",
        "raw_topic": RAW_TOPIC,
        "registered_topic": REGISTERED_TOPIC,
        "odometry_topic": ODOMETRY_TOPIC,
        "maximum_raw_registered_stamp_delta_sec": MAXIMUM_RAW_REGISTERED_STAMP_DELTA_SEC,
        "minimum_distance_m": MINIMUM_TRAJECTORY_DISTANCE_M,
        "maximum_collection_sim_seconds": MAXIMUM_COLLECTION_SIM_SECONDS,
    }
    write_json(output / "contract.json", contract)
    processes: list[subprocess.Popen[bytes] | None] = [None, None, None]
    streams: list[Any] = []
    start_sim: float | None = None
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
            f"rosbag record --lz4 --buffsize=2048 -O {bag_path} {topic_names} {RAW_TOPIC}",
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

        state: dict[str, Any] = {
            "clock": None,
            "waypoints": 0,
            "raw": 0,
            "registered": set(),
            "odometry": set(),
            "matched": 0,
        }
        rospy.init_node(f"aee_corrective_collector_{args.world}_{args.environment_seed}", anonymous=True, disable_signals=True)
        subscribers = [
            rospy.Subscriber("/clock", Clock, lambda msg: state.__setitem__("clock", msg.clock.to_sec())),
            rospy.Subscriber("/way_point", PointStamped, lambda _: state.__setitem__("waypoints", state["waypoints"] + 1)),
        ]

        def observe_raw(message: Any) -> None:
            if start_sim is not None and message.header.stamp.to_sec() + 1e-9 >= start_sim:
                state["raw"] += 1

        def observe_pair(message: Any, key: str) -> None:
            if start_sim is None or message.header.stamp.to_sec() + 1e-9 < start_sim:
                return
            stamp = int(message.header.stamp.to_nsec())
            state[key].add(stamp)
            other = "odometry" if key == "registered" else "registered"
            if stamp in state[other]:
                state[key].discard(stamp)
                state[other].discard(stamp)
                state["matched"] += 1

        subscribers.extend(
            [
                rospy.Subscriber(RAW_TOPIC, PointCloud2, observe_raw, queue_size=100),
                rospy.Subscriber(REGISTERED_TOPIC, PointCloud2, lambda msg: observe_pair(msg, "registered"), queue_size=100),
                rospy.Subscriber(ODOMETRY_TOPIC, Odometry, lambda msg: observe_pair(msg, "odometry"), queue_size=100),
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
        while state["raw"] < RAW_FRAMES_PER_TRAJECTORY or state["matched"] < RAW_FRAMES_PER_TRAJECTORY:
            if any(process is not None and process.poll() is not None for process in processes):
                raise RuntimeError("ROS process exited during collection")
            if state["clock"] is not None and float(state["clock"]) > deadline:
                raise RuntimeError(f"insufficient raw/matched frames before {MAXIMUM_COLLECTION_SIM_SECONDS} sim seconds")
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
    evidence = extract_sensor_shard(bag_path, output / "sensor_shard.npz", start_sim, args.trajectory_id)
    summary = {
        "schema_version": "aee_corrective_trajectory_summary_v1",
        "status": "PASS_AEE_CORRECTIVE_TRAJECTORY_V1",
        "contract": contract,
        "start_sim_sec": start_sim,
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
