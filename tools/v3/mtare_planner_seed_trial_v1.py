#!/usr/bin/env python3
"""Run one fixed-duration original-M-TARE seed qualification trial in ROS."""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import io
import json
import math
import os
from pathlib import Path
import signal
import struct
import subprocess
import time
from typing import Any


SOURCE_ENV = """source /opt/ros/noetic/setup.bash
source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash
export ROS_PACKAGE_PATH=/home/docker-user/mtare/tare_system/src:$ROS_PACKAGE_PATH
export LD_LIBRARY_PATH=/home/docker-user/mtare/tare_system/devel/lib:$LD_LIBRARY_PATH
export PATH=/home/docker-user/mtare/tare_system/devel/lib/tare_planner:$PATH
export PYTHONPATH=/workspace/src:$PYTHONPATH
"""
REPO = Path("/workspace")


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def start(command: str, log_path: Path) -> tuple[subprocess.Popen[bytes], Any]:
    stream = log_path.open("xb")
    process = subprocess.Popen(
        ["/bin/bash", "-lc", SOURCE_ENV + command],
        stdout=stream,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    return process, stream


def stop(process: subprocess.Popen[bytes] | None, timeout: float = 20.0) -> None:
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
            process.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            with suppress(ProcessLookupError):
                os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=5.0)


def wait_for_master(process: subprocess.Popen[bytes], timeout_sec: float = 90.0) -> None:
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


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def scan_identity(message: Any) -> str:
    buffer = io.BytesIO()
    buffer.write(struct.pack("<IIII?", message.height, message.width, message.point_step, message.row_step, message.is_dense))
    for field in message.fields:
        name = field.name.encode("utf-8")
        buffer.write(struct.pack("<I", len(name)))
        buffer.write(name)
        buffer.write(struct.pack("<III", field.offset, field.datatype, field.count))
    buffer.write(bytes(message.data))
    return sha256_bytes(buffer.getvalue())


def pose_identity(message: Any) -> str:
    p = message.pose.pose.position
    q = message.pose.pose.orientation
    values = (p.x, p.y, p.z, q.x, q.y, q.z, q.w)
    if not all(math.isfinite(value) for value in values):
        raise RuntimeError("non-finite odometry pose")
    return sha256_bytes(struct.pack("<7d", *values))


def summarize_bag(bag_path: Path, audit_frames: int, audit_waypoints: int) -> dict[str, Any]:
    import rosbag
    from mtare_topo.evaluation.closed_loop_recording import (
        TopicObservation,
        audit_topic_inventory,
        load_topic_contract,
    )

    scans: dict[int, str] = {}
    poses: dict[int, str] = {}
    waypoint_values: list[list[float]] = []
    topic_counts: dict[str, int] = {}
    topic_types: dict[str, str] = {}
    with rosbag.Bag(str(bag_path), "r") as bag:
        info = bag.get_type_and_topic_info().topics
        for name, value in info.items():
            topic_counts[name] = int(value.message_count)
            topic_types[name] = str(value.msg_type)
        for topic, message, _ in bag.read_messages(
            topics=["/registered_scan", "/state_estimation_at_scan", "/way_point"]
        ):
            if topic == "/registered_scan":
                stamp = int(message.header.stamp.to_nsec())
                if stamp in scans:
                    raise RuntimeError("duplicate registered-scan timestamp")
                scans[stamp] = scan_identity(message)
            elif topic == "/state_estimation_at_scan":
                stamp = int(message.header.stamp.to_nsec())
                if stamp in poses:
                    raise RuntimeError("duplicate scan-odometry timestamp")
                poses[stamp] = pose_identity(message)
            elif len(waypoint_values) < audit_waypoints:
                xyz = [float(message.point.x), float(message.point.y), float(message.point.z)]
                if not all(math.isfinite(value) for value in xyz):
                    raise RuntimeError("non-finite waypoint")
                waypoint_values.append(xyz)
    common = sorted(set(scans) & set(poses))
    if len(common) < audit_frames:
        raise RuntimeError(f"only {len(common)} synchronized frames; need {audit_frames}")
    if len(waypoint_values) < audit_waypoints:
        raise RuntimeError(f"only {len(waypoint_values)} waypoints; need {audit_waypoints}")
    selected = common[:audit_frames]
    origin = selected[0]
    rows = [
        {
            "relative_stamp_ns": stamp - origin,
            "scan_sha256": scans[stamp],
            "pose_sha256": poses[stamp],
        }
        for stamp in selected
    ]
    contract = load_topic_contract(REPO / "configs/v3/gate5/closed_loop_recording_topics_v1.json")
    recording_audit = audit_topic_inventory(
        contract,
        {
            name: TopicObservation(type=topic_types[name], messages=messages)
            for name, messages in topic_counts.items()
        },
    )
    if not recording_audit.passed:
        raise RuntimeError(f"closed-loop recording contract failed: {recording_audit.to_dict()}")
    return {
        "schema_version": "mtare_planner_seed_trial_summary_v1",
        "bag_sha256": hashlib.sha256(bag_path.read_bytes()).hexdigest(),
        "bag_size_bytes": bag_path.stat().st_size,
        "topic_counts": topic_counts,
        "topic_types": topic_types,
        "recording_audit": recording_audit.to_dict(),
        "synchronized_frame_count": len(common),
        "audit_frames": rows,
        "audit_waypoints_xyz_m": waypoint_values,
        "audit_frame_sequence_sha256": sha256_bytes(
            json.dumps(rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ),
        "audit_waypoint_sequence_sha256": sha256_bytes(
            json.dumps(waypoint_values, separators=(",", ":")).encode("utf-8")
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial-dir", required=True, type=Path)
    parser.add_argument("--seed", required=True, type=int)
    parser.add_argument("--duration-sec", type=float, default=60.0)
    parser.add_argument("--audit-frames", type=int, default=100)
    parser.add_argument("--audit-waypoints", type=int, default=20)
    args = parser.parse_args()
    if args.seed < 0 or args.duration_sec <= 0 or args.audit_frames <= 0 or args.audit_waypoints <= 0:
        raise ValueError("seed/counts/duration are outside the qualification contract")
    trial = args.trial_dir.resolve()
    trial.mkdir(parents=True, exist_ok=False)
    (trial / "logs").mkdir()
    bag_path = trial / "trial.bag"
    system: subprocess.Popen[bytes] | None = None
    planner: subprocess.Popen[bytes] | None = None
    recorder: subprocess.Popen[bytes] | None = None
    streams: list[Any] = []
    started_wall = time.monotonic()
    try:
        system, stream = start(
            "roslaunch /workspace/configs/v3/gate5/roslaunch/system_seeded.launch "
            f"world_name:=tunnel gazebo_seed:={args.seed} vehicleX:=0 vehicleY:=0 terrainZ:=0 vehicleYaw:=0 "
            "rviz:=false vis_tools:=false gazebo_gui:=false",
            trial / "logs/system.log",
        )
        streams.append(stream)
        wait_for_master(system)
        topics = json.loads((REPO / "configs/v3/gate5/closed_loop_recording_topics_v1.json").read_text())["topic_contract"]
        names = " ".join(item["name"] for item in topics)
        recorder, stream = start(
            f"rosbag record --lz4 --buffsize=2048 -O {bag_path} {names}",
            trial / "logs/recorder.log",
        )
        streams.append(stream)
        planner, stream = start(
            "roslaunch /workspace/configs/v3/gate5/roslaunch/explore_seeded.launch "
            f"scenario:=tunnel planner_seed:={args.seed} rviz:=false use_boundary:=false robot_num:=1 robot_id:=0 test_id:={args.seed}",
            trial / "logs/planner.log",
        )
        streams.append(stream)

        import rospy
        from rosgraph_msgs.msg import Clock

        latest = {"value": None}
        rospy.init_node(f"mtare_seed_trial_{args.seed}", anonymous=True, disable_signals=True)
        subscriber = rospy.Subscriber("/clock", Clock, lambda message: latest.__setitem__("value", message.clock.to_sec()))
        deadline = time.monotonic() + max(120.0, args.duration_sec * 3.0)
        while latest["value"] is None and time.monotonic() < deadline:
            time.sleep(0.05)
        if latest["value"] is None:
            raise TimeoutError("no simulation clock sample after planner start")
        start_sim = float(latest["value"])
        while float(latest["value"]) - start_sim < args.duration_sec:
            for name, process in (("system", system), ("planner", planner), ("recorder", recorder)):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited early with {process.returncode}")
            if time.monotonic() >= deadline:
                raise TimeoutError("simulation did not reach the fixed duration")
            time.sleep(0.05)
        subscriber.unregister()
    finally:
        stop(planner)
        stop(recorder)
        stop(system)
        for stream in streams:
            stream.close()
    if not bag_path.is_file() or bag_path.stat().st_size == 0:
        raise RuntimeError("rosbag was not finalized")
    summary = summarize_bag(bag_path, args.audit_frames, args.audit_waypoints)
    summary.update(
        {
            "seed": args.seed,
            "requested_sim_duration_sec": args.duration_sec,
            "wall_duration_sec": time.monotonic() - started_wall,
        }
    )
    write_json(trial / "summary.json", summary)
    print(json.dumps({key: summary[key] for key in ("seed", "synchronized_frame_count", "audit_frame_sequence_sha256", "audit_waypoint_sequence_sha256")}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
