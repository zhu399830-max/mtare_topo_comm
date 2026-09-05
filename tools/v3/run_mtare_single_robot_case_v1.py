#!/usr/bin/env python3
"""Run, summarize, losslessly compress, and seal one stochastic single-robot case."""

from __future__ import annotations

import argparse
from contextlib import suppress
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import time
from typing import Any

from summarize_mtare_closed_loop_bag_v1 import summarize, write_json


SOURCE_ENV = """source /opt/ros/noetic/setup.bash
source /home/docker-user/mtare/autonomous_exploration_development_environment/devel/setup.bash
source /home/docker-user/mtare/tare_system/devel/setup.bash --extend
export PYTHONPATH=/workspace/src:$PYTHONPATH
"""
DEFAULT_TOPIC_CONTRACT = Path("/workspace/configs/v3/gate5/closed_loop_recording_topics_v1.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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


def method_command(args: argparse.Namespace, planner_output: Path) -> str:
    if args.method_family == "original_mtare":
        return (
            "roslaunch /workspace/configs/v3/gate5/roslaunch/explore_seeded.launch "
            f"scenario:={args.world} planner_seed:={args.environment_seed} rviz:=false "
            "use_boundary:=false robot_num:=1 robot_id:=0 test_id:=0001"
        )
    if args.method_family == "m1d_topology":
        if not args.checkpoint or not args.checkpoint_sha256:
            raise ValueError("M1D case requires checkpoint identity")
        return (
            "python3 /workspace/tools/v3/semantic_topology_global_node_v3.py "
            f"--checkpoint /workspace/{args.checkpoint} --checkpoint-sha256 {args.checkpoint_sha256} "
            f"--output {planner_output} --publish-period-sec 1.0"
        )
    if args.method_family == "layered_gt_map_oracle":
        if not args.complete_map or not args.complete_map_sha256:
            raise ValueError("Oracle case requires complete-map identity")
        return (
            "python3 /workspace/tools/v3/layered_gt_map_global_node_v1.py "
            f"--complete-map {args.complete_map} --complete-map-sha256 {args.complete_map_sha256} "
            f"--output {planner_output} --publish-period-sec 1.0"
        )
    raise ValueError(f"unknown method family: {args.method_family}")


def verify_zstd_archive(archive: Path, expected_sha256: str) -> str:
    digest = hashlib.sha256()
    process = subprocess.Popen(["zstd", "-dc", str(archive)], stdout=subprocess.PIPE)
    assert process.stdout is not None
    for block in iter(lambda: process.stdout.read(1024 * 1024), b""):
        digest.update(block)
    returncode = process.wait()
    if returncode != 0 or digest.hexdigest() != expected_sha256:
        raise RuntimeError("zstd decompression hash verification failed")
    return digest.hexdigest()


def handoff_case_tree(case_dir: Path, host_uid: int, host_gid: int) -> dict[str, Any]:
    root = case_dir.resolve()
    try:
        root.relative_to(Path("/evidence/cases"))
    except ValueError as exc:
        raise RuntimeError("case ownership handoff escapes /evidence/cases") from exc
    if host_uid <= 0 or host_gid <= 0:
        raise ValueError("host archive requires positive non-root UID/GID")
    paths = [root, *sorted(root.rglob("*"))]
    if any(path.is_symlink() for path in paths):
        raise RuntimeError("case ownership handoff rejects symbolic links")
    record = {
        "schema_version": "mtare_case_host_ownership_handoff_v1",
        "status": "PASS_MTARE_CASE_HOST_OWNERSHIP_HANDOFF_V1",
        "host_uid": host_uid,
        "host_gid": host_gid,
        "entries": len(paths) + 1,
    }
    write_json(root / "host_ownership_handoff.json", record)
    for path in reversed([root, *sorted(root.rglob("*"))]):
        os.chown(path, host_uid, host_gid, follow_symlinks=False)
        os.chmod(path, 0o775 if path.is_dir() else 0o664)
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--world", choices=("tunnel", "garage"), required=True)
    parser.add_argument("--environment-seed", type=int, required=True)
    parser.add_argument("--method-id", required=True)
    parser.add_argument("--method-family", choices=("original_mtare", "m1d_topology", "layered_gt_map_oracle"), required=True)
    parser.add_argument("--execution-repeat", type=int, required=True)
    parser.add_argument("--checkpoint-seed", type=int)
    parser.add_argument("--block-id", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--complete-map")
    parser.add_argument("--complete-map-sha256")
    parser.add_argument("--runtime-sec", type=float, default=600.0)
    parser.add_argument("--topic-contract", type=Path, default=DEFAULT_TOPIC_CONTRACT)
    parser.add_argument("--archive-mode", choices=("container", "host"), default="container")
    parser.add_argument("--host-uid", type=int)
    parser.add_argument("--host-gid", type=int)
    args = parser.parse_args()
    topic_contract = args.topic_contract.resolve()
    try:
        topic_contract.relative_to(Path("/workspace"))
    except ValueError as exc:
        raise ValueError("topic contract must be a frozen /workspace path") from exc
    if not topic_contract.is_file():
        raise FileNotFoundError(topic_contract)
    if (
        args.environment_seed < 0
        or args.execution_repeat < 0
        or args.runtime_sec <= 0.0
        or not math.isfinite(args.runtime_sec)
    ):
        raise ValueError("invalid seed or runtime")
    if args.method_family == "m1d_topology" and args.checkpoint_seed not in (0, 1, 2):
        raise ValueError("M1D case requires checkpoint seed 0, 1 or 2")
    if args.method_family != "m1d_topology" and args.checkpoint_seed is not None:
        raise ValueError("only M1D cases may carry a checkpoint seed")
    if args.archive_mode == "host" and (args.host_uid is None or args.host_gid is None):
        raise ValueError("host archive mode requires host UID/GID")
    if args.archive_mode == "container" and (args.host_uid is not None or args.host_gid is not None):
        raise ValueError("host UID/GID are forbidden in container archive mode")
    case_dir = args.case_dir.resolve()
    case_dir.mkdir(parents=True, exist_ok=False)
    logs = case_dir / "logs"
    logs.mkdir()
    planner_output = case_dir / "planner"
    bag_path = case_dir / "raw.bag"
    contract = {
        "schema_version": "mtare_single_robot_case_contract_v1",
        "case_id": args.case_id,
        "world": args.world,
        "environment_seed": args.environment_seed,
        "method_id": args.method_id,
        "method_family": args.method_family,
        "execution_repeat": args.execution_repeat,
        "checkpoint_seed": args.checkpoint_seed,
        "block_id": args.block_id,
        "runtime_sec": args.runtime_sec,
        "checkpoint": args.checkpoint,
        "checkpoint_sha256": args.checkpoint_sha256,
        "complete_map": args.complete_map,
        "complete_map_sha256": args.complete_map_sha256,
        "test_id": "0001" if args.method_family == "original_mtare" else None,
        "topic_contract": str(topic_contract),
        "topic_contract_sha256": sha256(topic_contract),
    }
    write_json(case_dir / "case_contract.json", contract)
    system: subprocess.Popen[bytes] | None = None
    recorder: subprocess.Popen[bytes] | None = None
    method: subprocess.Popen[bytes] | None = None
    streams: list[Any] = []
    start_sim: float | None = None
    end_sim: float | None = None
    started_wall = time.monotonic()
    try:
        system, stream = start(
            "roslaunch /workspace/configs/v3/gate5/roslaunch/system_seeded.launch "
            f"world_name:={args.world} gazebo_seed:={args.environment_seed} "
            "vehicleX:=0 vehicleY:=0 terrainZ:=0 vehicleYaw:=0 rviz:=false vis_tools:=false gazebo_gui:=false",
            logs / "system.log",
        )
        streams.append(stream)
        wait_for_master(system)
        topics = json.loads(topic_contract.read_text(encoding="utf-8"))["topic_contract"]
        names = " ".join(item["name"] for item in topics)
        recorder, stream = start(
            f"rosbag record --lz4 --buffsize=2048 -O {bag_path} {names}", logs / "recorder.log"
        )
        streams.append(stream)
        method, stream = start(method_command(args, planner_output), logs / "method.log")
        streams.append(stream)

        import rospy
        from geometry_msgs.msg import PointStamped
        from rosgraph_msgs.msg import Clock

        state: dict[str, Any] = {"clock": None, "waypoints": 0}
        rospy.init_node(f"single_robot_case_{args.environment_seed}", anonymous=True, disable_signals=True)
        clock_sub = rospy.Subscriber("/clock", Clock, lambda msg: state.__setitem__("clock", msg.clock.to_sec()))
        waypoint_sub = rospy.Subscriber("/way_point", PointStamped, lambda _: state.__setitem__("waypoints", state["waypoints"] + 1))
        ready_deadline = time.monotonic() + 180.0
        while (state["clock"] is None or state["waypoints"] < 1) and time.monotonic() < ready_deadline:
            for name, process in (("system", system), ("recorder", recorder), ("method", method)):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited before method readiness with {process.returncode}")
            time.sleep(0.05)
        if state["clock"] is None or state["waypoints"] < 1:
            raise TimeoutError("method did not publish its first waypoint within 180 wall seconds")
        start_sim = float(state["clock"])
        end_sim = start_sim + args.runtime_sec
        run_deadline = time.monotonic() + max(1800.0, args.runtime_sec * 3.0)
        while float(state["clock"]) < end_sim:
            for name, process in (("system", system), ("recorder", recorder), ("method", method)):
                if process.poll() is not None:
                    raise RuntimeError(f"{name} exited early with {process.returncode}")
            if time.monotonic() >= run_deadline:
                raise TimeoutError("simulation did not reach the fixed budget")
            time.sleep(0.05)
        clock_sub.unregister()
        waypoint_sub.unregister()
    finally:
        stop(method)
        stop(recorder)
        stop(system)
        for stream in streams:
            stream.close()
    if start_sim is None or end_sim is None or not bag_path.is_file() or bag_path.stat().st_size == 0:
        raise RuntimeError("case did not finalize its simulation interval and bag")
    metrics = summarize(
        bag_path=bag_path,
        output_dir=case_dir / "evidence",
        topic_contract_path=topic_contract,
        start_sim_sec=start_sim,
        end_sim_sec=end_sim,
    )
    if args.method_family == "original_mtare":
        planner_evidence = {
            "kind": "original_mtare_internal",
            "method_log": "logs/method.log",
            "topology_snapshot": None,
            "decision_trace": None,
        }
    else:
        snapshot_path = planner_output / "topology_snapshot.json"
        trace_path = planner_output / "decision_trace.jsonl"
        if not snapshot_path.is_file() or not trace_path.is_file() or trace_path.stat().st_size == 0:
            raise RuntimeError("replacement planner did not finalize graph snapshot and decision trace")
        snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
        if snapshot.get("mode") != "closed_loop" or snapshot.get("failed_cycles") != 0:
            raise RuntimeError("replacement planner snapshot reports invalid mode or failed cycles")
        planner_evidence = {
            "kind": args.method_family,
            "topology_snapshot": str(snapshot_path.relative_to(case_dir)),
            "decision_trace": str(trace_path.relative_to(case_dir)),
            "decision_trace_bytes": trace_path.stat().st_size,
            "failed_cycles": snapshot["failed_cycles"],
        }
    original_sha = sha256(bag_path)
    original_size = bag_path.stat().st_size
    if args.archive_mode == "container":
        archive = case_dir / "raw.bag.zst"
        completed = subprocess.run(["zstd", "-10", "-T0", "-q", "-f", str(bag_path), "-o", str(archive)], check=False)
        if completed.returncode != 0 or not archive.is_file():
            raise RuntimeError("zstd compression failed")
        verified = verify_zstd_archive(archive, original_sha)
        storage = {
            "schema_version": "mtare_lossless_bag_archive_v1",
            "original_bag_sha256": original_sha,
            "original_bag_bytes": original_size,
            "archive_sha256": sha256(archive),
            "archive_bytes": archive.stat().st_size,
            "decompressed_sha256": verified,
            "zstd_level": 10,
        }
        write_json(case_dir / "storage.json", storage)
        bag_path.unlink()
        case_status = "PASS_SINGLE_ROBOT_CASE_V1"
    else:
        storage = {
            "schema_version": "mtare_host_archive_pending_v1",
            "status": "PENDING_HOST_ARCHIVE",
            "original_bag_sha256": original_sha,
            "original_bag_bytes": original_size,
            "archive_overwrite_permitted": False,
        }
        case_status = "PASS_SINGLE_ROBOT_CASE_V2_PENDING_HOST_ARCHIVE"
    result = {
        "schema_version": "mtare_single_robot_case_summary_v1",
        "case": contract,
        "metrics": metrics,
        "storage": storage,
        "planner_evidence": planner_evidence,
        "wall_duration_sec": time.monotonic() - started_wall,
        "status": case_status,
    }
    write_json(case_dir / "summary.json", result)
    if args.archive_mode == "host":
        handoff_case_tree(case_dir, int(args.host_uid), int(args.host_gid))
    print(json.dumps({"case_id": args.case_id, "status": result["status"], "metrics": metrics}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
