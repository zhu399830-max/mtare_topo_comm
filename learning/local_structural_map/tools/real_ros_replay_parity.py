import argparse
import json
import subprocess
import textwrap
from pathlib import Path

import numpy as np

from learning.local_structural_map.builder import LocalStructuralMapBuilder
from learning.local_structural_map.config import LocalMapConfig
from learning.local_structural_map.datasets.mtare_bag import MTAREBagAdapter
from learning.local_structural_map.tools.io_utils import write_json


def save_offline(bag: Path, out_npz: Path, max_samples: int) -> list[str]:
    builder = LocalStructuralMapBuilder(LocalMapConfig())
    tensors = []
    stamps = []
    names = None
    for frame in MTAREBagAdapter(bag, max_samples=max_samples):
        item = builder.update(frame)
        tensors.append(item.tensor)
        stamps.append(item.timestamp_ns)
        names = item.channel_names
    if not tensors:
        raise RuntimeError("offline path produced no samples")
    np.savez_compressed(out_npz, tensors=np.stack(tensors), stamps=np.asarray(stamps, dtype=np.int64), channel_names=np.asarray(names))
    return list(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bag", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--max-samples", type=int, default=20)
    parser.add_argument("--container", default="robot0")
    args = parser.parse_args()
    bag = Path(args.bag).resolve()
    work_dir = Path(args.work_dir).resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    offline_npz = work_dir / "offline_ros_parity.npz"
    online_npz = work_dir / "online_ros_parity.npz"
    subscriber_script = work_dir / "ros_parity_subscriber.py"
    channel_names = save_offline(bag, offline_npz, max(args.max_samples * 10, args.max_samples))
    repo = Path.cwd().resolve()
    container_work = Path(f"/tmp/local_map_parity_{work_dir.name}")
    bag_in_container = container_work / "input.bag"
    online_in_container = container_work / "online_ros_parity.npz"
    container_script = container_work / "ros_parity_subscriber.py"
    max_samples = int(args.max_samples)
    subscriber_script.write_text(textwrap.dedent(f"""
        import json
        import sys
        import threading
        import numpy as np
        import rospy
        from sensor_msgs.msg import PointCloud2
        from nav_msgs.msg import Odometry
        sys.path.insert(0, '{container_work}')
        from learning.local_structural_map.builder import LocalStructuralMapBuilder
        from learning.local_structural_map.config import LocalMapConfig
        from learning.local_structural_map.runtime.mtare_runtime import MTARERuntimeAdapter

        builder = LocalStructuralMapBuilder(LocalMapConfig())
        adapter = MTARERuntimeAdapter(builder)
        tensors = []
        stamps = []
        odom_count = 0
        scan_count = 0
        sync_count = 0
        out_path = '{online_in_container}'
        max_samples = {max_samples}
        lock = threading.Lock()

        def odom_cb(msg):
            global odom_count, sync_count
            with lock:
                odom_count += 1
                item = adapter.handle_odom(msg)
                if item is not None:
                    sync_count += 1
                    tensors.append(item.tensor)
                    stamps.append(item.timestamp_ns)
                    if len(tensors) >= max_samples:
                        rospy.signal_shutdown('enough samples')

        def scan_cb(msg):
            global scan_count, sync_count
            with lock:
                scan_count += 1
                item = adapter.handle_scan(msg)
                if item is not None:
                    sync_count += 1
                    tensors.append(item.tensor)
                    stamps.append(item.timestamp_ns)
                    if len(tensors) >= max_samples:
                        rospy.signal_shutdown('enough samples')

        rospy.init_node('local_structural_map_ros_parity_subscriber', anonymous=False)
        rospy.Subscriber('/state_estimation_at_scan', Odometry, odom_cb, queue_size=100)
        rospy.Subscriber('/registered_scan', PointCloud2, scan_cb, queue_size=100)
        rospy.spin()
        if tensors:
            np.savez_compressed(out_path, tensors=np.stack(tensors), stamps=np.asarray(stamps, dtype=np.int64),
                                channel_names=np.asarray({channel_names!r}),
                                diagnostics=np.asarray(json.dumps({{'odom_count': odom_count, 'scan_count': scan_count, 'sync_count': sync_count}})))
        else:
            np.savez_compressed(out_path, tensors=np.empty((0, {len(channel_names)}, 100, 100), dtype=np.float32),
                                stamps=np.empty((0,), dtype=np.int64), channel_names=np.asarray({channel_names!r}),
                                diagnostics=np.asarray(json.dumps({{'odom_count': odom_count, 'scan_count': scan_count, 'sync_count': sync_count}})))
    """))
    subprocess.run(["docker", "exec", args.container, "bash", "-lc", f"mkdir -p {container_work}"], check=True)
    subprocess.run(["docker", "cp", str(repo / "learning"), f"{args.container}:{container_work}/learning"], check=True)
    subprocess.run(["docker", "cp", str(bag), f"{args.container}:{bag_in_container}"], check=True)
    subprocess.run(["docker", "cp", str(subscriber_script), f"{args.container}:{container_script}"], check=True)
    cmd = textwrap.dedent(f"""
        set -Eeo pipefail
        export ROS_DISTRO=noetic
        export ROS_VERSION=1
        source /opt/ros/noetic/setup.bash
        export PYTHONPATH={container_work}:${{PYTHONPATH:-}}
        roscore > {container_work}/roscore.log 2>&1 &
        ROSCORE_PID=$!
        echo $ROSCORE_PID > {container_work}/roscore.pid
        sleep 4
        python3 {container_script} > {container_work}/subscriber.stdout.log 2> {container_work}/subscriber.stderr.log &
        SUB_PID=$!
        sleep 2
        rosbag play --clock {bag_in_container} /registered_scan:=/registered_scan /state_estimation_at_scan:=/state_estimation_at_scan > {container_work}/rosbag_play.stdout.log 2> {container_work}/rosbag_play.stderr.log &
        PLAY_PID=$!
        wait $SUB_PID || true
        kill $PLAY_PID >/dev/null 2>&1 || true
        kill $ROSCORE_PID >/dev/null 2>&1 || true
    """)
    try:
        proc = subprocess.run(["docker", "exec", args.container, "bash", "-lc", cmd], text=True, capture_output=True, timeout=180)
        docker_returncode = proc.returncode
        docker_stdout = proc.stdout
        docker_stderr = proc.stderr
    except subprocess.TimeoutExpired as exc:
        docker_returncode = 124
        docker_stdout = (exc.stdout or "") if isinstance(exc.stdout, str) else ""
        docker_stderr = ((exc.stderr or "") if isinstance(exc.stderr, str) else "") + "\nTIMEOUT_EXPIRED"
        subprocess.run(["docker", "exec", args.container, "bash", "-lc", f"pkill -f {container_work}/ros_parity_subscriber.py || true; pkill -f 'rosbag play --clock {bag_in_container}' || true; pkill -f roscore || true; pkill -f rosmaster || true"], text=True, capture_output=True)
    (work_dir / "docker.stdout.log").write_text(docker_stdout)
    (work_dir / "docker.stderr.log").write_text(docker_stderr)
    for name in ["online_ros_parity.npz", "roscore.log", "roscore.pid", "subscriber.stdout.log", "subscriber.stderr.log", "rosbag_play.stdout.log", "rosbag_play.stderr.log"]:
        subprocess.run(["docker", "cp", f"{args.container}:{container_work}/{name}", str(work_dir / name)], text=True, capture_output=True)
    result = {
        "docker_returncode": docker_returncode,
        "roscore_pid": (work_dir / "roscore.pid").read_text().strip() if (work_dir / "roscore.pid").exists() else None,
        "rosbag_play_command": f"rosbag play --clock {bag_in_container}",
        "container": args.container,
        "topics": ["/registered_scan", "/state_estimation_at_scan"],
    }
    if not online_npz.exists():
        result.update({"status": "FAILED", "reason": "online subscriber output missing"})
        write_json(args.output, result)
        return
    off = np.load(offline_npz, allow_pickle=False)
    on = np.load(online_npz, allow_pickle=False)
    offline_by_stamp = {int(s): off["tensors"][i] for i, s in enumerate(off["stamps"])}
    online_by_stamp = {int(s): on["tensors"][i] for i, s in enumerate(on["stamps"])}
    matched = sorted(set(offline_by_stamp) & set(online_by_stamp))
    diagnostics = json.loads(str(on["diagnostics"])) if "diagnostics" in on else {}
    if not matched:
        result.update({"status": "FAILED", "reason": "no timestamp matches", "subscriber_diagnostics": diagnostics})
        write_json(args.output, result)
        return
    diffs = np.stack([np.abs(offline_by_stamp[s] - online_by_stamp[s]) for s in matched])
    result.update({
        "status": "PASS" if float(np.max(diffs)) == 0.0 else "DIFF",
        "subscriber_diagnostics": diagnostics,
        "offline_count": int(len(off["stamps"])),
        "online_count": int(len(on["stamps"])),
        "matched_count": int(len(matched)),
        "unmatched_offline_count": int(len(set(offline_by_stamp) - set(online_by_stamp))),
        "unmatched_online_count": int(len(set(online_by_stamp) - set(offline_by_stamp))),
        "out_of_order_messages": "UNAVAILABLE: rospy callbacks do not expose bag connection order directly",
        "dropped_frames": int(max(0, len(off["stamps"]) - len(matched))),
        "channel_names": channel_names,
        "per_channel_max_abs_error": {name: float(np.max(diffs[:, i])) for i, name in enumerate(channel_names)},
        "per_channel_mean_abs_error": {name: float(np.mean(diffs[:, i])) for i, name in enumerate(channel_names)},
        "mask_mismatch_pixels": {
            name: int(sum(np.count_nonzero((offline_by_stamp[s][i] > 0.5) != (online_by_stamp[s][i] > 0.5)) for s in matched))
            for i, name in enumerate(channel_names)
            if "mask" in name or "occupancy" in name
        },
    })
    write_json(args.output, result)


if __name__ == "__main__":
    main()
