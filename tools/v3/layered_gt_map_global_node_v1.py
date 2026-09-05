#!/usr/bin/env python3
"""ROS1 topological global-planner upper bound from a frozen complete map.

The node intentionally shares the M1D node's causal graph, planner, handoff,
scan cadence and public topics.  Only the semantic predictor is replaced by a
pose-conditioned layered GT-map query.  It never launches tare_planner_node.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import threading
import time
from pathlib import Path
from typing import Any

from mtare_topo.integration.online_topology_runtime import (
    OnlineTopologyPlannerRuntime,
    SemanticPrediction,
    quaternion_yaw_deg,
)
from mtare_topo.oracle.layered_gt_map import LayeredGTMapConfig, LayeredGTMapOracle
from mtare_topo.planning.topological_frontier import TopologicalPlannerConfig
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig


GRAPH_CONFIG = CausalGraphConfig(
    stable_frames=2,
    minimum_event_travel_m=8.0,
    loop_merge_radius_m=6.0,
    branch_heading_merge_deg=20.0,
    turn_event_deg=45.0,
    distance_anchor_interval_m=20.0,
)
PLANNER_CONFIG = TopologicalPlannerConfig(
    waypoint_lookahead_m=4.0,
    graph_cost_scale_m=20.0,
    retry_penalty=0.25,
    minimum_frontier_confidence=0.05,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def quaternion_tuple(message: Any) -> tuple[float, float, float, float]:
    value = message.pose.pose.orientation
    return float(value.x), float(value.y), float(value.z), float(value.w)


def position_tuple(message: Any) -> tuple[float, float, float]:
    value = message.pose.pose.position
    return float(value.x), float(value.y), float(value.z)


def oracle_semantic_prediction(prediction: Any) -> SemanticPrediction:
    return SemanticPrediction(
        direction_logits=prediction.direction_logits,
        count_probabilities=prediction.count_probabilities,
        role_probabilities=prediction.role_probabilities,
        z_role=prediction.z_role,
    )


class LayeredGTMapGlobalNodeV1:
    def __init__(
        self,
        *,
        complete_map: Path,
        complete_map_sha256: str,
        output: Path,
        shadow: bool,
        publish_period_sec: float,
    ) -> None:
        import message_filters
        import rospy
        from geometry_msgs.msg import PointStamped
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2
        from std_msgs.msg import Bool, Float32

        if publish_period_sec <= 0.0 or not math.isfinite(publish_period_sec):
            raise ValueError("publish period must be finite and positive")
        actual_hash = sha256(complete_map)
        if actual_hash != complete_map_sha256:
            raise RuntimeError(f"complete-map hash drift: {actual_hash}")
        self.oracle = LayeredGTMapOracle.from_ply(complete_map, LayeredGTMapConfig())
        self.rospy = rospy
        self.PointStamped = PointStamped
        self.Bool = Bool
        self.Float32 = Float32
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.shadow = bool(shadow)
        self.publish_period_sec = float(publish_period_sec)
        self.last_publish_stamp = -math.inf
        self.runtime = OnlineTopologyPlannerRuntime(GRAPH_CONFIG, PLANNER_CONFIG)
        self.lock = threading.Lock()
        self.failed_cycles = 0
        self.input_points = 0
        self.started_monotonic = time.monotonic()
        self.map_identity = {
            "complete_map": str(complete_map),
            "complete_map_sha256": actual_hash,
            "point_count": int(len(self.oracle.points)),
            "oracle_config": self.oracle.config.to_dict(),
        }
        self.trace = (self.output / "decision_trace.jsonl").open("x", encoding="utf-8", buffering=1)
        self._snapshot_written = False

        prefix = "/layered_gt_map" if self.shadow else ""
        self.waypoint_pub = rospy.Publisher(prefix + "/way_point" if self.shadow else "/way_point", PointStamped, queue_size=1)
        self.runtime_pub = rospy.Publisher(prefix + "/runtime" if self.shadow else "/runtime", Float32, queue_size=2)
        self.finish_pub = rospy.Publisher(
            prefix + "/candidate_complete" if self.shadow else "/sensor_coverage_planner/exploration_finish",
            Bool,
            queue_size=2,
        )
        self.map_clearing_pub = rospy.Publisher(
            prefix + "/map_clearing_proposal" if self.shadow else "/map_clearing", Float32, queue_size=1
        )
        scan_sub = message_filters.Subscriber("/registered_scan", PointCloud2, queue_size=20)
        odom_sub = message_filters.Subscriber("/state_estimation_at_scan", Odometry, queue_size=20)
        self.synchronizer = message_filters.TimeSynchronizer([scan_sub, odom_sub], queue_size=100)
        self.synchronizer.registerCallback(self.scan_pose_callback)
        self.free_paths_sub = rospy.Subscriber("/free_paths", PointCloud2, self.free_paths_callback, queue_size=5)
        rospy.on_shutdown(self.write_snapshot)

    def scan_pose_callback(self, scan: Any, odom: Any) -> None:
        cycle_started = time.monotonic()
        try:
            if scan.header.stamp != odom.header.stamp:
                raise RuntimeError("TimeSynchronizer delivered unequal scan/odom stamps")
            if scan.header.frame_id.lstrip("/") != "map" or odom.header.frame_id.lstrip("/") != "map":
                raise RuntimeError("registered scan and odometry must both be in map frame")
            point_count = int(scan.width) * int(scan.height)
            if point_count < 20:
                raise RuntimeError("registered scan contains fewer than 20 returns")
            xyz = position_tuple(odom)
            orientation = quaternion_tuple(odom)
            yaw_deg = quaternion_yaw_deg(orientation)
            semantic = oracle_semantic_prediction(self.oracle.predict(xyz, yaw_deg))
            stamp_sec = float(scan.header.stamp.to_sec())
            with self.lock:
                cycle = self.runtime.update(
                    stamp_sec=stamp_sec,
                    sensor_xyz_m=xyz,
                    sensor_orientation_xyzw=orientation,
                    prediction=semantic,
                    cycle_started_monotonic=cycle_started,
                )
                self.input_points += point_count
                record = cycle.to_dict()
                record["oracle_exit_count"] = int(semantic.count_probabilities.argmax()) + 1
                self.trace.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
                self.publish_cycle(cycle)
        except Exception as exc:
            self.failed_cycles += 1
            self.rospy.logerr("layered GT-map oracle fatal cycle failure: %s", exc)
            self.rospy.signal_shutdown(f"layered GT-map oracle fatal cycle failure: {exc}")

    def publish_cycle(self, cycle: Any) -> None:
        self.runtime_pub.publish(self.Float32(data=float(cycle.handoff.runtime_sec)))
        candidate_complete = cycle.target.status == "CANDIDATE_COMPLETE"
        self.finish_pub.publish(self.Bool(data=candidate_complete if self.shadow else False))
        if cycle.stamp_sec - self.last_publish_stamp < self.publish_period_sec:
            return
        waypoint = cycle.handoff.waypoint
        if waypoint is None:
            raise RuntimeError("v2 handoff must provide TARGET or hold waypoint")
        message = self.PointStamped()
        message.header.stamp = self.rospy.Time.from_sec(float(waypoint["stamp_sec"]))
        message.header.frame_id = "map"
        message.point.x, message.point.y, message.point.z = waypoint["xyz_m"]
        self.waypoint_pub.publish(message)
        self.last_publish_stamp = cycle.stamp_sec

    def free_paths_callback(self, message: Any) -> None:
        try:
            with self.lock:
                clearing = self.runtime.observe_free_path_count(int(message.width) * int(message.height))
            if clearing is not None:
                self.map_clearing_pub.publish(self.Float32(data=float(clearing)))
        except Exception as exc:
            self.rospy.logerr("layered GT-map oracle free-path failure: %s", exc)
            self.rospy.signal_shutdown(f"layered GT-map oracle free-path failure: {exc}")

    def write_snapshot(self) -> None:
        with self.lock:
            if self._snapshot_written:
                return
            snapshot = {
                "schema_version": "layered_gt_map_global_node_v1_snapshot_v1",
                "mode": "shadow" if self.shadow else "closed_loop",
                "map": self.map_identity,
                "graph_config": GRAPH_CONFIG.to_dict(),
                "planner_config": PLANNER_CONFIG.to_dict(),
                "publish_period_sec": self.publish_period_sec,
                "failed_cycles": self.failed_cycles,
                "input_points": self.input_points,
                "wall_runtime_sec": time.monotonic() - self.started_monotonic,
                "runtime": self.runtime.snapshot(),
            }
            with (self.output / "topology_snapshot.json").open("x", encoding="utf-8") as stream:
                stream.write(json.dumps(snapshot, indent=2, sort_keys=True, allow_nan=False) + "\n")
            if not self.trace.closed:
                self.trace.close()
            self._snapshot_written = True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--complete-map", required=True, type=Path)
    parser.add_argument("--complete-map-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--publish-period-sec", type=float, default=1.0)
    args = parser.parse_args()
    import rospy

    rospy.init_node("layered_gt_map_global_node_v1", anonymous=False)
    LayeredGTMapGlobalNodeV1(
        complete_map=args.complete_map.resolve(),
        complete_map_sha256=args.complete_map_sha256,
        output=args.output.resolve(),
        shadow=args.shadow,
        publish_period_sec=args.publish_period_sec,
    )
    rospy.loginfo("layered GT-map global node started in %s mode", "shadow" if args.shadow else "closed_loop")
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
