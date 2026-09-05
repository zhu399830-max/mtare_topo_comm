#!/usr/bin/env python3
"""ROS1 global-planner replacement using the frozen M1D causal topology.

The node replaces only ``tare_planner_node``.  Simulator, registered LiDAR,
terrain analysis, localPlanner, pathFollower, collision checks, and control
remain the original M-TARE components.
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

import numpy as np
import torch

from mtare_topo.integration.online_topology_runtime import (
    OnlineTopologyPlannerRuntime,
    SemanticPrediction,
)
from mtare_topo.integration.range_image_adapter import registered_points_to_range_image
from mtare_topo.integration.aee_organized_scan_adapter import aee_organized_pointcloud2_to_range_image
from mtare_topo.deployment.m1d_checkpoint import COMPOSITE_V9_MODE, COMPOSITE_V9_RUNTIME_CONTRACT
from mtare_topo.integration.semantic_fallback import apply_composite_v9_semantics, apply_empty_direction_fallback
from mtare_topo.planning.topological_frontier import TopologicalPlannerConfig
from mtare_topo.representation.phase3_structural_semantics import StructuralSemanticNet
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
DEPLOYMENT_SCHEMA = "m1d_ros_deployment_state_v1"
SUPPORTED_DEPLOYMENT_MODES = ("M1D", "M1D_AEE_HEAD_ADAPTED_V1", COMPOSITE_V9_MODE)
MAXIMUM_RAW_POSE_STAMP_DELTA_SEC = 0.1


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


class SemanticTopologyGlobalNodeV3:
    def __init__(
        self,
        *,
        checkpoint: Path,
        checkpoint_sha256: str,
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

        if publish_period_sec <= 0 or not math.isfinite(publish_period_sec):
            raise ValueError("publish period must be finite and positive")
        actual_hash = sha256(checkpoint)
        if actual_hash != checkpoint_sha256:
            raise RuntimeError(f"deployment checkpoint hash drift: {actual_hash}")
        payload = torch.load(checkpoint, map_location="cpu")
        if payload.get("schema_version") != DEPLOYMENT_SCHEMA:
            raise RuntimeError("not a frozen M1D ROS deployment checkpoint")
        if payload.get("model_class") != "StructuralSemanticNet" or payload.get("mode") not in SUPPORTED_DEPLOYMENT_MODES:
            raise RuntimeError("deployment checkpoint identity mismatch")
        self.checkpoint_mode = str(payload["mode"])
        if self.checkpoint_mode == COMPOSITE_V9_MODE and payload.get("runtime_contract") != COMPOSITE_V9_RUNTIME_CONTRACT:
            raise RuntimeError("V9 deployment runtime contract mismatch")
        self.model = StructuralSemanticNet().cpu()
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.eval()

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
        self.accepted_points = 0
        self.unique_valid_cells = 0
        self.learned_empty_cycles = 0
        self.fallback_cycles = 0
        self.post_warmup_cycles = 0
        self.post_warmup_fallback_cycles = 0
        self.started_monotonic = time.monotonic()
        self.checkpoint_identity = {
            "deployment_checkpoint": str(checkpoint),
            "deployment_checkpoint_sha256": actual_hash,
            "source_checkpoint_sha256": payload.get("source_checkpoint_sha256"),
            "source_seed": payload.get("seed"),
            "mode": payload.get("mode"),
            "model_class": payload.get("model_class"),
            "runtime_contract": payload.get("runtime_contract"),
            "torch_runtime": torch.__version__,
        }
        self.trace = (self.output / "decision_trace.jsonl").open("x", encoding="utf-8", buffering=1)
        self._snapshot_written = False

        prefix = "/semantic_topology" if self.shadow else ""
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

        odom_sub = message_filters.Subscriber("/state_estimation_at_scan", Odometry, queue_size=20)
        if self.checkpoint_mode == COMPOSITE_V9_MODE:
            scan_sub = message_filters.Subscriber("/velodyne_points", PointCloud2, queue_size=20)
            self.synchronizer = message_filters.ApproximateTimeSynchronizer(
                [scan_sub, odom_sub],
                queue_size=100,
                slop=MAXIMUM_RAW_POSE_STAMP_DELTA_SEC,
                allow_headerless=False,
            )
            self.synchronizer.registerCallback(self.raw_scan_pose_callback)
            self.input_operator = "aee_organized_velodyne_16x350_to_16x720_v1"
        else:
            scan_sub = message_filters.Subscriber("/registered_scan", PointCloud2, queue_size=20)
            self.synchronizer = message_filters.TimeSynchronizer([scan_sub, odom_sub], queue_size=100)
            self.synchronizer.registerCallback(self.scan_pose_callback)
            self.input_operator = "registered_map_points_to_16x720_v1"
        self.free_paths_sub = rospy.Subscriber("/free_paths", PointCloud2, self.free_paths_callback, queue_size=5)
        rospy.on_shutdown(self.write_snapshot)

    @staticmethod
    def points_xyz(message: Any) -> np.ndarray:
        from sensor_msgs import point_cloud2

        values = list(point_cloud2.read_points(message, field_names=("x", "y", "z"), skip_nans=False))
        if not values:
            return np.empty((0, 3), dtype=np.float64)
        return np.asarray(values, dtype=np.float64).reshape(-1, 3)

    def infer(self, range_m: np.ndarray, valid_mask: np.ndarray) -> SemanticPrediction:
        student = np.stack((range_m / 50.0, valid_mask.astype(np.float32)), axis=0)[None]
        with torch.inference_mode():
            output = self.model(torch.from_numpy(student.astype(np.float32)))
        return SemanticPrediction(
            direction_logits=output["direction_logits"][0].cpu().numpy(),
            count_probabilities=torch.softmax(output["count_logits"][0], dim=0).cpu().numpy(),
            role_probabilities=torch.softmax(output["role_logits"][0], dim=0).cpu().numpy(),
            z_role=output["z_role"][0].cpu().numpy(),
        )

    def scan_pose_callback(self, scan: Any, odom: Any) -> None:
        cycle_started = time.monotonic()
        try:
            if scan.header.stamp != odom.header.stamp:
                raise RuntimeError("TimeSynchronizer delivered unequal scan/odom stamps")
            if scan.header.frame_id.lstrip("/") != "map" or odom.header.frame_id.lstrip("/") != "map":
                raise RuntimeError("registered scan and odometry must both be in map frame")
            points = self.points_xyz(scan)
            if len(points) < 20:
                raise RuntimeError("registered scan contains fewer than 20 returns")
            xyz = position_tuple(odom)
            orientation = quaternion_tuple(odom)
            range_m, valid_mask, audit = registered_points_to_range_image(
                points,
                sensor_origin_world=xyz,
                sensor_orientation_xyzw=orientation,
            )
            if audit.out_of_ring_points:
                raise RuntimeError(f"registered scan has {audit.out_of_ring_points} off-ring returns")
            self.process_model_input(
                range_m=range_m,
                valid_mask=valid_mask,
                audit=audit,
                odom=odom,
                stamp_sec=float(scan.header.stamp.to_sec()),
                cycle_started=cycle_started,
            )
        except Exception as exc:
            self.failed_cycles += 1
            self.rospy.logerr("semantic topology fatal cycle failure: %s", exc)
            self.rospy.signal_shutdown(f"semantic topology fatal cycle failure: {exc}")

    def raw_scan_pose_callback(self, scan: Any, odom: Any) -> None:
        cycle_started = time.monotonic()
        try:
            from sensor_msgs import point_cloud2

            delta = abs(float(scan.header.stamp.to_sec()) - float(odom.header.stamp.to_sec()))
            if delta > MAXIMUM_RAW_POSE_STAMP_DELTA_SEC + 1e-12:
                raise RuntimeError(f"raw scan/pose stamp delta exceeds frozen contract: {delta}")
            if not scan.header.frame_id or scan.header.frame_id.lstrip("/") == "map":
                raise RuntimeError(f"raw organized scan must remain in a sensor frame: {scan.header.frame_id}")
            if odom.header.frame_id.lstrip("/") != "map":
                raise RuntimeError("scan-synchronous odometry must be in map frame")
            range_m, valid_mask, audit = aee_organized_pointcloud2_to_range_image(
                scan, point_cloud2
            )
            self.process_model_input(
                range_m=range_m,
                valid_mask=valid_mask,
                audit=audit,
                odom=odom,
                stamp_sec=float(odom.header.stamp.to_sec()),
                cycle_started=cycle_started,
                input_extra={
                    "raw_stamp_sec": float(scan.header.stamp.to_sec()),
                    "pose_stamp_sec": float(odom.header.stamp.to_sec()),
                    "absolute_stamp_delta_sec": delta,
                },
            )
        except Exception as exc:
            self.failed_cycles += 1
            self.rospy.logerr("semantic topology fatal cycle failure: %s", exc)
            self.rospy.signal_shutdown(f"semantic topology fatal cycle failure: {exc}")

    def process_model_input(
        self,
        *,
        range_m: np.ndarray,
        valid_mask: np.ndarray,
        audit: Any,
        odom: Any,
        stamp_sec: float,
        cycle_started: float,
        input_extra: dict[str, Any] | None = None,
    ) -> None:
        xyz = position_tuple(odom)
        orientation = quaternion_tuple(odom)
        prediction = self.infer(range_m, valid_mask)
        composite_audit = None
        if self.checkpoint_mode == COMPOSITE_V9_MODE:
            prediction, fallback, composite_audit = apply_composite_v9_semantics(
                prediction, range_m, valid_mask,
                direction_threshold=self.runtime.direction_threshold,
            )
        else:
            prediction, fallback = apply_empty_direction_fallback(
                prediction, range_m, valid_mask,
                direction_threshold=self.runtime.direction_threshold,
            )
        with self.lock:
            cycle = self.runtime.update(
                stamp_sec=stamp_sec,
                sensor_xyz_m=xyz,
                sensor_orientation_xyzw=orientation,
                prediction=prediction,
                cycle_started_monotonic=cycle_started,
            )
            self.input_points += audit.input_points
            self.accepted_points += audit.accepted_points
            self.unique_valid_cells += audit.unique_valid_cells
            self.learned_empty_cycles += int(fallback.learned_empty)
            self.fallback_cycles += int(fallback.fallback_used)
            if cycle.frame_index >= 20:
                self.post_warmup_cycles += 1
                self.post_warmup_fallback_cycles += int(fallback.fallback_used)
            record = cycle.to_dict()
            record["input_audit"] = audit.to_dict()
            record["input_operator"] = self.input_operator
            if input_extra is not None:
                record["input_timing"] = input_extra
            record["semantic_fallback"] = fallback.to_dict()
            if composite_audit is not None:
                record["composite_semantics"] = composite_audit.to_dict()
            self.trace.write(json.dumps(record, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n")
            self.publish_cycle(cycle)

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
            count = int(message.width) * int(message.height)
            with self.lock:
                clearing = self.runtime.observe_free_path_count(count)
            if clearing is not None:
                self.map_clearing_pub.publish(self.Float32(data=float(clearing)))
        except Exception as exc:
            self.rospy.logerr("semantic topology free-path failure: %s", exc)
            self.rospy.signal_shutdown(f"semantic topology free-path failure: {exc}")

    def write_snapshot(self) -> None:
        with self.lock:
            if self._snapshot_written:
                return
            snapshot = {
                "schema_version": "semantic_topology_global_node_v3_snapshot_v1",
                "mode": "shadow" if self.shadow else "closed_loop",
                "checkpoint": self.checkpoint_identity,
                "graph_config": GRAPH_CONFIG.to_dict(),
                "planner_config": PLANNER_CONFIG.to_dict(),
                "publish_period_sec": self.publish_period_sec,
                "input_operator": self.input_operator,
                "failed_cycles": self.failed_cycles,
                "input_points": self.input_points,
                "accepted_points": self.accepted_points,
                "unique_valid_cells": self.unique_valid_cells,
                "learned_empty_cycles": self.learned_empty_cycles,
                "fallback_cycles": self.fallback_cycles,
                "post_warmup_cycles": self.post_warmup_cycles,
                "post_warmup_fallback_cycles": self.post_warmup_fallback_cycles,
                "post_warmup_fallback_rate": self.post_warmup_fallback_cycles / max(self.post_warmup_cycles, 1),
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
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--shadow", action="store_true")
    parser.add_argument("--publish-period-sec", type=float, default=1.0)
    args = parser.parse_args()
    import rospy

    rospy.init_node("semantic_topology_global_node_v3", anonymous=False)
    SemanticTopologyGlobalNodeV3(
        checkpoint=args.checkpoint.resolve(),
        checkpoint_sha256=args.checkpoint_sha256,
        output=args.output.resolve(),
        shadow=args.shadow,
        publish_period_sec=args.publish_period_sec,
    )
    rospy.loginfo("semantic topology v3 started in %s mode", "shadow" if args.shadow else "closed_loop")
    rospy.spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
