#!/usr/bin/env python3
"""ROS1 bridge: semantic role -> incremental topology -> safe local waypoints.

This node deliberately replaces only the *global* layer.  CMU's simulator,
terrain analysis, and collision-avoiding local planner remain untouched.  The
node consumes the same registered world-frame scans used in training and
publishes a short-horizon ``/way_point`` for the CMU local planner.

Run this inside the Noetic CMU/M-TARE container after copying the repository
and the frozen checkpoint.  ``--shadow`` builds and logs the topology but
never publishes a waypoint, which is the required integration gate before a
closed-loop run.
"""
from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
import torch

from learning.local_structural_map.datasets.common import pointcloud2_xyz, pose_from_odometry
from learning.local_structural_map.schema import Pose2D
from learning.structural_learning.surface_evidence import SurfaceEvidenceBuilder, SurfaceEvidenceConfig
from learning.structural_learning.topological_supervision import extract_exit_sectors
from learning.structural_learning.tools.run_semantic_bottleneck_role import Semantic, canonical_roles_numpy


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def sector_angle(sector: int, sectors: int = 32) -> float:
    """Training convention: x forward, y left, sector zero points forward."""
    return 2.0 * math.pi * float(sector) / float(sectors)


class ExplorationCoverage:
    """Planner-only 2-D observed-space memory; never enters the role model."""
    def __init__(self, resolution_m: float = .5) -> None:
        self.resolution_m = resolution_m
        self.known: set[tuple[int, int]] = set()

    def _cell(self, x: float, y: float) -> tuple[int, int]:
        return int(math.floor(x / self.resolution_m)), int(math.floor(y / self.resolution_m))

    def update(self, pose: Pose2D, points: np.ndarray) -> None:
        start = self._cell(pose.x, pose.y)
        for point in points[::40]:
            if math.hypot(float(point[0])-pose.x, float(point[1])-pose.y) > 20.: continue
            end = self._cell(float(point[0]), float(point[1])); dx=end[0]-start[0]; dy=end[1]-start[1]
            steps=max(abs(dx),abs(dy),1)
            for k in range(steps+1):
                self.known.add((int(round(start[0]+dx*k/steps)),int(round(start[1]+dy*k/steps))))

    def unknown_gain(self, pose: Pose2D, world_angle: float, distance_m: float = 8.) -> float:
        cells=[]
        for offset in (-.18, 0., .18):
            angle=world_angle+offset
            for radius in np.arange(1.,distance_m+self.resolution_m,self.resolution_m):
                cells.append(self._cell(pose.x+radius*math.cos(angle),pose.y+radius*math.sin(angle)))
        return float(np.mean([cell not in self.known for cell in cells])) if cells else 0.


class Topology:
    """Small persistent graph; node identities come from role and displacement.

    This is intentionally conservative: it never manufactures map edges from
    a role similarity alone.  An edge is added only for a traversed motion.
    """

    def __init__(self, min_event_travel_m: float = 4.0, event_distance: float = 0.16,
                 loop_radius_m: float = 2.5, role_merge_distance: float = 0.12) -> None:
        self.min_event_travel_m = min_event_travel_m
        self.event_distance = event_distance
        self.loop_radius_m = loop_radius_m
        self.role_merge_distance = role_merge_distance
        self.nodes: list[dict[str, Any]] = []
        self.edges: list[dict[str, Any]] = []
        self.current: int | None = None

    def update(self, pose: Pose2D, role: np.ndarray, semantic: dict[str, np.ndarray]) -> int:
        role = np.asarray(role, dtype=np.float32)
        if self.current is None:
            self.nodes.append(self._node(pose, role, semantic))
            self.current = 0
            return self.current
        previous = self.nodes[self.current]
        displacement = float(math.hypot(pose.x - previous['x'], pose.y - previous['y']))
        semantic_delta = self._semantic_distance(pose, semantic, previous)
        # A node is a structural event, never a fixed-distance sample.  Travel
        # only debounces local prediction noise; it cannot create a node by
        # itself.  This is the explicit semantic boundary of the mapper.
        new_event = displacement >= self.min_event_travel_m and semantic_delta >= self.event_distance
        if new_event:
            loop = self._nearby_match(pose, role)
            old = self.current
            if loop is None:
                self.nodes.append(self._node(pose, role, semantic))
                self.current = len(self.nodes) - 1
                edge_kind = 'structural_transition'
            else:
                self.current = loop
                edge_kind = 'loop_closure'
            if old != self.current:
                self.edges.append({'from': old, 'to': self.current, 'length_m': displacement, 'kind': edge_kind,
                                   'semantic_delta': semantic_delta})
                self._mark_traversed_exit(old, pose)
                # The same physical traversal also consumes the entrance at
                # the destination.  Without this, a dead-end node treats its
                # just-used entrance as a fresh frontier and oscillates rather
                # than asking the graph for another unexplored branch.
                old_pose = Pose2D(x=float(previous['x']), y=float(previous['y']),
                                  z=float(previous['z']), yaw=float(previous['yaw']))
                self._mark_traversed_exit(self.current, old_pose)
        else:
            # A node is a frozen observation at its anchor pose/yaw.  Updating
            # its robot-frame semantic arrays while retaining the old yaw
            # corrupts the later circular alignment and creates false
            # structural events.  Live semantics are used by the controller,
            # but never mutate the persistent node descriptor.
            pass
        return self.current

    def _nearby_match(self, pose: Pose2D, role: np.ndarray) -> int | None:
        for i, node in enumerate(self.nodes):
            if math.hypot(pose.x - node['x'], pose.y - node['y']) > self.loop_radius_m:
                continue
            d = 1.0 - float(np.dot(role, node['role']) / max(np.linalg.norm(role) * np.linalg.norm(node['role']), 1e-8))
            if d <= self.role_merge_distance:
                return i
        return None

    @staticmethod
    def _semantic_distance(pose: Pose2D, semantic: dict[str, np.ndarray], node: dict[str, Any]) -> float:
        old = node['semantic']
        # Directional facts live in each observation's robot frame.  Align the
        # node observation into the current frame before deciding whether the
        # physical connection pattern changed.
        turn = int(round((pose.yaw - float(node['yaw'])) / (2 * math.pi) * 32))
        old_direction = np.roll(old['direction'], -turn)
        old_distance = np.roll(old['distance'], -turn)
        # Connection pattern and continuous structural regime must both be
        # able to trigger an event.  Raw surface density is deliberately absent.
        # Canonical role remains the node descriptor/loop-closure key, but a
        # role-distance change alone cannot manufacture a node: its continuous
        # reach components vary as the local observation horizon advances.
        new_exit = extract_exit_sectors(semantic['direction'], semantic['distance'])['soft']
        old_exit = extract_exit_sectors(old_direction, old_distance)['soft']
        connection_d = .5 * float(np.mean(np.abs(semantic['direction'] - old_direction))) + .5 * float(np.mean(np.abs(new_exit - old_exit)))
        regime_d = .5 * float(abs(float(semantic['count']) - float(old['count'])) / 4.0) + .5 * float(np.mean(np.abs(semantic['static'] - old['static'])))
        return max(connection_d, regime_d)

    @staticmethod
    def _node(pose: Pose2D, role: np.ndarray, semantic: dict[str, np.ndarray]) -> dict[str, Any]:
        # Exit *objects* are contiguous connected sectors derived from predicted
        # traversability and reach.  The dataset's raw exit mask is identical
        # to direction_traversable, so using it as an independent score would
        # double-count the same fact.
        sectors = extract_exit_sectors(semantic['direction'], semantic['distance'])
        frontier = sectors['soft'] * (.35 + .65 * semantic['distance'])
        return {'x': float(pose.x), 'y': float(pose.y), 'z': float(pose.z), 'yaw': float(pose.yaw),
                'role': role, 'semantic': semantic, 'frontier_score': frontier,
                'traversed_exit': np.zeros_like(frontier, dtype=np.float32),
                'blocked_exit': np.zeros_like(frontier, dtype=np.float32)}

    def _mark_traversed_exit(self, node_id: int, destination: Pose2D) -> None:
        self._mark_exit_state(node_id, destination, 'traversed_exit')

    def mark_blocked_exit(self, node_id: int, destination: Pose2D) -> None:
        """Blacklist a locally rejected direction without calling it traversed."""
        self._mark_exit_state(node_id, destination, 'blocked_exit')

    def _mark_exit_state(self, node_id: int, destination: Pose2D, field: str) -> None:
        node = self.nodes[node_id]
        angle = math.atan2(destination.y - node['y'], destination.x - node['x']) - node['yaw']
        sector = int(round((angle % (2 * math.pi)) / (2 * math.pi) * 32)) % 32
        # A predicted exit is a circular connected component, not an
        # individual angular bin.  Consuming only +/- one bin leaves the rest
        # of a wide opening marked as novel and causes endless re-traversal of
        # the same physical branch.
        mask = np.asarray(node['frontier_score']) >= .05
        if np.any(mask) and not mask[sector]:
            candidates = np.flatnonzero(mask)
            circular_distance = np.minimum((candidates - sector) % 32,
                                           (sector - candidates) % 32)
            nearest = int(candidates[int(np.argmin(circular_distance))])
            if int(np.min(circular_distance)) <= 4:
                sector = nearest
        if mask[sector]:
            consumed = {sector}
            for step in (-1, 1):
                index = (sector + step) % 32
                # A fully open junction can form one 360-degree soft
                # component even though it contains several physical exits.
                # Cap consumption at +/-3 bins (67.5 degrees total) so one
                # traversal cannot erase every branch of such a junction.
                hops = 0
                while mask[index] and index not in consumed and hops < 3:
                    consumed.add(index)
                    index = (index + step) % 32
                    hops += 1
            for index in consumed:
                node[field][index] = 1.0
        else:
            for offset in (-1, 0, 1):
                node[field][(sector + offset) % 32] = 1.0

    def shortest_paths(self) -> tuple[dict[int, float], dict[int, int]]:
        """Return metric costs and predecessors from the current graph node."""
        if self.current is None:
            return {}, {}
        costs = {self.current: 0.0}
        previous: dict[int, int] = {}
        pending = [self.current]
        while pending:
            u = min(pending, key=lambda i: costs[i]); pending.remove(u)
            for e in self.edges:
                if e['from'] == u: v = e['to']
                elif e['to'] == u: v = e['from']
                else: continue
                c = costs[u] + float(e['length_m'])
                if c < costs.get(v, float('inf')):
                    costs[v] = c; previous[v] = u; pending.append(v)
        return costs, previous

    def best_unexplored_exit(self, min_reward: float = .05,
                             include_current: bool = True) -> tuple[int, int, int, float] | None:
        """Choose a graph frontier and the next graph hop toward it.

        The semantic network proposes structural exits.  Graph state decides
        whether an exit has already been traversed and how expensive it is to
        return to it; this separation keeps map/pose out of canonical role.
        """
        if self.current is None:
            return None
        costs, previous = self.shortest_paths()
        best = None
        for i, node in enumerate(self.nodes):
            if not include_current and i == self.current:
                continue
            available = (node['frontier_score'] * (1.0 - node['traversed_exit']) *
                         (1.0 - node['blocked_exit']))
            sector = int(np.argmax(available)); reward = float(available[sector])
            if reward < min_reward or i not in costs:
                continue
            utility = reward / (1.0 + .08 * costs.get(i, float('inf')))
            if best is None or utility > best[2]:
                best = (i, sector, utility)
        if best is None:
            return None
        target, sector, utility = best
        next_hop = target
        while previous.get(next_hop) is not None and previous[next_hop] != self.current:
            next_hop = previous[next_hop]
        return target, next_hop, sector, utility

    def snapshot(self) -> dict[str, Any]:
        return {'node_count': len(self.nodes), 'edge_count': len(self.edges), 'current_node': self.current,
                'nodes': [{'id': i, **{k: v for k, v in n.items() if k != 'role'}, 'role': n['role']}
                          for i, n in enumerate(self.nodes)], 'edges': self.edges}


class SemanticTopologyNode:
    def __init__(self, checkpoint: Path, output: Path, shadow: bool, waypoint_distance_m: float, device: str) -> None:
        import rospy
        from geometry_msgs.msg import PointStamped
        from geometry_msgs.msg import Point
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import PointCloud2
        from visualization_msgs.msg import Marker, MarkerArray

        self.rospy = rospy
        self.PointStamped = PointStamped
        self.Point = Point
        self.Marker = Marker
        self.MarkerArray = MarkerArray
        self.shadow = shadow
        self.waypoint_distance_m = waypoint_distance_m
        self.output = output
        self.output.mkdir(parents=True, exist_ok=True)
        self.builder = SurfaceEvidenceBuilder(SurfaceEvidenceConfig())
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=8)
        self.map_voxels: set[tuple[int, int]] = set()
        self.map_resolution_m = 0.25
        self.last_map_save_time = -float('inf')
        self.coverage = ExplorationCoverage()
        self.latest_pose: Pose2D | None = None
        self.latest_stamp: float | None = None
        self.topology = Topology()
        self.last_waypoint_time = -float('inf')
        self.last_decision: dict[str, Any] | None = None
        self.smoothed_semantic: dict[str, np.ndarray] | None = None
        self.active_target: dict[str, float | int] | None = None
        self.active_target_node: int | None = None
        self.active_target_since = -float('inf')
        self.active_target_origin: tuple[float, float] | None = None
        # Progress must be measured over a recent window.  Measuring only the
        # displacement from the original pose lets a robot move once and then
        # remain stuck forever without rejecting the target.
        self.active_target_best_remaining = float('inf')
        self.active_target_last_progress_time = -float('inf')
        self.rejected_targets: list[dict[str, Any]] = []

        self.device = torch.device(device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.model = Semantic().to(self.device)
        state = torch.load(checkpoint, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state['state'])
        self.model.eval()
        self.checkpoint = str(checkpoint)
        self.pub = rospy.Publisher('/way_point', PointStamped, queue_size=1)
        self.marker_pub = rospy.Publisher('/semantic_topology/markers', MarkerArray, queue_size=1)
        rospy.Subscriber('/state_estimation_at_scan', Odometry, self.odom_cb, queue_size=50)
        rospy.Subscriber('/registered_scan', PointCloud2, self.scan_cb, queue_size=10)
        rospy.on_shutdown(self.write_snapshot)

    def odom_cb(self, msg: Any) -> None:
        self.latest_pose = pose_from_odometry(msg)
        self.latest_stamp = msg.header.stamp.to_sec()

    def scan_cb(self, msg: Any) -> None:
        if self.latest_pose is None:
            return
        scan_time = msg.header.stamp.to_sec()
        if self.latest_stamp is None or abs(scan_time - self.latest_stamp) > 0.25:
            self.rospy.logwarn_throttle(5.0, 'semantic topology: scan/odom not synchronized')
            return
        points = pointcloud2_xyz(msg)
        points = points[np.isfinite(points).all(axis=1)]
        if len(points) < 20:
            return
        keys = np.unique(np.floor(points[::4, :2] / self.map_resolution_m).astype(np.int32), axis=0)
        self.map_voxels.update((int(row[0]), int(row[1])) for row in keys)
        self.coverage.update(self.latest_pose, points)
        if scan_time - self.last_map_save_time >= 5.0:
            map_xy = np.asarray(list(self.map_voxels), dtype=np.float32) * self.map_resolution_m
            np.save(self.output / 'accumulated_map_xy.npy', map_xy)
            self.last_map_save_time = scan_time
        self.frames.append((scan_time, points))
        current, _ = self.builder.build([points], self.latest_pose)
        history = []
        valid = []
        for _, old_points in self.frames:
            tensor, _ = self.builder.build([old_points], self.latest_pose)
            history.append(tensor[[0, 2, 3]])
            valid.append(1.0)
        while len(history) < 8:
            history.insert(0, np.zeros((3, 100, 100), dtype=np.float32))
            valid.insert(0, 0.0)
        batch = {'cur': torch.from_numpy(current[[0, 2, 3]][None]).to(self.device),
                 'hist': torch.from_numpy(np.stack(history)[None]).to(self.device),
                 'valid': torch.from_numpy(np.asarray(valid, dtype=np.float32)[None]).to(self.device)}
        with torch.no_grad():
            prediction = self.model(batch)
        raw_semantic = {k: prediction[k][0].detach().cpu().numpy()
                        for k in ('direction', 'distance', 'area', 'exit', 'count', 'static')}
        # The network emits topology facts per scan.  A causal EMA suppresses
        # single-frame surface noise before those facts become persistent map
        # events; it cannot introduce any encoder-to-role bypass.
        if self.smoothed_semantic is None:
            self.smoothed_semantic = {k: np.asarray(v, dtype=np.float32).copy()
                                      for k, v in raw_semantic.items()}
        else:
            for key, value in raw_semantic.items():
                self.smoothed_semantic[key] = (.75 * self.smoothed_semantic[key] +
                                               .25 * np.asarray(value, dtype=np.float32))
        semantic = {k: v.copy() for k, v in self.smoothed_semantic.items()}
        role = canonical_roles_numpy(semantic['direction'][None], semantic['distance'][None], semantic['exit'][None])[0]
        node_id = self.topology.update(self.latest_pose, role, semantic)
        decision = self.choose_waypoint(semantic, node_id, scan_time)
        self.last_decision = {'time': float(scan_time), 'node_id': node_id,
                              'pose': {'x': self.latest_pose.x, 'y': self.latest_pose.y,
                                       'z': self.latest_pose.z, 'yaw': self.latest_pose.yaw},
                              'role': role, 'semantic': semantic, 'decision': decision}
        trace = {'time': float(scan_time), 'node_id': int(node_id),
                 'node_count': len(self.topology.nodes), 'edge_count': len(self.topology.edges),
                 'pose': self.last_decision['pose'], 'decision': decision}
        with (self.output / 'decision_trace.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(json_safe(trace), separators=(',', ':')) + '\n')
        if not self.shadow and scan_time - self.last_waypoint_time >= 1.0:
            self.publish_waypoint(msg.header, decision)
            self.last_waypoint_time = scan_time
        self.publish_markers(msg.header)
        self.write_snapshot()

    def choose_waypoint(self, semantic: dict[str, np.ndarray], node_id: int, scan_time: float) -> dict[str, float | int]:
        # A global target is state, not a fresh classification each scan.  This
        # prevents noisy sector scores from rotating the robot in place.
        if self.active_target is not None:
            remaining = math.hypot(float(self.active_target['x']) - self.latest_pose.x,
                                   float(self.active_target['y']) - self.latest_pose.y)
            elapsed = scan_time - self.active_target_since
            progress = (math.hypot(self.latest_pose.x - self.active_target_origin[0],
                                   self.latest_pose.y - self.active_target_origin[1])
                        if self.active_target_origin is not None else 0.0)
            # A 25 cm decrease is meaningful at the 25 cm coverage resolution.
            # Reset the watchdog whenever such progress occurs, then reject an
            # unreachable semantic exit after eight seconds without progress.
            if remaining <= self.active_target_best_remaining - .25:
                self.active_target_best_remaining = remaining
                self.active_target_last_progress_time = scan_time
            stalled_for = scan_time - self.active_target_last_progress_time
            rejected = remaining > .8 and elapsed >= 8.0 and stalled_for >= 8.0
            if rejected and self.active_target_node is not None and self.active_target.get('mode') in (
                    'local_semantic_frontier', 'graph_frontier_exit'):
                destination = Pose2D(x=float(self.active_target['x']), y=float(self.active_target['y']),
                                     z=float(self.active_target['z']), yaw=0.0)
                self.topology.mark_blocked_exit(self.active_target_node, destination)
                self.rejected_targets.append({
                    'time': float(scan_time), 'node_id': int(self.active_target_node),
                    'mode': self.active_target['mode'], 'sector': int(self.active_target['sector']),
                    'elapsed_sec': float(elapsed), 'progress_m': float(progress),
                    'remaining_m': float(remaining), 'stalled_for_sec': float(stalled_for),
                    'best_remaining_m': float(self.active_target_best_remaining)})
                self.active_target = None
            elif remaining > .8 and elapsed < 20.0:
                return self.active_target
        # Traversability is the hard safety-facing signal; exit and reachable
        # distance rank globally useful directions.  Local planner remains the
        # collision authority and may reject the waypoint.
        direction = semantic['direction']
        reach = semantic['distance']
        sectors = extract_exit_sectors(direction, reach)
        score = np.zeros(32, dtype=np.float32)
        gains = np.zeros(32, dtype=np.float32)
        for center, length, valid in zip(sectors['centers'], sectors['lengths'], sectors['valid_mask']):
            if not valid: continue
            candidate = int(round(float(center) / (2 * math.pi) * 32)) % 32
            world_angle = self.latest_pose.yaw + sector_angle(candidate)
            gain = self.coverage.unknown_gain(self.latest_pose, world_angle, 5. + 7. * float(length))
            gains[candidate] = gain
            score[candidate] = gain * (.35 + .65 * float(length))
        # The graph stores which structural exits already became traversed
        # edges. Prefer an unconsumed predicted exit at the current structural
        # node; only fall back to the local score if none remains.
        node = self.topology.nodes[node_id]
        # ``traversed_exit`` is stored in the node's original robot frame;
        # rotate it into the *current* robot frame before masking fresh local
        # semantics.  Never use an old node's sector directly as a current
        # control direction.
        turn = int(round((self.latest_pose.yaw - node['yaw']) / (2 * math.pi) * 32))
        consumed = node['traversed_exit'][(np.arange(32) + turn) % 32]
        blocked = node['blocked_exit'][(np.arange(32) + turn) % 32]
        available = score * (1.0 - consumed) * (1.0 - blocked)
        # Prefer a genuinely novel exit at the current structural place.  If
        # none remains, route over already traversed graph edges to the next
        # node on the path to the best remote semantic frontier.
        if float(np.max(available)) < .04:
            # Visibility is not traversal: a lidar ray can see well into a
            # branch without the robot exploring it.  Only real graph edges
            # consume exits, and the exhausted current node is excluded from
            # remote frontier selection.
            remote = self.topology.best_unexplored_exit(include_current=True)
            if remote is not None:
                frontier_node, next_hop, frontier_sector, utility = remote
                if frontier_node == node_id:
                    # We have reached the graph frontier.  Execute the stored
                    # untraversed structural exit itself; merely routing to the
                    # node coordinate causes endless arrival/backtrack loops.
                    frontier = self.topology.nodes[frontier_node]
                    angle = float(frontier['yaw']) + sector_angle(frontier_sector)
                    length = float(self.waypoint_distance_m)
                    chosen = {'sector': int(round(((angle - self.latest_pose.yaw) % (2 * math.pi)) /
                                                   (2 * math.pi) * 32)) % 32,
                              'score': 0.0, 'unknown_gain': 0.0,
                              'graph_utility': float(utility), 'distance_m': length,
                              'x': float(self.latest_pose.x + length * math.cos(angle)),
                              'y': float(self.latest_pose.y + length * math.sin(angle)),
                              'z': float(self.latest_pose.z), 'mode': 'graph_frontier_exit',
                              'frontier_node': int(frontier_node), 'next_hop': int(next_hop),
                              'frontier_sector': int(frontier_sector)}
                    self.active_target, self.active_target_node, self.active_target_since = chosen, node_id, scan_time
                    self.active_target_origin = (self.latest_pose.x, self.latest_pose.y)
                    self.active_target_best_remaining = math.hypot(chosen['x'] - self.latest_pose.x,
                                                                  chosen['y'] - self.latest_pose.y)
                    self.active_target_last_progress_time = scan_time
                    return chosen
                if next_hop != node_id:
                    target = self.topology.nodes[next_hop]
                    angle = math.atan2(float(target['y']) - self.latest_pose.y,
                                       float(target['x']) - self.latest_pose.x)
                    remaining = math.hypot(float(target['x']) - self.latest_pose.x,
                                           float(target['y']) - self.latest_pose.y)
                    length = float(np.clip(remaining, 1.0, self.waypoint_distance_m))
                    chosen = {'sector': int(round(((angle - self.latest_pose.yaw) % (2 * math.pi)) /
                                                   (2 * math.pi) * 32)) % 32,
                              'score': 0.0, 'unknown_gain': 0.0,
                              'graph_utility': float(utility), 'distance_m': length,
                              'x': float(self.latest_pose.x + length * math.cos(angle)),
                              'y': float(self.latest_pose.y + length * math.sin(angle)),
                              'z': float(self.latest_pose.z), 'mode': 'graph_backtrack',
                              'frontier_node': int(frontier_node), 'next_hop': int(next_hop),
                              'frontier_sector': int(frontier_sector)}
                    self.active_target, self.active_target_node, self.active_target_since = chosen, node_id, scan_time
                    self.active_target_origin = (self.latest_pose.x, self.latest_pose.y)
                    self.active_target_best_remaining = math.hypot(chosen['x'] - self.latest_pose.x,
                                                                  chosen['y'] - self.latest_pose.y)
                    self.active_target_last_progress_time = scan_time
                    return chosen

            # Do not turn an all-zero semantic score into sector zero and
            # drive blindly into a wall.  Holding position is an explicit,
            # auditable failure/completion state.
            chosen = {'sector': -1, 'score': 0.0, 'unknown_gain': 0.0,
                      'graph_utility': 0.0, 'distance_m': 0.0,
                      'x': float(self.latest_pose.x), 'y': float(self.latest_pose.y),
                      'z': float(self.latest_pose.z), 'mode': 'no_semantic_frontier'}
            self.active_target, self.active_target_node, self.active_target_since = chosen, node_id, scan_time
            self.active_target_origin = (self.latest_pose.x, self.latest_pose.y)
            self.active_target_best_remaining = 0.0
            self.active_target_last_progress_time = scan_time
            return chosen

        sector = int(np.argmax(available if float(np.max(available)) > 1e-5 else score))
        graph_utility = float(available[sector])
        angle = self.latest_pose.yaw + sector_angle(sector)
        length = float(np.clip(1.5 + 3.0 * reach[sector], 1.5, self.waypoint_distance_m))
        chosen = {'sector': sector, 'score': float(score[sector]), 'unknown_gain': float(gains[sector]),
                  'graph_utility': graph_utility, 'distance_m': length,
                'x': float(self.latest_pose.x + length * math.cos(angle)),
                'y': float(self.latest_pose.y + length * math.sin(angle)), 'z': float(self.latest_pose.z),
                'mode': 'local_semantic_frontier'}
        self.active_target, self.active_target_node, self.active_target_since = chosen, node_id, scan_time
        self.active_target_origin = (self.latest_pose.x, self.latest_pose.y)
        self.active_target_best_remaining = math.hypot(chosen['x'] - self.latest_pose.x,
                                                      chosen['y'] - self.latest_pose.y)
        self.active_target_last_progress_time = scan_time
        return chosen

    def publish_waypoint(self, header: Any, decision: dict[str, float | int]) -> None:
        msg = self.PointStamped()
        msg.header.stamp = header.stamp
        msg.header.frame_id = 'map'
        msg.point.x, msg.point.y, msg.point.z = decision['x'], decision['y'], decision['z']
        self.pub.publish(msg)

    def publish_markers(self, header: Any) -> None:
        """Live RViz view of the semantic graph; no model features are exposed."""
        arr = self.MarkerArray(); frame = header.frame_id or 'map'
        edges = self.Marker(); edges.header = header; edges.header.frame_id = frame
        edges.ns = 'semantic_topology'; edges.id = 0; edges.type = self.Marker.LINE_LIST; edges.action = self.Marker.ADD
        edges.scale.x = .12; edges.color.r = .15; edges.color.g = .55; edges.color.b = 1.; edges.color.a = 1.
        for edge in self.topology.edges:
            for nid in (edge['from'], edge['to']):
                n = self.topology.nodes[nid]; p = self.Point(); p.x, p.y, p.z = n['x'], n['y'], n['z']; edges.points.append(p)
        arr.markers.append(edges)
        for i, n in enumerate(self.topology.nodes):
            m = self.Marker(); m.header = header; m.header.frame_id = frame; m.ns = 'semantic_nodes'; m.id = i
            m.type = self.Marker.SPHERE; m.action = self.Marker.ADD; m.pose.position.x, m.pose.position.y, m.pose.position.z = n['x'], n['y'], n['z']
            m.pose.orientation.w = 1.; m.scale.x = m.scale.y = m.scale.z = .7
            m.color.r, m.color.g, m.color.b, m.color.a = ((1., .15, .15, 1.) if i == self.topology.current else (.15, .9, .35, .95))
            arr.markers.append(m)
        self.marker_pub.publish(arr)

    def write_snapshot(self) -> None:
        payload = {'checkpoint': self.checkpoint, 'mode': 'shadow' if self.shadow else 'closed_loop',
                   'updated_unix_sec': time.time(), 'topology': self.topology.snapshot(), 'last_decision': self.last_decision}
        payload['execution_feedback'] = {'rejected_target_count': len(self.rejected_targets),
                                         'rejected_targets': self.rejected_targets}
        (self.output / 'topology_snapshot.json').write_text(json.dumps(json_safe(payload), indent=2), encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--shadow', action='store_true', help='infer/build topology but do not control /way_point')
    parser.add_argument('--waypoint-distance-m', type=float, default=4.0)
    parser.add_argument('--device', default='auto', choices=('auto', 'cpu', 'cuda'))
    args = parser.parse_args()
    import rospy
    rospy.init_node('semantic_topology_global_node', anonymous=False)
    SemanticTopologyNode(args.checkpoint, args.output, args.shadow, args.waypoint_distance_m, args.device)
    rospy.loginfo('semantic topology global node started (%s)', 'shadow' if args.shadow else 'closed loop')
    rospy.spin()


if __name__ == '__main__':
    main()
