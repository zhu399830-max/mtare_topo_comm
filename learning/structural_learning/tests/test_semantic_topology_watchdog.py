"""Regression test for a semantic waypoint that moves once and then stalls."""
from types import SimpleNamespace

import numpy as np

from learning.structural_learning.tools.semantic_topology_global_node import Pose2D, SemanticTopologyNode


class FakeTopology:
    def __init__(self):
        self.nodes = [{
            'yaw': 0.0,
            'traversed_exit': np.zeros(32, dtype=np.float32),
            'blocked_exit': np.zeros(32, dtype=np.float32),
        }]
        self.marked = []

    def mark_blocked_exit(self, node_id, destination):
        self.marked.append((node_id, destination.x, destination.y))

    def best_unexplored_exit(self, include_current=True):
        return None


def test_partial_progress_then_stall_is_rejected():
    node = SemanticTopologyNode.__new__(SemanticTopologyNode)
    node.active_target = {
        'x': 4.0, 'y': 0.0, 'z': 0.0,
        'mode': 'local_semantic_frontier', 'sector': 0,
    }
    node.active_target_node = 0
    node.active_target_since = 0.0
    node.active_target_origin = (0.0, 0.0)
    node.active_target_best_remaining = 4.0
    node.active_target_last_progress_time = 0.0
    node.rejected_targets = []
    node.latest_pose = Pose2D(0.6, 0.0, 0.0, 0.0)
    node.topology = FakeTopology()
    node.coverage = SimpleNamespace(unknown_gain=lambda *args: 0.0)
    node.waypoint_distance_m = 4.0
    semantic = {
        'direction': np.zeros(32, dtype=np.float32),
        'distance': np.zeros(32, dtype=np.float32),
    }

    # Initial movement resets the recent-progress watchdog.
    assert node.choose_waypoint(semantic, 0, 1.0)['mode'] == 'local_semantic_frontier'
    assert node.active_target_best_remaining < 3.5

    # Staying at that partially advanced pose for nine seconds must reject it.
    node.latest_pose = Pose2D(0.6, 0.0, 0.0, 0.0)
    assert node.choose_waypoint(semantic, 0, 10.0)['mode'] == 'no_semantic_frontier'
    assert len(node.topology.marked) == 1
    assert len(node.rejected_targets) == 1
    assert node.rejected_targets[0]['stalled_for_sec'] >= 8.0
