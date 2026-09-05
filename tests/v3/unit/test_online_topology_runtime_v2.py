from __future__ import annotations

import time

import numpy as np

from mtare_topo.integration.online_topology_runtime import SemanticPrediction
from mtare_topo.integration.online_topology_runtime_v2 import OnlineTopologyPlannerRuntimeV2
from mtare_topo.planning.topological_frontier import TopologicalPlannerConfig
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig


def prediction(headings, *, role, count):
    logits = np.full(720, -20.0, dtype=np.float32)
    for heading in headings:
        logits[int(round(heading * 2)) % 720] = 20.0
    counts = np.zeros(6, dtype=np.float32); counts[count - 1] = 1.0
    roles = np.zeros(3, dtype=np.float32); roles[role] = 1.0
    return SemanticPrediction(logits, counts, roles, np.ones(128, dtype=np.float32) / np.sqrt(128.0))


def runtime():
    return OnlineTopologyPlannerRuntimeV2(
        CausalGraphConfig(
            stable_frames=1,
            minimum_event_travel_m=6.0,
            loop_merge_radius_m=4.0,
            branch_heading_merge_deg=20.0,
            turn_event_deg=45.0,
            distance_anchor_interval_m=20.0,
        ),
        TopologicalPlannerConfig(4.0, 20.0, 0.25, 0.05),
    )


def update(instance, stamp, xyz, value):
    return instance.update(
        stamp_sec=stamp,
        sensor_xyz_m=xyz,
        sensor_orientation_xyzw=(0.0, 0.0, 0.0, 1.0),
        prediction=value,
        cycle_started_monotonic=time.monotonic(),
    )


def test_verified_backtrack_reanchors_and_selects_remaining_exit() -> None:
    instance = runtime()
    corridor = prediction((0, 180), role=0, count=2)
    terminal = prediction((180,), role=2, count=1)
    first = update(instance, 1.0, (0.0, 0.0, 0.0), corridor)
    second = update(instance, 2.0, (8.0, 0.0, 0.0), terminal)
    assert first.target.mode == "frontier_exit"
    assert second.graph_update["reason"] == "stable_terminal"
    assert second.target.mode == "graph_backtrack"
    assert second.target.next_hop_node_id == 0

    third = update(instance, 3.0, (3.9, 0.0, 0.0), corridor)
    state = instance.snapshot()
    assert third.graph_update["reason"] == "verified_backtrack_reanchor"
    assert third.graph_update["previous_node_id"] == 1
    assert third.graph_update["node_id"] == 0
    assert third.graph_update["eligibility"]["eligible"] is True
    assert third.target.mode == "frontier_exit"
    assert third.target.frontier.node_id == 0
    assert state["verified_reanchor_count"] == 1
    assert state["graph"]["current_node"] == 0
    assert state["graph"]["edges"][0]["verified_traversal_count"] == 2
    reverse = state["graph"]["edges"][0]["traversals"][-1]
    assert reverse["from"] == 1 and reverse["to"] == 0
    assert reverse["length_m"] > 0 and reverse["trace_frame_count"] >= 2


def test_same_pose_after_reanchor_is_idempotent() -> None:
    instance = runtime()
    corridor = prediction((0, 180), role=0, count=2)
    terminal = prediction((180,), role=2, count=1)
    update(instance, 1.0, (0.0, 0.0, 0.0), corridor)
    update(instance, 2.0, (8.0, 0.0, 0.0), terminal)
    update(instance, 3.0, (3.9, 0.0, 0.0), corridor)
    fourth = update(instance, 4.0, (3.9, 0.0, 0.0), corridor)
    state = instance.snapshot()
    assert fourth.graph_update["reason"] == "no_event"
    assert state["verified_reanchor_count"] == 1
    assert state["graph"]["edges"][0]["verified_traversal_count"] == 2


def test_runtime_v2_is_deterministic_for_same_causal_sequence() -> None:
    def replay():
        instance = runtime()
        corridor = prediction((0, 180), role=0, count=2)
        terminal = prediction((180,), role=2, count=1)
        update(instance, 1.0, (0.0, 0.0, 0.0), corridor)
        update(instance, 2.0, (8.0, 0.0, 0.0), terminal)
        update(instance, 3.0, (3.9, 0.0, 0.0), corridor)
        return instance.snapshot()

    first, second = replay(), replay()
    assert first == second
