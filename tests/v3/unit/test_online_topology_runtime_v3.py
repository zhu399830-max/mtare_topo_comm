from __future__ import annotations

import json
import time

import numpy as np

from mtare_topo.integration.online_topology_runtime import SemanticPrediction
from mtare_topo.integration.online_topology_runtime_v3 import OnlineTopologyPlannerRuntimeV3
from mtare_topo.planning.topological_frontier import (
    FrontierIdentity,
    GlobalTarget,
    TopologicalPlannerConfig,
)
from mtare_topo.topology.causal_graph_v2 import CausalGraphConfig
from mtare_topo.topology.frontier_execution_feedback import evaluate_frontier_execution_feedback
from mtare_topo.evaluation.topology_frontier_attempt_evidence_v1r3 import (
    audit_frontier_attempt_evidence_v1r3,
)


def target(stub_index=0):
    return GlobalTarget(
        "TARGET", "frontier_exit", (4.0, 0.0, 0.0),
        FrontierIdentity(0, stub_index), 0, 0, (0,), 0.0, 1.0, 1.0,
        "best_deterministic_rule_based_frontier",
    )


def graph():
    return {
        "nodes": [
            {"id": 0, "xyz_m": [0.0, 0.0, 0.0], "exit_stubs": [
                {"heading_world_deg": 0.0}, {"heading_world_deg": 180.0},
            ]},
            {"id": 1, "xyz_m": [-8.0, 0.0, 0.0], "exit_stubs": [
                {"heading_world_deg": 0.0},
            ]},
        ]
    }


def prediction(headings, *, role, count):
    logits = np.full(720, -20.0, dtype=np.float32)
    for heading in headings:
        logits[int(round(heading * 2)) % 720] = 20.0
    counts = np.zeros(6, dtype=np.float32); counts[count - 1] = 1.0
    roles = np.zeros(3, dtype=np.float32); roles[role] = 1.0
    return SemanticPrediction(logits, counts, roles, np.ones(128, dtype=np.float32) / np.sqrt(128.0))


def runtime():
    return OnlineTopologyPlannerRuntimeV3(
        CausalGraphConfig(
            stable_frames=1, minimum_event_travel_m=6.0, loop_merge_radius_m=4.0,
            branch_heading_merge_deg=20.0, turn_event_deg=45.0,
            distance_anchor_interval_m=20.0,
        ),
        TopologicalPlannerConfig(4.0, 20.0, 0.25, 0.05),
    )


def update(instance, stamp, xyz, value):
    return instance.update(
        stamp_sec=stamp, sensor_xyz_m=xyz,
        sensor_orientation_xyzw=(0.0, 0.0, 0.0, 1.0), prediction=value,
        cycle_started_monotonic=time.monotonic(),
    )


def test_feedback_classifies_matched_and_divergent_without_new_threshold():
    matched = evaluate_frontier_execution_feedback(
        target(1), previous_node_id=0,
        graph_update={"node_id": 1, "reason": "stable_terminal"}, graph_snapshot=graph(),
    )
    divergent = evaluate_frontier_execution_feedback(
        target(0), previous_node_id=0,
        graph_update={"node_id": 1, "reason": "stable_terminal"}, graph_snapshot=graph(),
    )
    assert matched.outcome == "matched_verified_departure" and not matched.record_rejection
    assert divergent.outcome == "divergent_verified_departure" and divergent.record_rejection
    assert divergent.actual_departure_stub_index == 1


def test_same_node_loop_merge_is_a_rejection_but_no_event_is_not():
    loop = evaluate_frontier_execution_feedback(
        target(), previous_node_id=0,
        graph_update={"node_id": 0, "reason": "loop_merge"}, graph_snapshot=graph(),
    )
    quiet = evaluate_frontier_execution_feedback(
        target(), previous_node_id=0,
        graph_update={"node_id": 0, "reason": "no_event"}, graph_snapshot=graph(),
    )
    assert loop.classified and loop.record_rejection and loop.outcome == "same_node_loop_merge"
    assert not quiet.classified and not quiet.record_rejection


def test_runtime_records_divergent_graph_event_in_existing_retry_ledger():
    instance = runtime()
    corridor = prediction((0, 180), role=0, count=2)
    terminal = prediction((0,), role=2, count=1)
    first = update(instance, 1.0, (0.0, 0.0, 0.0), corridor)
    assert first.target.frontier == FrontierIdentity(0, 0)
    second = update(instance, 2.0, (-8.0, 0.0, 0.0), terminal)
    feedback = second.graph_update["frontier_execution_feedback"]
    assert feedback["outcome"] == "divergent_verified_departure"
    assert instance.retry_ledger.counts()[FrontierIdentity(0, 0)] == 1
    assert instance.snapshot()["frontier_execution_rejection_count"] == 1
    assert instance.planner.config.retry_penalty == 0.25


def test_runtime_v3_is_deterministic_for_same_causal_sequence():
    def replay():
        instance = runtime()
        update(instance, 1.0, (0.0, 0.0, 0.0), prediction((0, 180), role=0, count=2))
        update(instance, 2.0, (-8.0, 0.0, 0.0), prediction((0,), role=2, count=1))
        return instance.snapshot()
    assert replay() == replay()


def test_online_feedback_matches_offline_exact_traversal_audit():
    instance = runtime()
    rows = json.loads(json.dumps([
        update(instance, 1.0, (0.0, 0.0, 0.0), prediction((0, 180), role=0, count=2)).to_dict(),
        update(instance, 2.0, (-8.0, 0.0, 0.0), prediction((0,), role=2, count=1)).to_dict(),
    ]))
    offline = audit_frontier_attempt_evidence_v1r3(rows, instance.snapshot()["graph"])
    assert offline["verified_traversal_event_bijection_count"] == 1
    assert offline["outcome_counts"] == {"divergent_verified_departure": 1}
    assert rows[1]["graph_update"]["frontier_execution_feedback"]["outcome"] == offline["events"][0]["outcome"]
