from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_topology_replay import (
    offline_topology_scientific_gate,
    replay_typed_observation_world,
)
from mtare_topo.semantics.exit_only_geometry_observation import exit_only_observation_from_arrays
from mtare_topo.semantics.geometric_semantics import EVENT_NAMES, ExitGeometryToken, GeometricSemanticObservation
from mtare_topo.topology.gse_graph import GSEGraphConfig
from mtare_topo.topology.gse_rule_graph import RuleAssociatedGeometryEventGraph


def _config(**changes) -> GSEGraphConfig:
    values = {
        "stable_event_frames": 1,
        "minimum_event_travel_m": 1.0,
        "metric_anchor_interval_m": 100.0,
        "event_probability_threshold": 0.5,
        "maximum_uncertainty": 0.5,
        "association_radius_m": 20.0,
        "descriptor_minimum_similarity": 0.9,
        "exit_heading_tolerance_deg": 20.0,
        "exit_descriptor_minimum_similarity": 0.9,
        "exit_width_log_tolerance": 0.3,
        "exit_vertical_profile_tolerance": 1.0,
        "maximum_exit_count_difference": 1,
        "ambiguity_similarity_margin": 0.1,
    }
    values.update(changes)
    return GSEGraphConfig(**values)


def _observation(descriptor: tuple[float, ...], event: str = "junction") -> GeometricSemanticObservation:
    probabilities = {name: 0.0 for name in EVENT_NAMES}
    probabilities[event] = 1.0
    return GeometricSemanticObservation(
        event_probabilities=probabilities,
        local_axis=(1.0, 0.0, 0.0),
        width_m=4.0,
        height_m=3.0,
        slope_deg=0.0,
        curvature_per_m=0.0,
        place_descriptor=descriptor,
        exit_tokens=(
            ExitGeometryToken(
                heading_robot_deg=0.0,
                opening_width_m=3.0,
                vertical_profile=(0.0, 0.0, 0.0, 0.0),
                descriptor=(1.0, 0.0),
                confidence=1.0,
            ),
        ),
        uncertainty=0.0,
    )


def test_rule_association_ignores_learned_descriptor_and_records_method() -> None:
    graph = RuleAssociatedGeometryEventGraph(_config())
    graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0)))
    graph.update(frame_index=1, route_arc_m=15.0, xyz_m=(15, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0), "corridor"))
    graph.update(frame_index=2, route_arc_m=30.0, xyz_m=(30, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0)))
    graph.update(frame_index=3, route_arc_m=45.0, xyz_m=(15, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0), "corridor"))
    result = graph.update(frame_index=4, route_arc_m=60.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=_observation((-1.0, 0.0)))
    assert result["reason"] == "rule_association"
    assert result["node_id"] == 0


def test_rule_association_rejects_equal_distance_candidates_as_ambiguous() -> None:
    graph = RuleAssociatedGeometryEventGraph(_config())
    graph.update(frame_index=0, route_arc_m=0.0, xyz_m=(0, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0)))
    graph.update(frame_index=1, route_arc_m=15.0, xyz_m=(15, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0), "corridor"))
    graph.update(frame_index=2, route_arc_m=30.0, xyz_m=(30, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0)))
    graph.update(frame_index=3, route_arc_m=45.0, xyz_m=(30, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0), "corridor"))
    result = graph.update(frame_index=4, route_arc_m=60.0, xyz_m=(15, 0, 0), yaw_deg=0, observation=_observation((1.0, 0.0)))
    assert result["reason"] == "ambiguous_association"
    assert result["association_candidate_count"] == 2
    assert graph.nodes[result["node_id"]]["association_status"] == "provisional"


def test_exit_only_adapter_exposes_only_role_direction_and_neutral_geometry() -> None:
    direction = np.full((1, 720), -20.0, dtype=np.float32)
    direction[0, 0:5] = 20.0
    direction[0, 358:363] = 20.0
    arrays = {
        "event_probabilities": np.asarray(((0.1, 0.8, 0.1, 0.0, 0.0),), dtype=np.float16),
        "direction_logits": direction,
        "z_role": np.asarray(((1.0, 2.0, 3.0),), dtype=np.float16),
    }
    observation = exit_only_observation_from_arrays(arrays, 0)
    assert observation.event.value == "junction"
    assert len(observation.exit_tokens) == 2
    assert observation.width_m == observation.height_m == 1.0
    assert observation.slope_deg == observation.curvature_per_m == 0.0
    assert np.isclose(np.linalg.norm(observation.place_descriptor), 1.0)


def _method_summary(
    node_f1: float,
    edge_f1: float,
    *,
    merges: int = 20,
    precision: float = 1.0,
    false_rate: float = 0.0,
    component_bias: float = 0.0,
    cycle_bias: float = 0.0,
) -> dict:
    return {
        "selected_aggregate": {
            "node": {"f1": node_f1},
            "edge": {"f1": edge_f1},
            "association": {
                "merge_attempts": merges,
                "precision": precision,
                "false_loop_merge_rate": false_rate,
            },
            "invariants": {
                "connected_component_mean_signed_error": component_bias,
                "cycle_rank_mean_signed_error": cycle_bias,
            },
        }
    }


def test_offline_topology_gate_compares_each_metric_to_its_strongest_baseline() -> None:
    methods = {
        "gse_learned_association": _method_summary(0.86, 0.81),
        "exit_only_rule_graph": _method_summary(0.80, 0.40),
        "nonlearning_geometry_rule_graph": _method_summary(0.50, 0.75),
    }
    gate = offline_topology_scientific_gate(methods)
    assert gate["passed"] is True
    assert np.isclose(gate["node_f1_gain_over_strongest_main_baseline"], 0.06)
    assert np.isclose(gate["edge_f1_gain_over_strongest_main_baseline"], 0.06)


def test_offline_topology_gate_rejects_vacuous_or_biased_graph() -> None:
    baselines = {
        "exit_only_rule_graph": _method_summary(0.60, 0.60),
        "nonlearning_geometry_rule_graph": _method_summary(0.60, 0.60),
    }
    vacuous = {
        "gse_learned_association": _method_summary(0.90, 0.90, merges=0),
        **baselines,
    }
    biased = {
        "gse_learned_association": _method_summary(0.90, 0.90, component_bias=-0.26),
        **baselines,
    }
    assert offline_topology_scientific_gate(vacuous)["passed"] is False
    assert offline_topology_scientific_gate(biased)["passed"] is False


def test_rule_replay_retains_identity_contamination_after_a_false_merge() -> None:
    traversals = [
        {"traversal_id": "e0:d0", "edge_id": "e0", "from_node_id": "n0", "to_node_id": "n1", "length_m": 12.0},
        {"traversal_id": "e0:d1", "edge_id": "e0", "from_node_id": "n1", "to_node_id": "n0", "length_m": 12.0},
    ]
    teacher = [
        {"traversal_id": "e0:d0", "edge_id": "e0", "traversal_arc_m": 2.0, "canonical_edge_arc_m": 2.0, "global_sequence_index": 0, "event": "junction", "identity": "a"},
        {"traversal_id": "e0:d0", "edge_id": "e0", "traversal_arc_m": 4.0, "canonical_edge_arc_m": 4.0, "global_sequence_index": 1, "event": "corridor", "identity": None},
        {"traversal_id": "e0:d0", "edge_id": "e0", "traversal_arc_m": 8.0, "canonical_edge_arc_m": 8.0, "global_sequence_index": 2, "event": "junction", "identity": "b"},
        {"traversal_id": "e0:d1", "edge_id": "e0", "traversal_arc_m": 1.0, "canonical_edge_arc_m": 11.0, "global_sequence_index": 3, "event": "corridor", "identity": None},
        {"traversal_id": "e0:d1", "edge_id": "e0", "traversal_arc_m": 4.0, "canonical_edge_arc_m": 8.0, "global_sequence_index": 4, "event": "junction", "identity": "b"},
        {"traversal_id": "e0:d1", "edge_id": "e0", "traversal_arc_m": 6.0, "canonical_edge_arc_m": 6.0, "global_sequence_index": 5, "event": "corridor", "identity": None},
        {"traversal_id": "e0:d1", "edge_id": "e0", "traversal_arc_m": 8.0, "canonical_edge_arc_m": 4.0, "global_sequence_index": 6, "event": "junction", "identity": "a"},
    ]
    poses = {
        0: {"axis_xyz_m": (0.0, 0.0, 0.0), "yaw_deg": 0.0},
        1: {"axis_xyz_m": (2.5, 0.0, 0.0), "yaw_deg": 0.0},
        2: {"axis_xyz_m": (5.0, 0.0, 0.0), "yaw_deg": 0.0},
        3: {"axis_xyz_m": (7.0, 0.0, 0.0), "yaw_deg": 180.0},
        4: {"axis_xyz_m": (5.0, 0.0, 0.0), "yaw_deg": 180.0},
        5: {"axis_xyz_m": (2.5, 0.0, 0.0), "yaw_deg": 180.0},
        6: {"axis_xyz_m": (0.0, 0.0, 0.0), "yaw_deg": 180.0},
    }
    observations = {
        index: _observation((1.0, 0.0), "corridor" if index in {1, 3, 5} else "junction")
        for index in range(7)
    }
    result = replay_typed_observation_world(
        traversal_records=traversals,
        teacher_observations=teacher,
        pose_by_sequence_index=poses,
        observation_by_sequence_index=observations,
        graph_config=_config(association_radius_m=20.0, ambiguity_similarity_margin=0.0),
        association_mode="rule",
    )
    assert result["identity_conflicts"]
    assert all(values == ["a", "b"] for values in result["identity_conflicts"].values())
    assert result["association_metrics"]["correct_merges"] == 0
    assert result["association_metrics"]["false_merges"] >= 2
