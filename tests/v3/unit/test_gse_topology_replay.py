from __future__ import annotations

from mtare_topo.evaluation.gse_topology_replay import (
    aggregate_gse_world_replays,
    collapse_to_structural_graph,
    directed_euler_order,
    gse_graph_parameter_grid,
    rule_graph_parameter_grid,
    strict_predicted_node_mapping,
    teacher_structural_graph,
    replay_gse_world,
    select_baseline_graph_sweep,
    select_gse_graph_sweep,
)
from mtare_topo.topology.gse_graph import GSEGraphConfig
import numpy as np


def test_directed_euler_order_is_deterministic_continuous_and_complete() -> None:
    records = [
        {"traversal_id": "e0:d0", "edge_id": "e0", "from_node_id": "a", "to_node_id": "b"},
        {"traversal_id": "e0:d1", "edge_id": "e0", "from_node_id": "b", "to_node_id": "a"},
        {"traversal_id": "e1:d0", "edge_id": "e1", "from_node_id": "b", "to_node_id": "c"},
        {"traversal_id": "e1:d1", "edge_id": "e1", "from_node_id": "c", "to_node_id": "b"},
    ]
    first = directed_euler_order(records)
    assert first == directed_euler_order(list(reversed(records)))
    assert set(first) == {record["traversal_id"] for record in records}


def test_teacher_structural_graph_collapses_corridor_and_deduplicates_reverse_views() -> None:
    rows = [
        {"edge_id": "e0", "canonical_edge_arc_m": 0.0, "event": "terminal", "identity": "a"},
        {"edge_id": "e0", "canonical_edge_arc_m": 1.0, "event": "terminal", "identity": "a"},
        {"edge_id": "e0", "canonical_edge_arc_m": 5.0, "event": "corridor", "identity": None},
        {"edge_id": "e0", "canonical_edge_arc_m": 9.0, "event": "junction", "identity": "b"},
        {"edge_id": "e0", "canonical_edge_arc_m": 10.0, "event": "junction", "identity": "b"},
    ]
    result = teacher_structural_graph(rows)
    assert result["node_ids"] == ["a", "b"]
    assert result["edges"] == [["a", "b"]]


def test_collapse_suppresses_degree_two_metric_anchor_but_keeps_branch_anchor() -> None:
    nodes = [
        {"id": 0, "node_kind": "structural"},
        {"id": 1, "node_kind": "anchor"},
        {"id": 2, "node_kind": "structural"},
        {"id": 3, "node_kind": "anchor"},
        {"id": 4, "node_kind": "structural"},
    ]
    edges = [
        {"from": 0, "to": 1},
        {"from": 1, "to": 2},
        {"from": 2, "to": 3},
        {"from": 3, "to": 4},
        {"from": 3, "to": 0},
    ]
    result = collapse_to_structural_graph(nodes, edges)
    assert result["suppressed_metric_anchor_ids"] == [1]
    assert result["node_ids"] == [0, 2, 3, 4]
    assert [0, 2] in result["edges"]


def test_strict_mapping_rejects_false_merge_identity_conflict() -> None:
    nodes = [
        {"id": 0, "event": "junction", "observation_frames": [1, 2]},
        {"id": 1, "event": "terminal", "observation_frames": [3]},
    ]
    teacher = {
        1: {"event": "junction", "identity": "a"},
        2: {"event": "junction", "identity": "b"},
        3: {"event": "terminal", "identity": "c"},
    }
    mapping, conflicts = strict_predicted_node_mapping(nodes, teacher)
    assert mapping == {1: "c"}
    assert conflicts == {0: ["a", "b"]}


def test_full_world_replay_builds_trace_verified_teacher_graph() -> None:
    traversals = [
        {"traversal_id": "e0:d0", "edge_id": "e0", "from_node_id": "a", "to_node_id": "b", "length_m": 12.0},
        {"traversal_id": "e0:d1", "edge_id": "e0", "from_node_id": "b", "to_node_id": "a", "length_m": 12.0},
    ]
    teacher = [
        {"traversal_id": "e0:d0", "edge_id": "e0", "traversal_arc_m": 2.0, "canonical_edge_arc_m": 2.0, "global_sequence_index": 0, "event": "terminal", "identity": "a"},
        {"traversal_id": "e0:d0", "edge_id": "e0", "traversal_arc_m": 6.0, "canonical_edge_arc_m": 6.0, "global_sequence_index": 1, "event": "corridor", "identity": None},
        {"traversal_id": "e0:d0", "edge_id": "e0", "traversal_arc_m": 10.0, "canonical_edge_arc_m": 10.0, "global_sequence_index": 2, "event": "junction", "identity": "b"},
        {"traversal_id": "e0:d1", "edge_id": "e0", "traversal_arc_m": 2.0, "canonical_edge_arc_m": 10.0, "global_sequence_index": 3, "event": "junction", "identity": "b"},
        {"traversal_id": "e0:d1", "edge_id": "e0", "traversal_arc_m": 6.0, "canonical_edge_arc_m": 6.0, "global_sequence_index": 4, "event": "corridor", "identity": None},
        {"traversal_id": "e0:d1", "edge_id": "e0", "traversal_arc_m": 10.0, "canonical_edge_arc_m": 2.0, "global_sequence_index": 5, "event": "terminal", "identity": "a"},
    ]
    event_logits = np.full((6, 5), -4.0)
    event_logits[(0, 5), 2] = 4.0
    event_logits[(1, 4), 0] = 4.0
    event_logits[(2, 3), 1] = 4.0
    outputs = {
        "event_logits": event_logits,
        "local_axis": np.tile((1.0, 0.0, 0.0), (6, 1)),
        "width_m": np.full(6, 5.0),
        "height_m": np.full(6, 4.0),
        "slope_deg": np.zeros(6),
        "curvature_per_m": np.zeros(6),
        "place_descriptor": np.asarray(((1, 0), (1, 1), (0, 1), (0, 1), (1, 1), (1, 0))),
        "uncertainty": np.zeros(6),
        "exit_confidence": np.full((6, 1), 0.1),
        "exit_heading_unit": np.tile((((0.0, 1.0),),), (6, 1, 1)),
        "exit_opening_width_m": np.full((6, 1), 3.0),
        "exit_vertical_profile": np.zeros((6, 1, 4)),
        "exit_descriptor": np.tile((((1.0, 0.0),),), (6, 1, 1)),
    }
    poses = {
        0: {"axis_xyz_m": (0, 0, 0), "yaw_deg": 0},
        1: {"axis_xyz_m": (5, 0, 0), "yaw_deg": 0},
        2: {"axis_xyz_m": (10, 0, 0), "yaw_deg": 0},
        3: {"axis_xyz_m": (10, 0, 0), "yaw_deg": 180},
        4: {"axis_xyz_m": (5, 0, 0), "yaw_deg": 180},
        5: {"axis_xyz_m": (0, 0, 0), "yaw_deg": 180},
    }
    config = GSEGraphConfig(
        stable_event_frames=1,
        minimum_event_travel_m=0.5,
        metric_anchor_interval_m=100.0,
        event_probability_threshold=0.5,
        maximum_uncertainty=0.5,
        association_radius_m=20.0,
        descriptor_minimum_similarity=0.8,
        exit_heading_tolerance_deg=20.0,
        exit_descriptor_minimum_similarity=0.8,
        exit_width_log_tolerance=0.3,
        exit_vertical_profile_tolerance=1.0,
        maximum_exit_count_difference=0,
        ambiguity_similarity_margin=0.1,
    )
    result = replay_gse_world(
        traversal_records=traversals,
        teacher_observations=teacher,
        pose_by_sequence_index=poses,
        model_outputs=outputs,
        output_row_by_sequence_index={index: index for index in range(6)},
        graph_config=config,
        event_temperature=1.0,
        exit_presence_threshold=0.5,
    )
    assert result["replayed_sequences"] == 6
    assert len(result["edges"]) == 1
    assert result["edges"][0]["traversal_count"] == 2
    assert result["association_metrics"]["association_precision"] == 1.0
    assert set(result["teacher_node_xyz_m"]) == {"a", "b"}
    assert result["teacher_node_xyz_m"]["a"][0] == 0.0
    assert result["teacher_node_xyz_m"]["b"][0] == 10.0
    assert result["topology_metrics"]["node"]["f1"] == 1.0
    assert result["topology_metrics"]["edge"]["f1"] == 1.0
    aggregate = aggregate_gse_world_replays([result])
    assert aggregate["node"]["f1"] == 1.0
    assert aggregate["edge"]["f1"] == 1.0
    assert aggregate["association"]["precision"] == 1.0


def test_gse_graph_grid_has_exactly_243_unique_configs() -> None:
    grid = gse_graph_parameter_grid(
        event_probability_threshold=0.6,
        maximum_uncertainty=0.4,
        descriptor_minimum_similarity=0.9,
        exit_descriptor_minimum_similarity=0.8,
        exit_heading_tolerance_deg=20.0,
        exit_width_log_tolerance=0.3,
        exit_vertical_profile_tolerance=1.0,
    )
    assert len(grid) == 243
    assert len({tuple(config.to_dict().items()) for config in grid}) == 243
    rule = rule_graph_parameter_grid(event_probability_threshold=0.5)
    assert len(rule) == 243
    assert all(item.exit_heading_tolerance_deg == 20.0 for item in rule)
    assert all(item.descriptor_minimum_similarity == -1.0 for item in rule)


def test_sweep_selection_rejects_unsafe_higher_f1_and_uses_safe_lexicographic_score() -> None:
    def row(index, node_f1, edge_f1, precision, false_rate):
        return {
            "grid_index": index,
            "aggregate": {
                "node": {"f1": node_f1},
                "edge": {"f1": edge_f1},
                "association": {"precision": precision, "false_loop_merge_rate": false_rate, "merge_attempts": 10},
                "invariants": {"connected_component_mean_absolute_error": 0.0, "cycle_rank_mean_absolute_error": 0.0},
            },
        }
    selected = select_gse_graph_sweep(
        (
            row(0, 0.99, 0.99, 0.97, 0.03),
            row(1, 0.80, 0.70, 0.99, 0.01),
            row(2, 0.75, 0.75, 0.99, 0.01),
        )
    )
    assert selected["grid_index"] == 2
    unrestricted = select_baseline_graph_sweep(
        (
            row(0, 0.99, 0.99, 0.20, 0.80),
            row(1, 0.80, 0.70, 0.99, 0.01),
        )
    )
    assert unrestricted["grid_index"] == 0
