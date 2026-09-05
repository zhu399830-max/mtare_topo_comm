from mtare_topo.evaluation.gse_objective_spatial_graph_score import (
    ObjectiveGraphNode,
    PredictedGraphNode,
    match_objective_graph_nodes,
    score_objective_spatial_graph,
)


def _truth(identity: str, x: float, *, event: str = "junction") -> ObjectiveGraphNode:
    return ObjectiveGraphNode(identity, "world", event, (x, 0.0, 0.0))


def _prediction(identifier: int, x: float, *, event: str = "junction") -> PredictedGraphNode:
    return PredictedGraphNode(identifier, "world", event, (x, 0.0, 0.0))


def test_assignment_maximizes_cardinality_before_distance() -> None:
    # Greedy nearest-first would take p0->t0 and strand p1.  The objective
    # assignment takes p0->t1 and p1->t0, recovering both valid identities.
    result = match_objective_graph_nodes(
        [_prediction(0, 0.0), _prediction(1, -3.9)],
        [_truth("t0", 0.0), _truth("t1", 3.9)],
        distance_cap_m=4.0,
    )
    assert result["matched_nodes"] == 2
    assert result["mapping"] == {0: "t1", 1: "t0"}


def test_duplicate_prediction_is_false_positive() -> None:
    result = score_objective_spatial_graph(
        predicted_nodes=[_prediction(0, 0.1), _prediction(1, 0.2)],
        predicted_edges=[], objective_nodes=[_truth("t0", 0.0)],
        objective_relations=set(), distance_cap_m=4.0,
    )
    assert result["correct_unique_nodes"] == 1
    assert result["node_precision"] == 0.5
    assert result["node_recall"] == 1.0


def test_world_and_event_are_hard_matching_constraints() -> None:
    prediction = PredictedGraphNode(0, "other", "junction", (0.0, 0.0, 0.0))
    result = match_objective_graph_nodes([prediction], [_truth("t0", 0.0)])
    assert result["matched_nodes"] == 0
    turn_like = _prediction(1, 0.0, event="terminal")
    result = match_objective_graph_nodes([turn_like], [_truth("t0", 0.0)])
    assert result["matched_nodes"] == 0


def test_edges_are_scored_only_through_one_to_one_mapping() -> None:
    result = score_objective_spatial_graph(
        predicted_nodes=[_prediction(10, 0.0), _prediction(11, 5.0)],
        predicted_edges=[{"from_hypothesis": 10, "to_hypothesis": 11}],
        objective_nodes=[_truth("a", 0.0), _truth("b", 5.0)],
        objective_relations={("a", "b")}, distance_cap_m=4.0,
    )
    assert result["node_precision"] == 1.0
    assert result["edge_precision"] == 1.0
    assert result["edge_recall"] == 1.0


def test_out_of_radius_prediction_remains_false_positive() -> None:
    result = score_objective_spatial_graph(
        predicted_nodes=[_prediction(0, 4.01)], predicted_edges=[],
        objective_nodes=[_truth("t0", 0.0)], objective_relations=set(),
        distance_cap_m=4.0,
    )
    assert result["correct_unique_nodes"] == 0
    assert result["node_precision"] == 0.0
