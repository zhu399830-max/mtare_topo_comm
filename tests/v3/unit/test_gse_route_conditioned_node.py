from __future__ import annotations

import numpy as np

from mtare_topo.representation.gse_route_conditioned_node import (
    RouteConditionedNodeConfig,
    evaluate_structured_decision_events,
    route_conditioned_action_scores,
    structured_decision_events,
)


def test_route_condition_removes_incoming_and_nms_duplicate_outgoing() -> None:
    confidence = np.asarray([[.9, .8, .7, .6, .1, .1]])
    degrees = np.radians([180, 0, 5, 90, 45, -45])
    heading = np.stack((np.sin(degrees), np.cos(degrees)), axis=1)[None]
    incoming = np.asarray([[0.0, -1.0]])
    back, outgoing = route_conditioned_action_scores(
        confidence, heading, incoming, incoming_half_angle_deg=20.0
    )
    assert back.tolist() == [np.float32(.9)]
    assert np.allclose(outgoing[0, :2], [.8, .6])


def test_structured_decoder_requires_consensus_and_persistence() -> None:
    incoming = np.full(4, .9)
    junction = np.asarray([[.8, .7, 0, 0, 0, 0]] * 4)
    terminal = np.zeros((4, 6))
    scores = {0: (incoming, junction), 1: (incoming, junction), 2: (incoming, terminal)}
    config = RouteConditionedNodeConfig(.5, 20.0, 2, 2)
    predicted = structured_decision_events(scores, ["a"] * 4, [0, 1, 2, 3], config)
    assert predicted.tolist() == [0, 1, 1, 1]


def test_structured_evaluation_collapses_contiguous_trigger() -> None:
    metrics = evaluate_structured_decision_events(
        np.asarray([0, 1, 1, 0, 2]), np.asarray([0, 1, 1, 0, 2]),
        np.asarray([-1, 0, 0, -1, 1]), ["a"] * 5, [0, 1, 2, 3, 4],
    )
    assert metrics["predicted_decision_triggers"] == 2
    assert metrics["decision_trigger_precision"] == 1.0
    assert metrics["decision_episode_recall"] == 1.0
