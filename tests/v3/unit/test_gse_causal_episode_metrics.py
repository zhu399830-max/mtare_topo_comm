from __future__ import annotations

import numpy as np

from mtare_topo.evaluation.gse_causal_episode_metrics import (
    evaluate_causal_event_triggers,
    evaluate_decision_event_triggers,
    evaluate_decision_mass_triggers,
    extract_decision_mass_triggers,
    extract_causal_event_triggers,
    select_structural_threshold,
)


def _fixture():
    probability = np.asarray([
        [.99, .005, .003, .001, .001],
        [.10, .80, .04, .03, .03],
        [.05, .88, .03, .02, .02],
        [.99, .005, .003, .001, .001],
        [.02, .01, .94, .02, .01],
        [.01, .01, .95, .02, .01],
    ])
    event = np.asarray([0, 1, 1, 0, 2, 2])
    episode = np.asarray([-1, 0, 0, -1, 1, 1])
    traversal = np.asarray(["a", "a", "a", "a", "b", "b"])
    sequence = np.asarray([0, 1, 2, 3, 0, 1])
    zeros = np.zeros(6)
    return probability, event, episode, traversal, sequence, zeros


def test_contiguous_responses_collapse_to_one_trigger_per_episode() -> None:
    probability, event, episode, traversal, sequence, zeros = _fixture()
    triggers = extract_causal_event_triggers(
        probability, traversal, sequence, zeros, zeros, structural_threshold=.5
    )
    assert [(row.row, row.predicted_event_index) for row in triggers] == [(2, 1), (5, 2)]
    metrics = evaluate_causal_event_triggers(
        probability, event, episode, traversal, sequence, zeros, zeros,
        structural_threshold=.5,
    )
    assert metrics["predicted_triggers"] == 2
    assert metrics["structural_trigger_precision"] == 1.0
    assert metrics["class_correct_episode_recall"] == 1.0


def test_threshold_selection_rejects_corridor_false_trigger() -> None:
    probability, event, episode, traversal, sequence, zeros = _fixture()
    probability = np.concatenate((probability, np.asarray([[.4, .5, .04, .03, .03]])))
    event = np.append(event, 0)
    episode = np.append(episode, -1)
    traversal = np.append(traversal, "c")
    sequence = np.append(sequence, 0)
    zeros = np.zeros(7)
    selection = select_structural_threshold(
        probability, event, episode, traversal, sequence, zeros, zeros,
        minimum_precision=1.0,
    )
    assert selection["structural_threshold"] > .6
    assert selection["structural_trigger_precision"] == 1.0
    assert selection["structural_episode_recall"] == 1.0


def test_decision_metrics_exclude_turn_but_count_wrong_junction_as_false() -> None:
    probability = np.asarray([
        [.01, .98, .005, .005, 0.0],  # correct junction
        [.01, .98, .005, .005, 0.0],  # terminal episode misclassified junction
        [.01, .005, .005, .98, 0.0],  # turn response is disabled
    ])
    target = np.asarray([1, 2, 3])
    episode = np.asarray([0, 1, 2])
    traversal = np.asarray(["a", "b", "c"])
    sequence = np.zeros(3, dtype=np.int64)
    zeros = np.zeros(3)
    metrics = evaluate_decision_event_triggers(
        probability, target, episode, traversal, sequence, zeros, zeros,
        structural_threshold=.9,
    )
    assert metrics["raw_structural_triggers"] == 3
    assert metrics["predicted_decision_triggers"] == 2
    assert metrics["true_decision_episodes"] == 2
    assert metrics["decision_trigger_precision"] == .5
    assert metrics["decision_episode_recall"] == .5


def test_decision_mass_gate_ignores_high_turn_probability() -> None:
    probability = np.asarray([
        [.005, .003, .002, .990, 0.0],
        [.005, .985, .005, .005, 0.0],
        [.004, .990, .002, .004, 0.0],
    ])
    traversal = np.asarray(["a", "a", "a"])
    sequence = np.asarray([0, 1, 2])
    triggers = extract_decision_mass_triggers(
        probability, traversal, sequence, np.zeros(3), decision_threshold=.98,
    )
    assert [(trigger.row, trigger.predicted_event_index) for trigger in triggers] == [(2, 1)]
    metrics = evaluate_decision_mass_triggers(
        probability, np.asarray([3, 1, 1]), np.asarray([0, 1, 1]),
        traversal, sequence, np.zeros(3), decision_threshold=.98,
    )
    assert metrics["predicted_decision_triggers"] == 1
    assert metrics["decision_trigger_precision"] == 1.0
