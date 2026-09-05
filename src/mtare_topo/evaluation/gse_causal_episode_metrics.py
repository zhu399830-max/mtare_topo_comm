"""Deployment-shaped metrics for causal structural-event node triggers."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np

from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


@dataclass(frozen=True)
class CausalEventTrigger:
    row: int
    traversal_id: str
    sequence_index: int
    predicted_event_index: int
    structural_probability: float
    uncertainty: float
    boundary_offset_m: float

    def to_dict(self) -> dict:
        return asdict(self)


def extract_causal_event_triggers(
    probabilities: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    boundary_offset_m: np.ndarray,
    uncertainty: np.ndarray,
    *,
    structural_threshold: float,
) -> tuple[CausalEventTrigger, ...]:
    """Collapse every contiguous accepted response into exactly one trigger."""

    probability = np.asarray(probabilities, dtype=np.float64)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    boundary = np.asarray(boundary_offset_m, dtype=np.float64)
    uncertain = np.asarray(uncertainty, dtype=np.float64)
    traversal = np.asarray(traversal_id, dtype=str)
    count = len(probability)
    if (
        probability.shape != (count, len(EVENT_NAMES))
        or traversal.shape != (count,)
        or sequence.shape != (count,)
        or boundary.shape != (count,)
        or uncertain.shape != (count,)
        or not 0.0 <= structural_threshold <= 1.0
        or not np.all(np.isfinite(probability))
        or not np.all(np.isfinite(boundary))
        or not np.all(np.isfinite(uncertain))
        or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5)
    ):
        raise ValueError("causal trigger input contract drift")
    structural = 1.0 - probability[:, 0]
    accepted = structural >= float(structural_threshold)
    groups: list[list[int]] = []
    active: list[int] = []
    for row in range(count):
        continues = bool(
            active
            and accepted[row]
            and traversal[row] == traversal[active[-1]]
            and sequence[row] == sequence[active[-1]] + 1
        )
        if not accepted[row] or not continues:
            if active:
                groups.append(active)
                active = []
        if accepted[row]:
            active.append(row)
    if active:
        groups.append(active)
    triggers = []
    for rows in groups:
        # Lexicographic tie break: maximum structural score, then earliest row.
        row = min(rows, key=lambda value: (-structural[value], value))
        event = int(np.argmax(probability[row, 1:])) + 1
        triggers.append(CausalEventTrigger(
            row=row,
            traversal_id=str(traversal[row]),
            sequence_index=int(sequence[row]),
            predicted_event_index=event,
            structural_probability=float(structural[row]),
            uncertainty=float(uncertain[row]),
            boundary_offset_m=float(boundary[row]),
        ))
    return tuple(triggers)


def extract_decision_mass_triggers(
    probabilities: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    uncertainty: np.ndarray,
    *,
    decision_threshold: float,
    decision_events: Sequence[str] = ("junction", "terminal"),
) -> tuple[CausalEventTrigger, ...]:
    """Collapse contiguous decision-class mass into one past-only node trigger."""

    probability = np.asarray(probabilities, dtype=np.float64)
    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    uncertain = np.asarray(uncertainty, dtype=np.float64)
    count = len(probability)
    try:
        indices = np.asarray([EVENT_NAMES.index(str(name)) for name in decision_events], dtype=np.int64)
    except ValueError as exc:
        raise ValueError("unknown decision event") from exc
    if (
        probability.shape != (count, len(EVENT_NAMES))
        or traversal.shape != (count,)
        or sequence.shape != (count,)
        or uncertain.shape != (count,)
        or len(indices) == 0
        or len(np.unique(indices)) != len(indices)
        or np.any(indices == 0)
        or not 0.0 <= decision_threshold <= 1.0
        or not np.all(np.isfinite(probability))
        or not np.all(np.isfinite(uncertain))
        or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=1), 1.0, rtol=0.0, atol=1e-5)
    ):
        raise ValueError("decision-mass trigger input contract drift")
    mass = probability[:, indices].sum(axis=1)
    accepted = mass >= float(decision_threshold)
    groups: list[list[int]] = []
    active: list[int] = []
    for row in range(count):
        continues = bool(
            active and accepted[row]
            and traversal[row] == traversal[active[-1]]
            and sequence[row] == sequence[active[-1]] + 1
        )
        if not accepted[row] or not continues:
            if active:
                groups.append(active)
                active = []
        if accepted[row]:
            active.append(row)
    if active:
        groups.append(active)
    triggers = []
    for rows in groups:
        row = min(rows, key=lambda value: (-mass[value], value))
        event = int(indices[int(np.argmax(probability[row, indices]))])
        triggers.append(CausalEventTrigger(
            row=row, traversal_id=str(traversal[row]), sequence_index=int(sequence[row]),
            predicted_event_index=event, structural_probability=float(mass[row]),
            uncertainty=float(uncertain[row]), boundary_offset_m=0.0,
        ))
    return tuple(triggers)


def evaluate_decision_mass_triggers(
    probabilities: np.ndarray,
    event_target: np.ndarray,
    episode_id: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    uncertainty: np.ndarray,
    *,
    decision_threshold: float,
    decision_events: Sequence[str] = ("junction", "terminal"),
) -> dict:
    """Evaluate the five-frame junction/terminal-specific deployment gate."""

    target = np.asarray(event_target, dtype=np.int64)
    episodes = np.asarray(episode_id, dtype=np.int64)
    if target.shape != episodes.shape or np.any((target == 0) != (episodes < 0)):
        raise ValueError("event/episode target contract drift")
    indices = tuple(EVENT_NAMES.index(str(name)) for name in decision_events)
    triggers = extract_decision_mass_triggers(
        probabilities, traversal_id, sequence_index, uncertainty,
        decision_threshold=decision_threshold, decision_events=decision_events,
    )
    episode_class = {
        int(value): int(np.unique(target[episodes == value]).item())
        for value in np.unique(episodes[episodes >= 0])
    }
    true_decision = {
        episode: event for episode, event in episode_class.items() if event in indices
    }
    matched: set[int] = set()
    correct: set[int] = set()
    predicted_by_class = {index: 0 for index in indices}
    correct_by_class = {index: 0 for index in indices}
    for trigger in triggers:
        predicted_by_class[trigger.predicted_event_index] += 1
        episode = int(episodes[trigger.row])
        if episode not in true_decision or episode in matched:
            continue
        matched.add(episode)
        if trigger.predicted_event_index == true_decision[episode]:
            correct.add(episode)
            correct_by_class[trigger.predicted_event_index] += 1
    per_event = {}
    f1_values = []
    for event in indices:
        truth = sum(value == event for value in true_decision.values())
        predicted = predicted_by_class[event]
        correct_count = correct_by_class[event]
        precision = correct_count / predicted if predicted else 0.0
        recall = correct_count / truth if truth else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_event[EVENT_NAMES[event]] = {
            "true_episodes": truth, "predicted_triggers": predicted,
            "correct_unique_episodes": correct_count, "precision": precision,
            "recall": recall, "f1": f1,
        }
    count = len(triggers)
    precision = len(correct) / count if count else 0.0
    return {
        "decision_threshold": float(decision_threshold),
        "predicted_decision_triggers": count,
        "true_decision_episodes": len(true_decision),
        "matched_unique_decision_episodes": len(matched),
        "correctly_classified_unique_decision_episodes": len(correct),
        "decision_trigger_precision": precision,
        "false_decision_trigger_fraction": 1.0 - precision if count else 1.0,
        "decision_episode_recall": len(correct) / len(true_decision) if true_decision else 0.0,
        "decision_episode_macro_f1": float(np.mean(f1_values)),
        "per_event": per_event,
    }


def select_decision_mass_threshold(
    probabilities: np.ndarray,
    event_target: np.ndarray,
    episode_id: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    uncertainty: np.ndarray,
    *,
    minimum_precision: float = .995,
    minimum_per_event_precision: float = .99,
    minimum_recall: float = .25,
    minimum_per_event_recall: float = .25,
) -> dict:
    """Select one five-frame decision threshold on the development split only."""

    candidates = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = evaluate_decision_mass_triggers(
            probabilities, event_target, episode_id, traversal_id, sequence_index,
            uncertainty, decision_threshold=float(threshold),
        )
        per_event = metrics["per_event"].values()
        if (
            metrics["predicted_decision_triggers"] > 0
            and metrics["decision_trigger_precision"] >= minimum_precision
            and metrics["decision_episode_recall"] >= minimum_recall
            and all(value["precision"] >= minimum_per_event_precision for value in per_event)
            and all(value["recall"] >= minimum_per_event_recall for value in per_event)
        ):
            candidates.append(metrics)
    if not candidates:
        raise RuntimeError("no five-frame decision threshold satisfies the frozen safety/recall margins")
    return max(
        candidates,
        key=lambda value: (
            value["decision_episode_recall"], value["decision_episode_macro_f1"],
            -value["predicted_decision_triggers"], -value["decision_threshold"],
        ),
    )


def evaluate_causal_event_triggers(
    probabilities: np.ndarray,
    event_target: np.ndarray,
    episode_id: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    boundary_offset_m: np.ndarray,
    uncertainty: np.ndarray,
    *,
    structural_threshold: float,
) -> dict:
    """Measure trigger precision and unique correctly classified episode recall."""

    target = np.asarray(event_target, dtype=np.int64)
    episodes = np.asarray(episode_id, dtype=np.int64)
    if target.shape != episodes.shape or np.any((target == 0) != (episodes < 0)):
        raise ValueError("event/episode target contract drift")
    triggers = extract_causal_event_triggers(
        probabilities, traversal_id, sequence_index, boundary_offset_m, uncertainty,
        structural_threshold=structural_threshold,
    )
    true_episode_ids = np.unique(episodes[episodes >= 0])
    episode_class = {
        int(value): int(np.unique(target[episodes == value]).item())
        for value in true_episode_ids
    }
    structural_matches: set[int] = set()
    correct_matches: set[int] = set()
    correct_trigger_count = 0
    structural_trigger_count = 0
    predicted_by_class = {index: 0 for index in range(1, len(EVENT_NAMES))}
    correct_by_class = {index: 0 for index in range(1, len(EVENT_NAMES))}
    for trigger in triggers:
        predicted_by_class[trigger.predicted_event_index] += 1
        episode = int(episodes[trigger.row])
        if episode < 0 or episode in structural_matches:
            continue
        structural_matches.add(episode)
        structural_trigger_count += 1
        if trigger.predicted_event_index == episode_class[episode]:
            correct_matches.add(episode)
            correct_trigger_count += 1
            correct_by_class[trigger.predicted_event_index] += 1
    class_metrics = {}
    class_f1 = []
    for event in range(1, len(EVENT_NAMES)):
        truth = sum(value == event for value in episode_class.values())
        predicted = predicted_by_class[event]
        correct = correct_by_class[event]
        precision = correct / predicted if predicted else 0.0
        recall = correct / truth if truth else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        class_f1.append(f1)
        class_metrics[EVENT_NAMES[event]] = {
            "true_episodes": truth,
            "predicted_triggers": predicted,
            "correct_unique_episodes": correct,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    trigger_count = len(triggers)
    return {
        "structural_threshold": float(structural_threshold),
        "predicted_triggers": trigger_count,
        "true_episodes": len(true_episode_ids),
        "structurally_matched_unique_episodes": len(structural_matches),
        "correctly_classified_unique_episodes": len(correct_matches),
        "structural_trigger_precision": structural_trigger_count / trigger_count if trigger_count else 0.0,
        "structural_episode_recall": len(structural_matches) / len(true_episode_ids) if len(true_episode_ids) else 0.0,
        "class_correct_trigger_precision": correct_trigger_count / trigger_count if trigger_count else 0.0,
        "class_correct_episode_recall": len(correct_matches) / len(true_episode_ids) if len(true_episode_ids) else 0.0,
        "structural_episode_macro_f1": float(np.mean(class_f1)),
        "per_event": class_metrics,
    }


def evaluate_decision_event_triggers(
    probabilities: np.ndarray,
    event_target: np.ndarray,
    episode_id: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    boundary_offset_m: np.ndarray,
    uncertainty: np.ndarray,
    *,
    structural_threshold: float,
    decision_events: Sequence[str] = ("junction", "terminal"),
) -> dict:
    """Evaluate only action-changing episode triggers used as graph nodes."""

    target = np.asarray(event_target, dtype=np.int64)
    episodes = np.asarray(episode_id, dtype=np.int64)
    if target.shape != episodes.shape or np.any((target == 0) != (episodes < 0)):
        raise ValueError("event/episode target contract drift")
    try:
        decision_indices = tuple(EVENT_NAMES.index(str(name)) for name in decision_events)
    except ValueError as exc:
        raise ValueError("unknown decision event") from exc
    if not decision_indices or 0 in decision_indices or len(set(decision_indices)) != len(decision_indices):
        raise ValueError("decision events must be unique non-corridor classes")
    all_triggers = extract_causal_event_triggers(
        probabilities, traversal_id, sequence_index, boundary_offset_m, uncertainty,
        structural_threshold=structural_threshold,
    )
    triggers = tuple(
        trigger for trigger in all_triggers
        if trigger.predicted_event_index in decision_indices
    )
    episode_class = {
        int(value): int(np.unique(target[episodes == value]).item())
        for value in np.unique(episodes[episodes >= 0])
    }
    true_decision = {
        episode: event for episode, event in episode_class.items()
        if event in decision_indices
    }
    matched: set[int] = set()
    correct: set[int] = set()
    predicted_by_class = {index: 0 for index in decision_indices}
    correct_by_class = {index: 0 for index in decision_indices}
    for trigger in triggers:
        predicted_by_class[trigger.predicted_event_index] += 1
        episode = int(episodes[trigger.row])
        if episode not in true_decision or episode in matched:
            continue
        matched.add(episode)
        if trigger.predicted_event_index == true_decision[episode]:
            correct.add(episode)
            correct_by_class[trigger.predicted_event_index] += 1
    per_event = {}
    f1_values = []
    for event in decision_indices:
        truth = sum(value == event for value in true_decision.values())
        predicted = predicted_by_class[event]
        correct_count = correct_by_class[event]
        precision = correct_count / predicted if predicted else 0.0
        recall = correct_count / truth if truth else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_event[EVENT_NAMES[event]] = {
            "true_episodes": truth,
            "predicted_triggers": predicted,
            "correct_unique_episodes": correct_count,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    trigger_count = len(triggers)
    return {
        "structural_threshold": float(structural_threshold),
        "raw_structural_triggers": len(all_triggers),
        "predicted_decision_triggers": trigger_count,
        "true_decision_episodes": len(true_decision),
        "matched_unique_decision_episodes": len(matched),
        "correctly_classified_unique_decision_episodes": len(correct),
        "decision_trigger_precision": len(correct) / trigger_count if trigger_count else 0.0,
        "decision_episode_recall": len(correct) / len(true_decision) if true_decision else 0.0,
        "false_decision_trigger_fraction": 1.0 - len(correct) / trigger_count if trigger_count else 1.0,
        "decision_episode_macro_f1": float(np.mean(f1_values)),
        "per_event": per_event,
    }


def select_structural_threshold(
    probabilities: np.ndarray,
    event_target: np.ndarray,
    episode_id: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    boundary_offset_m: np.ndarray,
    uncertainty: np.ndarray,
    *,
    minimum_precision: float = 0.98,
) -> dict:
    """Validation-only fixed-grid selection; maximize recall under precision."""

    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum precision must lie in (0,1]")
    candidates = []
    for threshold in np.linspace(0.0, 1.0, 1001):
        metrics = evaluate_causal_event_triggers(
            probabilities, event_target, episode_id, traversal_id, sequence_index,
            boundary_offset_m, uncertainty, structural_threshold=float(threshold),
        )
        if (
            metrics["predicted_triggers"] > 0
            and metrics["structural_trigger_precision"] >= minimum_precision
        ):
            candidates.append(metrics)
    if not candidates:
        raise RuntimeError("no non-vacuous structural threshold satisfies precision")
    return max(
        candidates,
        key=lambda value: (
            value["structural_episode_recall"],
            value["structural_episode_macro_f1"],
            -value["predicted_triggers"],
            -value["structural_threshold"],
        ),
    )


__all__ = [
    "CausalEventTrigger",
    "evaluate_causal_event_triggers",
    "evaluate_decision_event_triggers",
    "evaluate_decision_mass_triggers",
    "extract_decision_mass_triggers",
    "extract_causal_event_triggers",
    "select_decision_mass_threshold",
    "select_structural_threshold",
]
