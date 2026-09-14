"""Structured decision nodes from learned exits and the executed incoming action."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


TOKEN_NMS_DEGREES = 20.0


@dataclass(frozen=True)
class RouteConditionedNodeConfig:
    confidence_threshold: float
    incoming_half_angle_deg: float
    persistence_observations: int
    seed_consensus: int

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence threshold must lie in [0,1]")
        if self.incoming_half_angle_deg not in (20.0, 35.0, 50.0):
            raise ValueError("incoming half angle is outside the frozen grid")
        if self.persistence_observations not in (1, 2, 3):
            raise ValueError("persistence is outside the frozen grid")
        if self.seed_consensus not in (2, 3):
            raise ValueError("seed consensus is outside the frozen grid")


def _angle_degrees(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip(np.dot(left, right), -1.0, 1.0))))


def route_conditioned_action_scores(
    confidence: np.ndarray,
    heading_unit: np.ndarray,
    incoming_heading_unit: np.ndarray,
    *,
    incoming_half_angle_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return incoming confidence and NMS-separated outgoing confidences.

    Headings use the project's ``[sin(theta), cos(theta)]`` robot-frame
    convention.  The incoming action is supplied by causal execution state,
    not Teacher identity or a future observation.
    """

    score = np.asarray(confidence, dtype=np.float64)
    heading = np.asarray(heading_unit, dtype=np.float64)
    incoming = np.asarray(incoming_heading_unit, dtype=np.float64)
    rows = len(score)
    if (
        score.shape != (rows, 6) or heading.shape != (rows, 6, 2)
        or incoming.shape != (rows, 2)
        or np.any((score < 0.0) | (score > 1.0))
        or not np.all(np.isfinite(score)) or not np.all(np.isfinite(heading))
        or not np.all(np.isfinite(incoming))
        or not np.allclose(np.linalg.norm(heading, axis=2), 1.0, atol=2e-3)
        or not np.allclose(np.linalg.norm(incoming, axis=1), 1.0, atol=2e-3)
        or incoming_half_angle_deg not in (20.0, 35.0, 50.0)
    ):
        raise ValueError("route-conditioned action input contract drift")
    incoming_score = np.zeros(rows, dtype=np.float32)
    outgoing_score = np.zeros((rows, 6), dtype=np.float32)
    for row in range(rows):
        angles = np.asarray([
            _angle_degrees(heading[row, token], incoming[row]) for token in range(6)
        ])
        incoming_tokens = np.flatnonzero(angles <= incoming_half_angle_deg)
        if len(incoming_tokens):
            incoming_score[row] = float(np.max(score[row, incoming_tokens]))
        retained: list[int] = []
        for token in sorted(
            np.flatnonzero(angles > incoming_half_angle_deg).tolist(),
            key=lambda value: (-score[row, value], value),
        ):
            if all(
                _angle_degrees(heading[row, token], heading[row, previous]) >= TOKEN_NMS_DEGREES
                for previous in retained
            ):
                retained.append(token)
        for rank, token in enumerate(retained):
            outgoing_score[row, rank] = score[row, token]
    return incoming_score, outgoing_score


def structured_decision_events(
    scores_by_seed: Mapping[int, tuple[np.ndarray, np.ndarray]],
    traversal_id: Sequence[str],
    sequence_index: Sequence[int],
    config: RouteConditionedNodeConfig,
) -> np.ndarray:
    """Predict 0=no node, 1=junction or 2=terminal using only causal state."""

    if tuple(sorted(scores_by_seed)) != (0, 1, 2):
        raise ValueError("structured node decoder requires seeds 0/1/2")
    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    count = len(traversal)
    if sequence.shape != (count,):
        raise ValueError("structured node identity vectors drift")
    seed_event = np.zeros((3, count), dtype=np.int8)
    for seed in (0, 1, 2):
        incoming, outgoing = scores_by_seed[seed]
        incoming = np.asarray(incoming, dtype=np.float64)
        outgoing = np.asarray(outgoing, dtype=np.float64)
        if incoming.shape != (count,) or outgoing.shape != (count, 6):
            raise ValueError("structured node score population drift")
        has_incoming = incoming >= config.confidence_threshold
        outgoing_count = np.sum(outgoing >= config.confidence_threshold, axis=1)
        seed_event[seed, has_incoming & (outgoing_count == 0)] = 2
        seed_event[seed, has_incoming & (outgoing_count >= 2)] = 1
    consensus = np.zeros(count, dtype=np.int8)
    for event in (1, 2):
        accepted = np.sum(seed_event == event, axis=0) >= config.seed_consensus
        consensus[accepted] = event
    if np.any((np.sum(seed_event == 1, axis=0) >= config.seed_consensus) & (np.sum(seed_event == 2, axis=0) >= config.seed_consensus)):
        raise RuntimeError("structured seed consensus is ambiguous")
    starts = np.r_[True, traversal[1:] != traversal[:-1]]
    if np.any(sequence[starts] != 0):
        raise ValueError("structured decoder traversal must start at sequence zero")
    within = ~starts
    if np.any(sequence[1:][within[1:]] != sequence[:-1][within[1:]] + 1):
        raise ValueError("structured decoder history is not traversal-contiguous")
    result = consensus.copy()
    for lag in range(1, config.persistence_observations):
        valid = np.zeros(count, dtype=np.bool_)
        valid[lag:] = (
            (traversal[lag:] == traversal[:-lag])
            & (sequence[lag:] == sequence[:-lag] + lag)
            & (consensus[lag:] == consensus[:-lag])
        )
        result[~valid] = 0
    return result


def evaluate_structured_decision_events(
    predicted_event: np.ndarray,
    target_event: np.ndarray,
    episode_id: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: Sequence[int],
) -> dict:
    """Collapse contiguous predictions and score unique decision episodes."""

    predicted = np.asarray(predicted_event, dtype=np.int64)
    target = np.asarray(target_event, dtype=np.int64)
    episode = np.asarray(episode_id, dtype=np.int64)
    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    count = len(predicted)
    if (
        target.shape != (count,) or episode.shape != (count,)
        or traversal.shape != (count,) or sequence.shape != (count,)
        or np.any(~np.isin(predicted, (0, 1, 2)))
        or np.any(~np.isin(target, (0, 1, 2)))
        or np.any((target == 0) != (episode < 0))
    ):
        raise ValueError("structured decision evaluation contract drift")
    active = predicted > 0
    continues = np.zeros(count, dtype=np.bool_)
    continues[1:] = (
        active[1:] & active[:-1] & (predicted[1:] == predicted[:-1])
        & (traversal[1:] == traversal[:-1]) & (sequence[1:] == sequence[:-1] + 1)
    )
    trigger_rows = np.flatnonzero(active & ~continues).tolist()
    truth = {
        int(value): int(np.unique(target[episode == value]).item())
        for value in np.unique(episode[episode >= 0])
    }
    matched: set[int] = set()
    correct: set[int] = set()
    correct_by_event = {1: 0, 2: 0}
    for row in trigger_rows:
        bag = int(episode[row])
        if bag >= 0 and bag not in matched:
            matched.add(bag)
            if truth[bag] == int(predicted[row]):
                correct.add(bag)
                correct_by_event[int(predicted[row])] += 1
    per_event = {}
    for event, name in ((1, "junction"), (2, "terminal")):
        predicted_rows = [row for row in trigger_rows if predicted[row] == event]
        correct_event = correct_by_event[event]
        truth_count = sum(value == event for value in truth.values())
        precision = correct_event / len(predicted_rows) if predicted_rows else 0.0
        recall = correct_event / truth_count if truth_count else 0.0
        per_event[name] = {"true_episodes": truth_count, "predicted_triggers": len(predicted_rows), "correct_unique_episodes": correct_event, "precision": precision, "recall": recall}
    precision = len(correct) / len(trigger_rows) if trigger_rows else 0.0
    return {
        "predicted_decision_triggers": len(trigger_rows),
        "true_decision_episodes": len(truth),
        "correctly_classified_unique_decision_episodes": len(correct),
        "decision_trigger_precision": precision,
        "false_decision_trigger_fraction": 1.0 - precision if trigger_rows else 1.0,
        "decision_episode_recall": len(correct) / len(truth) if truth else 0.0,
        "per_event": per_event,
        "trigger_rows": trigger_rows,
    }


__all__ = ["RouteConditionedNodeConfig", "TOKEN_NMS_DEGREES", "evaluate_structured_decision_events", "route_conditioned_action_scores", "structured_decision_events"]
