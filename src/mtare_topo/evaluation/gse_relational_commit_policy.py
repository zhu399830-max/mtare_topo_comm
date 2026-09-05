"""Past-only commit policies for frozen relational structural-event outputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True, order=True)
class EventCommitPolicy:
    event: int
    probability_threshold: float
    consensus_votes: int
    stable_frames: int
    release_frames: int

    def __post_init__(self) -> None:
        if (
            self.event not in (1, 2)
            or not 0.0 <= self.probability_threshold <= 1.0
            or self.consensus_votes not in (2, 3)
            or self.stable_frames not in (1, 2, 3)
            or self.release_frames not in (2, 3, 5, 8)
        ):
            raise ValueError("event commit policy is outside the frozen grid")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class EventCommit:
    row: int
    event: int
    probability: float


def extract_event_commits(
    seed_probability: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    include_mask: np.ndarray,
    policy: EventCommitPolicy,
) -> tuple[EventCommit, ...]:
    """Emit one commit per accepted excursion using no future frame."""

    probability = np.asarray(seed_probability, dtype=np.float64)
    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    include = np.asarray(include_mask, dtype=np.bool_)
    count = probability.shape[1] if probability.ndim == 3 else 0
    if (
        probability.shape != (3, count, 3)
        or traversal.shape != (count,)
        or sequence.shape != (count,)
        or include.shape != (count,)
        or not np.any(include)
        or not np.all(np.isfinite(probability))
        or np.any(probability < 0.0)
        or not np.allclose(probability.sum(axis=2), 1.0, atol=1e-5)
    ):
        raise ValueError("commit-policy input contract drift")
    selected = np.flatnonzero(include)
    if np.any(np.diff(selected) <= 0):
        raise ValueError("commit-policy include rows must preserve source order")
    commits: list[EventCommit] = []
    previous_row = -1
    active_traversal = ""
    streak = 0
    latched = False
    clear = 0
    for row in selected.tolist():
        new_sequence = (
            traversal[row] != active_traversal
            or previous_row < 0
            or sequence[row] != sequence[previous_row] + 1
        )
        if new_sequence:
            active_traversal = traversal[row]
            streak = 0
            latched = False
            clear = 0
        values = probability[:, row, policy.event]
        accepted = int(np.sum(values >= policy.probability_threshold)) >= policy.consensus_votes
        if latched:
            if accepted:
                clear = 0
            else:
                clear += 1
                if clear >= policy.release_frames:
                    latched = False
                    clear = 0
                    streak = 0
        else:
            streak = streak + 1 if accepted else 0
            if streak >= policy.stable_frames:
                commits.append(EventCommit(
                    row=row,
                    event=policy.event,
                    probability=float(np.median(values)),
                ))
                latched = True
                clear = 0
                streak = 0
        previous_row = row
    return tuple(commits)


def merge_event_commits(*groups: Sequence[EventCommit]) -> tuple[EventCommit, ...]:
    """Merge class-specific streams, resolving same-row conflict by confidence."""

    by_row: dict[int, EventCommit] = {}
    for commit in (item for group in groups for item in group):
        previous = by_row.get(commit.row)
        if previous is None or (commit.probability, -commit.event) > (previous.probability, -previous.event):
            by_row[commit.row] = commit
    return tuple(by_row[row] for row in sorted(by_row))


def evaluate_event_commits(
    commits: Sequence[EventCommit],
    decision_target: np.ndarray,
    episode_id: np.ndarray,
    include_mask: np.ndarray,
) -> dict:
    """Score commits as unique correctly classified decision episodes."""

    target = np.asarray(decision_target, dtype=np.int64)
    episode = np.asarray(episode_id, dtype=np.int64)
    include = np.asarray(include_mask, dtype=np.bool_)
    if (
        target.shape != episode.shape
        or include.shape != target.shape
        or np.any((target == 0) != (episode < 0))
        or np.any((target < 0) | (target > 2))
    ):
        raise ValueError("commit evaluation target contract drift")
    true_episode = {
        int(value): int(np.unique(target[episode == value]).item())
        for value in np.unique(episode[include & (episode >= 0)])
    }
    matched: set[int] = set()
    correct: set[int] = set()
    predicted = {1: 0, 2: 0}
    correct_by_event = {1: 0, 2: 0}
    duplicate = corridor_false = wrong_event = 0
    for commit in commits:
        if not include[commit.row]:
            raise ValueError("commit lies outside evaluation partition")
        predicted[commit.event] += 1
        identity = int(episode[commit.row])
        if identity not in true_episode:
            corridor_false += 1
        elif identity in matched:
            duplicate += 1
        else:
            matched.add(identity)
            if commit.event == true_episode[identity]:
                correct.add(identity)
                correct_by_event[commit.event] += 1
            else:
                wrong_event += 1
    per_event = {}
    f1 = []
    for event, name in ((1, "junction"), (2, "terminal")):
        truth = sum(value == event for value in true_episode.values())
        precision = correct_by_event[event] / predicted[event] if predicted[event] else 0.0
        recall = correct_by_event[event] / truth if truth else 0.0
        score = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1.append(score)
        per_event[name] = {
            "true_episodes": truth,
            "predicted_commits": predicted[event],
            "correct_unique_episodes": correct_by_event[event],
            "precision": precision,
            "recall": recall,
            "f1": score,
        }
    count = len(commits)
    precision = len(correct) / count if count else 0.0
    return {
        "predicted_commits": count,
        "true_decision_episodes": len(true_episode),
        "correct_unique_episodes": len(correct),
        "precision": precision,
        "false_fraction": 1.0 - precision if count else 1.0,
        "recall": len(correct) / len(true_episode) if true_episode else 0.0,
        "macro_f1": float(np.mean(f1)),
        "per_event": per_event,
        "failure_decomposition": {
            "duplicate_same_episode": duplicate,
            "corridor_false": corridor_false,
            "wrong_event": wrong_event,
        },
    }


__all__ = [
    "EventCommit", "EventCommitPolicy", "evaluate_event_commits",
    "extract_event_commits", "merge_event_commits",
]
