"""Causal action-conditioned geometry-state segmentation evidence.

The module intentionally contains no trainable component.  It measures a
past-only discrepancy between two adjacent six-observation blocks and applies
one fit-only corridor false-alarm threshold.  Teacher fields are accepted only
by the evaluation functions, never by the state-score construction.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


HISTORY_OBSERVATIONS = 12
BLOCK_OBSERVATIONS = 6
GEOMETRY_COLUMNS = np.arange(5, 12, dtype=np.int64)
ACTION_COLUMNS = np.arange(141, 146, dtype=np.int64)


@dataclass(frozen=True)
class RobustDeltaScale:
    center: np.ndarray
    scale: np.ndarray

    def __post_init__(self) -> None:
        if (
            self.center.ndim != 1
            or self.scale.shape != self.center.shape
            or not np.all(np.isfinite(self.center))
            or not np.all(np.isfinite(self.scale))
            or np.any(self.scale <= 0.0)
        ):
            raise ValueError("robust delta scale is invalid")


@dataclass(frozen=True)
class StateDiscrepancyScores:
    geometry: np.ndarray
    action: np.ndarray
    combined: np.ndarray
    eligible: np.ndarray

    def __post_init__(self) -> None:
        size = len(self.eligible)
        if (
            self.eligible.shape != (size,)
            or self.eligible.dtype != np.bool_
            or any(value.shape != (size,) for value in (self.geometry, self.action, self.combined))
            or any(np.any(~np.isfinite(value[self.eligible])) for value in (self.geometry, self.action, self.combined))
            or any(np.any(value[self.eligible] < 0.0) for value in (self.geometry, self.action, self.combined))
            or any(np.any(np.isfinite(value[~self.eligible])) for value in (self.geometry, self.action, self.combined))
        ):
            raise ValueError("state discrepancy score contract drift")

    def as_dict(self) -> dict[str, np.ndarray]:
        return {"geometry": self.geometry, "action": self.action, "combined": self.combined}


def causal_two_block_delta(
    features: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    columns: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return recent-six minus previous-six means without crossing traversal.

    Rows within one traversal must be contiguous and have consecutive sequence
    indices.  The first eleven observations of every traversal are explicitly
    unavailable rather than padded with another traversal or future evidence.
    """

    values = np.asarray(features, dtype=np.float64)
    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    selected = np.asarray(columns, dtype=np.int64)
    count = len(values)
    if (
        values.ndim != 2
        or traversal.shape != (count,)
        or sequence.shape != (count,)
        or selected.ndim != 1
        or len(selected) == 0
        or np.any(selected < 0)
        or np.any(selected >= values.shape[1])
        or not np.all(np.isfinite(values[:, selected]))
    ):
        raise ValueError("causal state input contract drift")
    delta = np.full((count, len(selected)), np.nan, dtype=np.float64)
    eligible = np.zeros(count, dtype=np.bool_)
    seen: set[str] = set()
    start = 0
    while start < count:
        current = str(traversal[start])
        if current in seen:
            raise ValueError("one traversal appears in multiple row blocks")
        seen.add(current)
        end = start + 1
        while end < count and traversal[end] == current:
            end += 1
        local_sequence = sequence[start:end]
        if len(local_sequence) and not np.array_equal(
            local_sequence, np.arange(local_sequence[0], local_sequence[0] + len(local_sequence))
        ):
            raise ValueError("sequence indices must be consecutive within a traversal")
        local = values[start:end, selected]
        if len(local) >= HISTORY_OBSERVATIONS:
            prefix = np.concatenate(
                (np.zeros((1, len(selected)), dtype=np.float64), np.cumsum(local, axis=0)), axis=0
            )
            for offset in range(HISTORY_OBSERVATIONS - 1, len(local)):
                recent = (prefix[offset + 1] - prefix[offset + 1 - BLOCK_OBSERVATIONS]) / BLOCK_OBSERVATIONS
                previous = (
                    prefix[offset + 1 - BLOCK_OBSERVATIONS]
                    - prefix[offset + 1 - HISTORY_OBSERVATIONS]
                ) / BLOCK_OBSERVATIONS
                row = start + offset
                delta[row] = recent - previous
                eligible[row] = True
        start = end
    return delta, eligible


def fit_robust_delta_scale(delta: np.ndarray, fit_mask: np.ndarray) -> RobustDeltaScale:
    """Fit a deterministic robust scale using only eligible fit rows."""

    values = np.asarray(delta, dtype=np.float64)
    mask = np.asarray(fit_mask, dtype=np.bool_)
    if values.ndim != 2 or mask.shape != (len(values),):
        raise ValueError("delta scale input contract drift")
    selected = values[mask]
    if len(selected) == 0 or not np.all(np.isfinite(selected)):
        raise ValueError("delta scale requires finite fit rows")
    center = np.median(selected, axis=0)
    mad = np.median(np.abs(selected - center), axis=0)
    scale = 1.4826 * mad
    # Deterministic, data-derived fallbacks avoid an arbitrary numeric floor.
    q25, q75 = np.quantile(selected, (0.25, 0.75), axis=0)
    iqr_scale = (q75 - q25) / 1.349
    std_scale = np.std(selected, axis=0)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, iqr_scale)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, std_scale)
    scale = np.where(scale > np.finfo(np.float64).eps, scale, 1.0)
    return RobustDeltaScale(center=center, scale=scale)


def normalized_delta_score(delta: np.ndarray, eligible: np.ndarray, scale: RobustDeltaScale) -> np.ndarray:
    """Compute the root-mean-square robust standardized discrepancy."""

    values = np.asarray(delta, dtype=np.float64)
    mask = np.asarray(eligible, dtype=np.bool_)
    if values.ndim != 2 or mask.shape != (len(values),) or values.shape[1] != len(scale.center):
        raise ValueError("normalized score input contract drift")
    score = np.full(len(values), np.nan, dtype=np.float64)
    standardized = (values[mask] - scale.center) / scale.scale
    score[mask] = np.sqrt(np.mean(np.square(standardized), axis=1))
    return score


def action_conditioned_state_scores(
    features: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    fit_partition: np.ndarray,
) -> tuple[StateDiscrepancyScores, Mapping[str, RobustDeltaScale]]:
    """Create geometry, action and equal-group combined discrepancy scores."""

    fit = np.asarray(fit_partition, dtype=np.bool_)
    if fit.shape != (len(features),) or not np.any(fit) or np.all(fit):
        raise ValueError("fit partition must identify a proper nonempty subset")
    geometry_delta, geometry_eligible = causal_two_block_delta(
        features, traversal_id, sequence_index, GEOMETRY_COLUMNS
    )
    action_delta, action_eligible = causal_two_block_delta(
        features, traversal_id, sequence_index, ACTION_COLUMNS
    )
    if not np.array_equal(geometry_eligible, action_eligible):
        raise RuntimeError("geometry/action causal eligibility drift")
    scale_mask = fit & geometry_eligible
    geometry_scale = fit_robust_delta_scale(geometry_delta, scale_mask)
    action_scale = fit_robust_delta_scale(action_delta, scale_mask)
    geometry = normalized_delta_score(geometry_delta, geometry_eligible, geometry_scale)
    action = normalized_delta_score(action_delta, action_eligible, action_scale)
    combined = np.full(len(features), np.nan, dtype=np.float64)
    combined[geometry_eligible] = np.sqrt(
        (np.square(geometry[geometry_eligible]) + np.square(action[geometry_eligible])) / 2.0
    )
    return (
        StateDiscrepancyScores(geometry, action, combined, geometry_eligible),
        {"geometry": geometry_scale, "action": action_scale},
    )


def fit_corridor_false_alarm_threshold(
    score: np.ndarray,
    event_index: np.ndarray,
    fit_partition: np.ndarray,
    eligible: np.ndarray,
    *,
    corridor_quantile: float = 0.99,
) -> float:
    """Freeze the higher 99th percentile of eligible fit-corridor scores."""

    values = np.asarray(score, dtype=np.float64)
    event = np.asarray(event_index, dtype=np.int64)
    fit = np.asarray(fit_partition, dtype=np.bool_)
    valid = np.asarray(eligible, dtype=np.bool_)
    if (
        any(array.shape != (len(values),) for array in (event, fit, valid))
        or not 0.0 < corridor_quantile < 1.0
    ):
        raise ValueError("corridor threshold input contract drift")
    rows = fit & valid & (event == 0)
    if not np.any(rows) or not np.all(np.isfinite(values[rows])):
        raise ValueError("corridor threshold requires finite fit-corridor rows")
    return float(np.quantile(values[rows], corridor_quantile, method="higher"))


def extract_state_trigger_rows(
    score: np.ndarray,
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    eligible: np.ndarray,
    *,
    threshold: float,
) -> np.ndarray:
    """Collapse each contiguous accepted state response into one maximum row."""

    values = np.asarray(score, dtype=np.float64)
    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    valid = np.asarray(eligible, dtype=np.bool_)
    count = len(values)
    if (
        any(array.shape != (count,) for array in (traversal, sequence, valid))
        or not np.isfinite(threshold)
        or np.any(~np.isfinite(values[valid]))
    ):
        raise ValueError("state trigger input contract drift")
    accepted = valid & (values >= threshold)
    triggers: list[int] = []
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
                triggers.append(min(active, key=lambda item: (-values[item], item)))
                active = []
        if accepted[row]:
            active.append(row)
    if active:
        triggers.append(min(active, key=lambda item: (-values[item], item)))
    return np.asarray(triggers, dtype=np.int64)


def evaluate_state_triggers(
    score: np.ndarray,
    event_index: np.ndarray,
    episode_id: np.ndarray,
    identity: Sequence[str],
    traversal_id: Sequence[str],
    sequence_index: np.ndarray,
    partition: np.ndarray,
    eligible: np.ndarray,
    *,
    threshold: float,
) -> tuple[dict, np.ndarray]:
    """Evaluate class-agnostic topology-node triggers on one frozen partition."""

    values = np.asarray(score, dtype=np.float64)
    event = np.asarray(event_index, dtype=np.int64)
    episode = np.asarray(episode_id, dtype=np.int64)
    identities = np.asarray(identity, dtype=str)
    traversal = np.asarray(traversal_id, dtype=str)
    sequence = np.asarray(sequence_index, dtype=np.int64)
    selected = np.asarray(partition, dtype=np.bool_)
    valid = np.asarray(eligible, dtype=np.bool_)
    count = len(values)
    if (
        any(array.shape != (count,) for array in (event, episode, identities, traversal, sequence, selected, valid))
        or np.any((event == 0) != (episode < 0))
    ):
        raise ValueError("state trigger evaluation contract drift")
    rows = np.flatnonzero(selected)
    local_triggers = extract_state_trigger_rows(
        values[rows], traversal[rows], sequence[rows], valid[rows], threshold=threshold
    )
    triggers = rows[local_triggers]
    true_episodes = np.unique(episode[selected & (episode >= 0)])
    matched_episodes = np.unique(episode[triggers][episode[triggers] >= 0])
    matched_episode_set = set(int(value) for value in matched_episodes)
    correct_trigger_count = int(np.sum(episode[triggers] >= 0))
    corridor_eligible = selected & valid & (event == 0)
    false_trigger_count = int(np.sum(episode[triggers] < 0))
    per_event = {}
    for index in range(1, len(EVENT_NAMES)):
        event_episodes = np.unique(episode[selected & (event == index)])
        event_matched = np.asarray(
            [value for value in event_episodes if int(value) in matched_episode_set], dtype=np.int64
        )
        event_identities = set(identities[selected & (event == index)].tolist())
        matched_identities = set(
            identities[triggers][(event[triggers] == index) & (episode[triggers] >= 0)].tolist()
        )
        per_event[EVENT_NAMES[index]] = {
            "true_episodes": int(len(event_episodes)),
            "matched_unique_episodes": int(len(event_matched)),
            "episode_recall": float(len(event_matched) / len(event_episodes)) if len(event_episodes) else 0.0,
            "true_identities": int(len(event_identities)),
            "matched_unique_identities": int(len(matched_identities)),
            "identity_coverage": float(len(matched_identities) / len(event_identities)) if event_identities else 0.0,
        }
    metrics = {
        "threshold": float(threshold),
        "eligible_observations": int(np.sum(selected & valid)),
        "eligible_corridor_observations": int(np.sum(corridor_eligible)),
        "predicted_triggers": int(len(triggers)),
        "true_episodes": int(len(true_episodes)),
        "matched_unique_episodes": int(len(matched_episodes)),
        "structural_trigger_precision": float(correct_trigger_count / len(triggers)) if len(triggers) else 0.0,
        "structural_episode_recall": float(len(matched_episodes) / len(true_episodes)) if len(true_episodes) else 0.0,
        "false_trigger_count": false_trigger_count,
        "false_trigger_fraction_per_eligible_corridor_observation": (
            float(false_trigger_count / np.sum(corridor_eligible)) if np.any(corridor_eligible) else 0.0
        ),
        "per_event": per_event,
    }
    return metrics, triggers


__all__ = [
    "ACTION_COLUMNS",
    "BLOCK_OBSERVATIONS",
    "GEOMETRY_COLUMNS",
    "HISTORY_OBSERVATIONS",
    "RobustDeltaScale",
    "StateDiscrepancyScores",
    "action_conditioned_state_scores",
    "causal_two_block_delta",
    "evaluate_state_triggers",
    "extract_state_trigger_rows",
    "fit_corridor_false_alarm_threshold",
    "fit_robust_delta_scale",
    "normalized_delta_score",
]
