"""Analytic class mass for the causal episode MIL objective."""

from __future__ import annotations

import math

import numpy as np

from mtare_topo.semantics.geometric_semantics import EVENT_NAMES


def analytic_episode_class_mass(
    probability: np.ndarray,
    event_target: np.ndarray,
    episode_id: np.ndarray,
) -> dict:
    """Measure raw versus equal-class loss/gradient allocation without backprop."""

    values = np.asarray(probability, dtype=np.float64)
    target = np.asarray(event_target, dtype=np.int64)
    episodes = np.asarray(episode_id, dtype=np.int64)
    if (
        values.shape != (len(target), len(EVENT_NAMES))
        or episodes.shape != target.shape
        or not np.all(np.isfinite(values))
        or np.any(values <= 0.0)
        or not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=1e-5)
        or np.any((target == 0) != (episodes < 0))
    ):
        raise ValueError("episode class-mass input contract drift")
    records = {index: [] for index in range(1, len(EVENT_NAMES))}
    for episode in np.unique(episodes[episodes >= 0]):
        rows = np.flatnonzero(episodes == episode)
        classes = np.unique(target[rows])
        if len(classes) != 1 or int(classes[0]) == 0:
            raise ValueError("one positive episode must contain exactly one structural class")
        event = int(classes[0])
        row = int(rows[np.argmax(values[rows, event])])
        structural = 1.0 - values[row, 0]
        conditional = values[row, 1:] / structural
        conditional_gradient = conditional.copy()
        conditional_gradient[event - 1] -= 1.0
        structural_gradient = -values[row, 0]
        records[event].append({
            "loss": -math.log(values[row, event]),
            "structural_gradient_abs": abs(structural_gradient),
            "conditional_gradient_l2": float(np.linalg.norm(conditional_gradient)),
            "joint_gradient_l2": float(math.sqrt(
                structural_gradient**2 + float(np.dot(conditional_gradient, conditional_gradient))
            )),
        })
    counts = {event: len(rows) for event, rows in records.items()}
    if any(value == 0 for value in counts.values()):
        raise ValueError("every structural class must contain an episode")
    total = sum(counts.values())
    raw_gradient_total = sum(
        sum(row["joint_gradient_l2"] for row in rows) for rows in records.values()
    )
    equal_gradient_denominator = sum(
        np.mean([row["joint_gradient_l2"] for row in rows]) for rows in records.values()
    )
    result = {}
    for event, rows in records.items():
        mean_gradient = float(np.mean([row["joint_gradient_l2"] for row in rows]))
        gradient_sum = float(np.sum([row["joint_gradient_l2"] for row in rows]))
        result[EVENT_NAMES[event]] = {
            "episodes": len(rows),
            "raw_episode_mass_share": len(rows) / total,
            "equal_class_per_episode_weight_multiplier": total / ((len(EVENT_NAMES) - 1) * len(rows)),
            "mean_joint_loss": float(np.mean([row["loss"] for row in rows])),
            "mean_structural_gradient_abs": float(np.mean([row["structural_gradient_abs"] for row in rows])),
            "mean_conditional_gradient_l2": float(np.mean([row["conditional_gradient_l2"] for row in rows])),
            "mean_joint_gradient_l2": mean_gradient,
            "raw_joint_gradient_mass_share": gradient_sum / raw_gradient_total,
            "equal_class_joint_gradient_mass_share": mean_gradient / equal_gradient_denominator,
        }
    return {"episodes": total, "per_event": result}


__all__ = ["analytic_episode_class_mass"]
