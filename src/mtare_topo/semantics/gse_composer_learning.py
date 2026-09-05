"""Learning/evaluation contracts for the factorized GSE event Composers."""

from __future__ import annotations

from dataclasses import replace
from typing import Sequence

import numpy as np
import torch

from mtare_topo.representation.gse_typed_composers import (
    ActionSetComposerInput,
    MetricChangeComposerInput,
)


EVENT_NAMES = ("corridor", "junction", "terminal", "turn", "geometry_transition")
STEPS_AGO_SUPPORT = (4.0, 3.0, 2.0, 1.0, 0.0)


def fuse_factorized_event_logits(
    action_logits: torch.Tensor, metric_logits: torch.Tensor,
) -> torch.Tensor:
    """Combine mutually exclusive action/metric factors into five event logits."""

    if action_logits.ndim != 2 or action_logits.shape[1] != 3 or metric_logits.shape != action_logits.shape:
        raise ValueError("factorized Composer logits must align as [B,3]")
    action = torch.log_softmax(action_logits, dim=-1)
    metric = torch.log_softmax(metric_logits, dim=-1)
    joint = torch.stack((
        action[:, 0] + metric[:, 0],
        action[:, 1] + metric[:, 0],
        action[:, 2] + metric[:, 0],
        action[:, 0] + metric[:, 1],
        action[:, 0] + metric[:, 2],
    ), dim=-1)
    return joint - torch.logsumexp(joint, dim=-1, keepdim=True)


def fractional_backprojection_distribution(
    steps_ago: torch.Tensor, valid: torch.Tensor,
) -> torch.Tensor:
    """Linearly place a physical fractional delay on the fixed 4..0 support."""

    if steps_ago.ndim != 1 or valid.shape != steps_ago.shape or valid.dtype != torch.bool:
        raise ValueError("backprojection delay and validity must align as [B]")
    if not bool(torch.isfinite(steps_ago).all()) or bool((
        valid & ((steps_ago < 0.0) | (steps_ago > 4.0))
    ).any()):
        raise ValueError("valid backprojection delay must be finite and inside [0,4]")
    support = steps_ago.new_tensor(STEPS_AGO_SUPPORT)
    distance = (steps_ago[:, None] - support[None]).abs()
    weights = (1.0 - distance).clamp_min(0.0)
    weights = weights * valid[:, None]
    normalizer = weights.sum(dim=-1, keepdim=True)
    if bool(valid.any()) and not bool(torch.allclose(
        normalizer[valid], torch.ones_like(normalizer[valid]), atol=1e-6, rtol=0.0,
    )):
        raise RuntimeError("fractional backprojection interpolation lost unit mass")
    return weights


def neutralize_transport(inputs: ActionSetComposerInput) -> ActionSetComposerInput:
    """Remove relation information while retaining a valid typed probability input."""

    row = torch.full_like(inputs.transport_row_probability, 1.0 / 7.0)
    reveal = torch.full_like(inputs.transport_reveal_probability, 0.5)
    return replace(inputs, transport_row_probability=row, transport_reveal_probability=reveal)


def single_frame_action(inputs: ActionSetComposerInput) -> ActionSetComposerInput:
    mask = torch.zeros_like(inputs.valid_history_mask); mask[:, -1] = True
    return replace(neutralize_transport(inputs), valid_history_mask=mask)


def single_frame_metric(inputs: MetricChangeComposerInput) -> MetricChangeComposerInput:
    mask = torch.zeros_like(inputs.valid_history_mask); mask[:, -1] = True
    current = inputs.geometry_sequence[:, -1:, :].expand_as(inputs.geometry_sequence).clone()
    uncertainty = inputs.geometry_uncertainty[:, -1:, :].expand_as(inputs.geometry_uncertainty).clone()
    return replace(inputs, geometry_sequence=current, geometry_uncertainty=uncertainty, valid_history_mask=mask)


def macro_f1(truth: Sequence[int], prediction: Sequence[int], class_count: int = 5) -> dict[str, object]:
    expected = np.asarray(truth, dtype=np.int64); actual = np.asarray(prediction, dtype=np.int64)
    if expected.ndim != 1 or actual.shape != expected.shape or len(expected) == 0:
        raise ValueError("classification arrays must be aligned and nonempty")
    per_class = {}
    for index in range(class_count):
        tp = int(np.sum((expected == index) & (actual == index)))
        fp = int(np.sum((expected != index) & (actual == index)))
        fn = int(np.sum((expected == index) & (actual != index)))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[EVENT_NAMES[index] if class_count == 5 else str(index)] = {
            "precision": precision, "recall": recall, "f1": f1,
            "support": int(np.sum(expected == index)),
        }
    return {"macro_f1": float(np.mean([row["f1"] for row in per_class.values()])), "per_class": per_class}


def structural_acceptance_metrics(
    truth: Sequence[int], prediction: Sequence[int], reliability: Sequence[float], threshold: float,
) -> dict[str, float | int]:
    expected = np.asarray(truth, dtype=np.int64); actual = np.asarray(prediction, dtype=np.int64)
    score = np.asarray(reliability, dtype=np.float64)
    if expected.shape != actual.shape or score.shape != expected.shape or expected.ndim != 1:
        raise ValueError("selective event arrays must align")
    accepted = (actual != 0) & (score >= float(threshold))
    correct = accepted & (actual == expected) & (expected != 0)
    true_count = int(correct.sum()); accepted_count = int(accepted.sum()); eligible = int((expected != 0).sum())
    precision = true_count / accepted_count if accepted_count else 0.0
    recall = true_count / eligible if eligible else 0.0
    return {
        "threshold": float(threshold), "accepted": accepted_count, "true_positive": true_count,
        "false_positive": accepted_count - true_count, "precision": precision, "recall": recall,
        "false_accept_rate": (accepted_count - true_count) / accepted_count if accepted_count else 0.0,
    }


def identity_coverage(
    truth: Sequence[int], prediction: Sequence[int], reliability: Sequence[float], threshold: float,
    identities: Sequence[str],
) -> dict[str, dict[str, int | float]]:
    expected=np.asarray(truth,dtype=np.int64);actual=np.asarray(prediction,dtype=np.int64)
    score=np.asarray(reliability,dtype=np.float64);identity=np.asarray(identities).astype(str)
    if not (expected.shape==actual.shape==score.shape==identity.shape):raise ValueError("identity coverage arrays must align")
    output={}
    for index,name in enumerate(EVENT_NAMES[1:],start=1):
        total=set(identity[(expected==index)&(identity!="")].tolist())
        covered=set(identity[(expected==index)&(actual==index)&(score>=threshold)&(identity!="")].tolist())
        output[name]={"covered":len(covered),"total":len(total),"coverage":len(covered)/len(total) if total else 0.0}
    return output


__all__ = [
    "EVENT_NAMES", "STEPS_AGO_SUPPORT", "fractional_backprojection_distribution",
    "fuse_factorized_event_logits", "identity_coverage", "macro_f1", "neutralize_transport",
    "single_frame_action", "single_frame_metric", "structural_acceptance_metrics",
]
