"""Read-only diagnostics for the failed sparse-port relation model.

The helpers in this module never change model outputs.  They expose three
evaluation masks (deployed existence, Teacher cardinality, and a
Teacher-aligned proposal oracle), exact tie-safe score selection, and a sparse
best-link decoder that uses only the frozen learned relation score.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from mtare_topo.representation.primitive_relation_model import MAXIMUM_SLOTS


@dataclass(frozen=True)
class AttachmentScoreSlice:
    raw_score: np.ndarray
    safe_score: np.ndarray
    target: np.ndarray
    overlap_hard_negative: np.ndarray
    eligible_pairs: int
    target_pairs: int


def teacher_cardinality_topk_mask(
    existence_logits: torch.Tensor,
    target_mask: torch.Tensor,
) -> torch.Tensor:
    """Keep the predicted top-k slots, where k is the visible Teacher count.

    Teacher supplies only the count.  It does not reveal which query slots are
    matched and is therefore distinct from the proposal oracle.
    """

    if (
        existence_logits.shape != target_mask.shape
        or existence_logits.ndim != 2
        or existence_logits.shape[1] != MAXIMUM_SLOTS
    ):
        raise ValueError("cardinality attribution tensors must share [batch,32]")
    order = torch.argsort(existence_logits, dim=1, descending=True, stable=True)
    rank = torch.empty_like(order)
    source_rank = torch.arange(MAXIMUM_SLOTS, device=order.device)[None].expand_as(order)
    rank.scatter_(1, order, source_rank)
    count = target_mask.bool().sum(dim=1, keepdim=True)
    return rank < count


def attachment_upper_mask(device: torch.device) -> torch.Tensor:
    endpoint = torch.arange(2 * MAXIMUM_SLOTS, device=device)
    primitive = torch.div(endpoint, 2, rounding_mode="floor")
    return torch.triu(primitive[:, None] != primitive[None, :], diagonal=1)


def endpoint_pair_eligibility(primitive_mask: torch.Tensor) -> torch.Tensor:
    if primitive_mask.ndim != 2 or primitive_mask.shape[1] != MAXIMUM_SLOTS:
        raise ValueError("primitive mask must be [batch,32]")
    endpoint = primitive_mask.bool().repeat_interleave(2, dim=1)
    return endpoint[:, :, None] & endpoint[:, None, :] & attachment_upper_mask(endpoint.device)[None]


def best_link_pair_mask(score: torch.Tensor, eligible: torch.Tensor, *, mutual: bool) -> torch.Tensor:
    """Return sparse learned-score links before applying a score threshold.

    ``mutual=False`` keeps the undirected union of every endpoint's best link;
    ``mutual=True`` keeps only reciprocal best links.  No geometry threshold,
    Teacher relation, or hand-written topology label participates.
    """

    if score.shape != eligible.shape or score.ndim != 3 or score.shape[1:] != (64, 64):
        raise ValueError("best-link tensors must share [batch,64,64]")
    masked = score.masked_fill(~eligible, -torch.inf)
    best_score, best = masked.max(dim=2)
    valid = torch.isfinite(best_score)
    directed = torch.zeros_like(eligible)
    directed.scatter_(2, best[..., None], valid[..., None])
    if mutual:
        linked = directed & directed.transpose(1, 2)
    else:
        linked = directed | directed.transpose(1, 2)
    return linked & attachment_upper_mask(score.device)[None]


def attachment_score_slice(
    attachment_logits: torch.Tensor,
    attachment_uncertainty: torch.Tensor,
    aligned_attachment: torch.Tensor,
    aligned_overlap: torch.Tensor,
    primitive_mask: torch.Tensor,
    *,
    decoder: str = "independent",
) -> AttachmentScoreSlice:
    """Flatten eligible learned attachment scores and diagnostic labels."""

    batch = len(primitive_mask)
    raw = torch.sigmoid(attachment_logits).reshape(batch, 64, 64)
    uncertainty = attachment_uncertainty.reshape_as(raw)
    target = aligned_attachment.reshape_as(raw).bool() & attachment_upper_mask(raw.device)[None]
    eligible = endpoint_pair_eligibility(primitive_mask)
    if decoder == "best_link_union":
        eligible &= best_link_pair_mask(raw, eligible, mutual=False)
    elif decoder == "best_link_mutual":
        eligible &= best_link_pair_mask(raw, eligible, mutual=True)
    elif decoder != "independent":
        raise ValueError(f"unknown sparse-port attribution decoder: {decoder}")
    overlap = aligned_overlap.bool()
    overlap_endpoint = overlap.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    selected_raw = raw[eligible].detach().cpu().numpy().astype(np.float32, copy=False)
    selected_safe = (raw * (1.0 - uncertainty))[eligible].detach().cpu().numpy().astype(np.float32, copy=False)
    selected_target = target[eligible].detach().cpu().numpy().astype(np.bool_, copy=False)
    selected_overlap = overlap_endpoint[eligible].detach().cpu().numpy().astype(np.bool_, copy=False)
    return AttachmentScoreSlice(
        raw_score=selected_raw,
        safe_score=selected_safe,
        target=selected_target,
        overlap_hard_negative=selected_overlap,
        eligible_pairs=int(eligible.sum()),
        target_pairs=int(target.sum()),
    )


def exact_ranked_selection(
    score: np.ndarray,
    target: np.ndarray,
    *,
    total_positive: int,
    minimum_precision: float | None = None,
) -> dict[str, float | int | bool]:
    """Select an exact score threshold without splitting equal-score ties."""

    score = np.asarray(score, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.bool_).reshape(-1)
    if score.shape != target.shape or total_positive < int(np.count_nonzero(target)):
        raise ValueError("ranked-selection population drift")
    if score.size == 0:
        return {
            "available": False, "threshold": None, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "true_positive": 0,
            "false_positive": 0, "false_negative": int(total_positive),
            "selected_pairs": 0,
        }
    order = np.argsort(score, kind="stable")[::-1]
    ordered_score = score[order]
    ordered_target = target[order]
    cumulative_true = np.cumsum(ordered_target, dtype=np.int64)
    population = np.arange(1, len(order) + 1, dtype=np.int64)
    tie_end = np.empty(len(order), dtype=np.bool_)
    tie_end[-1] = True
    tie_end[:-1] = ordered_score[:-1] != ordered_score[1:]
    indices = np.flatnonzero(tie_end)
    true_positive = cumulative_true[indices]
    selected = population[indices]
    false_positive = selected - true_positive
    false_negative = int(total_positive) - true_positive
    precision = true_positive / selected
    recall = true_positive / max(int(total_positive), 1)
    denominator = 2 * true_positive + false_positive + false_negative
    f1 = np.divide(2 * true_positive, denominator, out=np.zeros_like(precision), where=denominator > 0)
    valid = np.ones(len(indices), dtype=np.bool_)
    if minimum_precision is not None:
        valid &= precision >= float(minimum_precision)
        valid &= true_positive > 0
    candidates = np.flatnonzero(valid)
    if candidates.size == 0:
        return {
            "available": False, "threshold": None, "precision": 0.0,
            "recall": 0.0, "f1": 0.0, "true_positive": 0,
            "false_positive": 0, "false_negative": int(total_positive),
            "selected_pairs": 0,
        }
    if minimum_precision is None:
        selected_index = int(max(candidates, key=lambda i: (f1[i], precision[i], ordered_score[indices[i]])))
    else:
        selected_index = int(max(candidates, key=lambda i: (recall[i], precision[i], ordered_score[indices[i]])))
    return {
        "available": True,
        "threshold": float(ordered_score[indices[selected_index]]),
        "precision": float(precision[selected_index]),
        "recall": float(recall[selected_index]),
        "f1": float(f1[selected_index]),
        "true_positive": int(true_positive[selected_index]),
        "false_positive": int(false_positive[selected_index]),
        "false_negative": int(false_negative[selected_index]),
        "selected_pairs": int(selected[selected_index]),
    }


def score_quantiles(score: np.ndarray, target: np.ndarray) -> dict[str, list[float] | int]:
    score = np.asarray(score, dtype=np.float32).reshape(-1)
    target = np.asarray(target, dtype=np.bool_).reshape(-1)
    if score.shape != target.shape:
        raise ValueError("score quantile population drift")
    quantiles = (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)

    def values(mask: np.ndarray) -> list[float]:
        selected = score[mask]
        return [float(value) for value in np.quantile(selected, quantiles)] if len(selected) else []

    return {
        "positive_count": int(np.count_nonzero(target)),
        "negative_count": int(np.count_nonzero(~target)),
        "quantile_probabilities": list(quantiles),
        "positive": values(target),
        "negative": values(~target),
    }


__all__ = [
    "AttachmentScoreSlice",
    "attachment_score_slice",
    "attachment_upper_mask",
    "best_link_pair_mask",
    "endpoint_pair_eligibility",
    "exact_ranked_selection",
    "score_quantiles",
    "teacher_cardinality_topk_mask",
]
