"""Pure diagnostics for attributing primitive-relation proposal failures.

The oracle masks in this module are evaluation-only.  They answer whether the
frozen relation logits would be useful if redundant primitive proposals were
removed; they are never a deployment input or a replacement for perception.
"""

from __future__ import annotations

from typing import Iterable

import torch

from mtare_topo.evaluation.primitive_relation_metrics import (
    BinaryCounts,
    THRESHOLD_GRID,
    threshold_sweep_counts,
)
from mtare_topo.representation.primitive_relation_model import (
    MAXIMUM_SLOTS,
    PrimitiveRelationPrediction,
)


def pair_space_counts(primitive_mask: torch.Tensor) -> dict[str, int]:
    """Count eligible unordered attachment and overlap pairs.

    Each primitive has two endpoints.  Attachment excludes the two endpoints
    of the same primitive, so an observation with ``n`` primitives has
    ``2*n*(n-1)`` unordered cross-primitive endpoint pairs.  Primitive overlap
    has ``n*(n-1)/2`` unordered pairs.
    """

    mask = primitive_mask.bool()
    if mask.ndim != 2 or mask.shape[1] != MAXIMUM_SLOTS:
        raise ValueError("primitive attribution mask must be [batch,32]")
    count = mask.sum(dim=1, dtype=torch.int64)
    attachment = 2 * count * (count - 1)
    overlap = count * (count - 1) // 2
    return {
        "rows": int(len(mask)),
        "primitives": int(count.sum()),
        "attachment_pairs": int(attachment.sum()),
        "overlap_pairs": int(overlap.sum()),
    }


def slot_population_counts(
    active_mask: torch.Tensor,
    target_mask: torch.Tensor,
) -> dict[str, int | list[int]]:
    """Count matched, missed, and redundant active primitive slots."""

    active = active_mask.bool()
    target = target_mask.bool()
    if active.shape != target.shape or active.ndim != 2 or active.shape[1] != MAXIMUM_SLOTS:
        raise ValueError("slot population masks must share [batch,32]")
    histogram = torch.bincount(active.sum(dim=1), minlength=MAXIMUM_SLOTS + 1)
    return {
        "rows": int(len(active)),
        "active": int(active.sum()),
        "targets": int(target.sum()),
        "matched_active": int((active & target).sum()),
        "missed_targets": int((~active & target).sum()),
        "redundant_active": int((active & ~target).sum()),
        "active_count_histogram": [int(value) for value in histogram.tolist()],
    }


def relation_sweeps_for_mask(
    prediction: PrimitiveRelationPrediction,
    aligned: dict[str, torch.Tensor],
    primitive_eligible: torch.Tensor,
    *,
    thresholds: Iterable[float] = THRESHOLD_GRID,
) -> dict[str, tuple[BinaryCounts, ...]]:
    """Evaluate frozen relation logits under an explicit primitive mask.

    Passing ``aligned['mask']`` is the proposal oracle: only slots assigned to
    a visible Teacher primitive are eligible.  The relation probabilities are
    still the model's unchanged predictions.
    """

    eligible = primitive_eligible.bool()
    target_mask = aligned["mask"].bool()
    if eligible.shape != target_mask.shape or eligible.ndim != 2 or eligible.shape[1] != MAXIMUM_SLOTS:
        raise ValueError("relation attribution mask must match aligned [batch,32]")
    endpoint_eligible = eligible.repeat_interleave(2, dim=1)
    endpoint = torch.arange(2 * MAXIMUM_SLOTS, device=eligible.device)
    primitive = torch.div(endpoint, 2, rounding_mode="floor")
    attachment_upper = torch.triu(primitive[:, None] != primitive[None, :], diagonal=1)
    attachment_eligible = (
        endpoint_eligible[:, :, None]
        & endpoint_eligible[:, None, :]
        & attachment_upper[None]
    )
    attachment_target = aligned["attachment"].reshape(
        len(eligible), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    ).bool() & attachment_upper[None]

    primitive_upper = torch.triu(
        torch.ones(MAXIMUM_SLOTS, MAXIMUM_SLOTS, dtype=torch.bool, device=eligible.device),
        diagonal=1,
    )
    overlap_eligible = eligible[:, :, None] & eligible[:, None, :] & primitive_upper[None]
    overlap_target = aligned["overlap"].bool() & primitive_upper[None]
    return {
        "attachment": threshold_sweep_counts(
            torch.sigmoid(prediction.endpoint_attachment_logits),
            attachment_target,
            eligible=attachment_eligible,
            thresholds=thresholds,
        ),
        "disconnected_overlap": threshold_sweep_counts(
            torch.sigmoid(prediction.disconnected_overlap_logits),
            overlap_target,
            eligible=overlap_eligible,
            thresholds=thresholds,
        ),
    }


__all__ = [
    "pair_space_counts",
    "relation_sweeps_for_mask",
    "slot_population_counts",
]
