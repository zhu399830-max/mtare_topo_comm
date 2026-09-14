"""Teacher-slot alignment and losses for the O(E) composition-anchor head."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mtare_topo.data.primitive_composition_anchor_batches import (
    CompositionAnchorNumpyBatch,
)

from mtare_topo.representation.primitive_composition_anchor_model import (
    CompositionAnchorPrediction,
    composition_anchor_losses,
)
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveAssignment,
    PrimitiveRelationLossTargets,
    align_primitive_relation_targets,
)
from mtare_topo.representation.primitive_relation_model import MAXIMUM_SLOTS
from mtare_topo.representation.primitive_relation_observable_training import (
    ObservablePrimitiveRelationTorchBatch,
    observable_numpy_batch_to_torch,
)


@dataclass(frozen=True)
class AlignedCompositionAnchorTargets:
    anchor_current_sensor_m: torch.Tensor
    primitive_mask: torch.Tensor
    endpoint_observed: torch.Tensor
    attachment: torch.Tensor
    disconnected_overlap: torch.Tensor


@dataclass(frozen=True)
class CompositionAnchorTorchBatch:
    base: ObservablePrimitiveRelationTorchBatch
    anchor_current_sensor_m: torch.Tensor


def composition_anchor_numpy_batch_to_torch(
    batch: CompositionAnchorNumpyBatch,
    *,
    device: torch.device,
) -> CompositionAnchorTorchBatch:
    base = observable_numpy_batch_to_torch(batch.base, device=device)
    anchor = torch.from_numpy(batch.anchor_current_sensor_m).to(
        device=device, dtype=torch.float32,
    )
    active = base.targets.primitive_mask.bool()
    if tuple(anchor.shape) != (len(active), MAXIMUM_SLOTS, 2, 3):
        raise ValueError("composition-anchor Torch target shape drift")
    if not bool(torch.isfinite(anchor[active]).all()) or bool((anchor[~active] != 0).any()):
        raise ValueError("composition-anchor Torch target value drift")
    return CompositionAnchorTorchBatch(base=base, anchor_current_sensor_m=anchor)


def align_composition_anchor_targets(
    anchor_current_sensor_m: torch.Tensor,
    targets: PrimitiveRelationLossTargets,
    assignments: tuple[PrimitiveAssignment, ...],
) -> AlignedCompositionAnchorTargets:
    """Move Teacher-order endpoint anchors into exchangeable query order."""

    batch = len(targets.primitive_mask)
    if anchor_current_sensor_m.shape != (batch, MAXIMUM_SLOTS, 2, 3):
        raise ValueError("Teacher composition anchors must be [B,32,2,3]")
    targets.validate()
    if targets.endpoint_observed is None:
        raise ValueError("composition-anchor alignment requires endpoint observability")
    active = targets.primitive_mask.bool()
    if not bool(torch.isfinite(anchor_current_sensor_m[active]).all()):
        raise ValueError("active Teacher composition anchors must be finite")
    if len(assignments) != batch:
        raise ValueError("composition-anchor assignment batch differs")

    aligned_relation = align_primitive_relation_targets(targets, assignments)
    aligned_anchor = torch.zeros_like(anchor_current_sensor_m)
    for batch_index, assignment in enumerate(assignments):
        matched = torch.nonzero(
            assignment.predicted_to_target >= 0, as_tuple=False,
        ).flatten()
        source = assignment.predicted_to_target[matched]
        value = anchor_current_sensor_m[batch_index, source]
        reverse = assignment.endpoint_reversed[matched]
        value = torch.where(reverse[:, None, None], value.flip(1), value)
        aligned_anchor[batch_index, matched] = value
    return AlignedCompositionAnchorTargets(
        anchor_current_sensor_m=aligned_anchor,
        primitive_mask=aligned_relation["mask"],
        endpoint_observed=aligned_relation["endpoint_observed"],
        attachment=aligned_relation["attachment"],
        disconnected_overlap=aligned_relation["overlap"],
    )


def composition_anchor_training_losses(
    prediction: CompositionAnchorPrediction,
    aligned: AlignedCompositionAnchorTargets,
) -> dict[str, torch.Tensor]:
    return composition_anchor_losses(
        prediction,
        anchor_target_current_sensor_m=aligned.anchor_current_sensor_m,
        primitive_mask=aligned.primitive_mask,
        endpoint_observed=aligned.endpoint_observed,
        attachment_target=aligned.attachment,
        disconnected_overlap=aligned.disconnected_overlap,
    )


__all__ = [
    "AlignedCompositionAnchorTargets",
    "CompositionAnchorTorchBatch",
    "align_composition_anchor_targets",
    "composition_anchor_numpy_batch_to_torch",
    "composition_anchor_training_losses",
]
