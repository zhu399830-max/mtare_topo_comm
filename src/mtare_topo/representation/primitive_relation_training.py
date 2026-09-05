"""Frozen batching and staged objective for primitive-relation training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from mtare_topo.data.primitive_relation_batches import PrimitiveRelationNumpyBatch
from mtare_topo.representation.primitive_relation_losses import PrimitiveRelationLossTargets


TRAINING_EPOCHS = 6
TRAINING_BATCH_SIZE = 16
EVALUATION_BATCH_SIZE = 128
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
GRADIENT_CLIP_NORM = 1.0


@dataclass(frozen=True)
class PrimitiveRelationTorchBatch:
    range_valid: torch.Tensor
    relative_translation_current_sensor_m: torch.Tensor
    relative_yaw_current_sensor_deg: torch.Tensor
    targets: PrimitiveRelationLossTargets


def numpy_batch_to_torch(
    batch: PrimitiveRelationNumpyBatch,
    *,
    device: torch.device,
) -> PrimitiveRelationTorchBatch:
    def floating(value) -> torch.Tensor:
        return torch.from_numpy(value).to(device=device, dtype=torch.float32, non_blocking=False)

    result = PrimitiveRelationTorchBatch(
        range_valid=floating(batch.range_valid),
        relative_translation_current_sensor_m=floating(
            batch.relative_translation_current_sensor_m,
        ),
        relative_yaw_current_sensor_deg=floating(batch.relative_yaw_current_sensor_deg),
        targets=PrimitiveRelationLossTargets(
            primitive_mask=floating(batch.primitive_mask),
            axis_control_current_sensor_m=floating(batch.axis_control_current_sensor_m),
            endpoint_half_axes_m=floating(batch.endpoint_half_axes_m),
            endpoint_shape_exponent=floating(batch.endpoint_shape_exponent),
            temporal_visibility=floating(batch.temporal_visibility),
            endpoint_attachment=floating(batch.endpoint_attachment),
            disconnected_overlap=floating(batch.disconnected_overlap),
        ),
    )
    result.targets.validate()
    return result


def training_stage(epoch: int) -> str:
    if epoch < 0 or epoch >= TRAINING_EPOCHS:
        raise ValueError("primitive-relation epoch outside frozen schedule")
    if epoch == 0:
        return "geometry_pretraining"
    if epoch == 1:
        return "relation_pretraining"
    if epoch == 2:
        return "temporal_uncertainty_pretraining"
    return "joint_finetuning"


def staged_training_loss(
    families: Mapping[str, torch.Tensor],
    *,
    epoch: int,
) -> torch.Tensor:
    required = {
        "primitive_set_parameters", "surface_reconstruction", "ray_free_space",
        "port_relations", "temporal_equivariance", "uncertainty_calibration", "total",
    }
    if set(families) != required:
        raise ValueError("primitive-relation loss-family inventory drift")
    stage = training_stage(epoch)
    if stage == "geometry_pretraining":
        names = ("primitive_set_parameters", "surface_reconstruction", "ray_free_space")
    elif stage == "relation_pretraining":
        names = ("port_relations",)
    elif stage == "temporal_uncertainty_pretraining":
        names = ("temporal_equivariance", "uncertainty_calibration")
    else:
        return families["total"]
    return torch.stack(tuple(families[name] for name in names)).mean()


__all__ = [
    "EVALUATION_BATCH_SIZE", "GRADIENT_CLIP_NORM", "LEARNING_RATE",
    "PrimitiveRelationTorchBatch", "TRAINING_BATCH_SIZE", "TRAINING_EPOCHS",
    "WEIGHT_DECAY", "numpy_batch_to_torch", "staged_training_loss",
    "training_stage",
]
