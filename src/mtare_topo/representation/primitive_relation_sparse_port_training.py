"""Frozen seven-epoch schedule for the sparse-port corrective."""

from __future__ import annotations

from typing import Mapping

import torch

from mtare_topo.representation.primitive_relation_training import (
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    TRAINING_BATCH_SIZE,
    WEIGHT_DECAY,
    numpy_batch_to_torch,
)


TRAINING_EPOCHS = 7


def training_stage(epoch: int) -> str:
    if epoch < 0 or epoch >= TRAINING_EPOCHS:
        raise ValueError("sparse-port epoch outside frozen schedule")
    if epoch == 0:
        return "geometry_pretraining"
    if epoch == 1:
        return "sparse_set_mdl_pretraining"
    if epoch == 2:
        return "endpoint_relation_pretraining"
    if epoch == 3:
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
        raise ValueError("sparse-port loss-family inventory drift")
    stage = training_stage(epoch)
    if stage == "geometry_pretraining":
        names = ("primitive_set_parameters", "surface_reconstruction", "ray_free_space")
    elif stage == "sparse_set_mdl_pretraining":
        names = ("primitive_set_parameters",)
    elif stage == "endpoint_relation_pretraining":
        names = ("port_relations",)
    elif stage == "temporal_uncertainty_pretraining":
        names = ("temporal_equivariance", "uncertainty_calibration")
    else:
        return families["total"]
    return torch.stack(tuple(families[name] for name in names)).mean()


__all__ = [
    "EVALUATION_BATCH_SIZE", "GRADIENT_CLIP_NORM", "LEARNING_RATE",
    "TRAINING_BATCH_SIZE", "TRAINING_EPOCHS", "WEIGHT_DECAY",
    "numpy_batch_to_torch", "staged_training_loss", "training_stage",
]
