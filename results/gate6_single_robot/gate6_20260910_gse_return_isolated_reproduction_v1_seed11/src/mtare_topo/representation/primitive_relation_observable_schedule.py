"""Frozen relation-only schedule after V2 geometry transfer."""

from __future__ import annotations

from typing import Mapping

import torch

from mtare_topo.representation.primitive_relation_training import (
    EVALUATION_BATCH_SIZE,
    GRADIENT_CLIP_NORM,
    LEARNING_RATE,
    TRAINING_BATCH_SIZE,
    WEIGHT_DECAY,
)


TRAINING_EPOCHS = 3


def training_stage(epoch: int) -> str:
    if epoch < 0 or epoch >= TRAINING_EPOCHS:
        raise ValueError("observable relation epoch outside frozen schedule")
    return "observable_relation_finetuning"


def relation_training_objective(
    families: Mapping[str, torch.Tensor],
) -> torch.Tensor:
    required = {
        "primitive_set_parameters", "surface_reconstruction", "ray_free_space",
        "port_relations", "temporal_equivariance", "uncertainty_calibration",
        "total",
    }
    if set(families) != required:
        raise ValueError("observable relation loss-family inventory drift")
    return torch.stack((
        families["port_relations"], families["uncertainty_calibration"],
    )).mean()


__all__ = [
    "EVALUATION_BATCH_SIZE", "GRADIENT_CLIP_NORM", "LEARNING_RATE",
    "TRAINING_BATCH_SIZE", "TRAINING_EPOCHS", "WEIGHT_DECAY",
    "relation_training_objective", "training_stage",
]
