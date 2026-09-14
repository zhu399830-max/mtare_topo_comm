"""Deterministic V2 geometry transfer and relation-only reset for V3."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import torch

from mtare_topo.representation.primitive_relation_observable_model import ObservableSparsePortRelationNet


RELATION_TRAINABLE_PREFIXES = (
    "endpoint_token_embedding.",
    "endpoint_relation_transformer.",
    "attachment_relation_head.",
    "attachment_uncertainty_head.",
    "primitive_relation_transformer.",
    "overlap_relation_head.",
    "overlap_uncertainty_head.",
    "endpoint_evidence_head.",
)


@dataclass(frozen=True)
class ObservableInitializationReport:
    checkpoint_seed: int
    checkpoint_epoch: int
    transferred_tensors: int
    reset_relation_tensors: int
    total_parameters: int
    trainable_parameters: int
    frozen_parameters: int


def is_relation_trainable_parameter(name: str) -> bool:
    return name.startswith(RELATION_TRAINABLE_PREFIXES)


def set_observable_relation_training_mode(
    model: ObservableSparsePortRelationNet,
) -> None:
    """Keep the transferred backbone deterministic while relation modules train."""

    model.eval()
    for name, module in model.named_modules():
        if name and is_relation_trainable_parameter(name + "."):
            module.train()


def initialize_observable_relation_from_state(
    checkpoint: Mapping[str, object],
    *,
    initialization_seed: int,
) -> tuple[ObservableSparsePortRelationNet, ObservableInitializationReport]:
    """Load only proven geometry/temporal weights; reset all relation layers."""

    if checkpoint.get("schema_version") != "primitive_relation_sparse_port_checkpoint_v1":
        raise ValueError("unsupported sparse-port checkpoint schema")
    state = checkpoint.get("model_state_dict")
    if not isinstance(state, Mapping) or not state:
        raise ValueError("sparse-port checkpoint model state is empty")
    torch.manual_seed(int(initialization_seed))
    model = ObservableSparsePortRelationNet()
    transfer = {str(name): value for name, value in state.items() if not is_relation_trainable_parameter(str(name))}
    expected_reset = {name for name in model.state_dict() if is_relation_trainable_parameter(name)}
    incompatible = model.load_state_dict(transfer, strict=False)
    if set(incompatible.missing_keys) != expected_reset or incompatible.unexpected_keys:
        raise RuntimeError(
            f"observable transfer boundary drift: missing={incompatible.missing_keys} unexpected={incompatible.unexpected_keys}"
        )
    model_state = model.state_dict()
    for name, value in transfer.items():
        if not torch.equal(model_state[name], value):
            raise RuntimeError(f"transferred geometry tensor differs: {name}")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(is_relation_trainable_parameter(name))
    total = sum(value.numel() for value in model.parameters())
    trainable = sum(value.numel() for value in model.parameters() if value.requires_grad)
    if trainable <= 0 or trainable >= total:
        raise RuntimeError("relation-only trainable parameter boundary is invalid")
    report = ObservableInitializationReport(
        checkpoint_seed=int(checkpoint.get("seed", -1)),
        checkpoint_epoch=int(checkpoint.get("epoch", -1)),
        transferred_tensors=len(transfer),
        reset_relation_tensors=len(expected_reset),
        total_parameters=total,
        trainable_parameters=trainable,
        frozen_parameters=total - trainable,
    )
    return model, report


def initialize_observable_relation_from_checkpoint(
    path: Path,
    *,
    initialization_seed: int,
    map_location: str | torch.device = "cpu",
) -> tuple[ObservableSparsePortRelationNet, ObservableInitializationReport]:
    checkpoint = torch.load(Path(path), map_location=map_location, weights_only=False)
    if not isinstance(checkpoint, Mapping):
        raise ValueError("sparse-port checkpoint must contain a mapping")
    return initialize_observable_relation_from_state(
        checkpoint, initialization_seed=initialization_seed,
    )


__all__ = [
    "ObservableInitializationReport", "RELATION_TRAINABLE_PREFIXES",
    "initialize_observable_relation_from_checkpoint",
    "initialize_observable_relation_from_state", "is_relation_trainable_parameter",
    "set_observable_relation_training_mode",
]
