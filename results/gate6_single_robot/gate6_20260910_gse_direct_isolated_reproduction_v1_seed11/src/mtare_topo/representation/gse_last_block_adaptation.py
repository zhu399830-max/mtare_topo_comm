"""Bounded trainability contract for the GSE causal geometry fallback."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from mtare_topo.representation.gse_causal_geometry_delta import (
    CausalGeometryDeltaEventHead,
)
from mtare_topo.representation.gse_graph import GeometrySemanticEventNet


EXPECTED_LAST_BLOCK_PARAMETERS = 271_104
EXPECTED_MULTITASK_HEAD_PARAMETERS = 444_425
EXPECTED_TOTAL_TRAINABLE_PARAMETERS = (
    EXPECTED_LAST_BLOCK_PARAMETERS + EXPECTED_MULTITASK_HEAD_PARAMETERS
)


def configure_last_block_adaptation(
    backbone: GeometrySemanticEventNet,
    head: CausalGeometryDeltaEventHead,
) -> dict[str, Any]:
    """Freeze everything except ``encoder[-1]`` and the fresh multitask head."""

    for parameter in backbone.parameters():
        parameter.requires_grad_(False)
    for parameter in backbone.encoder[-1].parameters():
        parameter.requires_grad_(True)
    for parameter in head.parameters():
        parameter.requires_grad_(True)

    last_block = sum(parameter.numel() for parameter in backbone.encoder[-1].parameters())
    head_count = sum(parameter.numel() for parameter in head.parameters())
    backbone_trainable = sum(
        parameter.numel() for parameter in backbone.parameters() if parameter.requires_grad
    )
    total_trainable = backbone_trainable + sum(
        parameter.numel() for parameter in head.parameters() if parameter.requires_grad
    )
    report = {
        "last_encoder_block_parameters": last_block,
        "multitask_head_parameters": head_count,
        "backbone_trainable_parameters": backbone_trainable,
        "total_trainable_parameters": total_trainable,
        "trainable_backbone_names": sorted(
            name for name, parameter in backbone.named_parameters() if parameter.requires_grad
        ),
        "frozen_backbone_parameters": sum(
            parameter.numel() for parameter in backbone.parameters() if not parameter.requires_grad
        ),
    }
    if (
        last_block != EXPECTED_LAST_BLOCK_PARAMETERS
        or head_count != EXPECTED_MULTITASK_HEAD_PARAMETERS
        or backbone_trainable != EXPECTED_LAST_BLOCK_PARAMETERS
        or total_trainable != EXPECTED_TOTAL_TRAINABLE_PARAMETERS
        or not report["trainable_backbone_names"]
        or any(
            not name.startswith(f"encoder.{len(backbone.encoder) - 1}.")
            for name in report["trainable_backbone_names"]
        )
    ):
        raise RuntimeError(f"last-block trainability contract drift: {report}")
    return report


def forward_last_block_adaptation(
    backbone: GeometrySemanticEventNet,
    head: CausalGeometryDeltaEventHead,
    student: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Run the causal stack with gradients reaching only the allowed parameters."""

    causal = backbone.encode_causal_features(student)
    baseline_logits = backbone.event_head(causal["context"])
    return head(causal["azimuth_sequence"], causal["context"], baseline_logits)


def audit_last_block_gradients(
    backbone: GeometrySemanticEventNet,
    head: CausalGeometryDeltaEventHead,
) -> dict[str, int]:
    """Fail closed if gradients escape the declared adaptation boundary."""

    unexpected = []
    trainable_with_gradient = []
    for name, parameter in backbone.named_parameters():
        if parameter.requires_grad:
            if parameter.grad is not None:
                if not torch.isfinite(parameter.grad).all():
                    raise RuntimeError(f"nonfinite last-block gradient: {name}")
                trainable_with_gradient.append(f"backbone.{name}")
        elif parameter.grad is not None:
            unexpected.append(f"backbone.{name}")
    for name, parameter in head.named_parameters():
        if not parameter.requires_grad:
            raise RuntimeError(f"fresh multitask head parameter is frozen: {name}")
        if parameter.grad is not None:
            if not torch.isfinite(parameter.grad).all():
                raise RuntimeError(f"nonfinite multitask-head gradient: {name}")
            trainable_with_gradient.append(f"head.{name}")
    if unexpected:
        raise RuntimeError(f"gradient escaped last-block boundary: {unexpected}")
    if not any(name.startswith("backbone.encoder.") for name in trainable_with_gradient):
        raise RuntimeError("last encoder block received no gradient")
    if not any(name.startswith("head.") for name in trainable_with_gradient):
        raise RuntimeError("multitask head received no gradient")
    return {
        "trainable_tensors_with_gradient": len(trainable_with_gradient),
        "unexpected_frozen_gradient_tensors": len(unexpected),
    }


def trainable_parameters(
    backbone: GeometrySemanticEventNet,
    head: CausalGeometryDeltaEventHead,
) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
    """Return disjoint optimizer groups for the last block and fresh head."""

    last_block = [parameter for parameter in backbone.encoder[-1].parameters() if parameter.requires_grad]
    head_parameters = [parameter for parameter in head.parameters() if parameter.requires_grad]
    if not last_block or not head_parameters:
        raise RuntimeError("last-block adaptation was not configured")
    if {id(value) for value in last_block} & {id(value) for value in head_parameters}:
        raise RuntimeError("optimizer parameter groups overlap")
    return last_block, head_parameters


__all__ = [
    "EXPECTED_LAST_BLOCK_PARAMETERS",
    "EXPECTED_MULTITASK_HEAD_PARAMETERS",
    "EXPECTED_TOTAL_TRAINABLE_PARAMETERS",
    "audit_last_block_gradients",
    "configure_last_block_adaptation",
    "forward_last_block_adaptation",
    "trainable_parameters",
]
