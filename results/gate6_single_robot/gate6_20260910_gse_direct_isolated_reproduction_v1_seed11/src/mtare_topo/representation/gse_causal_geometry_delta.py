"""Explicit causal geometry-delta multitask head for GSE-Graph."""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_directional_structural_event import (
    DirectionalStructuralEventHead,
    ENCODER_DIM,
    EVENT_COUNT,
    STRUCTURAL_EVENT_COUNT,
    directional_structural_event_loss,
)


GEOMETRY_DELTA_NAMES = (
    "delta_width_m",
    "delta_height_m",
    "delta_slope_deg",
    "delta_curvature_per_m",
)
# C01-C06 fit-only population standard deviations over the exact four-step
# geometry-valid lag contract (92,845 pairs).  These are normalization units,
# not learned or graph-tuned thresholds.
GEOMETRY_DELTA_SCALE = (
    3.047925538538476,
    0.5562241064740382,
    0.5709059013471545,
    0.005093988709664383,
)


class CausalGeometryDeltaEventHead(nn.Module):
    """Jointly predict signed local geometry change and structural events.

    The event branch and delta branch share the azimuth-preserving causal
    change encoder.  Geometry supervision therefore shapes the representation
    used to decide node creation rather than being attached as a post-hoc
    diagnostic.  Both residual event outputs and metric deltas start at zero.
    """

    def __init__(self) -> None:
        super().__init__()
        self.event_head = DirectionalStructuralEventHead()
        summary_dim = 3 * ENCODER_DIM
        self.geometry_delta_head = nn.Sequential(
            nn.Linear(summary_dim, ENCODER_DIM),
            nn.SiLU(),
            nn.Linear(ENCODER_DIM, len(GEOMETRY_DELTA_NAMES)),
        )
        nn.init.zeros_(self.geometry_delta_head[-1].weight)
        nn.init.zeros_(self.geometry_delta_head[-1].bias)

    def forward(
        self,
        azimuth_sequence: torch.Tensor,
        context: torch.Tensor,
        baseline_event_logits: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        event = self.event_head(azimuth_sequence, context, baseline_event_logits)
        directional_change = event["directional_change"]
        summary = torch.cat(
            (
                directional_change.mean(dim=-1),
                directional_change.amax(dim=-1),
                context,
            ),
            dim=1,
        )
        normalized_delta = self.geometry_delta_head(summary)
        scale = normalized_delta.new_tensor(GEOMETRY_DELTA_SCALE)
        return {
            **event,
            "geometry_delta_normalized": normalized_delta,
            "geometry_delta": normalized_delta * scale,
        }


def causal_geometry_delta_event_loss(
    outputs: Mapping[str, torch.Tensor],
    event_target: torch.Tensor,
    geometry_delta_target: torch.Tensor,
    geometry_delta_valid: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Joint event and normalized signed-delta loss with explicit validity."""

    event_losses = directional_structural_event_loss(outputs, event_target)
    geometry_delta = causal_geometry_delta_loss(
        outputs["geometry_delta_normalized"],
        geometry_delta_target,
        geometry_delta_valid,
    )
    total = event_losses["total"] + geometry_delta
    return {
        "structural": event_losses["structural"],
        "conditional_event": event_losses["conditional_event"],
        "geometry_delta": geometry_delta,
        "total": total,
    }


def causal_geometry_delta_loss(
    predicted_normalized: torch.Tensor,
    geometry_delta_target: torch.Tensor,
    geometry_delta_valid: torch.Tensor,
) -> torch.Tensor:
    """Normalized signed-delta regression loss for an independently masked batch."""

    predicted = predicted_normalized
    if geometry_delta_target.shape != predicted.shape or predicted.ndim != 2 or predicted.shape[1] != 4:
        raise ValueError("geometry_delta_target must align with [B,4] predictions")
    if geometry_delta_valid.shape != (len(predicted),):
        raise ValueError("geometry_delta_valid must have shape [B]")
    if not torch.isfinite(geometry_delta_target[geometry_delta_valid.bool()]).all():
        raise ValueError("valid geometry delta targets must be finite")
    scale = predicted.new_tensor(GEOMETRY_DELTA_SCALE)
    normalized_target = geometry_delta_target.to(dtype=predicted.dtype) / scale
    valid = geometry_delta_valid.bool()
    if bool(valid.any()):
        geometry_delta = F.smooth_l1_loss(predicted[valid], normalized_target[valid])
    else:
        geometry_delta = predicted.sum() * 0.0
    return geometry_delta


def geometry_delta_weighted_mae(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute delta error in fixed fit-only normalized units."""

    if prediction.shape != target.shape or prediction.ndim != 2 or prediction.shape[1] != 4:
        raise ValueError("prediction/target must be aligned [N,4]")
    if not torch.isfinite(prediction).all() or not torch.isfinite(target).all():
        raise ValueError("geometry delta metric inputs must be finite")
    scale = prediction.new_tensor(GEOMETRY_DELTA_SCALE)
    return torch.mean(torch.abs(prediction - target) / scale)


__all__ = [
    "CausalGeometryDeltaEventHead",
    "GEOMETRY_DELTA_NAMES",
    "GEOMETRY_DELTA_SCALE",
    "causal_geometry_delta_loss",
    "causal_geometry_delta_event_loss",
    "geometry_delta_weighted_mae",
]
