"""Soft circular heatmap and explicit hard-negative ranking for GSE peaks."""

from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F


BEARING_BINS = 180
SUPPORT_RADIUS_BINS = 4
FOCAL_ALPHA = 2.0
FOCAL_BETA = 4.0


def circular_soft_peak_target(presence: torch.Tensor) -> torch.Tensor:
    centers = presence.bool()
    if centers.ndim != 2 or centers.shape[1] != BEARING_BINS or not bool(centers.any(dim=1).all()):
        raise ValueError("soft circular peak target requires nonempty [B,180] centers")
    bins = torch.arange(BEARING_BINS, device=centers.device)
    distance = torch.minimum(
        torch.remainder(bins[:, None] - bins[None, :], BEARING_BINS),
        torch.remainder(bins[None, :] - bins[:, None], BEARING_BINS),
    )
    sigma = SUPPORT_RADIUS_BINS / 2.0
    kernel = torch.exp(-0.5 * (distance.to(torch.float32) / sigma) ** 2)
    kernel = torch.where(distance <= SUPPORT_RADIUS_BINS, kernel, torch.zeros_like(kernel))
    target = torch.amax(kernel[None] * centers[:, None, :].to(kernel.dtype), dim=-1)
    return target


def soft_angular_hard_negative_presence_loss(
    logits: torch.Tensor, presence: torch.Tensor
) -> dict[str, torch.Tensor]:
    centers = presence.bool()
    if logits.shape != centers.shape or logits.ndim != 2 or logits.shape[1] != BEARING_BINS:
        raise ValueError("soft angular peak loss shape drift")
    if not bool(torch.isfinite(logits).all()) or not bool(centers.any(dim=1).all()):
        raise ValueError("soft angular peak loss requires finite logits and nonempty rows")
    soft_target = circular_soft_peak_target(centers).to(dtype=logits.dtype)
    probability = torch.sigmoid(logits)
    positive_loss = -torch.log(probability.clamp_min(1e-8)) * (1.0 - probability).pow(FOCAL_ALPHA)
    negative_loss = -torch.log((1.0 - probability).clamp_min(1e-8)) * probability.pow(FOCAL_ALPHA) * (1.0 - soft_target).pow(FOCAL_BETA)
    positive_count = centers.sum(dim=1)
    heatmap = (
        (
            (positive_loss * centers).sum(dim=1)
            + (negative_loss * ~centers).sum(dim=1)
        )
        / positive_count
    ).mean()

    # Pad both variable-cardinality sets to the largest cardinality in the
    # batch, then mask padded pairs. This is algebraically identical to the
    # former row loop while avoiding hundreds of tiny GPU launches per batch.
    maximum_count = int(positive_count.max())
    outside_support = soft_target == 0
    if not bool((outside_support.sum(dim=1) >= positive_count).all()):
        raise ValueError("soft target leaves insufficient hard-negative support")
    negative_infinity = torch.finfo(logits.dtype).min
    positive = torch.topk(
        logits.masked_fill(~centers, negative_infinity),
        k=maximum_count,
        dim=1,
        largest=True,
        sorted=True,
    ).values
    hard = torch.topk(
        logits.masked_fill(~outside_support, negative_infinity),
        k=maximum_count,
        dim=1,
        largest=True,
        sorted=True,
    ).values
    rank = torch.arange(maximum_count, device=logits.device)[None]
    valid = rank < positive_count[:, None]
    pair_valid = valid[:, :, None] & valid[:, None, :]
    pair_loss = F.softplus(hard[:, :, None] - positive[:, None, :])
    hard_negative_ranking = (
        (pair_loss * pair_valid).sum(dim=(1, 2))
        / positive_count.to(logits.dtype).square()
    ).mean()
    return {
        "total": heatmap + hard_negative_ranking,
        "heatmap": heatmap,
        "hard_negative_ranking": hard_negative_ranking,
    }


def replace_v1_presence_with_v2(
    v1_losses: Mapping[str, torch.Tensor],
    logits: torch.Tensor,
    presence: torch.Tensor,
) -> dict[str, torch.Tensor]:
    v2 = soft_angular_hard_negative_presence_loss(logits, presence)
    required = ("total", "presence", "peak_geometry", "axis", "global_geometry")
    if any(name not in v1_losses for name in required):
        raise ValueError("V1 multitask loss interface drift")
    total = v1_losses["total"] - v1_losses["presence"] + v2["total"]
    return {
        "total": total,
        "presence": v2["total"],
        "soft_heatmap": v2["heatmap"],
        "hard_negative_ranking": v2["hard_negative_ranking"],
        "peak_geometry": v1_losses["peak_geometry"],
        "axis": v1_losses["axis"],
        "global_geometry": v1_losses["global_geometry"],
    }


__all__ = [
    "SUPPORT_RADIUS_BINS",
    "circular_soft_peak_target",
    "replace_v1_presence_with_v2",
    "soft_angular_hard_negative_presence_loss",
]
