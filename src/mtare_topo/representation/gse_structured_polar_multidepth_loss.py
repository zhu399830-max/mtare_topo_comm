"""Direct dense supervision for structured polar multi-depth events."""
from __future__ import annotations

from typing import Mapping

import torch
from torch.nn import functional as F

from mtare_topo.representation.gse_spatial_event_set import (
    _descriptor_loss_if_supported,
)
from mtare_topo.representation.gse_structured_polar_multidepth_event import (
    AZIMUTH_RESIDUAL_LIMIT_RAD,
    StructuredPolarLossWeights,
)


def rasterize_structured_polar_targets_vectorized(
    targets: Mapping[str, torch.Tensor],
    free_range_profile_m: torch.Tensor,
) -> dict[str, torch.Tensor]:
    mask = targets["event_mask"].bool()
    xyz = targets["event_relative_xyz_m"]
    identity = targets["event_identity_index"]
    if xyz.shape != (*mask.shape, 3) or mask.shape[1] != 16 or free_range_profile_m.shape != (len(mask), 180):
        raise ValueError("vectorized structured polar raster shape drift")
    radial = torch.linalg.vector_norm(xyz, dim=-1)
    bearing = torch.remainder(torch.atan2(xyz[..., 1], xyz[..., 0]), 2.0 * torch.pi)
    bins = torch.remainder(torch.floor((torch.rad2deg(bearing) + 0.25) / 2.0).long(), 180)
    same = mask[:, :, None] & mask[:, None, :] & bins[:, :, None].eq(bins[:, None, :])
    earlier = (radial[:, None, :] < radial[:, :, None]) | (
        radial[:, None, :].eq(radial[:, :, None])
        & (identity[:, None, :] < identity[:, :, None])
    )
    slots = (same & earlier).sum(dim=2).long()
    if bool((slots[mask] >= 2).any()):
        raise ValueError("vectorized structured polar Teacher exceeds two depth slots")
    batch_index, source_slot = torch.nonzero(mask, as_tuple=True)
    bin_index = bins[batch_index, source_slot]
    depth_slot = slots[batch_index, source_slot]
    dense_mask = torch.zeros((len(mask), 180, 2), dtype=torch.bool, device=xyz.device)
    dense_type = torch.full((len(mask), 180, 2), -1, dtype=torch.long, device=xyz.device)
    dense_identity = torch.full_like(dense_type, -1)
    dense_xyz = torch.zeros((len(mask), 180, 2, 3), dtype=xyz.dtype, device=xyz.device)
    dense_fraction = torch.zeros((len(mask), 180, 2), dtype=xyz.dtype, device=xyz.device)
    dense_residual = torch.zeros_like(dense_fraction)
    dense_elevation = torch.zeros_like(dense_fraction)
    if len(batch_index):
        anchor = free_range_profile_m[batch_index, bin_index]
        selected_radial = radial[batch_index, source_slot]
        if bool((selected_radial > anchor + 1e-3).any()):
            raise ValueError("vectorized structured polar target exceeds support")
        center = (bin_index.to(xyz.dtype) + 0.375) * (2.0 * torch.pi / 180.0)
        residual = torch.remainder(bearing[batch_index, source_slot] - center + torch.pi, 2.0 * torch.pi) - torch.pi
        if bool((torch.abs(residual) > AZIMUTH_RESIDUAL_LIMIT_RAD + 1e-6).any()):
            raise ValueError("vectorized structured polar azimuth overflow")
        selected_xyz = xyz[batch_index, source_slot]
        dense_mask[batch_index, bin_index, depth_slot] = True
        dense_type[batch_index, bin_index, depth_slot] = targets["event_type_index"][batch_index, source_slot]
        dense_identity[batch_index, bin_index, depth_slot] = identity[batch_index, source_slot]
        dense_xyz[batch_index, bin_index, depth_slot] = selected_xyz
        dense_fraction[batch_index, bin_index, depth_slot] = selected_radial / anchor.clamp_min(1e-8)
        dense_residual[batch_index, bin_index, depth_slot] = residual
        dense_elevation[batch_index, bin_index, depth_slot] = torch.atan2(selected_xyz[:, 2], torch.linalg.vector_norm(selected_xyz[:, :2], dim=-1))
    return {"event_mask": dense_mask, "event_type_index": dense_type, "event_identity_index": dense_identity, "event_relative_xyz_m": dense_xyz, "event_radial_fraction": dense_fraction, "event_azimuth_residual_rad": dense_residual, "event_elevation_rad": dense_elevation}


def balanced_dense_presence_loss(logits: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    active = mask.bool()
    if logits.shape != active.shape or logits.ndim != 3 or logits.shape[1:] != (180, 2):
        raise ValueError("balanced dense presence shape drift")
    positive = F.softplus(-logits)
    negative = F.softplus(logits)
    positive_count = active.sum(dim=(1, 2))
    negative_count = (~active).sum(dim=(1, 2))
    positive_mean = (positive * active).sum(dim=(1, 2)) / positive_count.clamp_min(1)
    negative_mean = (negative * (~active)).sum(dim=(1, 2)) / negative_count.clamp_min(1)
    has_positive = positive_count > 0
    per_row = torch.where(has_positive, 0.5 * (positive_mean + negative_mean), negative_mean)
    return per_row.mean()


def structured_polar_direct_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    event_type_class_weights: torch.Tensor | None = None,
    weights: StructuredPolarLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    loss_weights = weights or StructuredPolarLossWeights()
    logits = outputs["dense_event_presence_logits"]
    mask = targets["event_mask"].bool()
    presence = balanced_dense_presence_loss(logits, mask)
    zero = logits.sum() * 0.0
    if bool(mask.any()):
        event_type = F.cross_entropy(
            outputs["dense_event_type_logits"][mask],
            targets["event_type_index"][mask],
            weight=event_type_class_weights,
        )
        difference = (
            outputs["dense_event_relative_xyz_m"][mask]
            - targets["event_relative_xyz_m"][mask]
        ) / 50.0
        position = F.smooth_l1_loss(difference, torch.zeros_like(difference))
        error_m = torch.linalg.vector_norm(
            outputs["dense_event_relative_xyz_m"][mask]
            - targets["event_relative_xyz_m"][mask],
            dim=-1,
        )
        uncertainty_m = outputs["dense_event_uncertainty_m"][mask]
        uncertainty = torch.mean(
            error_m.detach() / uncertainty_m + torch.log(uncertainty_m)
        )
        descriptor = _descriptor_loss_if_supported(
            outputs["dense_event_descriptor"][mask],
            targets["event_identity_index"][mask],
            temperature=0.1,
        )
    else:
        event_type = position = uncertainty = descriptor = zero
    total = (
        loss_weights.presence * presence
        + loss_weights.event_type * event_type
        + loss_weights.position * position
        + loss_weights.descriptor * descriptor
        + loss_weights.uncertainty * uncertainty
    )
    return {
        "total": total,
        "presence": presence,
        "event_type": event_type,
        "position": position,
        "descriptor": descriptor,
        "uncertainty": uncertainty,
    }


__all__ = ["balanced_dense_presence_loss", "rasterize_structured_polar_targets_vectorized", "structured_polar_direct_loss"]
