"""Six-family objective for the sparse-port primitive relation corrective."""

from __future__ import annotations

import math
from typing import Mapping

import torch
from torch.nn import functional as F

from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG
from mtare_topo.representation.primitive_relation_losses import (
    PrimitiveRelationLossTargets,
    _balanced_binary_loss,
    align_primitive_relation_targets,
    match_primitives,
    sample_swept_superellipse_surface,
)
from mtare_topo.representation.primitive_relation_model import (
    MAXIMUM_RANGE_M,
    MAXIMUM_SLOTS,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    SparsePortRelationPrediction,
)


def sparse_cardinality_objective(
    existence_logits: torch.Tensor,
    matched: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Penalize false slots and excess description length without a new family.

    V1 used a class-balanced existence loss.  With about eight visible targets
    in 32 slots, that objective discarded the real inactive prior and activated
    almost every query.  The corrective uses calibrated BCE plus the exact
    Teacher cardinality and a one-sided redundant-description penalty.  These
    three terms remain inside the registered ``primitive_set_parameters``
    family.
    """

    if existence_logits.shape != matched.shape or existence_logits.ndim != 2:
        raise ValueError("existence logits and matched mask must share [B,32]")
    if existence_logits.shape[1] != MAXIMUM_SLOTS:
        raise ValueError("sparse cardinality requires the frozen 32-slot capacity")
    probability = torch.sigmoid(existence_logits)
    target = matched.to(existence_logits.dtype)
    binary = F.binary_cross_entropy_with_logits(existence_logits, target)
    predicted_count = probability.sum(dim=1)
    target_count = target.sum(dim=1)
    cardinality = F.smooth_l1_loss(
        predicted_count / MAXIMUM_SLOTS,
        target_count / MAXIMUM_SLOTS,
    )
    redundant_description = F.relu(predicted_count - target_count).mean() / MAXIMUM_SLOTS
    total = (binary + cardinality + redundant_description) / 3.0
    return {
        "binary": binary,
        "cardinality": cardinality,
        "redundant_description": redundant_description,
        "total": total,
    }


def _relation_masks(device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    primitive = torch.arange(MAXIMUM_SLOTS, device=device).repeat_interleave(2)
    endpoint_upper = torch.triu(
        primitive[:, None] != primitive[None, :], diagonal=1,
    )
    primitive_upper = torch.triu(
        torch.ones(MAXIMUM_SLOTS, MAXIMUM_SLOTS, dtype=torch.bool, device=device),
        diagonal=1,
    )
    return endpoint_upper, primitive_upper


def sparse_port_relation_losses(
    prediction: SparsePortRelationPrediction,
    targets: PrimitiveRelationLossTargets,
    range_valid: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Return the original six registered loss families and their equal mean."""

    assignments = match_primitives(prediction, targets)
    aligned = align_primitive_relation_targets(targets, assignments)
    matched = aligned["mask"]
    existence_terms = sparse_cardinality_objective(
        prediction.existence_logits, matched,
    )

    axis_error = torch.abs(
        prediction.axis_control_current_sensor_m - aligned["axis"],
    ).mean(dim=(-1, -2)) / MAXIMUM_RANGE_M
    axes_error = torch.abs(
        torch.log1p(prediction.endpoint_half_axes_m)
        - torch.log1p(aligned["half_axes"]),
    ).mean(dim=(-1, -2)) / math.log(11.0)
    exponent_error = torch.abs(
        prediction.endpoint_shape_exponent - aligned["exponent"],
    ).mean(dim=-1) / 8.0
    parameter_error = (axis_error + axes_error + exponent_error) / 3.0
    primitive_set_parameters = 0.5 * (
        existence_terms["total"] + parameter_error[matched].mean()
    )

    predicted_surface = sample_swept_superellipse_surface(
        prediction.axis_control_current_sensor_m,
        prediction.endpoint_half_axes_m,
        prediction.endpoint_shape_exponent,
    )
    target_surface = sample_swept_superellipse_surface(
        aligned["axis"],
        aligned["half_axes"].clamp_min(1e-4),
        aligned["exponent"].clamp_min(2.0),
    )
    surface_distance = torch.cdist(
        predicted_surface[matched], target_surface[matched],
    ) / MAXIMUM_RANGE_M
    surface_reconstruction = 0.5 * (
        surface_distance.amin(dim=-1).mean()
        + surface_distance.amin(dim=-2).mean()
    )

    surface = predicted_surface[matched]
    distance = torch.linalg.vector_norm(surface, dim=-1).clamp_min(1e-6)
    azimuth = torch.remainder(
        torch.atan2(surface[..., 1], surface[..., 0]), 2.0 * math.pi,
    )
    column = torch.round(
        azimuth / (2.0 * math.pi) * 720.0,
    ).long().remainder(720).detach()
    elevation = torch.rad2deg(
        torch.asin(torch.clamp(surface[..., 2] / distance, -1.0, 1.0)),
    )
    elevation_centers = torch.as_tensor(
        ELEVATION_DEG, device=surface.device, dtype=surface.dtype,
    )
    row = torch.argmin(
        torch.abs(elevation[..., None] - elevation_centers), dim=-1,
    ).detach()
    selected_batch = torch.nonzero(matched, as_tuple=False)[:, 0]
    current_range = range_valid[selected_batch, -1, 0]
    current_valid = range_valid[selected_batch, -1, 1].bool()
    observed = current_range[
        torch.arange(len(current_range), device=surface.device)[:, None],
        row,
        column,
    ]
    valid_projection = current_valid[
        torch.arange(len(current_valid), device=surface.device)[:, None],
        row,
        column,
    ] & (torch.abs(elevation - elevation_centers[row]) <= 1.0)
    free_error = F.relu(observed - distance / MAXIMUM_RANGE_M)
    ray_free_space = (
        free_error[valid_projection].mean()
        if bool(valid_projection.any())
        else free_error.sum() * 0.0
    )

    endpoint_upper, primitive_upper = _relation_masks(matched.device)
    attachment_logits = prediction.endpoint_attachment_logits.reshape(
        len(matched), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    )
    attachment_target = aligned["attachment"].reshape(
        len(matched), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    )
    attachment_mask = endpoint_upper[None].expand(len(matched), -1, -1)
    if targets.endpoint_observed is not None:
        # A physical positive whose crop endpoints have no ray support is
        # unknown in this five-frame observation, never a negative.  Keep
        # unmatched query pairs as valid negatives so relation logits on
        # inactive sparse slots remain constrained.
        matched_endpoint = matched[:, :, None].expand(-1, -1, 2).reshape(
            len(matched), 2 * MAXIMUM_SLOTS,
        )
        observed_endpoint = aligned["endpoint_observed"].reshape(
            len(matched), 2 * MAXIMUM_SLOTS,
        )
        both_matched = (
            matched_endpoint[:, :, None] & matched_endpoint[:, None, :]
        )
        both_observed = (
            observed_endpoint[:, :, None] & observed_endpoint[:, None, :]
        )
        attachment_mask = attachment_mask & (~both_matched | both_observed)
    overlap_mask = primitive_upper[None].expand(len(matched), -1, -1)
    attachment_loss = _balanced_binary_loss(
        attachment_logits[attachment_mask], attachment_target[attachment_mask],
    )
    overlap_loss = _balanced_binary_loss(
        prediction.disconnected_overlap_logits[overlap_mask],
        aligned["overlap"][overlap_mask],
    )
    port_relations = 0.5 * (attachment_loss + overlap_loss)

    temporal_target = torch.full(
        (len(matched), 5, MAXIMUM_SLOTS),
        MAXIMUM_SLOTS,
        dtype=torch.long,
        device=matched.device,
    )
    slot_index = torch.arange(MAXIMUM_SLOTS, device=matched.device)[None, None]
    temporal_target = torch.where(
        aligned["temporal"], slot_index, temporal_target,
    )
    selected_temporal = matched[:, None].expand(-1, 5, -1)
    temporal_correspondence = F.cross_entropy(
        prediction.temporal_correspondence_logits[selected_temporal],
        temporal_target[selected_temporal],
    )
    temporal_presence = _balanced_binary_loss(
        prediction.temporal_presence_logits[selected_temporal],
        aligned["temporal"][selected_temporal],
    )
    temporal_equivariance = 0.5 * (
        temporal_correspondence + temporal_presence
    )

    geometry_calibration = F.smooth_l1_loss(
        prediction.geometry_uncertainty[matched],
        parameter_error[matched].detach(),
    )
    attachment_error = torch.abs(
        torch.sigmoid(attachment_logits) - attachment_target,
    ).detach()
    attachment_uncertainty = prediction.endpoint_attachment_uncertainty.reshape(
        len(matched), 2 * MAXIMUM_SLOTS, 2 * MAXIMUM_SLOTS,
    )
    attachment_calibration = F.smooth_l1_loss(
        attachment_uncertainty[attachment_mask],
        attachment_error[attachment_mask],
    )
    overlap_error = torch.abs(
        torch.sigmoid(prediction.disconnected_overlap_logits) - aligned["overlap"],
    ).detach()
    overlap_calibration = F.smooth_l1_loss(
        prediction.disconnected_overlap_uncertainty[overlap_mask],
        overlap_error[overlap_mask],
    )
    uncertainty_calibration = (
        geometry_calibration + attachment_calibration + overlap_calibration
    ) / 3.0

    families = {
        "primitive_set_parameters": primitive_set_parameters,
        "surface_reconstruction": surface_reconstruction,
        "ray_free_space": ray_free_space,
        "port_relations": port_relations,
        "temporal_equivariance": temporal_equivariance,
        "uncertainty_calibration": uncertainty_calibration,
    }
    families["total"] = torch.stack(tuple(families.values())).mean()
    return families


__all__ = ["sparse_cardinality_objective", "sparse_port_relation_losses"]
