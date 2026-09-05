"""Permutation-safe losses for explicit swept primitives and their relations."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping

import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
from torch.nn import functional as F

from mtare_topo.data.cano_sensor_smoke import ELEVATION_DEG
from mtare_topo.representation.primitive_relation_model import (
    MAXIMUM_RANGE_M,
    MAXIMUM_SLOTS,
    PrimitiveRelationPrediction,
)


@dataclass(frozen=True)
class PrimitiveRelationLossTargets:
    primitive_mask: torch.Tensor
    axis_control_current_sensor_m: torch.Tensor
    endpoint_half_axes_m: torch.Tensor
    endpoint_shape_exponent: torch.Tensor
    temporal_visibility: torch.Tensor
    endpoint_attachment: torch.Tensor
    disconnected_overlap: torch.Tensor
    endpoint_observed: torch.Tensor | None = None

    def validate(self) -> None:
        batch = self.primitive_mask.shape[0] if self.primitive_mask.ndim else -1
        expected = {
            "primitive_mask": (batch, 32),
            "axis_control_current_sensor_m": (batch, 32, 3, 3),
            "endpoint_half_axes_m": (batch, 32, 2, 2),
            "endpoint_shape_exponent": (batch, 32, 2),
            "temporal_visibility": (batch, 5, 32),
            "endpoint_attachment": (batch, 32, 2, 32, 2),
            "disconnected_overlap": (batch, 32, 32),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"loss target shape drift: {name} {tuple(value.shape)}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"loss target contains non-finite values: {name}")
        mask = self.primitive_mask.bool()
        if bool((mask.sum(dim=1) == 0).any()) or bool((mask.sum(dim=1) > MAXIMUM_SLOTS).any()):
            raise ValueError("every readiness/training row must contain 1--32 visible primitives")
        if bool(((self.temporal_visibility != 0) & (self.temporal_visibility != 1)).any()):
            raise ValueError("temporal visibility must be binary")
        for value in (self.endpoint_attachment, self.disconnected_overlap):
            if bool(((value != 0) & (value != 1)).any()):
                raise ValueError("relation targets must be binary")
        if self.endpoint_observed is not None:
            if self.endpoint_observed.shape != (batch, 32, 2):
                raise ValueError("endpoint observed target must have shape [B,32,2]")
            if not bool(torch.isfinite(self.endpoint_observed).all()) or bool(
                ((self.endpoint_observed != 0) & (self.endpoint_observed != 1)).any()
            ):
                raise ValueError("endpoint observed target must be finite and binary")
            if bool((self.endpoint_observed.bool() & ~mask[:, :, None]).any()):
                raise ValueError("inactive primitive endpoint cannot be observed")


@dataclass(frozen=True)
class PrimitiveAssignment:
    predicted_to_target: torch.Tensor
    endpoint_reversed: torch.Tensor


def _pack_bytes(parts: list[bytes] | tuple[bytes, ...]) -> bytes:
    return b"".join(len(part).to_bytes(4, "little") + part for part in parts)


def _array_bytes(array: np.ndarray) -> bytes:
    array = np.ascontiguousarray(array)
    shape = np.asarray(array.shape, dtype="<i8").tobytes()
    return (
        array.dtype.str.encode("ascii")
        + len(array.shape).to_bytes(1, "little")
        + shape
        + array.tobytes()
    )


def _canonical_active_order(
    targets: PrimitiveRelationLossTargets,
    batch_index: int,
    active: torch.Tensor,
) -> torch.Tensor:
    """Order Teacher primitives by intrinsic geometry/relation content, never slot id.

    SciPy's Hungarian solver uses column order to resolve exactly or nearly tied
    optima.  Teacher slot order is arbitrary, so feeding active slots in storage
    order can leak that arbitrary order into relation losses.  The signatures
    below are endpoint-reversal invariant and refined by the local port graph.
    """

    slots = [int(value) for value in active.detach().cpu().tolist()]
    axis = targets.axis_control_current_sensor_m[batch_index].detach().cpu().numpy()
    half_axes = targets.endpoint_half_axes_m[batch_index].detach().cpu().numpy()
    exponent = targets.endpoint_shape_exponent[batch_index].detach().cpu().numpy()
    temporal_visibility = targets.temporal_visibility[batch_index].detach().cpu().numpy()
    attachment = targets.endpoint_attachment[batch_index].detach().cpu().numpy()
    overlap = targets.disconnected_overlap[batch_index].detach().cpu().numpy()
    endpoint_observed = (
        None if targets.endpoint_observed is None
        else targets.endpoint_observed[batch_index].detach().cpu().numpy()
    )
    base: dict[int, bytes] = {}
    canonical_reversed: dict[int, bool] = {}
    for slot in slots:
        temporal = _array_bytes(temporal_visibility[:, slot])
        direct = _pack_bytes((
            _array_bytes(axis[slot]),
            _array_bytes(half_axes[slot]),
            _array_bytes(exponent[slot]),
            temporal,
            *(() if endpoint_observed is None else (_array_bytes(endpoint_observed[slot]),)),
        ))
        reverse = _pack_bytes((
            _array_bytes(axis[slot][::-1]),
            _array_bytes(half_axes[slot][::-1]),
            _array_bytes(exponent[slot][::-1]),
            temporal,
            *(() if endpoint_observed is None else (_array_bytes(endpoint_observed[slot][::-1]),)),
        ))
        canonical_reversed[slot] = reverse < direct
        base[slot] = min(direct, reverse)

    colors = {slot: hashlib.sha256(base[slot]).digest() for slot in slots}
    for _ in range(len(slots)):
        refined: dict[int, bytes] = {}
        for slot in slots:
            endpoint_neighborhoods: list[bytes] = []
            for canonical_endpoint in range(2):
                source_endpoint = (
                    1 - canonical_endpoint
                    if canonical_reversed[slot]
                    else canonical_endpoint
                )
                neighbors: list[bytes] = []
                for other in slots:
                    for other_source_endpoint in range(2):
                        if attachment[
                            slot, source_endpoint, other, other_source_endpoint,
                        ] > 0.5:
                            other_canonical_endpoint = (
                                1 - other_source_endpoint
                                if canonical_reversed[other]
                                else other_source_endpoint
                            )
                            neighbors.append(
                                colors[other] + bytes((other_canonical_endpoint,))
                            )
                endpoint_neighborhoods.append(_pack_bytes(sorted(neighbors)))
            overlap_neighbors = sorted(
                colors[other]
                for other in slots
                if overlap[slot, other] > 0.5
            )
            relation = _pack_bytes(
                tuple(sorted(endpoint_neighborhoods))
                + (_pack_bytes(overlap_neighbors),)
            )
            refined[slot] = hashlib.sha256(base[slot] + relation).digest()
        colors = refined

    ordered = sorted(slots, key=lambda slot: (colors[slot], base[slot]))
    return torch.as_tensor(ordered, dtype=torch.long, device=active.device)


def _pair_cost(
    prediction: PrimitiveRelationPrediction,
    targets: PrimitiveRelationLossTargets,
    batch_index: int,
    active: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    predicted_axis = prediction.axis_control_current_sensor_m[batch_index, :, None]
    target_axis = targets.axis_control_current_sensor_m[batch_index, active][None]
    predicted_axes = prediction.endpoint_half_axes_m[batch_index, :, None]
    target_axes = targets.endpoint_half_axes_m[batch_index, active][None]
    predicted_exponent = prediction.endpoint_shape_exponent[batch_index, :, None]
    target_exponent = targets.endpoint_shape_exponent[batch_index, active][None]

    def normalized_cost(axis: torch.Tensor, axes: torch.Tensor, exponent: torch.Tensor) -> torch.Tensor:
        axis_error = torch.abs(predicted_axis - axis).mean(dim=(-1, -2)) / MAXIMUM_RANGE_M
        axes_error = torch.abs(torch.log1p(predicted_axes) - torch.log1p(axes)).mean(dim=(-1, -2)) / math.log(11.0)
        exponent_error = torch.abs(predicted_exponent - exponent).mean(dim=-1) / 8.0
        return (axis_error + axes_error + exponent_error) / 3.0

    direct = normalized_cost(target_axis, target_axes, target_exponent)
    reverse = normalized_cost(
        target_axis.flip(-2), target_axes.flip(-2), target_exponent.flip(-1),
    )
    return torch.minimum(direct, reverse), reverse < direct


def match_primitives(
    prediction: PrimitiveRelationPrediction,
    targets: PrimitiveRelationLossTargets,
) -> tuple[PrimitiveAssignment, ...]:
    """Hungarian-match every exchangeable prediction with endpoint reversal."""

    targets.validate()
    assignments: list[PrimitiveAssignment] = []
    for batch_index in range(len(targets.primitive_mask)):
        active = torch.nonzero(targets.primitive_mask[batch_index].bool(), as_tuple=False).flatten()
        active = _canonical_active_order(targets, batch_index, active)
        cost, reversed_pair = _pair_cost(prediction, targets, batch_index, active)
        rows, columns = linear_sum_assignment(cost.detach().cpu().numpy())
        mapping = torch.full((MAXIMUM_SLOTS,), -1, dtype=torch.long, device=cost.device)
        reversed_slots = torch.zeros(MAXIMUM_SLOTS, dtype=torch.bool, device=cost.device)
        row_tensor = torch.as_tensor(rows, device=cost.device, dtype=torch.long)
        column_tensor = torch.as_tensor(columns, device=cost.device, dtype=torch.long)
        mapping[row_tensor] = active[column_tensor]
        reversed_slots[row_tensor] = reversed_pair[row_tensor, column_tensor]
        assignments.append(PrimitiveAssignment(mapping, reversed_slots))
    return tuple(assignments)


def _aligned_targets(
    targets: PrimitiveRelationLossTargets,
    assignments: tuple[PrimitiveAssignment, ...],
) -> dict[str, torch.Tensor]:
    batch = len(assignments)
    device = targets.primitive_mask.device
    mask = torch.zeros(batch, MAXIMUM_SLOTS, dtype=torch.bool, device=device)
    axis = torch.zeros(batch, MAXIMUM_SLOTS, 3, 3, device=device, dtype=targets.axis_control_current_sensor_m.dtype)
    half_axes = torch.zeros(batch, MAXIMUM_SLOTS, 2, 2, device=device, dtype=targets.endpoint_half_axes_m.dtype)
    exponent = torch.zeros(batch, MAXIMUM_SLOTS, 2, device=device, dtype=targets.endpoint_shape_exponent.dtype)
    temporal = torch.zeros(batch, 5, MAXIMUM_SLOTS, dtype=torch.bool, device=device)
    attachment = torch.zeros(batch, MAXIMUM_SLOTS, 2, MAXIMUM_SLOTS, 2, device=device)
    overlap = torch.zeros(batch, MAXIMUM_SLOTS, MAXIMUM_SLOTS, device=device)
    endpoint_observed = torch.zeros(batch, MAXIMUM_SLOTS, 2, dtype=torch.bool, device=device)
    for batch_index, assignment in enumerate(assignments):
        matched = torch.nonzero(assignment.predicted_to_target >= 0, as_tuple=False).flatten()
        source = assignment.predicted_to_target[matched]
        mask[batch_index, matched] = True
        axis_values = targets.axis_control_current_sensor_m[batch_index, source]
        axes_values = targets.endpoint_half_axes_m[batch_index, source]
        exponent_values = targets.endpoint_shape_exponent[batch_index, source]
        reverse = assignment.endpoint_reversed[matched]
        axis_values = torch.where(reverse[:, None, None], axis_values.flip(1), axis_values)
        axes_values = torch.where(reverse[:, None, None], axes_values.flip(1), axes_values)
        exponent_values = torch.where(reverse[:, None], exponent_values.flip(1), exponent_values)
        axis[batch_index, matched] = axis_values
        half_axes[batch_index, matched] = axes_values
        exponent[batch_index, matched] = exponent_values
        temporal[batch_index, :, matched] = targets.temporal_visibility[batch_index, :, source].bool()
        if targets.endpoint_observed is None:
            endpoint_observed[batch_index, matched] = True
        else:
            observed_values = targets.endpoint_observed[batch_index, source].bool()
            observed_values = torch.where(reverse[:, None], observed_values.flip(1), observed_values)
            endpoint_observed[batch_index, matched] = observed_values
        for local_i, predicted_i in enumerate(matched.tolist()):
            target_i = int(source[local_i]); flip_i = bool(reverse[local_i])
            for local_j, predicted_j in enumerate(matched.tolist()):
                target_j = int(source[local_j]); flip_j = bool(reverse[local_j])
                overlap[batch_index, predicted_i, predicted_j] = targets.disconnected_overlap[batch_index, target_i, target_j]
                for endpoint_i in range(2):
                    for endpoint_j in range(2):
                        source_i = 1 - endpoint_i if flip_i else endpoint_i
                        source_j = 1 - endpoint_j if flip_j else endpoint_j
                        attachment[batch_index, predicted_i, endpoint_i, predicted_j, endpoint_j] = targets.endpoint_attachment[
                            batch_index, target_i, source_i, target_j, source_j
                        ]
    return {
        "mask": mask, "axis": axis, "half_axes": half_axes, "exponent": exponent,
        "temporal": temporal, "attachment": attachment, "overlap": overlap,
        "endpoint_observed": endpoint_observed,
    }


def align_primitive_relation_targets(
    targets: PrimitiveRelationLossTargets,
    assignments: tuple[PrimitiveAssignment, ...],
) -> dict[str, torch.Tensor]:
    """Expose the exact loss-side alignment to the frozen evaluator."""

    targets.validate()
    if len(assignments) != len(targets.primitive_mask):
        raise ValueError("assignment batch length drift")
    return _aligned_targets(targets, assignments)


def sample_swept_superellipse_surface(
    axis_control: torch.Tensor,
    endpoint_half_axes: torch.Tensor,
    endpoint_exponent: torch.Tensor,
    angular_samples: int = 12,
) -> torch.Tensor:
    """Differentiably sample the three supervised cross-sections."""

    if axis_control.shape[-2:] != (3, 3) or endpoint_half_axes.shape[-2:] != (2, 2) or endpoint_exponent.shape[-1] != 2:
        raise ValueError("surface sampler primitive shapes are invalid")
    tangent = F.normalize(axis_control[..., 2, :] - axis_control[..., 0, :], dim=-1, eps=1e-6)
    gravity = torch.zeros_like(tangent); gravity[..., 2] = 1.0
    side = torch.linalg.cross(gravity, tangent, dim=-1)
    fallback = torch.zeros_like(tangent); fallback[..., 0] = 1.0
    fallback_side = torch.linalg.cross(fallback, tangent, dim=-1)
    side = torch.where((torch.linalg.vector_norm(side, dim=-1) < 1e-4)[..., None], fallback_side, side)
    side = F.normalize(side, dim=-1, eps=1e-6)
    up = F.normalize(torch.linalg.cross(tangent, side, dim=-1), dim=-1, eps=1e-6)
    half_axes = torch.stack(
        (endpoint_half_axes[..., 0, :], endpoint_half_axes.mean(dim=-2), endpoint_half_axes[..., 1, :]), dim=-2,
    )
    exponent = torch.stack(
        (endpoint_exponent[..., 0], endpoint_exponent.mean(dim=-1), endpoint_exponent[..., 1]), dim=-1,
    )
    angle = torch.arange(angular_samples, device=axis_control.device, dtype=axis_control.dtype) * (2.0 * math.pi / angular_samples)
    cosine, sine = torch.cos(angle), torch.sin(angle)
    power = 2.0 / exponent[..., :, None]
    transverse = half_axes[..., :, 0, None] * torch.sign(cosine) * torch.abs(cosine).pow(power)
    vertical = half_axes[..., :, 1, None] * torch.sign(sine) * torch.abs(sine).pow(power)
    surface = (
        axis_control[..., :, None, :]
        + transverse[..., None] * side[..., None, None, :]
        + vertical[..., None] * up[..., None, None, :]
    )
    return surface.flatten(start_dim=-3, end_dim=-2)


def _balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.bool()
    positive, negative = target, ~target
    terms = []
    if bool(positive.any()): terms.append(F.softplus(-logits[positive]).mean())
    if bool(negative.any()): terms.append(F.softplus(logits[negative]).mean())
    if not terms: raise ValueError("binary loss has no qualified elements")
    return torch.stack(terms).mean()


def primitive_relation_losses(
    prediction: PrimitiveRelationPrediction,
    targets: PrimitiveRelationLossTargets,
    range_valid: torch.Tensor,
) -> Mapping[str, torch.Tensor]:
    """Return the six frozen, dimensionless loss families and equal mean."""

    assignments = match_primitives(prediction, targets)
    aligned = _aligned_targets(targets, assignments)
    matched = aligned["mask"]
    existence = _balanced_binary_loss(prediction.existence_logits, matched)
    axis_error = torch.abs(prediction.axis_control_current_sensor_m - aligned["axis"]).mean(dim=(-1, -2)) / MAXIMUM_RANGE_M
    axes_error = torch.abs(torch.log1p(prediction.endpoint_half_axes_m) - torch.log1p(aligned["half_axes"])).mean(dim=(-1, -2)) / math.log(11.0)
    exponent_error = torch.abs(prediction.endpoint_shape_exponent - aligned["exponent"]).mean(dim=-1) / 8.0
    parameter_error = (axis_error + axes_error + exponent_error) / 3.0
    primitive_set_parameters = 0.5 * (existence + parameter_error[matched].mean())

    predicted_surface = sample_swept_superellipse_surface(
        prediction.axis_control_current_sensor_m, prediction.endpoint_half_axes_m,
        prediction.endpoint_shape_exponent,
    )
    target_surface = sample_swept_superellipse_surface(
        aligned["axis"], aligned["half_axes"].clamp_min(1e-4), aligned["exponent"].clamp_min(2.0),
    )
    surface_distance = torch.cdist(predicted_surface[matched], target_surface[matched]) / MAXIMUM_RANGE_M
    surface_reconstruction = 0.5 * (surface_distance.amin(dim=-1).mean() + surface_distance.amin(dim=-2).mean())

    surface = predicted_surface[matched]
    distance = torch.linalg.vector_norm(surface, dim=-1).clamp_min(1e-6)
    azimuth = torch.remainder(torch.atan2(surface[..., 1], surface[..., 0]), 2.0 * math.pi)
    column = torch.round(azimuth / (2.0 * math.pi) * 720.0).long().remainder(720).detach()
    elevation = torch.rad2deg(torch.asin(torch.clamp(surface[..., 2] / distance, -1.0, 1.0)))
    elevation_centers = torch.as_tensor(ELEVATION_DEG, device=surface.device, dtype=surface.dtype)
    row = torch.argmin(torch.abs(elevation[..., None] - elevation_centers), dim=-1).detach()
    selected_batch = torch.nonzero(matched, as_tuple=False)[:, 0]
    current_range = range_valid[selected_batch, -1, 0]
    current_valid = range_valid[selected_batch, -1, 1].bool()
    observed = current_range.gather(1, row * 720 + column) if current_range.ndim == 2 else current_range[
        torch.arange(len(current_range), device=surface.device)[:, None], row, column
    ]
    valid_projection = current_valid[
        torch.arange(len(current_valid), device=surface.device)[:, None], row, column
    ] & (torch.abs(elevation - elevation_centers[row]) <= 1.0)
    free_error = F.relu(observed - distance / MAXIMUM_RANGE_M)
    ray_free_space = free_error[valid_projection].mean() if bool(valid_projection.any()) else free_error.sum() * 0.0

    pair_mask = matched[:, :, None] & matched[:, None, :]
    pair_mask &= ~torch.eye(MAXIMUM_SLOTS, dtype=torch.bool, device=matched.device)[None]
    endpoint_pair_mask = pair_mask[:, :, None, :, None].expand(-1, -1, 2, -1, 2) & (
        aligned["endpoint_observed"][:, :, :, None, None]
        & aligned["endpoint_observed"][:, None, None, :, :]
    )
    attachment_loss = (
        _balanced_binary_loss(
            prediction.endpoint_attachment_logits[endpoint_pair_mask],
            aligned["attachment"][endpoint_pair_mask],
        )
        if bool(endpoint_pair_mask.any())
        else prediction.endpoint_attachment_logits.sum() * 0.0
    )
    overlap_loss = _balanced_binary_loss(
        prediction.disconnected_overlap_logits[pair_mask], aligned["overlap"][pair_mask],
    )
    port_relations = 0.5 * (attachment_loss + overlap_loss)

    temporal_target = torch.full(
        (len(matched), 5, MAXIMUM_SLOTS), MAXIMUM_SLOTS,
        dtype=torch.long, device=matched.device,
    )
    slot_index = torch.arange(MAXIMUM_SLOTS, device=matched.device)[None, None]
    temporal_target = torch.where(aligned["temporal"], slot_index, temporal_target)
    temporal_correspondence = F.cross_entropy(
        prediction.temporal_correspondence_logits[matched[:, None].expand(-1, 5, -1)],
        temporal_target[matched[:, None].expand(-1, 5, -1)],
    )
    temporal_presence = _balanced_binary_loss(
        prediction.temporal_presence_logits[matched[:, None].expand(-1, 5, -1)],
        aligned["temporal"][matched[:, None].expand(-1, 5, -1)],
    )
    temporal_equivariance = 0.5 * (temporal_correspondence + temporal_presence)

    uncertainty_calibration = F.smooth_l1_loss(
        prediction.geometry_uncertainty[matched], parameter_error[matched].detach(),
    )
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


__all__ = [
    "PrimitiveAssignment", "PrimitiveRelationLossTargets",
    "align_primitive_relation_targets", "match_primitives",
    "primitive_relation_losses", "sample_swept_superellipse_surface",
]
