"""Cyclic-ordered unimodal slot transport for executable exit geometry."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_cardinality_conditioned_circular_slot_transport import (
    BIN_WIDTH_DEG,
    BRANCH_SLICE,
    MAX_EXIT_COUNT,
    MIN_EXIT_COUNT,
    TOTAL_CARDINALITY_SLOTS,
    CardinalityConditionedCircularSlotTransportNet,
    CircularSlotTransportObservation,
    _decode_slots,
    _global_geometry_losses,
    _soft_circular_target,
)
from mtare_topo.representation.gse_circular_peak_geometry_model import (
    BEARING_BINS,
    ENCODER_DIM,
    EXIT_DESCRIPTOR_DIM,
    PROFILE_DIM,
    PROFILE_SCALE_M,
    SLOPE_SCALE_DEG,
)


@dataclass(frozen=True)
class CyclicOrderedUnimodalSlotContract:
    cardinality: tuple[int, int]
    assignment_group: str
    orientation: str
    distribution: str
    continuous_bearing: str
    confidence: str
    free_query_existence: bool


CONTRACT = CyclicOrderedUnimodalSlotContract(
    cardinality=(MIN_EXIT_COUNT, MAX_EXIT_COUNT),
    assignment_group="K orientation-preserving cyclic shifts for cardinality K",
    orientation="sensor-azimuth counter-clockwise order",
    distribution="discrete unimodal von-Mises projection over 180 fixed bins",
    continuous_bearing="first circular moment of the projected distribution",
    confidence="count confidence times minimum proper circular concentration",
    free_query_existence=False,
)


def discrete_von_mises(
    mean_unit_xy: torch.Tensor,
    kappa: torch.Tensor,
    azimuth_rad: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return normalized log mass, mass, bearing and concentration.

    The discrete normalization through log_softmax is stable for both uniform
    and extreme concentrations and avoids a platform-dependent Bessel ratio.
    """
    unit = F.normalize(mean_unit_xy, dim=-1, eps=1e-8)
    phase = unit[..., 0, None] * torch.cos(azimuth_rad) + unit[..., 1, None] * torch.sin(azimuth_rad)
    log_mass = F.log_softmax(kappa[..., None] * phase, dim=-1)
    mass = torch.exp(log_mass)
    cosine = (mass * torch.cos(azimuth_rad)).sum(dim=-1)
    sine = (mass * torch.sin(azimuth_rad)).sum(dim=-1)
    bearing = torch.remainder(torch.rad2deg(torch.atan2(sine, cosine)), 360.0)
    concentration = torch.sqrt(cosine.square() + sine.square()).clamp(0.0, 1.0)
    return log_mass, mass, bearing, concentration


class CyclicOrderedUnimodalSlotTransportNet(CardinalityConditionedCircularSlotTransportNet):
    """Project equivariant raw angular evidence into proper unimodal slots."""

    def __init__(self) -> None:
        super().__init__()
        self.slot_kappa_head = nn.Linear(ENCODER_DIM, TOTAL_CARDINALITY_SLOTS)

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        context, directional = self.encode_causal_features(scans)
        raw_logits = self.slot_logit_head(directional)
        raw_mass = torch.softmax(raw_logits, dim=-1)
        azimuth = self.bearing_azimuth_rad.to(dtype=directional.dtype)
        raw_mean_xy = torch.stack(
            ((raw_mass * torch.cos(azimuth)).sum(dim=-1), (raw_mass * torch.sin(azimuth)).sum(dim=-1)),
            dim=-1,
        )
        kappa = F.softplus(self.slot_kappa_head(context)) + 1e-3
        slot_log_mass, slot_mass, slot_bearing, slot_concentration = discrete_von_mises(raw_mean_xy, kappa, azimuth)
        slot_resultant_xy = torch.stack(
            ((slot_mass * torch.cos(azimuth)).sum(dim=-1), (slot_mass * torch.sin(azimuth)).sum(dim=-1)),
            dim=-1,
        )
        slot_feature = torch.einsum("bsk,bck->bsc", slot_mass, directional)
        raw_slot_geometry = self.slot_geometry_head(slot_feature)
        offset = 0
        opening_width_m = F.softplus(raw_slot_geometry[..., offset]) + 1e-4
        offset += 1
        vertical_profile_m = PROFILE_SCALE_M * torch.tanh(raw_slot_geometry[..., offset : offset + PROFILE_DIM])
        offset += PROFILE_DIM
        descriptor = F.normalize(raw_slot_geometry[..., offset : offset + EXIT_DESCRIPTOR_DIM], dim=-1, eps=1e-8)
        offset += EXIT_DESCRIPTOR_DIM
        uncertainty = F.softplus(raw_slot_geometry[..., offset : offset + 6]) + 0.05

        axis_probability = torch.softmax(self.axis_azimuth_head(directional).squeeze(1), dim=-1)
        horizontal_axis = F.normalize(
            torch.stack(((axis_probability * torch.cos(azimuth)).sum(-1), (axis_probability * torch.sin(azimuth)).sum(-1)), dim=-1),
            dim=-1, eps=1e-8,
        )
        local_axis = F.normalize(torch.cat((horizontal_axis, torch.tanh(self.axis_vertical_head(context))), dim=-1), dim=-1, eps=1e-8)
        geometry = self.geometry_head(context)
        count_logits = self.exit_count_head(context)
        count_probability = torch.softmax(count_logits, dim=-1)
        decoded = _decode_slots(
            count_probability=count_probability, bearing_deg=slot_bearing, concentration=slot_concentration,
            opening_width_m=opening_width_m, vertical_profile_m=vertical_profile_m,
            descriptor=descriptor, uncertainty=uncertainty,
        )
        return {
            "raw_slot_logits": raw_logits,
            "raw_slot_mass": raw_mass,
            "raw_slot_mean_xy": raw_mean_xy,
            "slot_kappa": kappa,
            "slot_log_mass": slot_log_mass,
            "slot_mass": slot_mass,
            "slot_bearing_deg": slot_bearing,
            "slot_concentration": slot_concentration,
            "slot_resultant_xy": slot_resultant_xy,
            "slot_opening_width_m": opening_width_m,
            "slot_vertical_profile_m": vertical_profile_m,
            "slot_descriptor": descriptor,
            "slot_uncertainty": uncertainty,
            "exit_count_logits": count_logits,
            "exit_count_probability": count_probability,
            "decoded_count": decoded.count,
            "decoded_bearing_deg": decoded.bearing_deg,
            "decoded_concentration": decoded.concentration,
            "decoded_valid_mask": decoded.valid_mask,
            "decoded_opening_width_m": decoded.opening_width_m,
            "decoded_vertical_profile_m": decoded.vertical_profile_m,
            "decoded_descriptor": decoded.descriptor,
            "decoded_uncertainty": decoded.uncertainty,
            "decoded_count_confidence": decoded.count_confidence,
            "decoded_set_confidence": decoded.set_confidence,
            "local_axis": local_axis,
            "width_m": F.softplus(geometry[:, 0]) + 1e-4,
            "height_m": F.softplus(geometry[:, 1]) + 1e-4,
            "slope_deg": SLOPE_SCALE_DEG * torch.tanh(geometry[:, 2]),
            "curvature_per_m": F.softplus(geometry[:, 3]),
            "place_descriptor": F.normalize(self.place_head(context), dim=-1, eps=1e-8),
            "observation_uncertainty": torch.sigmoid(self.observation_uncertainty_head(context).squeeze(-1)),
        }


def cyclic_orders(cardinality: int) -> tuple[tuple[int, ...], ...]:
    """All and only orientation-preserving cyclic target assignments."""
    if cardinality < MIN_EXIT_COUNT or cardinality > MAX_EXIT_COUNT:
        raise ValueError("cyclic cardinality must be in [1,4]")
    return tuple(tuple((slot + shift) % cardinality for slot in range(cardinality)) for shift in range(cardinality))


def cyclic_ordered_unimodal_slot_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Proper discrete circular NLL with cyclic-order-preserving transport."""
    presence = targets["presence"].bool()
    count = presence.sum(dim=1)
    if not bool(((count >= MIN_EXIT_COUNT) & (count <= MAX_EXIT_COUNT)).all()):
        raise ValueError("COUST cardinality must be in [1,4]")
    log_mass = outputs["slot_log_mass"]
    if log_mass.shape != (len(presence), TOTAL_CARDINALITY_SLOTS, BEARING_BINS):
        raise ValueError("COUST slot field shape drift")

    assignment_parts = []
    bearing_parts = []
    geometry_parts = []
    for cardinality_value in range(MIN_EXIT_COUNT, MAX_EXIT_COUNT + 1):
        rows = torch.nonzero(count == cardinality_value, as_tuple=False).squeeze(1)
        if not len(rows):
            continue
        cardinality = int(cardinality_value)
        target_bins = torch.nonzero(presence[rows], as_tuple=False)[:, 1].reshape(len(rows), cardinality)
        residual = torch.gather(targets["heading_residual_deg"][rows], 1, target_bins)
        lower, upper, lower_weight, upper_weight = _soft_circular_target(target_bins, residual, dtype=log_mass.dtype)
        target_bearing = torch.remainder(target_bins.to(log_mass.dtype) * BIN_WIDTH_DEG + residual.to(log_mass.dtype), 360.0)
        branch = torch.arange(BRANCH_SLICE[cardinality].start, BRANCH_SLICE[cardinality].stop, device=log_mass.device)
        selected_log_mass = log_mass[rows[:, None], branch[None, :]]
        lower_log = torch.gather(selected_log_mass, 2, lower[:, None, :].expand(-1, cardinality, -1))
        upper_log = torch.gather(selected_log_mass, 2, upper[:, None, :].expand(-1, cardinality, -1))
        assignment_cost = -(lower_weight[:, None, :] * lower_log + upper_weight[:, None, :] * upper_log)
        slot_bearing = outputs["slot_bearing_deg"][rows[:, None], branch[None, :]]
        bearing_cost = 1.0 - torch.cos(torch.deg2rad(slot_bearing[:, :, None] - target_bearing[:, None, :]))
        predicted_width = outputs["slot_opening_width_m"][rows[:, None], branch[None, :]]
        target_width = torch.gather(targets["opening_width_m"][rows], 1, target_bins).to(log_mass.dtype)
        width_valid = torch.gather(targets["width_valid_mask"][rows], 1, target_bins).bool()
        width_cost = F.smooth_l1_loss(
            torch.log1p(predicted_width[:, :, None].expand(-1, -1, cardinality)) / math.log1p(60.0),
            torch.log1p(target_width[:, None, :].expand(-1, cardinality, -1).clamp_min(0.0)) / math.log1p(60.0), reduction="none",
        ) * width_valid[:, None, :]
        predicted_profile = outputs["slot_vertical_profile_m"][rows[:, None], branch[None, :]]
        target_profile = torch.gather(
            targets["vertical_profile_m"][rows], 1,
            target_bins[:, :, None].expand(-1, -1, PROFILE_DIM),
        ).to(log_mass.dtype)
        profile_cost = F.smooth_l1_loss(
            predicted_profile[:, :, None, :].expand(-1, -1, cardinality, -1) / PROFILE_SCALE_M,
            target_profile[:, None, :, :].expand(-1, cardinality, -1, -1) / PROFILE_SCALE_M, reduction="none",
        ).mean(dim=-1)
        geometry_cost = width_cost + profile_cost
        total_cost = assignment_cost + bearing_cost + geometry_cost
        orders = torch.tensor(cyclic_orders(cardinality), device=log_mass.device, dtype=torch.long)
        candidates = []
        for order in orders:
            index = order[None, :, None].expand(len(rows), -1, 1)
            candidates.append(torch.stack((
                torch.gather(assignment_cost, 2, index).squeeze(-1).mean(dim=1),
                torch.gather(bearing_cost, 2, index).squeeze(-1).mean(dim=1),
                torch.gather(geometry_cost, 2, index).squeeze(-1).mean(dim=1),
                torch.gather(total_cost, 2, index).squeeze(-1).mean(dim=1),
            ), dim=-1))
        candidate_values = torch.stack(candidates, dim=1)
        chosen = torch.argmin(candidate_values[..., 3].detach(), dim=1)
        selected = candidate_values[torch.arange(len(rows), device=log_mass.device), chosen]
        assignment_parts.append(selected[:, 0])
        bearing_parts.append(selected[:, 1])
        geometry_parts.append(selected[:, 2])

    assignment = torch.cat(assignment_parts).mean()
    bearing = torch.cat(bearing_parts).mean()
    slot_geometry = torch.cat(geometry_parts).mean()
    cardinality_loss = F.cross_entropy(outputs["exit_count_logits"], count - MIN_EXIT_COUNT)
    axis, global_geometry = _global_geometry_losses(outputs, targets)
    total = assignment + cardinality_loss + bearing + slot_geometry + axis + global_geometry
    return {
        "total": total, "assignment": assignment, "cardinality": cardinality_loss,
        "bearing": bearing, "slot_geometry": slot_geometry, "axis": axis,
        "global_geometry": global_geometry,
    }
