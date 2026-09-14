"""Cardinality-conditioned circular slot transport for executable exit sets."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_circular_peak_geometry_model import (
    BEARING_BINS,
    CURVATURE_SCALE_PER_M,
    ENCODER_DIM,
    EXIT_DESCRIPTOR_DIM,
    METRIC_DISTANCE_SCALE_M,
    PLACE_DESCRIPTOR_DIM,
    PROFILE_DIM,
    PROFILE_SCALE_M,
    SLOPE_SCALE_DEG,
    CircularPeakGeometryConfig,
    CircularPeakGeometrySemanticNet,
)


MIN_EXIT_COUNT = 1
MAX_EXIT_COUNT = 4
TOTAL_CARDINALITY_SLOTS = sum(range(MIN_EXIT_COUNT, MAX_EXIT_COUNT + 1))
BIN_WIDTH_DEG = 360.0 / BEARING_BINS
BRANCH_SLICE = {
    count: slice(sum(range(1, count)), sum(range(1, count + 1)))
    for count in range(MIN_EXIT_COUNT, MAX_EXIT_COUNT + 1)
}


@dataclass(frozen=True)
class CircularSlotTransportObservation:
    count: torch.Tensor
    bearing_deg: torch.Tensor
    concentration: torch.Tensor
    valid_mask: torch.Tensor
    opening_width_m: torch.Tensor
    vertical_profile_m: torch.Tensor
    descriptor: torch.Tensor
    uncertainty: torch.Tensor
    count_confidence: torch.Tensor
    set_confidence: torch.Tensor


class CardinalityConditionedCircularSlotTransportNet(CircularPeakGeometrySemanticNet):
    """Predict K azimuth-domain distributions after explicitly predicting K.

    The slots have no existence/objectness output.  For the selected
    cardinality, every slot is required to transport to one distinct exit.
    """

    def __init__(self, config: CircularPeakGeometryConfig | None = None) -> None:
        super().__init__(config=config)
        del self.peak_head
        self.slot_logit_head = nn.Sequential(
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, TOTAL_CARDINALITY_SLOTS, 1),
        )
        self.exit_count_head = nn.Linear(ENCODER_DIM, MAX_EXIT_COUNT)
        self.slot_geometry_head = nn.Sequential(
            nn.Linear(ENCODER_DIM, ENCODER_DIM),
            nn.SiLU(),
            nn.Linear(ENCODER_DIM, 1 + PROFILE_DIM + EXIT_DESCRIPTOR_DIM + 6),
        )

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        context, directional = self.encode_causal_features(scans)
        slot_logits = self.slot_logit_head(directional)
        slot_log_mass = torch.log_softmax(slot_logits, dim=-1)
        slot_mass = torch.softmax(slot_logits, dim=-1)
        azimuth = self.bearing_azimuth_rad.to(dtype=directional.dtype)
        cosine = (slot_mass * torch.cos(azimuth)).sum(dim=-1)
        sine = (slot_mass * torch.sin(azimuth)).sum(dim=-1)
        concentration = torch.sqrt(cosine.square() + sine.square()).clamp(0.0, 1.0)
        bearing_deg = torch.remainder(torch.rad2deg(torch.atan2(sine, cosine)), 360.0)
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
            dim=-1,
            eps=1e-8,
        )
        local_axis = F.normalize(torch.cat((horizontal_axis, torch.tanh(self.axis_vertical_head(context))), dim=-1), dim=-1, eps=1e-8)
        geometry = self.geometry_head(context)
        count_logits = self.exit_count_head(context)
        count_probability = torch.softmax(count_logits, dim=-1)
        decoded = _decode_slots(
            count_probability=count_probability,
            bearing_deg=bearing_deg,
            concentration=concentration,
            opening_width_m=opening_width_m,
            vertical_profile_m=vertical_profile_m,
            descriptor=descriptor,
            uncertainty=uncertainty,
        )
        return {
            "slot_logits": slot_logits,
            "slot_log_mass": slot_log_mass,
            "slot_mass": slot_mass,
            "slot_bearing_deg": bearing_deg,
            "slot_concentration": concentration,
            "slot_resultant_xy": torch.stack((cosine, sine), dim=-1),
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


def _decode_slots(
    *,
    count_probability: torch.Tensor,
    bearing_deg: torch.Tensor,
    concentration: torch.Tensor,
    opening_width_m: torch.Tensor,
    vertical_profile_m: torch.Tensor,
    descriptor: torch.Tensor,
    uncertainty: torch.Tensor,
) -> CircularSlotTransportObservation:
    count = torch.argmax(count_probability, dim=-1) + MIN_EXIT_COUNT
    batch = len(count)
    device = count.device
    padded_bearing = bearing_deg.new_zeros((batch, MAX_EXIT_COUNT))
    padded_concentration = concentration.new_zeros((batch, MAX_EXIT_COUNT))
    padded_width = opening_width_m.new_zeros((batch, MAX_EXIT_COUNT))
    padded_profile = vertical_profile_m.new_zeros((batch, MAX_EXIT_COUNT, PROFILE_DIM))
    padded_descriptor = descriptor.new_zeros((batch, MAX_EXIT_COUNT, EXIT_DESCRIPTOR_DIM))
    padded_uncertainty = uncertainty.new_zeros((batch, MAX_EXIT_COUNT, 6))
    valid = torch.zeros((batch, MAX_EXIT_COUNT), dtype=torch.bool, device=device)
    for row in range(batch):
        cardinality = int(count[row])
        branch = BRANCH_SLICE[cardinality]
        valid[row, :cardinality] = True
        padded_bearing[row, :cardinality] = bearing_deg[row, branch]
        padded_concentration[row, :cardinality] = concentration[row, branch]
        padded_width[row, :cardinality] = opening_width_m[row, branch]
        padded_profile[row, :cardinality] = vertical_profile_m[row, branch]
        padded_descriptor[row, :cardinality] = descriptor[row, branch]
        padded_uncertainty[row, :cardinality] = uncertainty[row, branch]
    count_confidence = count_probability.max(dim=-1).values
    minimum_concentration = torch.stack([padded_concentration[row, : int(count[row])].min() for row in range(batch)])
    return CircularSlotTransportObservation(
        count=count,
        bearing_deg=padded_bearing,
        concentration=padded_concentration,
        valid_mask=valid,
        opening_width_m=padded_width,
        vertical_profile_m=padded_profile,
        descriptor=padded_descriptor,
        uncertainty=padded_uncertainty,
        count_confidence=count_confidence,
        set_confidence=count_confidence * minimum_concentration,
    )


def _soft_circular_target(bin_index: torch.Tensor, residual_deg: torch.Tensor, *, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    continuous = bin_index.to(dtype) + residual_deg.to(dtype) / BIN_WIDTH_DEG
    lower_unwrapped = torch.floor(continuous)
    fraction = continuous - lower_unwrapped
    lower = torch.remainder(lower_unwrapped.to(torch.long), BEARING_BINS)
    upper = torch.remainder(lower + 1, BEARING_BINS)
    return lower, upper, 1.0 - fraction, fraction


def _global_geometry_losses(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    target_axis = F.normalize(targets["local_axis"].to(outputs["local_axis"].dtype), dim=-1)
    axis = (1.0 - (outputs["local_axis"] * target_axis).sum(dim=-1)).mean()
    target_geometry = targets["geometry"].to(outputs["width_m"].dtype)
    valid = targets["geometry_valid_mask"].bool()
    prediction = torch.stack((outputs["width_m"] / METRIC_DISTANCE_SCALE_M, outputs["height_m"] / METRIC_DISTANCE_SCALE_M, outputs["slope_deg"] / SLOPE_SCALE_DEG, outputs["curvature_per_m"] / CURVATURE_SCALE_PER_M), dim=-1)
    normalized_target = torch.stack((target_geometry[:, 0] / METRIC_DISTANCE_SCALE_M, target_geometry[:, 1] / METRIC_DISTANCE_SCALE_M, target_geometry[:, 2] / SLOPE_SCALE_DEG, target_geometry[:, 3] / CURVATURE_SCALE_PER_M), dim=-1)
    global_geometry = torch.stack([F.smooth_l1_loss(prediction[:, index][valid[:, index]], normalized_target[:, index][valid[:, index]]) for index in range(4)]).mean()
    return axis, global_geometry


def _circular_slot_transport_loss_reference(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    presence = targets["presence"].bool()
    count = presence.sum(dim=1)
    if not bool(((count >= MIN_EXIT_COUNT) & (count <= MAX_EXIT_COUNT)).all()):
        raise ValueError("slot transport cardinality must be in [1,4]")
    log_mass = outputs["slot_log_mass"]
    if log_mass.shape != (len(presence), TOTAL_CARDINALITY_SLOTS, BEARING_BINS):
        raise ValueError("slot transport field shape drift")
    assignment_losses = []
    bearing_losses = []
    geometry_losses = []
    for row in range(len(presence)):
        cardinality = int(count[row])
        target_bins = torch.nonzero(presence[row], as_tuple=False).squeeze(1)
        residual = targets["heading_residual_deg"][row, target_bins]
        lower, upper, lower_weight, upper_weight = _soft_circular_target(target_bins, residual, dtype=log_mass.dtype)
        target_bearing = torch.remainder(target_bins.to(log_mass.dtype) * BIN_WIDTH_DEG + residual.to(log_mass.dtype), 360.0)
        branch_indices = list(range(BRANCH_SLICE[cardinality].start, BRANCH_SLICE[cardinality].stop))
        candidates = []
        for order in permutations(range(cardinality)):
            nll = log_mass.new_zeros(())
            bearing = log_mass.new_zeros(())
            geometry = log_mass.new_zeros(())
            for local_slot, target_index in enumerate(order):
                slot = branch_indices[local_slot]
                nll = nll - lower_weight[target_index] * log_mass[row, slot, lower[target_index]] - upper_weight[target_index] * log_mass[row, slot, upper[target_index]]
                delta_rad = torch.deg2rad(outputs["slot_bearing_deg"][row, slot] - target_bearing[target_index])
                bearing = bearing + (1.0 - torch.cos(delta_rad))
                target_bin = target_bins[target_index]
                if bool(targets["width_valid_mask"][row, target_bin]):
                    geometry = geometry + F.smooth_l1_loss(torch.log1p(outputs["slot_opening_width_m"][row, slot]) / math.log1p(60.0), torch.log1p(targets["opening_width_m"][row, target_bin].clamp_min(0.0)) / math.log1p(60.0))
                geometry = geometry + F.smooth_l1_loss(outputs["slot_vertical_profile_m"][row, slot] / PROFILE_SCALE_M, targets["vertical_profile_m"][row, target_bin].to(log_mass.dtype) / PROFILE_SCALE_M)
            candidates.append((nll / cardinality, bearing / cardinality, geometry / cardinality))
        totals = torch.stack([sum(candidate) for candidate in candidates])
        chosen = int(torch.argmin(totals.detach()))
        assignment_losses.append(candidates[chosen][0])
        bearing_losses.append(candidates[chosen][1])
        geometry_losses.append(candidates[chosen][2])
    assignment = torch.stack(assignment_losses).mean()
    bearing = torch.stack(bearing_losses).mean()
    slot_geometry = torch.stack(geometry_losses).mean()
    cardinality = F.cross_entropy(outputs["exit_count_logits"], count - MIN_EXIT_COUNT)
    axis, global_geometry = _global_geometry_losses(outputs, targets)
    total = assignment + cardinality + bearing + slot_geometry + axis + global_geometry
    return {"total": total, "assignment": assignment, "cardinality": cardinality, "bearing": bearing, "slot_geometry": slot_geometry, "axis": axis, "global_geometry": global_geometry}


def circular_slot_transport_loss(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Vectorized equivalent of the explicit bijective reference loss."""
    presence = targets["presence"].bool()
    count = presence.sum(dim=1)
    if not bool(((count >= MIN_EXIT_COUNT) & (count <= MAX_EXIT_COUNT)).all()):
        raise ValueError("slot transport cardinality must be in [1,4]")
    log_mass = outputs["slot_log_mass"]
    if log_mass.shape != (len(presence), TOTAL_CARDINALITY_SLOTS, BEARING_BINS):
        raise ValueError("slot transport field shape drift")
    assignment_parts = []
    bearing_parts = []
    geometry_parts = []
    for cardinality_value in range(MIN_EXIT_COUNT, MAX_EXIT_COUNT + 1):
        rows = torch.nonzero(count == cardinality_value, as_tuple=False).squeeze(1)
        if not len(rows):
            continue
        cardinality_int = int(cardinality_value)
        target_bins = torch.nonzero(presence[rows], as_tuple=False)[:, 1].reshape(len(rows), cardinality_int)
        residual = torch.gather(targets["heading_residual_deg"][rows], 1, target_bins)
        lower, upper, lower_weight, upper_weight = _soft_circular_target(target_bins, residual, dtype=log_mass.dtype)
        target_bearing = torch.remainder(target_bins.to(log_mass.dtype) * BIN_WIDTH_DEG + residual.to(log_mass.dtype), 360.0)
        branch = BRANCH_SLICE[cardinality_int]
        branch_log_mass = log_mass[rows, branch, :]
        gather_shape = (len(rows), cardinality_int, cardinality_int)
        low_index = lower[:, None, :].expand(gather_shape)
        high_index = upper[:, None, :].expand(gather_shape)
        assignment_cost = -lower_weight[:, None, :] * torch.gather(branch_log_mass, 2, low_index) - upper_weight[:, None, :] * torch.gather(branch_log_mass, 2, high_index)
        slot_bearing = outputs["slot_bearing_deg"][rows, branch]
        bearing_cost = 1.0 - torch.cos(torch.deg2rad(slot_bearing[:, :, None] - target_bearing[:, None, :]))
        slot_width = torch.log1p(outputs["slot_opening_width_m"][rows, branch]) / math.log1p(60.0)
        target_width = torch.log1p(torch.gather(targets["opening_width_m"][rows], 1, target_bins).clamp_min(0.0)) / math.log1p(60.0)
        width_valid = torch.gather(targets["width_valid_mask"][rows], 1, target_bins).to(log_mass.dtype)
        width_cost = F.smooth_l1_loss(slot_width[:, :, None].expand(-1, -1, cardinality_int), target_width[:, None, :].expand(-1, cardinality_int, -1), reduction="none") * width_valid[:, None, :]
        slot_profile = outputs["slot_vertical_profile_m"][rows, branch] / PROFILE_SCALE_M
        target_profile = torch.gather(targets["vertical_profile_m"][rows], 1, target_bins[:, :, None].expand(-1, -1, PROFILE_DIM)).to(log_mass.dtype) / PROFILE_SCALE_M
        profile_cost = F.smooth_l1_loss(slot_profile[:, :, None, :].expand(-1, -1, cardinality_int, -1), target_profile[:, None, :, :].expand(-1, cardinality_int, -1, -1), reduction="none").mean(dim=-1)
        geometry_cost = width_cost + profile_cost
        total_cost = assignment_cost + bearing_cost + geometry_cost
        permutation_tensor = torch.tensor(list(permutations(range(cardinality_int))), dtype=torch.long, device=log_mass.device)
        permutation_count = len(permutation_tensor)
        expanded_index = permutation_tensor[None, :, :, None].expand(len(rows), -1, -1, 1)
        candidate_total = total_cost[:, None, :, :].expand(-1, permutation_count, -1, -1).gather(3, expanded_index).squeeze(-1).sum(dim=-1)
        chosen = permutation_tensor[torch.argmin(candidate_total.detach(), dim=1)]
        chosen_index = chosen[:, :, None]
        assignment_parts.append(torch.gather(assignment_cost, 2, chosen_index).squeeze(-1).mean(dim=1))
        bearing_parts.append(torch.gather(bearing_cost, 2, chosen_index).squeeze(-1).mean(dim=1))
        geometry_parts.append(torch.gather(geometry_cost, 2, chosen_index).squeeze(-1).mean(dim=1))
    assignment = torch.cat(assignment_parts).mean()
    bearing = torch.cat(bearing_parts).mean()
    slot_geometry = torch.cat(geometry_parts).mean()
    cardinality = F.cross_entropy(outputs["exit_count_logits"], count - MIN_EXIT_COUNT)
    axis, global_geometry = _global_geometry_losses(outputs, targets)
    total = assignment + cardinality + bearing + slot_geometry + axis + global_geometry
    return {"total": total, "assignment": assignment, "cardinality": cardinality, "bearing": bearing, "slot_geometry": slot_geometry, "axis": axis, "global_geometry": global_geometry}


def circular_slot_transport_contract() -> dict[str, object]:
    return {
        "student_input": "five causal range/valid images only",
        "cardinality": (MIN_EXIT_COUNT, MAX_EXIT_COUNT),
        "slots_by_cardinality": {count: count for count in range(MIN_EXIT_COUNT, MAX_EXIT_COUNT + 1)},
        "slot_domain": "fixed 180-bin sensor azimuth distributions",
        "existence_objectness": None,
        "assignment": "permutation-invariant bijection between K required slots and K exits",
        "continuous_bearing": "circular mean of each matched slot distribution",
        "confidence": "predicted-count confidence times minimum slot resultant concentration",
        "node_rule": "past-only stable executable slot-set change",
        "edge_rule": "physical traversal only",
        "forbidden_inputs": ("pose", "world_id", "traversal_id", "TNG_identity", "exit_identity", "future_frame", "graph_state", "C09", "C10", "M-TARE"),
    }


__all__ = [
    "BIN_WIDTH_DEG",
    "BRANCH_SLICE",
    "CardinalityConditionedCircularSlotTransportNet",
    "CircularSlotTransportObservation",
    "TOTAL_CARDINALITY_SLOTS",
    "_circular_slot_transport_loss_reference",
    "circular_slot_transport_contract",
    "circular_slot_transport_loss",
]
