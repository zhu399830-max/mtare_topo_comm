"""Joint cyclic gap-simplex decoder for executable exit geometry sets."""
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
    _global_geometry_losses,
)
from mtare_topo.representation.gse_circular_peak_geometry_model import (
    BEARING_BINS,
    ENCODER_DIM,
    EXIT_DESCRIPTOR_DIM,
    PROFILE_DIM,
    PROFILE_SCALE_M,
    SLOPE_SCALE_DEG,
    CircularPeakGeometryConfig,
    CircularPeakGeometrySemanticNet,
)


NUMERIC_GAP_FLOOR = 1e-5
SAFE_PHASE_RESULTANT_SQUARED_NORM = torch.finfo(torch.float32).eps


@dataclass(frozen=True)
class JointCyclicGapSimplexContract:
    cardinality: tuple[int, int]
    representation: str
    closure: str
    assignment: str
    geometry_binding: str
    confidence: str
    independent_slot_localizers: bool


CONTRACT = JointCyclicGapSimplexContract(
    cardinality=(MIN_EXIT_COUNT, MAX_EXIT_COUNT),
    representation="one circular phase plus K positive normalized angular gaps",
    closure="gaps sum exactly to 360 degrees for each cardinality branch",
    assignment="orientation-preserving cyclic target relabels only",
    geometry_binding="periodic differentiable sampling at jointly decoded bearings",
    confidence="proper heteroscedastic phase and gap scales",
    independent_slot_localizers=False,
)


def positive_gap_simplex(logits: torch.Tensor) -> torch.Tensor:
    """Map K logits to strictly positive fractions that sum to one."""
    cardinality = logits.shape[-1]
    if cardinality < MIN_EXIT_COUNT or cardinality > MAX_EXIT_COUNT:
        raise ValueError("gap cardinality must be in [1,4]")
    mass = torch.softmax(logits, dim=-1)
    return NUMERIC_GAP_FLOOR + (1.0 - cardinality * NUMERIC_GAP_FLOOR) * mass


def decode_cyclic_bearings(phase_deg: torch.Tensor, gap_fraction: torch.Tensor) -> torch.Tensor:
    """Decode ordered bearings; the final unreturned gap closes the circle."""
    if phase_deg.shape != gap_fraction.shape[:-1]:
        raise ValueError("phase/gap batch shape mismatch")
    prefix = torch.cat((gap_fraction.new_zeros((*gap_fraction.shape[:-1], 1)), torch.cumsum(gap_fraction[..., :-1], dim=-1)), dim=-1)
    return torch.remainder(phase_deg[..., None] + 360.0 * prefix, 360.0)


def circular_gaps_from_ordered_bearings(bearing_deg: torch.Tensor) -> torch.Tensor:
    """Return positive wrap-aware gaps for a counter-clockwise ordered set."""
    if bearing_deg.shape[-1] == 1:
        return torch.full_like(bearing_deg, 360.0)
    return torch.remainder(torch.roll(bearing_deg, shifts=-1, dims=-1) - bearing_deg, 360.0)


def _circular_absolute_error_deg(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    delta = torch.deg2rad(predicted - target)
    return torch.rad2deg(torch.abs(torch.atan2(torch.sin(delta), torch.cos(delta))))


class _StableAtan2(torch.autograd.Function):
    """atan2 with a bounded derivative where a float32 phase has no direction."""

    @staticmethod
    def forward(ctx, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(y, x)
        return torch.atan2(y, x)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y, x = ctx.saved_tensors
        denominator = (x.square() + y.square()).clamp_min(SAFE_PHASE_RESULTANT_SQUARED_NORM)
        return gradient * x / denominator, -gradient * y / denominator


def stable_phase_atan2(y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Return the ordinary phase angle with finite gradients at zero resultant."""
    return _StableAtan2.apply(y, x)


def periodic_linear_coordinates(bearing_deg: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return canonical interpolation coordinates for the 180-bin circle.

    ``torch.remainder`` can round a negative angle infinitesimally below zero
    to exactly ``BEARING_BINS``.  That value is the same circular coordinate
    as zero, but is not a legal tensor index.  Canonicalising the integer
    coordinate modulo the number of bins preserves the circle rather than
    clipping a genuinely invalid index.
    """
    if not bool(torch.isfinite(bearing_deg).all()):
        raise FloatingPointError("non-finite bearing before periodic indexing")
    continuous = torch.remainder(bearing_deg / BIN_WIDTH_DEG, BEARING_BINS)
    lower_float = torch.floor(continuous)
    fraction = continuous - lower_float
    lower = torch.remainder(lower_float.to(torch.long), BEARING_BINS)
    upper = torch.remainder(lower + 1, BEARING_BINS)
    return fraction, lower, upper


def periodic_linear_sample(directional: torch.Tensor, bearing_deg: torch.Tensor) -> torch.Tensor:
    """Differentiably sample [B,C,180] circular features at [B,S] bearings."""
    if directional.ndim != 3 or directional.shape[-1] != BEARING_BINS:
        raise ValueError("directional field must be [B,C,180]")
    if bearing_deg.ndim != 2 or bearing_deg.shape[0] != directional.shape[0]:
        raise ValueError("bearing field must be [B,S]")
    fraction, lower, upper = periodic_linear_coordinates(bearing_deg)
    channels = directional.shape[1]
    low_value = torch.gather(directional, 2, lower[:, None, :].expand(-1, channels, -1))
    high_value = torch.gather(directional, 2, upper[:, None, :].expand(-1, channels, -1))
    fraction = fraction.to(directional.dtype)
    return ((1.0 - fraction[:, None, :]) * low_value + fraction[:, None, :] * high_value).transpose(1, 2)


class JointCyclicGapSimplexNet(CircularPeakGeometrySemanticNet):
    """Jointly decode a closed ordered exit set for each predicted cardinality."""

    def __init__(self, config: CircularPeakGeometryConfig | None = None) -> None:
        super().__init__(config=config)
        del self.peak_head
        self.phase_logit_head = nn.Sequential(
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, MAX_EXIT_COUNT, 1),
        )
        self.exit_count_head = nn.Linear(ENCODER_DIM, MAX_EXIT_COUNT)
        self.gap_logit_head = nn.Linear(ENCODER_DIM, TOTAL_CARDINALITY_SLOTS)
        self.phase_scale_head = nn.Linear(ENCODER_DIM, MAX_EXIT_COUNT)
        self.gap_scale_head = nn.Linear(ENCODER_DIM, TOTAL_CARDINALITY_SLOTS)
        self.exit_geometry_head = nn.Sequential(
            nn.Linear(ENCODER_DIM, ENCODER_DIM),
            nn.SiLU(),
            nn.Linear(ENCODER_DIM, 1 + PROFILE_DIM + EXIT_DESCRIPTOR_DIM + 6),
        )

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        context, directional = self.encode_causal_features(scans)
        phase_logits = self.phase_logit_head(directional)
        phase_log_mass = torch.log_softmax(phase_logits, dim=-1)
        phase_mass = torch.exp(phase_log_mass)
        azimuth = self.bearing_azimuth_rad.to(dtype=directional.dtype)
        azimuth64 = self.bearing_azimuth_rad.to(dtype=torch.float64)
        phase_mass64 = phase_mass.to(torch.float64)
        phase_xy = torch.stack(((phase_mass64 * torch.cos(azimuth64)).sum(-1), (phase_mass64 * torch.sin(azimuth64)).sum(-1)), dim=-1)
        phase_bearing = torch.remainder(torch.rad2deg(stable_phase_atan2(phase_xy[..., 1], phase_xy[..., 0])), 360.0)
        phase_concentration = torch.linalg.vector_norm(phase_xy, dim=-1).clamp(0.0, 1.0)
        phase_scale = F.softplus(self.phase_scale_head(context)) + 0.05

        raw_gap_logits = self.gap_logit_head(context)
        raw_gap_scale = F.softplus(self.gap_scale_head(context)) + 0.05
        gap_fraction = raw_gap_logits.new_zeros(raw_gap_logits.shape)
        bearing = torch.zeros(raw_gap_logits.shape, dtype=torch.float64, device=raw_gap_logits.device)
        bearing_scale = raw_gap_logits.new_zeros(raw_gap_logits.shape)
        for cardinality in range(MIN_EXIT_COUNT, MAX_EXIT_COUNT + 1):
            branch = BRANCH_SLICE[cardinality]
            gaps = positive_gap_simplex(raw_gap_logits[:, branch])
            gap_fraction[:, branch] = gaps
            bearing[:, branch] = decode_cyclic_bearings(phase_bearing[:, cardinality - 1], gaps.to(torch.float64))
            cumulative_scale = torch.sqrt(phase_scale[:, cardinality - 1, None].square() + torch.cumsum(raw_gap_scale[:, branch].square(), dim=-1) - raw_gap_scale[:, branch].square())
            bearing_scale[:, branch] = cumulative_scale

        sampled = periodic_linear_sample(directional, bearing)
        raw_geometry = self.exit_geometry_head(sampled)
        offset = 0
        opening_width_m = F.softplus(raw_geometry[..., offset]) + 1e-4
        offset += 1
        vertical_profile_m = PROFILE_SCALE_M * torch.tanh(raw_geometry[..., offset : offset + PROFILE_DIM])
        offset += PROFILE_DIM
        descriptor = F.normalize(raw_geometry[..., offset : offset + EXIT_DESCRIPTOR_DIM], dim=-1, eps=1e-8)
        offset += EXIT_DESCRIPTOR_DIM
        uncertainty = F.softplus(raw_geometry[..., offset : offset + 6]) + 0.05

        axis_probability = torch.softmax(self.axis_azimuth_head(directional).squeeze(1), dim=-1)
        horizontal_axis = F.normalize(torch.stack(((axis_probability * torch.cos(azimuth)).sum(-1), (axis_probability * torch.sin(azimuth)).sum(-1)), dim=-1), dim=-1, eps=1e-8)
        local_axis = F.normalize(torch.cat((horizontal_axis, torch.tanh(self.axis_vertical_head(context))), dim=-1), dim=-1, eps=1e-8)
        global_geometry = self.geometry_head(context)
        count_logits = self.exit_count_head(context)
        count_probability = torch.softmax(count_logits, dim=-1)
        decoded_count = torch.argmax(count_probability, dim=-1) + MIN_EXIT_COUNT
        batch = len(scans)
        decoded_bearing = bearing.new_zeros((batch, MAX_EXIT_COUNT))
        decoded_scale = bearing.new_zeros((batch, MAX_EXIT_COUNT))
        decoded_phase_concentration = phase_concentration.new_zeros(batch)
        decoded_width = opening_width_m.new_zeros((batch, MAX_EXIT_COUNT))
        decoded_profile = vertical_profile_m.new_zeros((batch, MAX_EXIT_COUNT, PROFILE_DIM))
        decoded_descriptor = descriptor.new_zeros((batch, MAX_EXIT_COUNT, EXIT_DESCRIPTOR_DIM))
        decoded_uncertainty = uncertainty.new_zeros((batch, MAX_EXIT_COUNT, 6))
        decoded_valid = torch.zeros((batch, MAX_EXIT_COUNT), dtype=torch.bool, device=scans.device)
        for row in range(batch):
            cardinality = int(decoded_count[row])
            branch = BRANCH_SLICE[cardinality]
            decoded_valid[row, :cardinality] = True
            decoded_phase_concentration[row] = phase_concentration[row, cardinality - 1]
            decoded_bearing[row, :cardinality] = bearing[row, branch]
            decoded_scale[row, :cardinality] = bearing_scale[row, branch]
            decoded_width[row, :cardinality] = opening_width_m[row, branch]
            decoded_profile[row, :cardinality] = vertical_profile_m[row, branch]
            decoded_descriptor[row, :cardinality] = descriptor[row, branch]
            decoded_uncertainty[row, :cardinality] = uncertainty[row, branch]
        count_confidence = count_probability.max(dim=-1).values
        max_scale = torch.stack([decoded_scale[row, : int(decoded_count[row])].max() for row in range(batch)])
        set_confidence = count_confidence * decoded_phase_concentration * torch.exp(-max_scale / 180.0)
        return {
            "phase_logits": phase_logits, "phase_log_mass": phase_log_mass, "phase_mass": phase_mass, "phase_bearing_deg": phase_bearing,
            "phase_concentration": phase_concentration, "phase_scale_deg": phase_scale,
            "gap_logits": raw_gap_logits, "gap_fraction": gap_fraction, "gap_scale_deg": raw_gap_scale,
            "joint_bearing_deg": bearing, "joint_bearing_scale_deg": bearing_scale,
            "exit_opening_width_m": opening_width_m, "exit_vertical_profile_m": vertical_profile_m,
            "exit_descriptor": descriptor, "exit_uncertainty": uncertainty,
            "exit_count_logits": count_logits, "exit_count_probability": count_probability,
            "decoded_count": decoded_count, "decoded_bearing_deg": decoded_bearing,
            "decoded_bearing_scale_deg": decoded_scale, "decoded_valid_mask": decoded_valid,
            "decoded_phase_concentration": decoded_phase_concentration,
            "decoded_opening_width_m": decoded_width, "decoded_vertical_profile_m": decoded_profile,
            "decoded_descriptor": decoded_descriptor, "decoded_uncertainty": decoded_uncertainty,
            "decoded_count_confidence": count_confidence, "decoded_set_confidence": set_confidence,
            "local_axis": local_axis,
            "width_m": F.softplus(global_geometry[:, 0]) + 1e-4,
            "height_m": F.softplus(global_geometry[:, 1]) + 1e-4,
            "slope_deg": SLOPE_SCALE_DEG * torch.tanh(global_geometry[:, 2]),
            "curvature_per_m": F.softplus(global_geometry[:, 3]),
            "place_descriptor": F.normalize(self.place_head(context), dim=-1, eps=1e-8),
            "observation_uncertainty": torch.sigmoid(self.observation_uncertainty_head(context).squeeze(-1)),
        }


def joint_cyclic_gap_simplex_loss(outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Cyclic-invariant joint phase/gap likelihood plus bound exit geometry."""
    presence = targets["presence"].bool()
    count = presence.sum(dim=1)
    if not bool(((count >= MIN_EXIT_COUNT) & (count <= MAX_EXIT_COUNT)).all()):
        raise ValueError("joint gap cardinality must be in [1,4]")
    phase_likelihood_parts, localization_parts, uncertainty_parts, geometry_parts = [], [], [], []
    for cardinality_value in range(MIN_EXIT_COUNT, MAX_EXIT_COUNT + 1):
        rows = torch.nonzero(count == cardinality_value, as_tuple=False).flatten()
        if not len(rows):
            continue
        cardinality = int(cardinality_value)
        target_bins = torch.nonzero(presence[rows], as_tuple=False)[:, 1].reshape(len(rows), cardinality)
        residual = torch.gather(targets["heading_residual_deg"][rows], 1, target_bins).to(outputs["joint_bearing_deg"].dtype)
        target_bearing = target_bins.to(residual.dtype) * BIN_WIDTH_DEG + residual
        branch = BRANCH_SLICE[cardinality]
        predicted_bearing = outputs["joint_bearing_deg"][rows, branch]
        predicted_gap = 360.0 * outputs["gap_fraction"][rows, branch]
        phase_scale = outputs["phase_scale_deg"][rows, cardinality - 1]
        gap_scale = outputs["gap_scale_deg"][rows, branch]
        candidate_values = []
        for shift in range(cardinality):
            ordered_target = torch.roll(target_bearing, shifts=-shift, dims=1)
            target_gap = circular_gaps_from_ordered_bearings(ordered_target)
            bearing_error = _circular_absolute_error_deg(predicted_bearing, ordered_target)
            phase_error = bearing_error[:, 0]
            phase_fraction, lower_phase, upper_phase = periodic_linear_coordinates(ordered_target[:, 0])
            selected_phase_log_mass = outputs["phase_log_mass"][rows, cardinality - 1]
            phase_likelihood = -(
                (1.0 - phase_fraction) * torch.gather(selected_phase_log_mass, 1, lower_phase[:, None]).squeeze(1)
                + phase_fraction * torch.gather(selected_phase_log_mass, 1, upper_phase[:, None]).squeeze(1)
            )
            gap_error = torch.abs(predicted_gap - target_gap)
            localization = (bearing_error / 180.0).mean(dim=1) + (gap_error / 360.0).mean(dim=1)
            uncertainty_nll = phase_error / phase_scale + torch.log(phase_scale / 180.0)
            uncertainty_nll = uncertainty_nll + (gap_error / gap_scale + torch.log(gap_scale / 360.0)).mean(dim=1)
            predicted_width = outputs["exit_opening_width_m"][rows, branch]
            target_width = torch.gather(targets["opening_width_m"][rows], 1, target_bins)
            target_width = torch.roll(target_width, shifts=-shift, dims=1).to(predicted_width.dtype)
            width_valid = torch.gather(targets["width_valid_mask"][rows], 1, target_bins)
            width_valid = torch.roll(width_valid, shifts=-shift, dims=1).to(predicted_width.dtype)
            width_cost = F.smooth_l1_loss(torch.log1p(predicted_width) / math.log1p(60.0), torch.log1p(target_width.clamp_min(0.0)) / math.log1p(60.0), reduction="none") * width_valid
            predicted_profile = outputs["exit_vertical_profile_m"][rows, branch]
            target_profile = torch.gather(targets["vertical_profile_m"][rows], 1, target_bins[:, :, None].expand(-1, -1, PROFILE_DIM))
            target_profile = torch.roll(target_profile, shifts=-shift, dims=1).to(predicted_profile.dtype)
            profile_cost = F.smooth_l1_loss(predicted_profile / PROFILE_SCALE_M, target_profile / PROFILE_SCALE_M, reduction="none").mean(dim=-1)
            geometry = (width_cost + profile_cost).mean(dim=1)
            candidate_values.append(torch.stack((phase_likelihood, localization, uncertainty_nll, geometry), dim=-1))
        candidates = torch.stack(candidate_values, dim=1)
        chosen = torch.argmin((candidates[..., 0] + candidates[..., 1] + candidates[..., 3]).detach(), dim=1)
        selected = candidates[torch.arange(len(rows), device=rows.device), chosen]
        phase_likelihood_parts.append(selected[:, 0]); localization_parts.append(selected[:, 1]); uncertainty_parts.append(selected[:, 2]); geometry_parts.append(selected[:, 3])
    phase_likelihood = torch.cat(phase_likelihood_parts).mean()
    localization = torch.cat(localization_parts).mean()
    uncertainty_nll = torch.cat(uncertainty_parts).mean()
    exit_geometry = torch.cat(geometry_parts).mean()
    cardinality_loss = F.cross_entropy(outputs["exit_count_logits"], count - MIN_EXIT_COUNT)
    axis, global_geometry = _global_geometry_losses(outputs, targets)
    total = phase_likelihood + localization + uncertainty_nll + exit_geometry + cardinality_loss + axis + global_geometry
    return {"total": total, "phase_likelihood": phase_likelihood, "localization": localization, "uncertainty_nll": uncertainty_nll, "exit_geometry": exit_geometry, "cardinality": cardinality_loss, "axis": axis, "global_geometry": global_geometry}


def joint_cyclic_gap_simplex_contract() -> dict[str, object]:
    return {
        "student_input": "five causal range/valid images only",
        "cardinality": CONTRACT.cardinality,
        "representation": CONTRACT.representation,
        "closure": CONTRACT.closure,
        "assignment": CONTRACT.assignment,
        "geometry_binding": CONTRACT.geometry_binding,
        "confidence": CONTRACT.confidence,
        "independent_slot_localizers": CONTRACT.independent_slot_localizers,
    }
