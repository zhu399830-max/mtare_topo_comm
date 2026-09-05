"""Structured circular event field with two radial-depth slots per azimuth."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_spatial_event_set import (
    EVENT_TYPE_NAMES,
    validate_spatial_event_targets,
)


HISTORY_FRAMES = 5
ELEVATION_ROWS = 16
AZIMUTH_COLUMNS = 720
POLAR_BINS = 180
ELEVATION_BANDS = 4
DEPTH_SLOTS = 2
TOPK_EVENTS = 16
DESCRIPTOR_DIM = 64
ENCODER_DIM = 128
SENSOR_MAXIMUM_RANGE_M = 50.0
SENSOR_NEAR_RANGE_M = 0.3
LOS_SUPPORT_MARGIN_M = 0.25
LOCAL_SUPPORT_RADIUS_BINS = 1
AZIMUTH_RESIDUAL_LIMIT_RAD = math.pi / POLAR_BINS
ELEVATION_LIMIT_RAD = math.radians(15.0)
ELEVATION_DEGREES = tuple(float(value) for value in range(-15, 16, 2))


@dataclass(frozen=True)
class StructuredPolarMultiDepthConfig:
    polar_bins: int = POLAR_BINS
    depth_slots: int = DEPTH_SLOTS
    topk_events: int = TOPK_EVENTS
    encoder_dim: int = ENCODER_DIM
    descriptor_dim: int = DESCRIPTOR_DIM
    sensor_maximum_range_m: float = SENSOR_MAXIMUM_RANGE_M

    def __post_init__(self) -> None:
        if (
            self.polar_bins != POLAR_BINS
            or self.depth_slots != DEPTH_SLOTS
            or self.topk_events != TOPK_EVENTS
            or self.encoder_dim != ENCODER_DIM
            or self.descriptor_dim != DESCRIPTOR_DIM
            or self.sensor_maximum_range_m != SENSOR_MAXIMUM_RANGE_M
        ):
            raise ValueError("structured polar multi-depth config is frozen")


@dataclass(frozen=True)
class StructuredPolarLossWeights:
    presence: float = 1.0
    event_type: float = 1.0
    position: float = 5.0
    descriptor: float = 0.1
    uncertainty: float = 0.01


def _gather_flat(values: torch.Tensor, indices: torch.Tensor) -> torch.Tensor:
    flat = values.reshape(values.shape[0], POLAR_BINS * DEPTH_SLOTS, *values.shape[3:])
    if values.ndim == 3:
        return torch.gather(flat, 1, indices)
    gather = indices.reshape(*indices.shape, *([1] * (flat.ndim - 2))).expand(
        *indices.shape, *flat.shape[2:]
    )
    return torch.gather(flat, 1, gather)


class StructuredPolarMultiDepthEventEncoder(nn.Module):
    """Predict a dense 180x2 event field and expose a top-16 graph interface."""

    def __init__(self, config: StructuredPolarMultiDepthConfig | None = None) -> None:
        super().__init__()
        self.config = config or StructuredPolarMultiDepthConfig()
        input_channels = HISTORY_FRAMES * ELEVATION_BANDS * 3
        self.circular_encoder = nn.Sequential(
            nn.Conv1d(input_channels, ENCODER_DIM, 5, padding=2, padding_mode="circular"),
            nn.GroupNorm(8, ENCODER_DIM),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, 3, padding=1, padding_mode="circular"),
            nn.GroupNorm(8, ENCODER_DIM),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
        )
        # presence + two classes + range + elevation + azimuth residual + descriptor + uncertainty
        self.values_per_slot = 1 + len(EVENT_TYPE_NAMES) + 1 + 1 + 1 + DESCRIPTOR_DIM + 1
        self.dense_head = nn.Sequential(
            nn.Conv1d(ENCODER_DIM, ENCODER_DIM, 1),
            nn.SiLU(),
            nn.Conv1d(ENCODER_DIM, DEPTH_SLOTS * self.values_per_slot, 1),
        )
        azimuth = (torch.arange(POLAR_BINS, dtype=torch.float32) + 0.375) * (2.0 * torch.pi / POLAR_BINS)
        elevation = torch.tensor(ELEVATION_DEGREES, dtype=torch.float32).reshape(ELEVATION_BANDS, ELEVATION_ROWS // ELEVATION_BANDS)
        self.register_buffer("polar_azimuth_rad", azimuth, persistent=True)
        self.register_buffer("elevation_band_rad", torch.deg2rad(elevation).mean(dim=1), persistent=True)

    def _validate_inputs(self, range_m: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        expected = (HISTORY_FRAMES, ELEVATION_ROWS, AZIMUTH_COLUMNS)
        if (
            range_m.ndim != 4
            or tuple(range_m.shape[1:]) != expected
            or valid_mask.shape != range_m.shape
            or not torch.is_floating_point(range_m)
            or not bool(torch.isfinite(range_m).all())
            or bool((range_m < SENSOR_NEAR_RANGE_M - 1e-4).any())
            or bool((range_m > SENSOR_MAXIMUM_RANGE_M + 1e-4).any())
        ):
            raise ValueError("structured polar LiDAR input contract drift")
        if torch.is_floating_point(valid_mask):
            if not bool(torch.isfinite(valid_mask).all()) or bool(((valid_mask != 0) & (valid_mask != 1)).any()):
                raise ValueError("valid mask must be finite binary")
        elif valid_mask.dtype not in (torch.bool, torch.uint8):
            raise ValueError("valid mask must be bool, uint8, or binary float")
        return valid_mask.bool()

    def polar_inputs(self, range_m: torch.Tensor, valid_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        valid = self._validate_inputs(range_m, valid_mask)
        batch = range_m.shape[0]
        ranges = range_m.reshape(batch, HISTORY_FRAMES, ELEVATION_BANDS, 4, POLAR_BINS, 4)
        valids = valid.reshape_as(ranges)
        observed = torch.where(valids, torch.clamp(ranges, max=SENSOR_MAXIMUM_RANGE_M), range_m.new_full((), SENSOR_MAXIMUM_RANGE_M))
        band_range = observed.mean(dim=(3, 5)) / SENSOR_MAXIMUM_RANGE_M
        band_valid = valids.to(range_m.dtype).mean(dim=(3, 5))
        elevation = self.elevation_band_rad.to(dtype=range_m.dtype)[None, None, :, None]
        polar = torch.stack((band_range * torch.cos(elevation), band_range * torch.sin(elevation), band_valid), dim=3).reshape(batch, HISTORY_FRAMES * ELEVATION_BANDS * 3, POLAR_BINS)

        causal_range = range_m.reshape(batch, HISTORY_FRAMES, ELEVATION_ROWS, POLAR_BINS, 4)
        causal_valid = valid.reshape_as(causal_range)
        causal_support = torch.where(causal_valid, torch.clamp(causal_range, max=SENSOR_MAXIMUM_RANGE_M), range_m.new_full((), SENSOR_MAXIMUM_RANGE_M))
        profile = causal_support.amax(dim=(1, 2, 4))
        profile = F.max_pool1d(
            F.pad(profile[:, None], (1, 1), mode="circular"),
            kernel_size=3,
            stride=1,
        ).squeeze(1)
        return polar, torch.clamp(profile + LOS_SUPPORT_MARGIN_M, max=SENSOR_MAXIMUM_RANGE_M)

    def forward(self, range_m: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        polar, free_range_profile_m = self.polar_inputs(range_m, valid_mask)
        directional = self.circular_encoder(polar)
        raw = self.dense_head(directional).reshape(
            range_m.shape[0], DEPTH_SLOTS, self.values_per_slot, POLAR_BINS
        ).permute(0, 3, 1, 2)
        offset = 0
        presence_logits = raw[..., offset]
        offset += 1
        type_logits = raw[..., offset : offset + len(EVENT_TYPE_NAMES)]
        offset += len(EVENT_TYPE_NAMES)
        radial_fraction = torch.sigmoid(raw[..., offset])
        offset += 1
        elevation_rad = ELEVATION_LIMIT_RAD * torch.tanh(raw[..., offset])
        offset += 1
        azimuth_residual_rad = AZIMUTH_RESIDUAL_LIMIT_RAD * torch.tanh(raw[..., offset])
        offset += 1
        descriptor = F.normalize(raw[..., offset : offset + DESCRIPTOR_DIM], dim=-1, eps=1e-8)
        offset += DESCRIPTOR_DIM
        uncertainty_m = F.softplus(raw[..., offset]) + 1e-4
        anchor = free_range_profile_m[:, :, None]
        radial_distance_m = radial_fraction * anchor
        azimuth = self.polar_azimuth_rad.to(dtype=range_m.dtype)[None, :, None] + azimuth_residual_rad
        horizontal = radial_distance_m * torch.cos(elevation_rad)
        relative_xyz_m = torch.stack((horizontal * torch.cos(azimuth), horizontal * torch.sin(azimuth), radial_distance_m * torch.sin(elevation_rad)), dim=-1)
        confidence = torch.sigmoid(presence_logits)
        flat_confidence = confidence.reshape(range_m.shape[0], POLAR_BINS * DEPTH_SLOTS)
        top_confidence, top_index = torch.topk(flat_confidence, TOPK_EVENTS, dim=1, largest=True, sorted=True)
        top_bin = torch.div(top_index, DEPTH_SLOTS, rounding_mode="floor")
        top_slot = top_index % DEPTH_SLOTS
        return {
            "dense_event_presence_logits": presence_logits,
            "dense_event_confidence": confidence,
            "dense_event_type_logits": type_logits,
            "dense_event_type_probabilities": torch.softmax(type_logits, dim=-1),
            "dense_event_relative_xyz_m": relative_xyz_m,
            "dense_event_radial_distance_m": radial_distance_m,
            "dense_event_radial_fraction": radial_fraction,
            "dense_event_elevation_rad": elevation_rad,
            "dense_event_azimuth_residual_rad": azimuth_residual_rad,
            "dense_event_descriptor": descriptor,
            "dense_event_uncertainty_m": uncertainty_m,
            "free_range_profile_m": free_range_profile_m,
            "event_presence_logits": _gather_flat(presence_logits, top_index),
            "event_confidence": top_confidence,
            "event_type_logits": _gather_flat(type_logits, top_index),
            "event_type_probabilities": _gather_flat(torch.softmax(type_logits, dim=-1), top_index),
            "event_relative_xyz_m": _gather_flat(relative_xyz_m, top_index),
            "event_radial_distance_m": _gather_flat(radial_distance_m, top_index),
            "event_radial_fraction": _gather_flat(radial_fraction, top_index),
            "event_elevation_rad": _gather_flat(elevation_rad, top_index),
            "event_azimuth_residual_rad": _gather_flat(azimuth_residual_rad, top_index),
            "event_descriptor": _gather_flat(descriptor, top_index),
            "event_uncertainty_m": _gather_flat(uncertainty_m, top_index),
            "event_azimuth_bin_index": top_bin,
            "event_depth_slot_index": top_slot,
        }


def rasterize_structured_polar_targets(
    targets: Mapping[str, torch.Tensor],
    free_range_profile_m: torch.Tensor,
) -> dict[str, torch.Tensor]:
    validate_spatial_event_targets(targets)
    mask = targets["event_mask"].bool()
    batch = len(mask)
    if free_range_profile_m.shape != (batch, POLAR_BINS) or not bool(torch.isfinite(free_range_profile_m).all()):
        raise ValueError("structured polar target support shape drift")
    device = targets["event_relative_xyz_m"].device
    dense_mask = torch.zeros((batch, POLAR_BINS, DEPTH_SLOTS), dtype=torch.bool, device=device)
    dense_type = torch.full((batch, POLAR_BINS, DEPTH_SLOTS), -1, dtype=torch.long, device=device)
    dense_identity = torch.full_like(dense_type, -1)
    dense_xyz = torch.zeros((batch, POLAR_BINS, DEPTH_SLOTS, 3), dtype=targets["event_relative_xyz_m"].dtype, device=device)
    dense_radial_fraction = torch.zeros((batch, POLAR_BINS, DEPTH_SLOTS), dtype=dense_xyz.dtype, device=device)
    dense_azimuth_residual = torch.zeros_like(dense_radial_fraction)
    dense_elevation = torch.zeros_like(dense_radial_fraction)
    for row in range(batch):
        active = torch.nonzero(mask[row], as_tuple=False).flatten()
        if not len(active):
            continue
        xyz = targets["event_relative_xyz_m"][row, active]
        radial = torch.linalg.vector_norm(xyz, dim=-1)
        bearing = torch.remainder(torch.atan2(xyz[:, 1], xyz[:, 0]), 2.0 * torch.pi)
        bearing_deg = torch.rad2deg(bearing)
        bins = torch.remainder(torch.floor((bearing_deg + 0.25) / 2.0).long(), POLAR_BINS)
        ordered = sorted(range(len(active)), key=lambda index: (int(bins[index]), float(radial[index]), int(targets["event_identity_index"][row, active[index]])))
        occupancy = [0] * POLAR_BINS
        for local in ordered:
            bin_index = int(bins[local])
            slot = occupancy[bin_index]
            if slot >= DEPTH_SLOTS:
                raise ValueError("structured polar Teacher exceeds two depth slots")
            occupancy[bin_index] += 1
            target_index = int(active[local])
            anchor = free_range_profile_m[row, bin_index]
            if bool(radial[local] > anchor + 1e-3):
                raise ValueError("structured polar target exceeds free-range support")
            center = (bin_index + 0.375) * (2.0 * torch.pi / POLAR_BINS)
            residual = torch.remainder(bearing[local] - center + torch.pi, 2.0 * torch.pi) - torch.pi
            if bool(torch.abs(residual) > AZIMUTH_RESIDUAL_LIMIT_RAD + 1e-6):
                raise ValueError("structured polar target azimuth residual overflow")
            horizontal = torch.linalg.vector_norm(xyz[local, :2])
            dense_mask[row, bin_index, slot] = True
            dense_type[row, bin_index, slot] = targets["event_type_index"][row, target_index]
            dense_identity[row, bin_index, slot] = targets["event_identity_index"][row, target_index]
            dense_xyz[row, bin_index, slot] = xyz[local]
            dense_radial_fraction[row, bin_index, slot] = radial[local] / anchor.clamp_min(1e-8)
            dense_azimuth_residual[row, bin_index, slot] = residual
            dense_elevation[row, bin_index, slot] = torch.atan2(xyz[local, 2], horizontal)
    return {
        "event_mask": dense_mask,
        "event_type_index": dense_type,
        "event_identity_index": dense_identity,
        "event_relative_xyz_m": dense_xyz,
        "event_radial_fraction": dense_radial_fraction,
        "event_azimuth_residual_rad": dense_azimuth_residual,
        "event_elevation_rad": dense_elevation,
    }


def structured_polar_multidepth_loss(
    outputs: Mapping[str, torch.Tensor],
    targets: Mapping[str, torch.Tensor],
    *,
    presence_positive_weight: float = 1.0,
    event_type_class_weights: torch.Tensor | None = None,
    weights: StructuredPolarLossWeights | None = None,
) -> dict[str, torch.Tensor]:
    loss_weights = weights or StructuredPolarLossWeights()
    logits = outputs["dense_event_presence_logits"]
    mask = targets["event_mask"].bool()
    if logits.shape != mask.shape or logits.shape[1:] != (POLAR_BINS, DEPTH_SLOTS):
        raise ValueError("structured polar dense loss shape drift")
    presence = F.binary_cross_entropy_with_logits(logits, mask.to(logits.dtype), pos_weight=logits.new_tensor(float(presence_positive_weight)))
    zero = logits.sum() * 0.0
    if bool(mask.any()):
        event_type = F.cross_entropy(outputs["dense_event_type_logits"][mask], targets["event_type_index"][mask], weight=event_type_class_weights)
        error = torch.linalg.vector_norm(outputs["dense_event_relative_xyz_m"][mask] - targets["event_relative_xyz_m"][mask], dim=-1)
        position = F.smooth_l1_loss(error / SENSOR_MAXIMUM_RANGE_M, torch.zeros_like(error))
        uncertainty = outputs["dense_event_uncertainty_m"][mask]
        uncertainty_loss = torch.mean(error.detach() / uncertainty + torch.log(uncertainty))
        descriptors = outputs["dense_event_descriptor"][mask]
        identities = targets["event_identity_index"][mask]
        descriptor_terms = []
        for identity in torch.unique(identities, sorted=True):
            selected = descriptors[identities == identity]
            if len(selected) >= 2:
                descriptor_terms.append(1.0 - torch.mean(selected @ selected.T))
        descriptor = torch.stack(descriptor_terms).mean() if descriptor_terms else zero
    else:
        event_type = position = uncertainty_loss = descriptor = zero
    total = loss_weights.presence * presence + loss_weights.event_type * event_type + loss_weights.position * position + loss_weights.descriptor * descriptor + loss_weights.uncertainty * uncertainty_loss
    return {"total": total, "presence": presence, "event_type": event_type, "position": position, "descriptor": descriptor, "uncertainty": uncertainty_loss}


def structured_polar_multidepth_input_contract() -> Mapping[str, object]:
    return {
        "student_inputs": ("five_frame_range_m", "five_frame_valid_mask"),
        "forbidden_inputs": ("world_pose", "sensor_pose", "route_identity", "traversal_identity", "TNG_identity", "future_frame", "teacher_feature"),
        "range_shape": (HISTORY_FRAMES, ELEVATION_ROWS, AZIMUTH_COLUMNS),
        "dense_layout": (POLAR_BINS, DEPTH_SLOTS),
        "depth_slot_rule": "ascending_teacher_radial_distance_within_each_azimuth_bin",
        "proposal_rule": "confidence_top16_over_360_without_neighbor_suppression",
        "azimuth_residual_limit_deg": 1.0,
        "range_rule": "sigmoid_fraction_of_five_frame_local_free_range_envelope",
    }


__all__ = [
    "DEPTH_SLOTS",
    "POLAR_BINS",
    "StructuredPolarLossWeights",
    "StructuredPolarMultiDepthConfig",
    "StructuredPolarMultiDepthEventEncoder",
    "rasterize_structured_polar_targets",
    "structured_polar_multidepth_input_contract",
    "structured_polar_multidepth_loss",
]
