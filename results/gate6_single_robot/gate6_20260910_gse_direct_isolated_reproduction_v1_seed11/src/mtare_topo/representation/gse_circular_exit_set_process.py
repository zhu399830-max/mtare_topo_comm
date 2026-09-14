"""Cardinality-conditioned causal circular exit-set process for GSE-Graph."""

from __future__ import annotations

from dataclasses import dataclass
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
    HEADING_RESIDUAL_LIMIT_DEG,
    METRIC_DISTANCE_SCALE_M,
    PLACE_DESCRIPTOR_DIM,
    PROFILE_DIM,
    PROFILE_SCALE_M,
    SLOPE_SCALE_DEG,
    CircularPeakGeometryConfig,
    CircularPeakGeometrySemanticNet,
    circular_peak_geometry_loss,
)


MIN_EXIT_COUNT = 1
MAX_EXIT_COUNT = 4
SUPPRESSION_RADIUS_BINS = 1
BIN_WIDTH_DEG = 360.0 / BEARING_BINS


@dataclass(frozen=True)
class DecodedCircularExitSet:
    """Padded typed exit set; ``valid_mask`` selects exactly ``count`` rows."""

    count: torch.Tensor
    bearing_deg: torch.Tensor
    bin_index: torch.Tensor
    valid_mask: torch.Tensor
    set_mass: torch.Tensor
    opening_width_m: torch.Tensor
    vertical_profile_m: torch.Tensor
    descriptor: torch.Tensor
    geometry_uncertainty: torch.Tensor
    count_confidence: torch.Tensor


class CausalCircularExitSetProcessNet(CircularPeakGeometrySemanticNet):
    """Reuse the proven causal circular backbone without independent objectness.

    The 180 directional logits form one normalized finite-set intensity.  A
    separate invariant head predicts set cardinality 1--4.  There are no free
    queries and no per-peak existence threshold.
    """

    def __init__(self, config: CircularPeakGeometryConfig | None = None) -> None:
        super().__init__(config=config)
        self.exit_count_head = nn.Linear(ENCODER_DIM, MAX_EXIT_COUNT)

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        context, directional = self.encode_causal_features(scans)
        raw = self.peak_head(directional).transpose(1, 2)
        offset = 0
        intensity_logits = raw[..., offset]
        offset += 1
        heading_residual_deg = HEADING_RESIDUAL_LIMIT_DEG * torch.tanh(raw[..., offset])
        offset += 1
        opening_width_m = F.softplus(raw[..., offset]) + 1e-4
        offset += 1
        vertical_profile_m = PROFILE_SCALE_M * torch.tanh(raw[..., offset : offset + PROFILE_DIM])
        offset += PROFILE_DIM
        exit_descriptor = F.normalize(raw[..., offset : offset + EXIT_DESCRIPTOR_DIM], dim=-1, eps=1e-8)
        offset += EXIT_DESCRIPTOR_DIM
        geometry_uncertainty = F.softplus(raw[..., offset : offset + 6]) + 0.05

        azimuth = self.bearing_azimuth_rad.to(dtype=directional.dtype)
        axis_probability = torch.softmax(self.axis_azimuth_head(directional).squeeze(1), dim=-1)
        horizontal_axis = F.normalize(
            torch.stack(
                (
                    (axis_probability * torch.cos(azimuth)).sum(dim=-1),
                    (axis_probability * torch.sin(azimuth)).sum(dim=-1),
                ),
                dim=-1,
            ),
            dim=-1,
            eps=1e-8,
        )
        local_axis = F.normalize(
            torch.cat((horizontal_axis, torch.tanh(self.axis_vertical_head(context))), dim=-1),
            dim=-1,
            eps=1e-8,
        )
        geometry = self.geometry_head(context)
        count_logits = self.exit_count_head(context)
        return {
            "exit_intensity_logits": intensity_logits,
            "exit_log_mass": torch.log_softmax(intensity_logits, dim=-1),
            "exit_mass": torch.softmax(intensity_logits, dim=-1),
            "exit_count_logits": count_logits,
            "exit_count_probability": torch.softmax(count_logits, dim=-1),
            # Preserve the proven geometry-loss interface without preserving
            # the rejected independent-BCE semantics.
            "peak_presence_logits": intensity_logits,
            "peak_heading_residual_deg": heading_residual_deg,
            "peak_opening_width_m": opening_width_m,
            "peak_vertical_profile_m": vertical_profile_m,
            "peak_descriptor": exit_descriptor,
            "peak_geometry_uncertainty": geometry_uncertainty,
            "local_axis": local_axis,
            "width_m": F.softplus(geometry[:, 0]) + 1e-4,
            "height_m": F.softplus(geometry[:, 1]) + 1e-4,
            "slope_deg": SLOPE_SCALE_DEG * torch.tanh(geometry[:, 2]),
            "curvature_per_m": F.softplus(geometry[:, 3]),
            "place_descriptor": F.normalize(self.place_head(context), dim=-1, eps=1e-8),
            "observation_uncertainty": torch.sigmoid(self.observation_uncertainty_head(context).squeeze(-1)),
        }


def circular_exit_set_process_loss(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Permutation-invariant finite-set likelihood plus unchanged geometry.

    All observed exits in a row share one normalized probability budget.  A
    ghost direction therefore necessarily removes likelihood mass from true
    exits, unlike independent binary peak classification.
    """

    presence = targets["presence"].bool()
    log_mass = outputs["exit_log_mass"]
    if log_mass.shape != presence.shape or log_mass.ndim != 2 or log_mass.shape[1] != BEARING_BINS:
        raise ValueError("circular exit-set target shape drift")
    count = presence.sum(dim=1)
    if not bool(((count >= MIN_EXIT_COUNT) & (count <= MAX_EXIT_COUNT)).all()):
        raise ValueError("circular exit-set cardinality must be in [1,4]")
    set_nll = -(log_mass * presence).sum(dim=1).div(count).mean()
    cardinality = F.cross_entropy(outputs["exit_count_logits"], count - MIN_EXIT_COUNT)
    v1 = circular_peak_geometry_loss(outputs, targets)
    total = v1["total"] - v1["presence"] + set_nll + cardinality
    return {
        "total": total,
        "set_nll": set_nll,
        "cardinality": cardinality,
        "peak_geometry": v1["peak_geometry"],
        "axis": v1["axis"],
        "global_geometry": v1["global_geometry"],
    }


def decode_circular_exit_set(outputs: Mapping[str, torch.Tensor]) -> DecodedCircularExitSet:
    """Deterministically decode count-conditioned continuous bearings.

    Radius one is fixed by the Teacher proof that no physical exits occupy the
    same or adjacent 2-degree bins; it is not a selected hyperparameter.
    """

    mass = outputs["exit_mass"]
    count_probability = outputs["exit_count_probability"]
    if mass.ndim != 2 or mass.shape[1] != BEARING_BINS or count_probability.shape != (len(mass), MAX_EXIT_COUNT):
        raise ValueError("circular exit-set decode shape drift")
    count = torch.argmax(count_probability, dim=-1) + MIN_EXIT_COUNT
    batch = len(mass)
    device = mass.device
    bins = torch.full((batch, MAX_EXIT_COUNT), -1, dtype=torch.long, device=device)
    valid = torch.zeros((batch, MAX_EXIT_COUNT), dtype=torch.bool, device=device)
    for row in range(batch):
        available = torch.ones(BEARING_BINS, dtype=torch.bool, device=device)
        for slot in range(int(count[row])):
            selected = int(torch.argmax(mass[row].masked_fill(~available, -1.0)))
            bins[row, slot] = selected
            valid[row, slot] = True
            for delta in range(-SUPPRESSION_RADIUS_BINS, SUPPRESSION_RADIUS_BINS + 1):
                available[(selected + delta) % BEARING_BINS] = False
    safe_bins = bins.clamp_min(0)
    row_index = torch.arange(batch, device=device)[:, None].expand(-1, MAX_EXIT_COUNT)
    bearing = torch.remainder(
        safe_bins.to(mass.dtype) * BIN_WIDTH_DEG
        + outputs["peak_heading_residual_deg"][row_index, safe_bins],
        360.0,
    )
    bearing = torch.where(valid, bearing, torch.zeros_like(bearing))
    return DecodedCircularExitSet(
        count=count,
        bearing_deg=bearing,
        bin_index=bins,
        valid_mask=valid,
        set_mass=torch.where(valid, mass[row_index, safe_bins], torch.zeros_like(bearing)),
        opening_width_m=torch.where(valid, outputs["peak_opening_width_m"][row_index, safe_bins], torch.zeros_like(bearing)),
        vertical_profile_m=torch.where(valid[..., None], outputs["peak_vertical_profile_m"][row_index, safe_bins], torch.zeros((*valid.shape, PROFILE_DIM), dtype=mass.dtype, device=device)),
        descriptor=torch.where(valid[..., None], outputs["peak_descriptor"][row_index, safe_bins], torch.zeros((*valid.shape, EXIT_DESCRIPTOR_DIM), dtype=mass.dtype, device=device)),
        geometry_uncertainty=torch.where(valid[..., None], outputs["peak_geometry_uncertainty"][row_index, safe_bins], torch.zeros((*valid.shape, 6), dtype=mass.dtype, device=device)),
        count_confidence=count_probability.max(dim=-1).values,
    )


def circular_exit_set_process_contract() -> dict[str, object]:
    return {
        "student_input": "five causal range/valid images only",
        "cardinality": (MIN_EXIT_COUNT, MAX_EXIT_COUNT),
        "bearing_representation": "normalized 180-bin circular finite-set intensity plus bounded continuous residual",
        "set_supervision": "permutation-invariant mean log likelihood over all true exits",
        "decode": "predicted cardinality plus deterministic radius-one diverse mass modes; no existence threshold",
        "node_rule": "past-only stable decoded exit-set change",
        "edge_rule": "physical traversal only",
        "forbidden_inputs": ("pose", "world_id", "traversal_id", "TNG_identity", "exit_identity", "future_frame", "graph_state", "C09", "C10", "M-TARE"),
    }


__all__ = [
    "BIN_WIDTH_DEG",
    "CausalCircularExitSetProcessNet",
    "DecodedCircularExitSet",
    "MAX_EXIT_COUNT",
    "MIN_EXIT_COUNT",
    "SUPPRESSION_RADIUS_BINS",
    "circular_exit_set_process_contract",
    "circular_exit_set_process_loss",
    "decode_circular_exit_set",
]
