"""Causal circular LiDAR model for executable exit peaks and tunnel geometry."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
from typing import Mapping

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.phase3_structural_semantics import (
    CircularAzimuthConv2d,
    ResidualRangeBlock,
)


HISTORY_FRAMES = 5
ELEVATION_ROWS = 16
AZIMUTH_COLUMNS = 720
BEARING_BINS = 180
ENCODER_DIM = 128
PLACE_DESCRIPTOR_DIM = 128
EXIT_DESCRIPTOR_DIM = 32
PROFILE_DIM = 4
HEADING_RESIDUAL_LIMIT_DEG = 1.0
PROFILE_SCALE_M = 5.0
METRIC_DISTANCE_SCALE_M = 30.0
SLOPE_SCALE_DEG = 45.0
CURVATURE_SCALE_PER_M = 0.1


@dataclass(frozen=True)
class CircularPeakGeometryConfig:
    history_frames: int = HISTORY_FRAMES
    bearing_bins: int = BEARING_BINS
    encoder_dim: int = ENCODER_DIM
    place_descriptor_dim: int = PLACE_DESCRIPTOR_DIM
    exit_descriptor_dim: int = EXIT_DESCRIPTOR_DIM

    def __post_init__(self) -> None:
        if self != CircularPeakGeometryConfig.__new_defaults__():
            raise ValueError("circular peak-geometry model config is frozen")

    @staticmethod
    def __new_defaults__() -> "CircularPeakGeometryConfig":
        instance = object.__new__(CircularPeakGeometryConfig)
        object.__setattr__(instance, "history_frames", HISTORY_FRAMES)
        object.__setattr__(instance, "bearing_bins", BEARING_BINS)
        object.__setattr__(instance, "encoder_dim", ENCODER_DIM)
        object.__setattr__(instance, "place_descriptor_dim", PLACE_DESCRIPTOR_DIM)
        object.__setattr__(instance, "exit_descriptor_dim", EXIT_DESCRIPTOR_DIM)
        return instance


class CircularPeakGeometrySemanticNet(nn.Module):
    """Map five past/current range images to a typed geometry-semantic field.

    ``scans`` is ``[B,5,2,16,720]`` with normalized finite range followed by a
    binary valid-return channel.  Pose, world, traversal, graph and Teacher
    identity never enter ``forward``.
    """

    def __init__(self, config: CircularPeakGeometryConfig | None = None) -> None:
        super().__init__()
        self.config = config or CircularPeakGeometryConfig()
        dim = self.config.encoder_dim
        self.encoder = nn.Sequential(
            CircularAzimuthConv2d(2, 32, stride=(1, 2)),
            nn.GroupNorm(8, 32),
            nn.SiLU(),
            ResidualRangeBlock(32, 64, (2, 2)),
            ResidualRangeBlock(64, 96, (2, 1)),
            ResidualRangeBlock(96, dim, (2, 1)),
        )
        self.temporal = nn.GRU(dim, dim, batch_first=True)
        self.directional_temporal = nn.Sequential(
            nn.Conv1d(HISTORY_FRAMES * dim, dim, 1),
            nn.SiLU(),
            nn.Conv1d(dim, dim, 1),
        )
        self.axis_azimuth_head = nn.Sequential(
            nn.Conv1d(dim, 64, 1), nn.SiLU(), nn.Conv1d(64, 1, 1)
        )
        self.axis_vertical_head = nn.Linear(dim, 1)
        self.geometry_head = nn.Linear(dim, 4)
        self.place_head = nn.Sequential(
            nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, PLACE_DESCRIPTOR_DIM)
        )
        self.observation_uncertainty_head = nn.Linear(dim, 1)

        # presence, heading residual, width, profile(4), descriptor(32),
        # geometry uncertainty scales for heading/width/profile(6)
        values_per_bin = 1 + 1 + 1 + PROFILE_DIM + EXIT_DESCRIPTOR_DIM + 6
        self.peak_head = nn.Sequential(
            nn.Conv1d(dim, dim, 3, padding=1, padding_mode="circular"),
            nn.SiLU(),
            nn.Conv1d(dim, values_per_bin, 1),
        )
        azimuth = torch.arange(BEARING_BINS, dtype=torch.float32) * (
            2.0 * torch.pi / BEARING_BINS
        )
        self.register_buffer("bearing_azimuth_rad", azimuth, persistent=True)

    @staticmethod
    def _validate_scans(scans: torch.Tensor) -> None:
        expected = (HISTORY_FRAMES, 2, ELEVATION_ROWS, AZIMUTH_COLUMNS)
        if scans.ndim != 5 or tuple(scans.shape[1:]) != expected:
            raise ValueError(f"expected [B,5,2,16,720], got {tuple(scans.shape)}")
        if not torch.is_floating_point(scans) or not bool(torch.isfinite(scans).all()):
            raise ValueError("circular peak input must be finite floating point")
        ranges = scans[:, :, 0]
        valid = scans[:, :, 1]
        if bool((ranges < 0.0).any()) or bool((ranges > 1.0).any()):
            raise ValueError("normalized range channel must be in [0,1]")
        if bool(((valid != 0.0) & (valid != 1.0)).any()):
            raise ValueError("valid-return channel must be binary")

    def encode_causal_features(self, scans: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        self._validate_scans(scans)
        batch = scans.shape[0]
        encoded = self.encoder(scans.reshape(batch * HISTORY_FRAMES, 2, 16, 720))
        if encoded.shape[-1] != BEARING_BINS:
            raise RuntimeError("circular encoder bearing resolution drift")
        pooled = encoded.mean(dim=(2, 3)).reshape(batch, HISTORY_FRAMES, ENCODER_DIM)
        temporal, _ = self.temporal(pooled)
        context = temporal[:, -1]
        sequence = encoded.reshape(
            batch, HISTORY_FRAMES, ENCODER_DIM, encoded.shape[-2], BEARING_BINS
        ).mean(dim=3)
        directional = self.directional_temporal(
            sequence.reshape(batch, HISTORY_FRAMES * ENCODER_DIM, BEARING_BINS)
        )
        return context, directional

    def forward(self, scans: torch.Tensor) -> dict[str, torch.Tensor]:
        context, directional = self.encode_causal_features(scans)
        raw = self.peak_head(directional).transpose(1, 2)
        offset = 0
        presence_logits = raw[..., offset]
        offset += 1
        heading_residual_deg = HEADING_RESIDUAL_LIMIT_DEG * torch.tanh(raw[..., offset])
        offset += 1
        opening_width_m = F.softplus(raw[..., offset]) + 1e-4
        offset += 1
        vertical_profile_m = PROFILE_SCALE_M * torch.tanh(raw[..., offset : offset + PROFILE_DIM])
        offset += PROFILE_DIM
        exit_descriptor = F.normalize(
            raw[..., offset : offset + EXIT_DESCRIPTOR_DIM], dim=-1, eps=1e-8
        )
        offset += EXIT_DESCRIPTOR_DIM
        geometry_uncertainty = F.softplus(raw[..., offset : offset + 6]) + 0.05

        azimuth = self.bearing_azimuth_rad.to(dtype=directional.dtype)
        axis_probability = torch.softmax(
            self.axis_azimuth_head(directional).squeeze(1), dim=-1
        )
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
        return {
            "peak_presence_logits": presence_logits,
            "peak_confidence": torch.sigmoid(presence_logits),
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
            "observation_uncertainty": torch.sigmoid(
                self.observation_uncertainty_head(context).squeeze(-1)
            ),
        }


def balanced_peak_presence_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Equal-mass positive/negative BCE without a tunable class weight."""
    if logits.shape != target.shape or logits.ndim != 2 or logits.shape[1] != BEARING_BINS:
        raise ValueError("peak presence loss shape drift")
    positive = target.bool()
    negative = ~positive
    if not bool(positive.any()) or not bool(negative.any()):
        raise ValueError("balanced peak loss requires positive and negative bins")
    return 0.5 * (F.softplus(-logits[positive]).mean() + F.softplus(logits[negative]).mean())


def circular_peak_geometry_loss(
    outputs: Mapping[str, torch.Tensor], targets: Mapping[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    """Masked, unit-normalized geometry loss with learned per-peak uncertainty."""
    logits = outputs["peak_presence_logits"]
    presence_target = targets["presence"].bool()
    presence = balanced_peak_presence_loss(logits, presence_target)
    width_valid = targets["width_valid_mask"].bool()
    if width_valid.shape != presence_target.shape or bool((width_valid & ~presence_target).any()):
        raise ValueError("width-valid mask must be a subset of peak presence")
    residual_target = targets["heading_residual_deg"].to(logits.dtype)
    width_target = targets["opening_width_m"].to(logits.dtype)
    profile_target = targets["vertical_profile_m"].to(logits.dtype)
    if residual_target.shape != presence_target.shape or width_target.shape != presence_target.shape:
        raise ValueError("peak scalar target shape drift")
    if profile_target.shape != (*presence_target.shape, PROFILE_DIM):
        raise ValueError("peak profile target shape drift")

    errors = torch.cat(
        (
            ((outputs["peak_heading_residual_deg"] - residual_target) / HEADING_RESIDUAL_LIMIT_DEG)[..., None],
            ((torch.log1p(outputs["peak_opening_width_m"]) - torch.log1p(width_target.clamp_min(0.0))) / torch.log1p(logits.new_tensor(60.0)))[..., None],
            (outputs["peak_vertical_profile_m"] - profile_target) / PROFILE_SCALE_M,
        ),
        dim=-1,
    )
    valid = torch.cat(
        (
            presence_target[..., None],
            width_valid[..., None],
            presence_target[..., None].expand(-1, -1, PROFILE_DIM),
        ),
        dim=-1,
    )
    uncertainty = outputs["peak_geometry_uncertainty"]
    if uncertainty.shape != errors.shape or not bool(valid.any(dim=(0, 1)).all()):
        raise ValueError("peak uncertainty/mask contract drift")
    element = F.smooth_l1_loss(errors / uncertainty, torch.zeros_like(errors), reduction="none") + torch.log(uncertainty)
    peak_geometry = torch.stack(
        [element[..., index][valid[..., index]].mean() for index in range(errors.shape[-1])]
    ).mean()

    target_axis = F.normalize(targets["local_axis"].to(logits.dtype), dim=-1)
    if target_axis.shape != outputs["local_axis"].shape:
        raise ValueError("local-axis target shape drift")
    axis = (1.0 - (outputs["local_axis"] * target_axis).sum(dim=-1)).mean()
    target_geometry = targets["geometry"].to(logits.dtype)
    geometry_valid = targets["geometry_valid_mask"].bool()
    if target_geometry.shape != geometry_valid.shape or target_geometry.shape != (len(logits), 4):
        raise ValueError("global geometry target shape drift")
    prediction = torch.stack(
        (
            outputs["width_m"] / METRIC_DISTANCE_SCALE_M,
            outputs["height_m"] / METRIC_DISTANCE_SCALE_M,
            outputs["slope_deg"] / SLOPE_SCALE_DEG,
            outputs["curvature_per_m"] / CURVATURE_SCALE_PER_M,
        ),
        dim=-1,
    )
    normalized_target = torch.stack(
        (
            target_geometry[:, 0] / METRIC_DISTANCE_SCALE_M,
            target_geometry[:, 1] / METRIC_DISTANCE_SCALE_M,
            target_geometry[:, 2] / SLOPE_SCALE_DEG,
            target_geometry[:, 3] / CURVATURE_SCALE_PER_M,
        ),
        dim=-1,
    )
    if not bool(geometry_valid.any(dim=0).all()):
        raise ValueError("real batch must cover every global geometry field")
    global_element = F.smooth_l1_loss(prediction, normalized_target, reduction="none")
    global_geometry = torch.stack(
        [global_element[:, index][geometry_valid[:, index]].mean() for index in range(4)]
    ).mean()
    total = presence + peak_geometry + axis + global_geometry
    return {
        "total": total,
        "presence": presence,
        "peak_geometry": peak_geometry,
        "axis": axis,
        "global_geometry": global_geometry,
    }


def circular_peak_geometry_input_contract() -> dict[str, object]:
    return {
        "forward_parameters": tuple(inspect.signature(CircularPeakGeometrySemanticNet.forward).parameters)[1:],
        "student_shape": (HISTORY_FRAMES, 2, ELEVATION_ROWS, AZIMUTH_COLUMNS),
        "peak_layout": (BEARING_BINS,),
        "causal_frames": "current plus four past frames only",
        "node_rule": "stable local maxima create executable exit tokens; no free query",
        "edge_rule": "physical traversal only",
        "forbidden_inputs": (
            "pose", "world_id", "traversal_id", "TNG_identity", "exit_identity",
            "future_frame", "graph_state", "C09", "C10", "M-TARE",
        ),
    }


__all__ = [
    "BEARING_BINS",
    "CircularPeakGeometryConfig",
    "CircularPeakGeometrySemanticNet",
    "balanced_peak_presence_loss",
    "circular_peak_geometry_input_contract",
    "circular_peak_geometry_loss",
]
