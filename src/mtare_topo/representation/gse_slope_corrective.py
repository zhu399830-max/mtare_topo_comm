"""Physics-guided causal slope corrective for the frozen GSE representation.

The module deliberately leaves every existing GSE output unchanged.  It uses
five deterministic range-geometry observations, treats their mean slope as a
causal physics prior, and learns only a bounded residual plus an error scale.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


HISTORY_FRAMES = 5
RAW_FEATURES_PER_FRAME = 6
MAXIMUM_RESIDUAL_DEG = 10.0


class RangeGeometryPredictor(Protocol):
    def predict(self, range_m: np.ndarray, valid_mask: np.ndarray) -> dict[str, Any]: ...


def slope_frame_features(result: dict[str, Any]) -> np.ndarray:
    """Convert one deterministic geometry result to the frozen six features."""

    row = np.asarray(
        (
            float(result["slope_deg"]),
            float(result["width_m"]),
            float(result["height_m"]),
            float(result["curvature_per_m"]),
            math.log1p(int(result["local_return_count"])),
            float(result["populated_sections"]),
        ),
        dtype=np.float32,
    )
    if row.shape != (RAW_FEATURES_PER_FRAME,) or not np.all(np.isfinite(row)):
        raise RuntimeError("range geometry predictor returned non-finite corrective features")
    return row


def slope_sequence_features(
    range_m: np.ndarray,
    valid_mask: np.ndarray,
    *,
    predictor: RangeGeometryPredictor,
) -> tuple[np.ndarray, float]:
    """Return five raw geometry feature rows and the mean-slope prior.

    No pose, identity, topology or future frame is accepted.  The six features
    are slope, width, height, curvature, log return count and populated section
    count.  Normalization is intentionally deferred to fit-world statistics.
    """

    ranges = np.asarray(range_m, dtype=np.float64)
    valid = np.asarray(valid_mask)
    if ranges.shape != (HISTORY_FRAMES, 16, 720) or valid.shape != ranges.shape:
        raise ValueError("slope corrective expects five [16,720] scans and masks")
    if not np.all(np.isfinite(ranges)):
        raise ValueError("slope corrective ranges must be finite")
    if valid.dtype not in (np.dtype(np.bool_), np.dtype(np.uint8)):
        raise ValueError("slope corrective masks must be bool or uint8")
    if not np.all((valid == 0) | (valid == 1)):
        raise ValueError("slope corrective masks must be binary")
    rows: list[list[float]] = []
    slopes: list[float] = []
    for frame_range, frame_valid in zip(ranges, valid, strict=True):
        result = predictor.predict(frame_range, frame_valid)
        row = slope_frame_features(result)
        slope = float(row[0])
        rows.append(row.tolist())
        slopes.append(slope)
    features = np.asarray(rows, dtype=np.float32)
    prior = float(np.mean(slopes))
    if features.shape != (HISTORY_FRAMES, RAW_FEATURES_PER_FRAME) or not math.isfinite(prior):
        raise RuntimeError("slope corrective feature construction drift")
    return features, prior


@dataclass(frozen=True)
class SlopeCorrectiveConfig:
    feature_dim: int = RAW_FEATURES_PER_FRAME
    hidden_dim: int = 32
    maximum_residual_deg: float = MAXIMUM_RESIDUAL_DEG
    minimum_error_scale_deg: float = 0.05

    def __post_init__(self) -> None:
        if self.feature_dim != RAW_FEATURES_PER_FRAME or self.hidden_dim < 1:
            raise ValueError("invalid slope corrective dimensions")
        if (
            not math.isfinite(self.maximum_residual_deg)
            or not math.isfinite(self.minimum_error_scale_deg)
            or self.maximum_residual_deg <= 0.0
            or self.minimum_error_scale_deg <= 0.0
        ):
            raise ValueError("slope corrective scales must be positive")


class PhysicsGuidedSlopeResidualNet(nn.Module):
    """Small temporal residual model on deterministic geometry features."""

    def __init__(self, config: SlopeCorrectiveConfig | None = None) -> None:
        super().__init__()
        self.config = config or SlopeCorrectiveConfig()
        self.temporal = nn.GRU(self.config.feature_dim, self.config.hidden_dim, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(self.config.hidden_dim + 1, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, 2),
        )
        # The initial model is exactly the physics prior.  Only the final layer
        # starts at zero; the temporal encoder remains trainable from step one.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        normalized_features: torch.Tensor,
        prior_slope_deg: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if normalized_features.ndim != 3 or normalized_features.shape[1:] != (
            HISTORY_FRAMES,
            self.config.feature_dim,
        ):
            raise ValueError("normalized slope features must be [B,5,6]")
        if prior_slope_deg.ndim != 1 or len(prior_slope_deg) != len(normalized_features):
            raise ValueError("prior slope must be [B]")
        if not bool(torch.isfinite(normalized_features).all()) or not bool(
            torch.isfinite(prior_slope_deg).all()
        ):
            raise ValueError("slope corrective inputs must be finite")
        _, state = self.temporal(normalized_features)
        context = torch.cat((state[-1], prior_slope_deg[:, None] / 20.0), dim=1)
        raw = self.head(context)
        correction = self.config.maximum_residual_deg * torch.tanh(raw[:, 0])
        error_scale = F.softplus(raw[:, 1]) + self.config.minimum_error_scale_deg
        return {
            "slope_deg": prior_slope_deg + correction,
            "prior_slope_deg": prior_slope_deg,
            "correction_deg": correction,
            "error_scale_deg": error_scale,
        }


def slope_corrective_loss(
    outputs: dict[str, torch.Tensor],
    target_slope_deg: torch.Tensor,
    *,
    maximum_residual_deg: float = MAXIMUM_RESIDUAL_DEG,
) -> dict[str, torch.Tensor]:
    """Fit slope first; train uncertainty without letting it hide slope error."""

    target = target_slope_deg.float()
    predicted = outputs["slope_deg"].float()
    correction = outputs["correction_deg"].float()
    scale = outputs["error_scale_deg"].float()
    if target.ndim != 1 or predicted.shape != target.shape or scale.shape != target.shape:
        raise ValueError("slope corrective outputs and target must be [B]")
    if not math.isfinite(maximum_residual_deg) or maximum_residual_deg <= 0.0:
        raise ValueError("maximum residual must be finite and positive")
    if not all(
        bool(torch.isfinite(value).all())
        for value in (target, predicted, correction, scale)
    ):
        raise ValueError("slope corrective loss inputs must be finite")
    primary = F.smooth_l1_loss(predicted, target, beta=1.0)
    absolute_error = torch.abs(predicted.detach() - target)
    uncertainty = F.smooth_l1_loss(scale, absolute_error, beta=0.5)
    residual_regularization = torch.mean((correction / maximum_residual_deg) ** 2)
    total = primary + 0.1 * uncertainty + 0.01 * residual_regularization
    return {
        "total": total,
        "slope": primary,
        "uncertainty": uncertainty,
        "residual_regularization": residual_regularization,
    }


__all__ = [
    "HISTORY_FRAMES",
    "MAXIMUM_RESIDUAL_DEG",
    "PhysicsGuidedSlopeResidualNet",
    "RAW_FEATURES_PER_FRAME",
    "SlopeCorrectiveConfig",
    "slope_corrective_loss",
    "slope_frame_features",
    "slope_sequence_features",
]
