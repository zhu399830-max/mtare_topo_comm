"""Linear-output composition anchors for observable primitive endpoints.

The failed independent pair head emitted one learned value for every endpoint
pair.  This module instead emits one 3-D anchor residual and one uncertainty
per endpoint.  Pair compatibility is derived from the two predicted Gaussian
anchors, so the learned output remains O(E) while the resulting relation is
symmetric by construction.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.primitive_relation_model import (
    ENDPOINT_DESCRIPTOR_DIM,
    MAXIMUM_RANGE_M,
    MAXIMUM_SHAPE_EXPONENT,
    MAXIMUM_SLOTS,
    MINIMUM_SHAPE_EXPONENT,
    MODEL_DIM,
)
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationPrediction,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    outward_endpoint_tangents,
)


ENDPOINTS = 2 * MAXIMUM_SLOTS
COMPOSITION_ANCHOR_FEATURE_DIM = ENDPOINT_DESCRIPTOR_DIM + 8
COMPOSITION_ANCHOR_OUTPUTS_PER_ENDPOINT = 4
MINIMUM_ANCHOR_SCALE_M = 1e-3


@dataclass(frozen=True)
class CompositionAnchorPrediction:
    anchor_current_sensor_m: torch.Tensor
    residual_local_m: torch.Tensor
    scale_m: torch.Tensor
    compatibility_logits: torch.Tensor

    def validate(self) -> None:
        batch = self.anchor_current_sensor_m.shape[0]
        expected = {
            "anchor_current_sensor_m": (batch, MAXIMUM_SLOTS, 2, 3),
            "residual_local_m": (batch, MAXIMUM_SLOTS, 2, 3),
            "scale_m": (batch, MAXIMUM_SLOTS, 2),
            "compatibility_logits": (batch, ENDPOINTS, ENDPOINTS),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not torch.is_floating_point(value):
                raise ValueError(f"composition-anchor prediction shape drift: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"composition-anchor prediction is non-finite: {name}")
        if bool((self.scale_m < MINIMUM_ANCHOR_SCALE_M).any()):
            raise ValueError("composition-anchor scale is below its numerical floor")
        if not torch.equal(self.compatibility_logits, self.compatibility_logits.transpose(1, 2)):
            raise ValueError("composition-anchor compatibility must be exactly symmetric")


@dataclass(frozen=True)
class CompositionAnchorModelPrediction:
    primitive: ObservableSparsePortRelationPrediction
    composition: CompositionAnchorPrediction


def endpoint_local_frame(axis_control_current_sensor_m: torch.Tensor) -> torch.Tensor:
    """Return a yaw-equivariant [sensor-radial, lateral, gravity] frame.

    Predicted primitive control points may collapse even for a matched query,
    so their tangent is not a total coordinate system.  Current-sensor polar
    position remains defined for every observable endpoint and spans arbitrary
    horizontal corrections together with the gravity axis.
    """

    if axis_control_current_sensor_m.ndim != 4 or axis_control_current_sensor_m.shape[1:] != (
        MAXIMUM_SLOTS, 3, 3,
    ):
        raise ValueError("composition-anchor axes must be [B,32,3,3]")
    endpoints = axis_control_current_sensor_m[:, :, (0, 2)]
    radial = endpoints.clone()
    radial[..., 2] = 0.0
    radial_norm = torch.linalg.vector_norm(radial, dim=-1)
    radial_unit = F.normalize(radial, dim=-1, eps=1e-8)
    gravity = torch.zeros_like(radial_unit)
    gravity[..., 2] = 1.0
    radial_available = radial_norm >= 1e-4
    radial_unit = torch.where(
        radial_available[..., None], radial_unit, torch.zeros_like(radial_unit),
    )
    lateral = F.normalize(
        torch.linalg.cross(gravity, radial_unit, dim=-1), dim=-1, eps=1e-8,
    )
    return torch.stack((radial_unit, lateral, gravity), dim=-2)


def composition_anchor_features(
    prediction: ObservableSparsePortRelationPrediction,
) -> torch.Tensor:
    """Build endpoint-wise yaw-invariant features from deployable outputs."""

    axis = prediction.axis_control_current_sensor_m
    endpoints = axis[:, :, (0, 2)]
    horizontal_range = torch.linalg.vector_norm(endpoints[..., :2], dim=-1, keepdim=True)
    existence = torch.sigmoid(prediction.existence_logits)[:, :, None, None].expand(-1, -1, 2, -1)
    evidence = torch.sigmoid(prediction.endpoint_evidence_logits)[..., None]
    uncertainty = prediction.geometry_uncertainty[:, :, None, None].expand(-1, -1, 2, -1)
    features = torch.cat((
        prediction.endpoint_descriptor,
        horizontal_range / MAXIMUM_RANGE_M,
        endpoints[..., 2:3] / MAXIMUM_RANGE_M,
        torch.log1p(prediction.endpoint_half_axes_m) / math.log(11.0),
        (
            prediction.endpoint_shape_exponent[..., None] - MINIMUM_SHAPE_EXPONENT
        ) / (MAXIMUM_SHAPE_EXPONENT - MINIMUM_SHAPE_EXPONENT),
        uncertainty,
        existence,
        evidence,
    ), dim=-1)
    if features.shape[-1] != COMPOSITION_ANCHOR_FEATURE_DIM:
        raise RuntimeError("composition-anchor feature dimension drift")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("composition-anchor features must be finite")
    return features


def gaussian_anchor_compatibility_logits(
    anchor_current_sensor_m: torch.Tensor,
    scale_m: torch.Tensor,
    *,
    log_temperature: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """Return symmetric same-anchor logits from isotropic Gaussian outputs."""

    batch = len(anchor_current_sensor_m)
    if anchor_current_sensor_m.shape != (batch, MAXIMUM_SLOTS, 2, 3):
        raise ValueError("composition anchors must be [B,32,2,3]")
    if scale_m.shape != (batch, MAXIMUM_SLOTS, 2):
        raise ValueError("composition anchor scales must be [B,32,2]")
    anchor = anchor_current_sensor_m.reshape(batch, ENDPOINTS, 3)
    scale = scale_m.reshape(batch, ENDPOINTS).clamp_min(MINIMUM_ANCHOR_SCALE_M)
    delta = anchor[:, :, None] - anchor[:, None, :]
    variance = scale[:, :, None].square() + scale[:, None, :].square()
    mahalanobis = delta.square().sum(dim=-1) / variance
    log_density = -0.5 * (mahalanobis + 3.0 * torch.log(variance))
    temperature = F.softplus(log_temperature) + 1e-4
    logits = temperature * log_density + bias
    return 0.5 * (logits + logits.transpose(1, 2))


class CompositionAnchorResidualHead(nn.Module):
    """Predict one local-frame anchor residual and scale per endpoint."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint_head = nn.Sequential(
            nn.Linear(COMPOSITION_ANCHOR_FEATURE_DIM, MODEL_DIM),
            nn.SiLU(),
            nn.Linear(MODEL_DIM, MODEL_DIM),
            nn.SiLU(),
            nn.Linear(MODEL_DIM, COMPOSITION_ANCHOR_OUTPUTS_PER_ENDPOINT),
        )
        # Start near a flat probability calibration so micron-scale float32
        # coordinate perturbations are not amplified into relation changes.
        # The scalar remains fully learnable from positive/hard-negative pairs.
        self.compatibility_log_temperature = nn.Parameter(torch.tensor(-5.0))
        self.compatibility_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        prediction: ObservableSparsePortRelationPrediction,
    ) -> CompositionAnchorPrediction:
        features = composition_anchor_features(prediction)
        raw = self.endpoint_head(features)
        residual_local_m = raw[..., :3]
        scale_m = F.softplus(raw[..., 3]) + MINIMUM_ANCHOR_SCALE_M
        frame = endpoint_local_frame(prediction.axis_control_current_sensor_m)
        residual_sensor_m = torch.einsum("bsej,bsejk->bsek", residual_local_m, frame)
        endpoints = prediction.axis_control_current_sensor_m[:, :, (0, 2)]
        anchors = endpoints + residual_sensor_m
        compatibility = gaussian_anchor_compatibility_logits(
            anchors,
            scale_m,
            log_temperature=self.compatibility_log_temperature,
            bias=self.compatibility_bias,
        )
        result = CompositionAnchorPrediction(
            anchor_current_sensor_m=anchors,
            residual_local_m=residual_local_m,
            scale_m=scale_m,
            compatibility_logits=compatibility,
        )
        result.validate()
        return result


class FrozenObservableCompositionAnchorNet(nn.Module):
    """Attach the O(E) corrective to a frozen mature primitive backbone.

    The old pair outputs are retained inside ``primitive`` only for exact
    checkpoint compatibility and ablation.  They do not enter the new anchor
    head or its loss.  A later deployment pruning pass can remove their
    compute after the anchor hypothesis passes C07.
    """

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.anchor_head = CompositionAnchorResidualHead()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.anchor_head.train(mode)
        return self

    def forward(
        self,
        range_valid: torch.Tensor,
        relative_translation_current_sensor_m: torch.Tensor,
        relative_yaw_current_sensor_deg: torch.Tensor,
        *,
        query_permutation: torch.Tensor | None = None,
    ) -> CompositionAnchorModelPrediction:
        with torch.no_grad():
            primitive = self.backbone(
                range_valid,
                relative_translation_current_sensor_m,
                relative_yaw_current_sensor_deg,
                query_permutation=query_permutation,
            )
        composition = self.anchor_head(primitive)
        return CompositionAnchorModelPrediction(primitive=primitive, composition=composition)


def composition_anchor_safe_score(
    prediction: CompositionAnchorModelPrediction,
) -> torch.Tensor:
    """Deployable pair score: anchor compatibility times endpoint evidence."""

    prediction.composition.validate()
    batch = len(prediction.composition.compatibility_logits)
    evidence = torch.sigmoid(
        prediction.primitive.endpoint_evidence_logits,
    ).reshape(batch, ENDPOINTS)
    # Form the evidence outer product before combining it with compatibility.
    # The mathematically equivalent left-associated expression
    # ``compatibility * evidence_i * evidence_j`` rounds in a different order
    # after transposition and can therefore violate the exact-symmetry
    # deployment contract by one float32 ULP on real CUDA batches.
    evidence_pair = evidence[:, :, None] * evidence[:, None, :]
    score = torch.sigmoid(prediction.composition.compatibility_logits) * evidence_pair
    if tuple(score.shape) != (batch, ENDPOINTS, ENDPOINTS):
        raise RuntimeError("composition-anchor safe score shape drift")
    if not torch.equal(score, score.transpose(1, 2)) or not bool(torch.isfinite(score).all()):
        raise ValueError("composition-anchor safe score must be finite and symmetric")
    return score


def _balanced_binary_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    target = target.bool()
    terms = []
    if bool(target.any()):
        terms.append(F.softplus(-logits[target]).mean())
    if bool((~target).any()):
        terms.append(F.softplus(logits[~target]).mean())
    if not terms:
        raise ValueError("composition-anchor binary loss has no elements")
    return torch.stack(terms).mean()


def composition_anchor_losses(
    prediction: CompositionAnchorPrediction,
    *,
    anchor_target_current_sensor_m: torch.Tensor,
    primitive_mask: torch.Tensor,
    endpoint_observed: torch.Tensor,
    attachment_target: torch.Tensor,
    disconnected_overlap: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Supervise anchors, uncertainty and probabilistic physical compatibility.

    All targets are already aligned to prediction slots.  Construction and
    primitive identities are intentionally absent from this interface.
    """

    prediction.validate()
    batch = len(prediction.anchor_current_sensor_m)
    expected = {
        "anchor_target_current_sensor_m": (batch, MAXIMUM_SLOTS, 2, 3),
        "primitive_mask": (batch, MAXIMUM_SLOTS),
        "endpoint_observed": (batch, MAXIMUM_SLOTS, 2),
        "attachment_target": (batch, MAXIMUM_SLOTS, 2, MAXIMUM_SLOTS, 2),
        "disconnected_overlap": (batch, MAXIMUM_SLOTS, MAXIMUM_SLOTS),
    }
    values = {
        "anchor_target_current_sensor_m": anchor_target_current_sensor_m,
        "primitive_mask": primitive_mask,
        "endpoint_observed": endpoint_observed,
        "attachment_target": attachment_target,
        "disconnected_overlap": disconnected_overlap,
    }
    for name, shape in expected.items():
        if tuple(values[name].shape) != shape:
            raise ValueError(f"composition-anchor loss target shape drift: {name}")
    active = primitive_mask.bool()
    observed = endpoint_observed.bool()
    if bool((observed & ~active[:, :, None]).any()):
        raise ValueError("inactive composition-anchor endpoint cannot be observed")
    if not bool(observed.any()) or not bool(torch.isfinite(anchor_target_current_sensor_m[observed]).all()):
        raise ValueError("composition-anchor loss requires finite observed targets")

    error = prediction.anchor_current_sensor_m - anchor_target_current_sensor_m
    scale = prediction.scale_m[..., None]
    anchor_nll = (
        0.5 * (error[observed] / scale[observed]).square()
        + torch.log(scale[observed])
    ).mean()
    endpoint_error = torch.linalg.vector_norm(error.detach(), dim=-1) / math.sqrt(3.0)
    uncertainty_calibration = F.smooth_l1_loss(
        prediction.scale_m[observed], endpoint_error[observed],
    )

    endpoint_observed_flat = observed.reshape(batch, ENDPOINTS)
    endpoint_index = torch.arange(ENDPOINTS, device=observed.device)
    primitive_index = endpoint_index // 2
    upper_cross_primitive = (
        (primitive_index[:, None] != primitive_index[None, :])
        & torch.triu(torch.ones(ENDPOINTS, ENDPOINTS, dtype=torch.bool, device=observed.device), diagonal=1)
    )
    eligible = (
        endpoint_observed_flat[:, :, None]
        & endpoint_observed_flat[:, None, :]
        & upper_cross_primitive[None]
    )
    attachment = attachment_target.reshape(batch, ENDPOINTS, ENDPOINTS).bool()
    logits = prediction.compatibility_logits[eligible]
    target = attachment[eligible]
    compatibility = _balanced_binary_loss(logits, target)
    endpoint_overlap = disconnected_overlap.bool().repeat_interleave(2, dim=1).repeat_interleave(2, dim=2)
    hard_negative = eligible & endpoint_overlap & ~attachment
    if bool(hard_negative.any()):
        hard_negative_loss = F.softplus(prediction.compatibility_logits[hard_negative]).mean()
    else:
        hard_negative_loss = prediction.compatibility_logits.sum() * 0.0
    relation = 0.5 * (compatibility + hard_negative_loss)
    total = torch.stack((anchor_nll, uncertainty_calibration, relation)).mean()
    return {
        "anchor_nll": anchor_nll,
        "uncertainty_calibration": uncertainty_calibration,
        "compatibility": compatibility,
        "overlap_hard_negative": hard_negative_loss,
        "relation": relation,
        "total": total,
    }


__all__ = [
    "COMPOSITION_ANCHOR_FEATURE_DIM",
    "COMPOSITION_ANCHOR_OUTPUTS_PER_ENDPOINT",
    "CompositionAnchorModelPrediction",
    "CompositionAnchorPrediction",
    "CompositionAnchorResidualHead",
    "FrozenObservableCompositionAnchorNet",
    "composition_anchor_features",
    "composition_anchor_losses",
    "composition_anchor_safe_score",
    "endpoint_local_frame",
    "gaussian_anchor_compatibility_logits",
]
