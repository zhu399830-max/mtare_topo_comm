"""Endpoint-evidence-aware sparse primitive relation model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn

from mtare_topo.representation.primitive_relation_model import (
    ENDPOINT_DESCRIPTOR_DIM,
    MAXIMUM_RANGE_M,
    MODEL_DIM,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    SparsePortRelationNet,
    SparsePortRelationPrediction,
    outward_endpoint_tangents,
)


ENDPOINT_EVIDENCE_FEATURE_DIM = ENDPOINT_DESCRIPTOR_DIM + 3 + 3 + 2 + 1 + 1 + 1


@dataclass(frozen=True)
class ObservableSparsePortRelationPrediction:
    existence_logits: torch.Tensor
    axis_control_current_sensor_m: torch.Tensor
    endpoint_half_axes_m: torch.Tensor
    endpoint_shape_exponent: torch.Tensor
    endpoint_descriptor: torch.Tensor
    geometry_uncertainty: torch.Tensor
    endpoint_evidence_logits: torch.Tensor
    endpoint_attachment_logits: torch.Tensor
    endpoint_attachment_uncertainty: torch.Tensor
    disconnected_overlap_logits: torch.Tensor
    disconnected_overlap_uncertainty: torch.Tensor
    temporal_correspondence_logits: torch.Tensor
    temporal_presence_logits: torch.Tensor


class ObservableSparsePortRelationNet(SparsePortRelationNet):
    """V3 relation model with a deployable endpoint-evidence probability.

    The support sidecar is Teacher-only.  At inference this head estimates
    whether each predicted crop end contains enough local evidence to behave
    as a physical endpoint; safe attachment scores multiply both endpoint
    probabilities with the learned pair score and relation confidence.
    """

    def __init__(self) -> None:
        super().__init__()
        self.endpoint_evidence_head = nn.Sequential(
            nn.Linear(ENDPOINT_EVIDENCE_FEATURE_DIM, MODEL_DIM),
            nn.SiLU(),
            nn.Linear(MODEL_DIM, 1),
        )

    def forward(self, *args, **kwargs) -> ObservableSparsePortRelationPrediction:
        base: SparsePortRelationPrediction = super().forward(*args, **kwargs)
        endpoints = base.axis_control_current_sensor_m[:, :, (0, 2)]
        tangent = outward_endpoint_tangents(base.axis_control_current_sensor_m)
        existence = torch.sigmoid(base.existence_logits)[:, :, None, None].expand(-1, -1, 2, -1)
        uncertainty = base.geometry_uncertainty[:, :, None, None].expand(-1, -1, 2, -1)
        features = torch.cat((
            base.endpoint_descriptor,
            endpoints / MAXIMUM_RANGE_M,
            tangent,
            torch.log1p(base.endpoint_half_axes_m) / math.log(11.0),
            base.endpoint_shape_exponent[..., None],
            uncertainty,
            existence,
        ), dim=-1)
        if features.shape[-1] != ENDPOINT_EVIDENCE_FEATURE_DIM:
            raise RuntimeError("endpoint evidence feature dimension drift")
        evidence = self.endpoint_evidence_head(features).squeeze(-1)
        return ObservableSparsePortRelationPrediction(
            **base.__dict__, endpoint_evidence_logits=evidence,
        )


def safe_attachment_score(prediction: ObservableSparsePortRelationPrediction) -> torch.Tensor:
    """Deployment score requiring two learned endpoint-evidence probabilities."""

    evidence = torch.sigmoid(prediction.endpoint_evidence_logits)
    endpoint_pair = evidence[:, :, :, None, None] * evidence[:, None, None, :, :]
    relation = torch.sigmoid(prediction.endpoint_attachment_logits)
    confidence = 1.0 - prediction.endpoint_attachment_uncertainty
    return relation * endpoint_pair * confidence


__all__ = [
    "ENDPOINT_EVIDENCE_FEATURE_DIM", "ObservableSparsePortRelationNet",
    "ObservableSparsePortRelationPrediction", "safe_attachment_score",
]
