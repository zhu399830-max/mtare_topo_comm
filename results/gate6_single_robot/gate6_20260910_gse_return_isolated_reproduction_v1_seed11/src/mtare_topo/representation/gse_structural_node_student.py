"""Small permutation-invariant student for structural-node evidence."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_structural_node_evidence import NODE_EVIDENCE_DIM
from mtare_topo.representation.primitive_composition_anchor_model import (
    COMPOSITION_ANCHOR_FEATURE_DIM,
    composition_anchor_features,
    endpoint_local_frame,
)
from mtare_topo.representation.primitive_relation_model import MODEL_DIM
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationPrediction,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import (
    outward_endpoint_tangents,
)


ENDPOINTS = 64
NODE_STUDENT_ENDPOINT_FEATURE_DIM = COMPOSITION_ANCHOR_FEATURE_DIM + 4


@dataclass(frozen=True)
class StructuralNodeStudentPrediction:
    descriptor: torch.Tensor
    degree_logits: torch.Tensor
    association_scale: torch.Tensor
    association_bias: torch.Tensor

    def validate(self) -> None:
        batch = self.descriptor.shape[0]
        if self.descriptor.shape != (batch, NODE_EVIDENCE_DIM):
            raise ValueError("node student descriptor shape drift")
        if self.degree_logits.shape != (batch, 4):
            raise ValueError("node student degree shape drift")
        if self.association_scale.ndim != 0 or self.association_bias.ndim != 0:
            raise ValueError("node student association calibration shape drift")
        if not all(torch.isfinite(value).all() for value in (
            self.descriptor, self.degree_logits,
            self.association_scale, self.association_bias,
        )):
            raise ValueError("node student prediction is nonfinite")


def structural_node_student_inputs(
    prediction: ObservableSparsePortRelationPrediction,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build yaw-invariant endpoint tokens from frozen deployable outputs."""

    base = composition_anchor_features(prediction)
    tangent = outward_endpoint_tangents(prediction.axis_control_current_sensor_m)
    frame = endpoint_local_frame(prediction.axis_control_current_sensor_m)
    tangent_local = torch.einsum("bsej,bsekj->bsek", tangent, frame)
    axis = prediction.axis_control_current_sensor_m
    left = axis[:, :, 1] - axis[:, :, 0]
    right = axis[:, :, 2] - axis[:, :, 1]
    nl = torch.linalg.vector_norm(left, dim=-1).clamp_min(1e-6)
    nr = torch.linalg.vector_norm(right, dim=-1).clamp_min(1e-6)
    cosine = torch.sum(left * right, dim=-1) / (nl * nr)
    angle = torch.acos(torch.clamp(cosine, -1.0, 1.0))
    curvature = angle / ((nl + nr) * .5)
    curvature = torch.clamp(curvature / .5, 0.0, 2.0)[:, :, None, None].expand(-1, -1, 2, -1)
    features = torch.cat((base, tangent_local, curvature), dim=-1).reshape(
        -1, ENDPOINTS, NODE_STUDENT_ENDPOINT_FEATURE_DIM,
    )
    existence = torch.sigmoid(prediction.existence_logits)[:, :, None].expand(-1, -1, 2)
    evidence = torch.sigmoid(prediction.endpoint_evidence_logits)
    confidence = (existence * evidence).reshape(-1, ENDPOINTS)
    if (
        features.shape[1:] != (ENDPOINTS, NODE_STUDENT_ENDPOINT_FEATURE_DIM)
        or confidence.shape != features.shape[:2]
        or not torch.isfinite(features).all()
        or not torch.isfinite(confidence).all()
    ):
        raise RuntimeError("node student input contract drift")
    return features, confidence


class StructuralNodeAggregationHead(nn.Module):
    """Permutation-invariant endpoint-set aggregator with refusal confidence."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(NODE_STUDENT_ENDPOINT_FEATURE_DIM, MODEL_DIM),
            nn.GELU(),
            nn.Linear(MODEL_DIM, MODEL_DIM),
        )
        layer = nn.TransformerEncoderLayer(
            MODEL_DIM, nhead=8, dim_feedforward=4 * MODEL_DIM,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.attention = nn.Linear(MODEL_DIM, 1)
        self.descriptor_head = nn.Sequential(
            nn.Linear(MODEL_DIM, MODEL_DIM), nn.GELU(),
            nn.Linear(MODEL_DIM, NODE_EVIDENCE_DIM),
        )
        self.degree_head = nn.Linear(MODEL_DIM, 4)
        self.association_scale_raw = nn.Parameter(torch.zeros(()))
        self.association_bias = nn.Parameter(torch.zeros(()))

    def forward(self, endpoint_features: torch.Tensor, endpoint_confidence: torch.Tensor) -> StructuralNodeStudentPrediction:
        batch = endpoint_features.shape[0]
        if (
            endpoint_features.shape != (batch, ENDPOINTS, NODE_STUDENT_ENDPOINT_FEATURE_DIM)
            or endpoint_confidence.shape != (batch, ENDPOINTS)
            or not torch.isfinite(endpoint_features).all()
            or not torch.isfinite(endpoint_confidence).all()
            or bool(((endpoint_confidence < 0.0) | (endpoint_confidence > 1.0)).any())
        ):
            raise ValueError("node aggregation input contract drift")
        encoded = self.context(self.endpoint_encoder(endpoint_features))
        logits = self.attention(encoded).squeeze(-1) + torch.log(endpoint_confidence.clamp_min(1e-6))
        weights = torch.softmax(logits, dim=1)
        pooled = torch.sum(weights[..., None] * encoded, dim=1)
        result = StructuralNodeStudentPrediction(
            descriptor=self.descriptor_head(pooled),
            degree_logits=self.degree_head(pooled),
            association_scale=F.softplus(self.association_scale_raw) + 1e-4,
            association_bias=self.association_bias,
        )
        result.validate()
        return result


def structural_node_student_losses(
    prediction: StructuralNodeStudentPrediction,
    target_descriptor: torch.Tensor,
    target_degree: torch.Tensor,
    same_node: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Tiny-overfit objective; identities enter only the pairwise loss target."""

    prediction.validate()
    batch = len(prediction.descriptor)
    if (
        target_descriptor.shape != (batch, NODE_EVIDENCE_DIM)
        or target_degree.shape != (batch,)
        or same_node.shape != (batch, batch)
        or target_degree.dtype != torch.long
        or same_node.dtype != torch.bool
        or bool(((target_degree < 1) | (target_degree > 4)).any())
        or not torch.equal(same_node, same_node.T)
        or not torch.isfinite(target_descriptor).all()
    ):
        raise ValueError("node student target contract drift")
    geometry = F.smooth_l1_loss(prediction.descriptor[:, :-1], target_descriptor[:, :-1])
    uncertainty = F.smooth_l1_loss(prediction.descriptor[:, -1], target_descriptor[:, -1])
    degree = F.cross_entropy(prediction.degree_logits, target_degree - 1)
    delta = prediction.descriptor[:, None, :-1] - prediction.descriptor[None, :, :-1]
    rms = torch.sqrt(torch.mean(delta.square(), dim=-1) + 1e-12)
    logits = prediction.association_bias - prediction.association_scale * rms
    off_diagonal = ~torch.eye(batch, dtype=torch.bool, device=logits.device)
    positive = same_node & off_diagonal
    negative = ~same_node & off_diagonal
    relation_terms = []
    if positive.any():
        relation_terms.append(F.softplus(-logits[positive]).mean())
    if negative.any():
        relation_terms.append(F.softplus(logits[negative]).mean())
    if len(relation_terms) != 2:
        raise ValueError("tiny-overfit batch requires positive and negative node pairs")
    relation = torch.stack(relation_terms).mean()
    total = geometry + .25 * uncertainty + .25 * degree + .25 * relation
    return {
        "geometry": geometry, "uncertainty": uncertainty,
        "degree": degree, "relation": relation, "total": total,
    }


__all__ = [
    "NODE_STUDENT_ENDPOINT_FEATURE_DIM", "StructuralNodeAggregationHead",
    "StructuralNodeStudentPrediction", "structural_node_student_inputs",
    "structural_node_student_losses",
]
