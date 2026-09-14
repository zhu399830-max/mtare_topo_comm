"""No-slot endpoint relation metric for primitive composition learning.

Each observable primitive endpoint receives a continuous embedding.  Physical
composition is represented by embedding similarity, not by an arbitrary slot
index or a shared dustbin class.  The pair logit is symmetric by construction
and is consumed through one global confidence threshold.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.primitive_composition_anchor_model import (
    COMPOSITION_ANCHOR_FEATURE_DIM,
    composition_anchor_features,
)
from mtare_topo.representation.primitive_local_composition_slot_model import ENDPOINT_COUNT
from mtare_topo.representation.primitive_relation_model import MODEL_DIM
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationNet,
    ObservableSparsePortRelationPrediction,
)


RELATION_EMBEDDING_DIM = 64


@dataclass(frozen=True)
class EndpointRelationMetricPrediction:
    embedding: torch.Tensor
    cosine_similarity: torch.Tensor
    pair_logits: torch.Tensor

    def validate(self) -> None:
        batch = self.embedding.shape[0]
        expected = {
            "embedding": (batch, ENDPOINT_COUNT, RELATION_EMBEDDING_DIM),
            "cosine_similarity": (batch, ENDPOINT_COUNT, ENDPOINT_COUNT),
            "pair_logits": (batch, ENDPOINT_COUNT, ENDPOINT_COUNT),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not torch.is_floating_point(value):
                raise ValueError(f"endpoint relation metric shape drift: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"endpoint relation metric is non-finite: {name}")
        for name in ("cosine_similarity", "pair_logits"):
            value = getattr(self, name)
            if not torch.equal(value, value.transpose(1, 2)):
                raise ValueError(f"endpoint relation metric is not exactly symmetric: {name}")


@dataclass(frozen=True)
class EndpointRelationMetricModelPrediction:
    primitive: ObservableSparsePortRelationPrediction
    relation: EndpointRelationMetricPrediction


class EndpointRelationMetricHead(nn.Module):
    """Permutation-equivariant endpoint encoder with a symmetric metric."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(COMPOSITION_ANCHOR_FEATURE_DIM, MODEL_DIM),
            nn.SiLU(),
            nn.Linear(MODEL_DIM, MODEL_DIM),
        )
        layer = nn.TransformerEncoderLayer(
            MODEL_DIM, nhead=8, dim_feedforward=4 * MODEL_DIM,
            dropout=0.0, batch_first=True, activation="gelu", norm_first=True,
        )
        self.endpoint_context = nn.TransformerEncoder(
            layer, num_layers=2, norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.relation_embedding = nn.Linear(MODEL_DIM, RELATION_EMBEDDING_DIM)
        self.logit_scale_raw = nn.Parameter(torch.zeros(()))
        self.logit_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        primitive: ObservableSparsePortRelationPrediction,
    ) -> EndpointRelationMetricPrediction:
        features = composition_anchor_features(primitive).reshape(
            -1, ENDPOINT_COUNT, COMPOSITION_ANCHOR_FEATURE_DIM,
        )
        context = self.endpoint_context(self.endpoint_embedding(features))
        embedding = F.normalize(self.relation_embedding(context), dim=-1, eps=1e-8)
        cosine = torch.einsum("bed,bfd->bef", embedding, embedding)
        cosine = 0.5 * (cosine + cosine.transpose(1, 2))
        scale = F.softplus(self.logit_scale_raw) + 1e-4
        logits = scale * cosine + self.logit_bias
        logits = 0.5 * (logits + logits.transpose(1, 2))
        result = EndpointRelationMetricPrediction(embedding, cosine, logits)
        result.validate()
        return result


class FrozenObservableEndpointRelationMetricNet(nn.Module):
    def __init__(self, backbone: ObservableSparsePortRelationNet) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.relation_head = EndpointRelationMetricHead()

    def train(self, mode: bool = True):
        super().train(mode); self.backbone.eval(); self.relation_head.train(mode)
        return self

    def forward(
        self,
        range_valid: torch.Tensor,
        relative_translation_current_sensor_m: torch.Tensor,
        relative_yaw_current_sensor_deg: torch.Tensor,
        *,
        query_permutation: torch.Tensor | None = None,
    ) -> EndpointRelationMetricModelPrediction:
        with torch.no_grad():
            primitive = self.backbone(
                range_valid, relative_translation_current_sensor_m,
                relative_yaw_current_sensor_deg, query_permutation=query_permutation,
            )
        return EndpointRelationMetricModelPrediction(
            primitive=primitive, relation=self.relation_head(primitive),
        )


def endpoint_relation_pair_score(
    prediction: EndpointRelationMetricModelPrediction | EndpointRelationMetricPrediction,
) -> torch.Tensor:
    relation = prediction.relation if isinstance(
        prediction, EndpointRelationMetricModelPrediction,
    ) else prediction
    relation.validate()
    score = torch.sigmoid(relation.pair_logits)
    endpoint = torch.arange(ENDPOINT_COUNT, device=score.device)
    same_primitive = endpoint[:, None] // 2 == endpoint[None, :] // 2
    score = score.masked_fill(same_primitive[None], 0.0)
    score = 0.5 * (score + score.transpose(1, 2))
    if not torch.equal(score, score.transpose(1, 2)) or not bool(torch.isfinite(score).all()):
        raise ValueError("endpoint relation pair score must be finite and symmetric")
    return score


__all__ = [
    "EndpointRelationMetricHead", "EndpointRelationMetricModelPrediction",
    "EndpointRelationMetricPrediction", "FrozenObservableEndpointRelationMetricNet",
    "RELATION_EMBEDDING_DIM", "endpoint_relation_pair_score",
]
