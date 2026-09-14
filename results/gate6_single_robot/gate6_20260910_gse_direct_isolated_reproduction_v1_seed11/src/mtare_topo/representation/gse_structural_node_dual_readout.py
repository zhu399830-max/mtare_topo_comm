"""Dual readout for geometry-semantic structural-node observations.

The geometry vector and the place-association embedding are deliberately
separate.  Geometry answers *what local structure is present*.  Association
only proposes *which previous node may be the same place* and must still be
checked by graph consistency before a merge is committed.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.gse_structural_node_evidence import (
    MAX_NODE_PORTS,
    NODE_EVIDENCE_DIM,
    PAIR_CAPACITY,
    PORT_FEATURE_DIM,
)
from mtare_topo.representation.gse_structural_node_student import (
    ENDPOINTS,
    NODE_STUDENT_ENDPOINT_FEATURE_DIM,
)


ASSOCIATION_DIM = 64
MODEL_DIM = 128
POOL_QUERIES = 8


def _geometry_scale() -> torch.Tensor:
    """Analytic descriptor scales; no fitted split statistics are used."""

    port = (1.0, 1.0, 2.0, 2.0, 1.0)
    values = [1.0]
    values.extend(port * MAX_NODE_PORTS)
    values.extend([1.0] * PAIR_CAPACITY)  # direction cosine
    values.extend([4.0] * PAIR_CAPACITY)  # four-feature shape delta
    values.extend(port)  # mean
    values.extend(port)  # standard deviation
    values.append(1.0)  # uncertainty
    result = torch.tensor(values, dtype=torch.float32)
    if result.shape != (NODE_EVIDENCE_DIM,) or torch.any(result <= 0):
        raise RuntimeError("structural-node geometry scale contract drift")
    return result


GEOMETRY_SCALE = _geometry_scale()


@dataclass(frozen=True)
class StructuralNodeDualPrediction:
    geometry: torch.Tensor
    association: torch.Tensor
    degree_logits: torch.Tensor
    association_temperature: torch.Tensor
    association_bias: torch.Tensor

    def validate(self) -> None:
        batch = self.geometry.shape[0]
        if self.geometry.shape != (batch, NODE_EVIDENCE_DIM):
            raise ValueError("dual-readout geometry shape drift")
        if self.association.shape != (batch, ASSOCIATION_DIM):
            raise ValueError("dual-readout association shape drift")
        if self.degree_logits.shape != (batch, 4):
            raise ValueError("dual-readout degree shape drift")
        if self.association_temperature.ndim != 0 or self.association_bias.ndim != 0:
            raise ValueError("dual-readout calibration shape drift")
        values = (
            self.geometry,
            self.association,
            self.degree_logits,
            self.association_temperature,
            self.association_bias,
        )
        if not all(torch.isfinite(value).all() for value in values):
            raise ValueError("dual-readout prediction is nonfinite")
        norm = torch.linalg.vector_norm(self.association, dim=-1)
        if not torch.allclose(norm, torch.ones_like(norm), atol=2e-5, rtol=2e-5):
            raise ValueError("association embedding is not normalized")


class StructuralNodeDualReadout(nn.Module):
    """Permutation-invariant set readout with distinct scientific roles."""

    def __init__(self) -> None:
        super().__init__()
        self.endpoint_encoder = nn.Sequential(
            nn.Linear(NODE_STUDENT_ENDPOINT_FEATURE_DIM + 1, MODEL_DIM),
            nn.LayerNorm(MODEL_DIM),
            nn.GELU(),
            nn.Linear(MODEL_DIM, MODEL_DIM),
        )
        layer = nn.TransformerEncoderLayer(
            MODEL_DIM,
            nhead=8,
            dim_feedforward=4 * MODEL_DIM,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.context = nn.TransformerEncoder(
            layer,
            num_layers=2,
            norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.pool_queries = nn.Parameter(torch.empty(POOL_QUERIES, MODEL_DIM))
        nn.init.normal_(self.pool_queries, std=0.02)
        pooled_dim = (POOL_QUERIES + 3) * MODEL_DIM
        self.trunk = nn.Sequential(
            nn.Linear(pooled_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
        )
        self.geometry_head = nn.Sequential(
            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, NODE_EVIDENCE_DIM),
        )
        self.association_head = nn.Sequential(
            nn.Linear(512, 512), nn.GELU(), nn.Linear(512, ASSOCIATION_DIM),
        )
        self.degree_head = nn.Linear(512, 4)
        self.association_temperature_raw = nn.Parameter(torch.tensor(2.0))
        self.association_bias = nn.Parameter(torch.zeros(()))

    def forward(
        self,
        endpoint_features: torch.Tensor,
        endpoint_confidence: torch.Tensor,
    ) -> StructuralNodeDualPrediction:
        batch = endpoint_features.shape[0]
        if (
            endpoint_features.shape
            != (batch, ENDPOINTS, NODE_STUDENT_ENDPOINT_FEATURE_DIM)
            or endpoint_confidence.shape != (batch, ENDPOINTS)
            or not torch.isfinite(endpoint_features).all()
            or not torch.isfinite(endpoint_confidence).all()
            or bool(((endpoint_confidence < 0.0) | (endpoint_confidence > 1.0)).any())
        ):
            raise ValueError("dual-readout input contract drift")

        token = torch.cat((endpoint_features, endpoint_confidence[..., None]), dim=-1)
        encoded = self.context(self.endpoint_encoder(token))
        confidence = endpoint_confidence.clamp_min(1e-6)
        query_logits = torch.einsum(
            "bsm,qm->bqs", encoded, self.pool_queries,
        ) / math.sqrt(MODEL_DIM)
        query_logits = query_logits + torch.log(confidence[:, None, :])
        query_pool = torch.einsum(
            "bqs,bsm->bqm", torch.softmax(query_logits, dim=-1), encoded,
        ).flatten(1)
        weight = confidence / confidence.sum(dim=1, keepdim=True)
        mean = torch.sum(weight[..., None] * encoded, dim=1)
        variance = torch.sum(weight[..., None] * (encoded - mean[:, None]).square(), dim=1)
        standard_deviation = torch.sqrt(variance + 1e-6)
        maximum = torch.amax(encoded + torch.log(confidence)[..., None], dim=1)
        latent = self.trunk(torch.cat((query_pool, mean, standard_deviation, maximum), dim=-1))
        result = StructuralNodeDualPrediction(
            geometry=self.geometry_head(latent),
            association=F.normalize(self.association_head(latent), dim=-1, eps=1e-6),
            degree_logits=self.degree_head(latent),
            association_temperature=F.softplus(self.association_temperature_raw) + 1e-4,
            association_bias=self.association_bias,
        )
        result.validate()
        return result


def structural_node_dual_losses(
    prediction: StructuralNodeDualPrediction,
    target_geometry: torch.Tensor,
    target_degree: torch.Tensor,
    same_node: torch.Tensor,
    candidate_pair: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Balanced geometry, event and within-graph association objectives."""

    prediction.validate()
    batch = prediction.geometry.shape[0]
    if (
        target_geometry.shape != (batch, NODE_EVIDENCE_DIM)
        or target_degree.shape != (batch,)
        or same_node.shape != (batch, batch)
        or candidate_pair.shape != (batch, batch)
        or target_degree.dtype != torch.long
        or same_node.dtype != torch.bool
        or candidate_pair.dtype != torch.bool
        or not torch.equal(same_node, same_node.T)
        or not torch.equal(candidate_pair, candidate_pair.T)
        or bool((same_node & ~candidate_pair).any())
        or not torch.isfinite(target_geometry).all()
    ):
        raise ValueError("dual-readout target contract drift")

    scale = GEOMETRY_SCALE.to(device=target_geometry.device, dtype=target_geometry.dtype)
    geometry = F.mse_loss(
        prediction.geometry[:, :-1] / scale[:-1],
        target_geometry[:, :-1] / scale[:-1],
    )
    uncertainty = F.mse_loss(prediction.geometry[:, -1], target_geometry[:, -1])
    degree = F.cross_entropy(prediction.degree_logits, target_degree - 1)

    diagonal = torch.eye(batch, dtype=torch.bool, device=target_geometry.device)
    candidate = candidate_pair & ~diagonal
    positive = same_node & candidate
    negative = ~same_node & candidate
    if not positive.any() or not negative.any():
        raise ValueError("dual-readout batch requires candidate positives and negatives")
    cosine = prediction.association @ prediction.association.T
    logits = prediction.association_temperature * cosine + prediction.association_bias
    balanced_bce = 0.5 * (
        F.softplus(-logits[positive]).mean() + F.softplus(logits[negative]).mean()
    )
    # Every repeated node in the tiny/full batches has exactly one or more
    # positives.  This supervised contrastive term forces its positive ahead
    # of all competing nodes in the same online graph without node IDs at
    # inference time.
    positive_count = positive.sum(dim=1)
    anchors = positive_count > 0
    masked_logits = logits.masked_fill(~candidate, -torch.inf)
    log_denominator = torch.logsumexp(masked_logits[anchors], dim=1)
    positive_logits = logits.masked_fill(~positive, -torch.inf)
    log_numerator = torch.logsumexp(positive_logits[anchors], dim=1)
    contrastive = (log_denominator - log_numerator).mean()
    relation = 0.5 * (balanced_bce + contrastive)
    total = geometry + 0.1 * uncertainty + 0.25 * degree + 0.5 * relation
    return {
        "geometry": geometry,
        "uncertainty": uncertainty,
        "degree": degree,
        "relation_bce": balanced_bce,
        "relation_contrastive": contrastive,
        "relation": relation,
        "total": total,
    }


def association_distance(embedding: torch.Tensor) -> torch.Tensor:
    """Cosine distance matrix for graph-candidate scoring."""

    if embedding.ndim != 2 or embedding.shape[1] != ASSOCIATION_DIM:
        raise ValueError("association embedding contract drift")
    return torch.clamp(1.0 - embedding @ embedding.T, min=0.0, max=2.0)


__all__ = [
    "ASSOCIATION_DIM",
    "GEOMETRY_SCALE",
    "StructuralNodeDualPrediction",
    "StructuralNodeDualReadout",
    "association_distance",
    "structural_node_dual_losses",
]
