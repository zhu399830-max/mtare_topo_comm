"""Exchangeable local composition slots for observable primitive endpoints.

The previous relation corrective regressed a global 3-D anchor independently
for every endpoint.  That target was geometrically consistent, but the frozen
LiDAR primitive predictions were not accurate enough for sub-metre coordinate
agreement.  This module keeps the learned primitive representation and changes
only the relation interface: endpoints select one of a set of exchangeable
local composition slots, or a dustbin when no complete local attachment is
observable.  Equality of a non-dustbin slot induces a candidate attachment.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F

from mtare_topo.representation.primitive_composition_anchor_model import (
    COMPOSITION_ANCHOR_FEATURE_DIM,
    composition_anchor_features,
)
from mtare_topo.representation.primitive_relation_model import MODEL_DIM
from mtare_topo.representation.primitive_relation_observable_model import (
    ObservableSparsePortRelationPrediction,
)


ENDPOINT_COUNT = 64
COMPOSITION_SLOT_COUNT = 32
DUSTBIN_INDEX = COMPOSITION_SLOT_COUNT
ASSIGNMENT_CLASS_COUNT = COMPOSITION_SLOT_COUNT + 1


@dataclass(frozen=True)
class LocalCompositionSlotPrediction:
    assignment_logits: torch.Tensor
    slot_presence_logits: torch.Tensor
    endpoint_uncertainty: torch.Tensor

    def validate(self) -> None:
        batch = self.assignment_logits.shape[0]
        expected = {
            "assignment_logits": (batch, ENDPOINT_COUNT, ASSIGNMENT_CLASS_COUNT),
            "slot_presence_logits": (batch, COMPOSITION_SLOT_COUNT),
            "endpoint_uncertainty": (batch, ENDPOINT_COUNT),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if tuple(value.shape) != shape or not torch.is_floating_point(value):
                raise ValueError(f"local composition-slot shape drift: {name}")
            if not bool(torch.isfinite(value).all()):
                raise ValueError(f"local composition-slot value is non-finite: {name}")
        if bool((self.endpoint_uncertainty < 0.0).any()) or bool(
            (self.endpoint_uncertainty > 1.0 + 1e-6).any()
        ):
            raise ValueError("local composition-slot uncertainty must lie in [0,1]")


@dataclass(frozen=True)
class LocalCompositionSlotModelPrediction:
    primitive: ObservableSparsePortRelationPrediction
    composition: LocalCompositionSlotPrediction


class LocalCompositionSlotHead(nn.Module):
    """Predict an exchangeable set of endpoint-composition assignments.

    No endpoint-pair classifier is emitted.  Endpoint and slot tokens interact
    through set attention, then one categorical score is produced for each
    endpoint/slot pair plus one endpoint-specific dustbin score.  The output
    size is O(EK), and permuting either primitive queries or composition-slot
    queries only permutes the corresponding output axis.
    """

    def __init__(self) -> None:
        super().__init__()
        self.endpoint_embedding = nn.Sequential(
            nn.Linear(COMPOSITION_ANCHOR_FEATURE_DIM, MODEL_DIM),
            nn.SiLU(),
            nn.Linear(MODEL_DIM, MODEL_DIM),
        )
        endpoint_layer = nn.TransformerEncoderLayer(
            MODEL_DIM,
            nhead=8,
            dim_feedforward=4 * MODEL_DIM,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.endpoint_context = nn.TransformerEncoder(
            endpoint_layer,
            num_layers=2,
            norm=nn.LayerNorm(MODEL_DIM),
            enable_nested_tensor=False,
        )
        self.slot_query = nn.Parameter(torch.empty(COMPOSITION_SLOT_COUNT, MODEL_DIM))
        nn.init.normal_(self.slot_query, mean=0.0, std=0.02)
        slot_layer = nn.TransformerDecoderLayer(
            MODEL_DIM,
            nhead=8,
            dim_feedforward=4 * MODEL_DIM,
            dropout=0.0,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.slot_decoder = nn.TransformerDecoder(
            slot_layer,
            num_layers=2,
            norm=nn.LayerNorm(MODEL_DIM),
        )
        self.endpoint_assignment = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.slot_assignment = nn.Linear(MODEL_DIM, MODEL_DIM, bias=False)
        self.assignment_temperature = nn.Parameter(torch.zeros(()))
        self.assignment_bias = nn.Parameter(torch.zeros(COMPOSITION_SLOT_COUNT))
        self.dustbin_head = nn.Sequential(
            nn.Linear(MODEL_DIM, MODEL_DIM // 2),
            nn.SiLU(),
            nn.Linear(MODEL_DIM // 2, 1),
        )
        self.slot_presence_head = nn.Sequential(
            nn.Linear(MODEL_DIM, MODEL_DIM // 2),
            nn.SiLU(),
            nn.Linear(MODEL_DIM // 2, 1),
        )

    @staticmethod
    def _validate_slot_permutation(permutation: torch.Tensor) -> None:
        if permutation.shape != (COMPOSITION_SLOT_COUNT,) or permutation.dtype != torch.long:
            raise ValueError("composition-slot permutation must be int64 [32]")
        expected = torch.arange(COMPOSITION_SLOT_COUNT, device=permutation.device)
        if not torch.equal(torch.sort(permutation).values, expected):
            raise ValueError("composition-slot permutation must be a bijection")

    def forward(
        self,
        primitive: ObservableSparsePortRelationPrediction,
        *,
        composition_slot_permutation: torch.Tensor | None = None,
    ) -> LocalCompositionSlotPrediction:
        features = composition_anchor_features(primitive).reshape(
            -1, ENDPOINT_COUNT, COMPOSITION_ANCHOR_FEATURE_DIM,
        )
        endpoint = self.endpoint_context(self.endpoint_embedding(features))
        query = self.slot_query
        if composition_slot_permutation is not None:
            self._validate_slot_permutation(composition_slot_permutation)
            query = query[composition_slot_permutation]
        query = query[None].expand(len(endpoint), -1, -1)
        slots = self.slot_decoder(query, endpoint)
        endpoint_key = F.normalize(self.endpoint_assignment(endpoint), dim=-1, eps=1e-8)
        slot_key = F.normalize(self.slot_assignment(slots), dim=-1, eps=1e-8)
        temperature = F.softplus(self.assignment_temperature) + 1e-4
        slot_logits = temperature * torch.einsum("bed,bkd->bek", endpoint_key, slot_key)
        bias = self.assignment_bias
        if composition_slot_permutation is not None:
            bias = bias[composition_slot_permutation]
        slot_logits = slot_logits + bias[None, None]
        dustbin = self.dustbin_head(endpoint)
        assignment_logits = torch.cat((slot_logits, dustbin), dim=-1)
        probability = torch.softmax(assignment_logits, dim=-1)
        entropy = -(probability * torch.log(probability.clamp_min(1e-12))).sum(dim=-1)
        uncertainty = entropy / math.log(ASSIGNMENT_CLASS_COUNT)
        result = LocalCompositionSlotPrediction(
            assignment_logits=assignment_logits,
            slot_presence_logits=self.slot_presence_head(slots).squeeze(-1),
            endpoint_uncertainty=uncertainty,
        )
        result.validate()
        return result


class FrozenObservableLocalCompositionSlotNet(nn.Module):
    """Attach the local composition head to a frozen observable backbone."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        for parameter in self.backbone.parameters():
            parameter.requires_grad_(False)
        self.backbone.eval()
        self.composition_head = LocalCompositionSlotHead()

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.composition_head.train(mode)
        return self

    def forward(
        self,
        range_valid: torch.Tensor,
        relative_translation_current_sensor_m: torch.Tensor,
        relative_yaw_current_sensor_deg: torch.Tensor,
        *,
        query_permutation: torch.Tensor | None = None,
        composition_slot_permutation: torch.Tensor | None = None,
    ) -> LocalCompositionSlotModelPrediction:
        with torch.no_grad():
            primitive = self.backbone(
                range_valid,
                relative_translation_current_sensor_m,
                relative_yaw_current_sensor_deg,
                query_permutation=query_permutation,
            )
        composition = self.composition_head(
            primitive,
            composition_slot_permutation=composition_slot_permutation,
        )
        return LocalCompositionSlotModelPrediction(
            primitive=primitive,
            composition=composition,
        )


def composition_slot_probabilities(
    prediction: LocalCompositionSlotPrediction,
) -> torch.Tensor:
    prediction.validate()
    return torch.softmax(prediction.assignment_logits, dim=-1)


def composition_slot_relation_probability(
    prediction: LocalCompositionSlotPrediction,
) -> torch.Tensor:
    """Return symmetric same-non-dustbin probabilities in O(EK)."""

    probability = composition_slot_probabilities(prediction)[..., :COMPOSITION_SLOT_COUNT]
    presence = torch.sigmoid(prediction.slot_presence_logits)[:, None]
    weighted = probability * presence
    relation = torch.einsum("bek,bfk->bef", weighted, weighted)
    relation = 0.5 * (relation + relation.transpose(1, 2))
    diagonal = torch.eye(ENDPOINT_COUNT, dtype=torch.bool, device=relation.device)[None]
    relation = relation.masked_fill(diagonal, 0.0)
    if not torch.equal(relation, relation.transpose(1, 2)) or not bool(
        torch.isfinite(relation).all()
    ):
        raise ValueError("composition-slot relation probability must be finite and symmetric")
    return relation


def local_composition_slot_safe_score(
    prediction: LocalCompositionSlotModelPrediction,
) -> torch.Tensor:
    """Require learned endpoint evidence and low assignment ambiguity."""

    relation = composition_slot_relation_probability(prediction.composition)
    evidence = torch.sigmoid(prediction.primitive.endpoint_evidence_logits).reshape(
        -1, ENDPOINT_COUNT,
    )
    confidence = (1.0 - prediction.composition.endpoint_uncertainty).clamp(0.0, 1.0)
    endpoint = evidence * confidence
    pair = endpoint[:, :, None] * endpoint[:, None, :]
    score = relation * pair
    score = 0.5 * (score + score.transpose(1, 2))
    if not torch.equal(score, score.transpose(1, 2)) or not bool(torch.isfinite(score).all()):
        raise ValueError("local composition-slot safe score must be finite and symmetric")
    return score


def decode_local_composition_slots(
    prediction: LocalCompositionSlotModelPrediction,
    *,
    assignment_probability_threshold: float,
    assignment_margin_threshold: float,
    endpoint_evidence_threshold: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Decode conservative endpoint labels and the induced attachment matrix.

    Thresholds are explicit arguments because they are frozen from validation
    only in the later training run.  This readiness implementation does not
    select or silently embed deployment thresholds.
    """

    for value in (
        assignment_probability_threshold,
        assignment_margin_threshold,
        endpoint_evidence_threshold,
    ):
        if value < 0.0 or value > 1.0:
            raise ValueError("composition-slot decode thresholds must lie in [0,1]")
    probability = composition_slot_probabilities(prediction.composition)
    slot_probability = probability[..., :COMPOSITION_SLOT_COUNT]
    top = torch.topk(slot_probability, k=2, dim=-1)
    best_probability, best_slot = top.values[..., 0], top.indices[..., 0]
    margin = best_probability - top.values[..., 1]
    dustbin = probability[..., DUSTBIN_INDEX]
    evidence = torch.sigmoid(prediction.primitive.endpoint_evidence_logits).reshape(
        -1, ENDPOINT_COUNT,
    )
    accepted = (
        (best_probability >= assignment_probability_threshold)
        & (margin >= assignment_margin_threshold)
        & (best_probability > dustbin)
        & (evidence >= endpoint_evidence_threshold)
    )
    labels = torch.where(accepted, best_slot, torch.full_like(best_slot, -1))
    attachment = (
        (labels[:, :, None] >= 0)
        & (labels[:, :, None] == labels[:, None, :])
    )
    diagonal = torch.eye(ENDPOINT_COUNT, dtype=torch.bool, device=labels.device)[None]
    attachment = attachment & ~diagonal
    return labels, attachment


__all__ = [
    "ASSIGNMENT_CLASS_COUNT",
    "COMPOSITION_SLOT_COUNT",
    "DUSTBIN_INDEX",
    "ENDPOINT_COUNT",
    "FrozenObservableLocalCompositionSlotNet",
    "LocalCompositionSlotHead",
    "LocalCompositionSlotModelPrediction",
    "LocalCompositionSlotPrediction",
    "composition_slot_probabilities",
    "composition_slot_relation_probability",
    "decode_local_composition_slots",
    "local_composition_slot_safe_score",
]
