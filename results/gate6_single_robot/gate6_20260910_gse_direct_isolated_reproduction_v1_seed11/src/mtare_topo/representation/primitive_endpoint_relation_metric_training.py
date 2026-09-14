"""Teacher-only contrastive losses for the no-slot endpoint relation metric."""

from __future__ import annotations

import torch
from torch.nn import functional as F

from mtare_topo.representation.primitive_endpoint_relation_metric_model import (
    EndpointRelationMetricPrediction,
)
from mtare_topo.representation.primitive_local_composition_slot_model import ENDPOINT_COUNT
from mtare_topo.representation.primitive_local_composition_slot_training import (
    LocalCompositionSlotTargets,
)


def _balanced_logistic(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    terms = []
    if bool(target.any()): terms.append(F.softplus(-logits[target]).mean())
    if bool((~target).any()): terms.append(F.softplus(logits[~target]).mean())
    if not terms: raise ValueError("endpoint relation metric has no supervised pairs")
    return torch.stack(terms).mean()


def endpoint_relation_metric_losses(
    prediction: EndpointRelationMetricPrediction,
    targets: LocalCompositionSlotTargets,
) -> dict[str, torch.Tensor]:
    """Attract same-composition endpoints and repel every observed alternative."""

    prediction.validate(); targets.validate()
    if len(prediction.embedding) != len(targets.labels):
        raise ValueError("endpoint relation metric prediction/target batch mismatch")
    device = prediction.embedding.device
    endpoint = torch.arange(ENDPOINT_COUNT, device=device)
    cross_primitive = endpoint[:, None] // 2 != endpoint[None, :] // 2
    upper = torch.triu(torch.ones(ENDPOINT_COUNT, ENDPOINT_COUNT, dtype=torch.bool, device=device), 1)
    observed = targets.endpoint_supervised
    eligible = observed[:, :, None] & observed[:, None, :] & cross_primitive[None] & upper[None]
    same = (
        (targets.labels[:, :, None] >= 0)
        & (targets.labels[:, :, None] == targets.labels[:, None, :])
        & eligible
    )
    different = eligible & ~same
    logistic = _balanced_logistic(prediction.pair_logits[eligible], same[eligible])
    compactness = (1.0 - prediction.cosine_similarity[same]).mean() if bool(same.any()) else logistic * 0.0
    separation = F.softplus(prediction.cosine_similarity[different]).mean() if bool(different.any()) else logistic * 0.0
    overlap = targets.disconnected_overlap & eligible
    overlap_rejection = F.softplus(prediction.pair_logits[overlap]).mean() if bool(overlap.any()) else logistic * 0.0
    components = (logistic, compactness, separation, overlap_rejection)
    total = torch.stack(components).mean()
    return {
        "balanced_pair_logistic": logistic,
        "same_composition_compactness": compactness,
        "different_composition_separation": separation,
        "disconnected_overlap_rejection": overlap_rejection,
        "total": total,
    }


__all__ = ["endpoint_relation_metric_losses"]
