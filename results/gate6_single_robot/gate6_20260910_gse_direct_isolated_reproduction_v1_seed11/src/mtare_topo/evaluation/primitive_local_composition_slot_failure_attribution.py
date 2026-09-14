"""Factorized diagnostics for a frozen local-composition-slot prediction.

The functions in this module do not define a replacement deployment decoder.
They expose the factors already present in the frozen model so a C07-only
audit can distinguish assignment failure from confidence-product collapse.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from mtare_topo.representation.primitive_local_composition_slot_decoding import (
    structured_same_slot_pair_score,
)
from mtare_topo.representation.primitive_local_composition_slot_model import (
    COMPOSITION_SLOT_COUNT,
    DUSTBIN_INDEX,
    ENDPOINT_COUNT,
    LocalCompositionSlotModelPrediction,
    composition_slot_probabilities,
    composition_slot_relation_probability,
    local_composition_slot_safe_score,
)


@dataclass(frozen=True)
class LocalSlotConfidenceFactors:
    best_slot: torch.Tensor
    best_probability: torch.Tensor
    second_probability: torch.Tensor
    dustbin_probability: torch.Tensor
    margin_over_second: torch.Tensor
    margin_over_dustbin: torch.Tensor
    joint_margin: torch.Tensor
    slot_presence: torch.Tensor
    primitive_existence: torch.Tensor
    endpoint_evidence: torch.Tensor
    certainty: torch.Tensor

    def endpoint_scores(self) -> dict[str, torch.Tensor]:
        full = (
            self.best_probability * self.joint_margin * self.slot_presence
            * self.primitive_existence * self.endpoint_evidence * self.certainty
        )
        return {
            "best_probability": self.best_probability,
            "margin_over_second": self.margin_over_second,
            "margin_over_dustbin": self.margin_over_dustbin,
            "joint_margin": self.joint_margin,
            "slot_presence": self.slot_presence,
            "primitive_existence": self.primitive_existence,
            "endpoint_evidence": self.endpoint_evidence,
            "certainty": self.certainty,
            "assignment_only": self.best_probability * self.joint_margin,
            "without_margin": (
                self.best_probability * self.slot_presence * self.primitive_existence
                * self.endpoint_evidence * self.certainty
            ),
            "learned_slot_only": (
                self.best_probability * self.joint_margin * self.slot_presence * self.certainty
            ),
            "frozen_endpoint_only": self.primitive_existence * self.endpoint_evidence,
            "structured_full": full,
        }


def factorize_local_slot_confidence(
    prediction: LocalCompositionSlotModelPrediction,
) -> LocalSlotConfidenceFactors:
    probability = composition_slot_probabilities(prediction.composition)
    slot_probability = probability[..., :COMPOSITION_SLOT_COUNT]
    top = torch.topk(slot_probability, k=2, dim=-1)
    best_probability, best_slot = top.values[..., 0], top.indices[..., 0]
    second_probability = top.values[..., 1]
    dustbin_probability = probability[..., DUSTBIN_INDEX]
    margin_over_second = (best_probability - second_probability).clamp_min(0.0)
    margin_over_dustbin = (best_probability - dustbin_probability).clamp_min(0.0)
    joint_margin = (best_probability - torch.maximum(
        second_probability, dustbin_probability,
    )).clamp_min(0.0)
    slot_presence = torch.sigmoid(prediction.composition.slot_presence_logits).gather(1, best_slot)
    primitive_existence = torch.sigmoid(
        prediction.primitive.existence_logits,
    ).repeat_interleave(2, dim=1)
    endpoint_evidence = torch.sigmoid(
        prediction.primitive.endpoint_evidence_logits,
    ).reshape(-1, ENDPOINT_COUNT)
    certainty = (1.0 - prediction.composition.endpoint_uncertainty).clamp(0.0, 1.0)
    result = LocalSlotConfidenceFactors(
        best_slot=best_slot,
        best_probability=best_probability,
        second_probability=second_probability,
        dustbin_probability=dustbin_probability,
        margin_over_second=margin_over_second,
        margin_over_dustbin=margin_over_dustbin,
        joint_margin=joint_margin,
        slot_presence=slot_presence,
        primitive_existence=primitive_existence,
        endpoint_evidence=endpoint_evidence,
        certainty=certainty,
    )
    for name, value in result.endpoint_scores().items():
        if tuple(value.shape) != tuple(best_slot.shape) or not bool(torch.isfinite(value).all()):
            raise ValueError(f"invalid local-slot confidence factor: {name}")
        if bool((value < 0.0).any()) or bool((value > 1.0 + 1e-6).any()):
            raise ValueError(f"local-slot confidence factor outside [0,1]: {name}")
    return result


def _symmetric_pair_min(value: torch.Tensor, same_slot: torch.Tensor) -> torch.Tensor:
    score = torch.minimum(value[:, :, None], value[:, None, :])
    score = torch.where(same_slot, score, torch.zeros_like(score))
    diagonal = torch.eye(ENDPOINT_COUNT, dtype=torch.bool, device=value.device)[None]
    score = score.masked_fill(diagonal, 0.0)
    score = 0.5 * (score + score.transpose(1, 2))
    return score


def local_slot_diagnostic_pair_scores(
    prediction: LocalCompositionSlotModelPrediction,
) -> dict[str, torch.Tensor]:
    """Return frozen factor ablations; none is a selected deployment formula."""

    factors = factorize_local_slot_confidence(prediction)
    same = factors.best_slot[:, :, None] == factors.best_slot[:, None, :]
    scores = {
        "hard_slot_identity": _symmetric_pair_min(torch.ones_like(factors.best_probability), same),
    }
    for name, value in factors.endpoint_scores().items():
        scores[f"hard_slot_{name}"] = _symmetric_pair_min(value, same)

    probability = composition_slot_probabilities(prediction.composition)[..., :COMPOSITION_SLOT_COUNT]
    soft_affinity = torch.einsum("bek,bfk->bef", probability, probability)
    soft_affinity = 0.5 * (soft_affinity + soft_affinity.transpose(1, 2))
    diagonal = torch.eye(ENDPOINT_COUNT, dtype=torch.bool, device=soft_affinity.device)[None]
    scores["soft_slot_affinity_no_presence"] = soft_affinity.masked_fill(diagonal, 0.0)
    scores["soft_slot_relation_with_presence"] = composition_slot_relation_probability(
        prediction.composition,
    )
    scores["soft_slot_safe_legacy"] = local_composition_slot_safe_score(prediction)
    scores["structured_deployed"] = structured_same_slot_pair_score(prediction)
    for name, score in scores.items():
        if tuple(score.shape) != (len(factors.best_slot), ENDPOINT_COUNT, ENDPOINT_COUNT):
            raise ValueError(f"invalid diagnostic pair-score shape: {name}")
        if not bool(torch.isfinite(score).all()) or not torch.equal(score, score.transpose(1, 2)):
            raise ValueError(f"diagnostic pair score is not finite/exact symmetric: {name}")
    if not torch.equal(scores["hard_slot_structured_full"], scores["structured_deployed"]):
        raise ValueError("factorized full score does not reproduce deployed score")
    return scores


__all__ = [
    "LocalSlotConfidenceFactors",
    "factorize_local_slot_confidence",
    "local_slot_diagnostic_pair_scores",
]
