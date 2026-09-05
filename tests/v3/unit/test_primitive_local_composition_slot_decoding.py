from dataclasses import replace

import torch

from mtare_topo.representation.primitive_local_composition_slot_decoding import (
    decode_structured_local_composition,
    structured_endpoint_slot_confidence,
    structured_same_slot_pair_score,
)
from mtare_topo.representation.primitive_local_composition_slot_model import (
    LocalCompositionSlotModelPrediction,
    LocalCompositionSlotPrediction,
)
from tests.v3.unit.test_primitive_local_composition_slot_model import _primitive


def _prediction() -> LocalCompositionSlotModelPrediction:
    logits = torch.full((1, 64, 33), -8.0)
    logits[..., 32] = 0.0
    logits[0, 0, 2] = logits[0, 1, 2] = logits[0, 2, 2] = 10.0
    logits[0, 3, 7] = logits[0, 4, 7] = 10.0
    probability = torch.softmax(logits, dim=-1)
    uncertainty = -(probability * torch.log(probability.clamp_min(1e-12))).sum(-1) / torch.log(
        torch.tensor(33.0)
    )
    primitive = replace(
        _primitive(),
        existence_logits=torch.ones(1, 32) * 8.0,
        endpoint_evidence_logits=torch.ones(1, 32, 2) * 8.0,
    )
    composition = LocalCompositionSlotPrediction(
        assignment_logits=logits,
        slot_presence_logits=torch.ones(1, 32) * 8.0,
        endpoint_uncertainty=uncertainty,
    )
    return LocalCompositionSlotModelPrediction(primitive, composition)


def test_one_threshold_decode_is_cluster_transitive() -> None:
    prediction = _prediction()
    labels, attachment = decode_structured_local_composition(
        prediction, confidence_threshold=0.5,
    )
    assert labels[0, :5].tolist() == [2, 2, 2, 7, 7]
    assert attachment[0, 0, 1] and attachment[0, 1, 2] and attachment[0, 0, 2]
    assert attachment[0, 3, 4]
    assert not attachment[0, 2, 3]


def test_pair_threshold_matches_endpoint_acceptance() -> None:
    prediction = _prediction()
    _, confidence = structured_endpoint_slot_confidence(prediction)
    pair = structured_same_slot_pair_score(prediction)
    threshold = 0.5
    labels, decoded = decode_structured_local_composition(
        prediction, confidence_threshold=threshold,
    )
    assert torch.equal(pair >= threshold, decoded)
    assert torch.all(confidence[labels < 0] < threshold)


def test_low_existence_or_evidence_rejects_endpoint() -> None:
    prediction = _prediction()
    primitive = replace(
        prediction.primitive,
        existence_logits=prediction.primitive.existence_logits.clone(),
        endpoint_evidence_logits=prediction.primitive.endpoint_evidence_logits.clone(),
    )
    primitive.existence_logits[0, 0] = -20.0
    primitive.endpoint_evidence_logits[0, 1, 0] = -20.0
    changed = LocalCompositionSlotModelPrediction(primitive, prediction.composition)
    labels, _ = decode_structured_local_composition(
        changed, confidence_threshold=0.5,
    )
    assert labels[0, 0] == labels[0, 1] == labels[0, 2] == -1
    assert labels[0, 3] == 7


def test_structured_pair_score_is_exactly_symmetric() -> None:
    score = structured_same_slot_pair_score(_prediction())
    assert torch.equal(score, score.transpose(1, 2))
    assert torch.equal(torch.diagonal(score, dim1=1, dim2=2), torch.zeros(1, 64))
