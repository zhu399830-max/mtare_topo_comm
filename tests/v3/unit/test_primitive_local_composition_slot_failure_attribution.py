from dataclasses import replace

import torch

from mtare_topo.evaluation.primitive_local_composition_slot_failure_attribution import (
    factorize_local_slot_confidence,
    local_slot_diagnostic_pair_scores,
)
from tests.v3.unit.test_primitive_local_composition_slot_decoding import _prediction


def test_factorization_exactly_reproduces_deployed_structured_score() -> None:
    prediction = _prediction()
    factors = factorize_local_slot_confidence(prediction)
    scores = local_slot_diagnostic_pair_scores(prediction)
    assert torch.equal(scores["hard_slot_structured_full"], scores["structured_deployed"])
    assert factors.joint_margin.shape == (1, 64)
    assert all(torch.isfinite(value).all() for value in factors.endpoint_scores().values())


def test_dustbin_dominance_isolated_from_second_slot_margin() -> None:
    prediction = _prediction()
    logits = prediction.composition.assignment_logits.clone()
    logits[0, 0, 32] = 20.0
    changed = replace(
        prediction,
        composition=replace(prediction.composition, assignment_logits=logits),
    )
    factors = factorize_local_slot_confidence(changed)
    assert factors.margin_over_second[0, 0] > 0
    assert factors.margin_over_dustbin[0, 0] == 0
    assert factors.joint_margin[0, 0] == 0


def test_soft_affinity_can_measure_fragmentation_without_changing_hard_decoder() -> None:
    scores = local_slot_diagnostic_pair_scores(_prediction())
    assert scores["hard_slot_identity"][0, 0, 1] == 1
    assert scores["hard_slot_identity"][0, 0, 3] == 0
    assert scores["soft_slot_affinity_no_presence"][0, 0, 1] > scores[
        "soft_slot_affinity_no_presence"
    ][0, 0, 3]


def test_all_pair_diagnostics_are_exactly_symmetric_with_zero_diagonal() -> None:
    for score in local_slot_diagnostic_pair_scores(_prediction()).values():
        assert torch.equal(score, score.transpose(1, 2))
        assert torch.equal(torch.diagonal(score, dim1=1, dim2=2), torch.zeros(1, 64))
