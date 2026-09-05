from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from mtare_topo.evaluation.primitive_relation_observable_failure_attribution import (
    diagnose_observable_relation,
    observable_attachment_score_slice,
    observable_endpoint_evidence_slice,
)


def _fixture():
    attachment = torch.zeros(1, 32, 2, 32, 2, dtype=torch.bool)
    attachment[0, 0, 1, 1, 0] = True
    overlap = torch.zeros(1, 32, 32, dtype=torch.bool)
    mask = torch.zeros(1, 32, dtype=torch.bool); mask[0, :3] = True
    observed = torch.zeros(1, 32, 2, dtype=torch.bool)
    observed[0, 0, 1] = True; observed[0, 1, 0] = True
    aligned = {
        "attachment": attachment,
        "overlap": overlap,
        "mask": mask,
        "endpoint_observed": observed,
    }
    logits = torch.full((1, 32, 2, 32, 2), -4.0)
    logits[0, 0, 1, 1, 0] = 4.0
    prediction = SimpleNamespace(
        endpoint_attachment_logits=logits,
        endpoint_attachment_uncertainty=torch.zeros_like(logits),
        endpoint_evidence_logits=torch.zeros(1, 32, 2),
    )
    return prediction, aligned, mask


def test_observable_slice_keeps_only_visible_matched_target() -> None:
    prediction, aligned, mask = _fixture()
    result = observable_attachment_score_slice(prediction, aligned, mask)
    assert result.all_observable_target_pairs == 1
    assert result.eligible_true_pairs == 1
    # Matched pairs with either hidden endpoint are unknown and excluded.
    assert result.eligible_pairs == 1
    assert set(result.scores) == {"raw", "uncertainty_adjusted", "learned_safe"}
    assert np.max(result.scores["raw"]) > 0.98
    assert np.max(result.scores["learned_safe"]) > 0.24


def test_observable_proposal_oracle_removes_redundant_slot_pairs() -> None:
    prediction, aligned, mask = _fixture()
    deployed = mask.clone(); deployed[0, 3] = True
    oracle = observable_attachment_score_slice(prediction, aligned, mask)
    actual = observable_attachment_score_slice(prediction, aligned, deployed)
    assert oracle.eligible_pairs == 1
    assert actual.eligible_pairs > oracle.eligible_pairs
    assert actual.all_observable_target_pairs == oracle.all_observable_target_pairs == 1


def test_endpoint_evidence_matched_only_population() -> None:
    prediction, aligned, _ = _fixture()
    all_score, all_target, total = observable_endpoint_evidence_slice(
        prediction, aligned, matched_only=False,
    )
    matched_score, matched_target, matched_total = observable_endpoint_evidence_slice(
        prediction, aligned, matched_only=True,
    )
    assert len(all_score) == 64 and len(matched_score) == 6
    assert int(all_target.sum()) == int(matched_target.sum()) == total == matched_total == 2


def _counts(**overrides: int) -> dict[str, int]:
    result = {
        "deployed__independent__raw": 0,
        "deployed__independent__uncertainty_adjusted": 0,
        "deployed__independent__learned_safe": 0,
        "deployed__best_link_union__raw": 0,
        "teacher_cardinality__best_link_union__raw": 0,
        "proposal_oracle__independent__raw": 0,
        "proposal_oracle__independent__uncertainty_adjusted": 0,
        "proposal_oracle__independent__learned_safe": 0,
        "proposal_oracle__best_link_union__raw": 0,
    }
    result.update(overrides)
    return result


def test_diagnosis_resolves_endpoint_evidence_suppression() -> None:
    diagnosis, decision = diagnose_observable_relation(_counts(**{
        "proposal_oracle__independent__raw": 2,
        "proposal_oracle__independent__uncertainty_adjusted": 2,
    }))
    assert diagnosis == "ENDPOINT_EVIDENCE_SUPPRESSES_ORACLE_SAFE_RELATIONS"
    assert decision == "ALLOW_ENDPOINT_EVIDENCE_CALIBRATION_READINESS"


def test_diagnosis_stops_relation_head_when_oracle_fails() -> None:
    diagnosis, decision = diagnose_observable_relation(_counts())
    assert diagnosis == "RELATION_SCORE_FAILS_EVEN_WITH_OBSERVABLE_PROPOSAL_ORACLE"
    assert decision == "STOP_DIRECT_PAIR_RELATION_HEAD_AND_REASSESS_ARCHITECTURE"
