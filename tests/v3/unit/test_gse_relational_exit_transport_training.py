from __future__ import annotations

import torch

from evaluate_gse_relational_exit_transport_selection_v1 import augment_summary
from train_gse_relational_exit_transport_event_v1 import _decision_probability


def _summary(macro: float, *, base_pass: bool = True) -> dict:
    return {
        "status": "PASS_GSE_ACTION_SET_NODE_SELECTION_V1" if base_pass else "FAIL_GSE_ACTION_SET_NODE_SELECTION_V1",
        "ensemble": {"decision_episode_macro_f1": macro},
        "seed_at_ensemble_threshold": {
            str(seed): {"decision_trigger_precision": 0.99, "decision_episode_recall": 0.5}
            for seed in range(3)
        },
        "gates": {"safety": base_pass},
    }


def test_relational_gain_gate_requires_five_points() -> None:
    assert augment_summary(_summary(0.81), 0.74)["scientific_pass"] is True
    assert augment_summary(_summary(0.78), 0.74)["scientific_pass"] is False


def test_relational_gain_cannot_override_safety_failure() -> None:
    assert augment_summary(_summary(0.95, base_pass=False), 0.74)["scientific_pass"] is False


def test_commit_weighted_event_probability_is_a_simplex() -> None:
    event = torch.tensor([[0.2, 0.3, 0.5], [0.9, 0.05, 0.05]])
    commit = torch.tensor([0.8, 0.25])
    probability = _decision_probability({
        "event_probability": event,
        "commit_probability": commit,
    })
    assert torch.allclose(probability.sum(dim=1), torch.ones(2))
    assert torch.allclose(probability[:, 1:], commit[:, None] * event[:, 1:])
    assert bool((probability >= 0.0).all())
