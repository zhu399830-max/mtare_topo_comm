from __future__ import annotations

import numpy as np
import torch

from train_gse_structured_exact_one_event_v1 import _decision_probability, exact_one_nll


def test_numpy_exact_one_nll_prefers_one_peak() -> None:
    target = np.asarray([0, 1, 1, 1])
    episode = np.asarray([-1, 0, 0, 0])
    one = np.asarray([[.99, .005, .005], [.01, .98, .01], [.99, .005, .005], [.99, .005, .005]])
    two = one.copy(); two[2] = [.01, .98, .01]
    assert exact_one_nll(one, target, episode)["total"] < exact_one_nll(two, target, episode)["total"]


def test_decision_probability_matches_deployment_simplex() -> None:
    event = torch.tensor([[.2, .3, .5], [.9, .05, .05]])
    commit = torch.tensor([.8, .25])
    probability = _decision_probability({"event_probability": event, "commit_probability": commit})
    assert torch.allclose(probability.sum(dim=1), torch.ones(2))
    assert torch.allclose(probability[:, 1:], commit[:, None] * event[:, 1:])
