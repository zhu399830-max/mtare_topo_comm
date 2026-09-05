from __future__ import annotations

import numpy as np
import torch

from mtare_topo.representation.gse_rare_event_corrective import (
    GSERareEventCorrective,
    identity_class_balanced_weights,
)


def test_zero_initialized_corrective_reproduces_baseline_logits() -> None:
    model = GSERareEventCorrective()
    features = torch.zeros((3, 292))
    probability = torch.asarray(
        [[0.2, 0.3, 0.1, 0.15, 0.25]] * 3, dtype=torch.float32
    )
    assert torch.allclose(model(features, probability), probability.log())
    loss = torch.nn.functional.cross_entropy(model(features, probability), torch.tensor([0, 1, 4]))
    loss.backward()
    assert all(parameter.grad is not None and torch.all(torch.isfinite(parameter.grad)) for parameter in model.parameters())


def test_identity_class_balancing_equalizes_classes_and_identities() -> None:
    event = np.asarray([0, 0, 1, 1, 1, 1, 2, 3, 4])
    identity = np.asarray([-1, -1, 10, 10, 10, 11, 20, 30, 40])
    weight = identity_class_balanced_weights(event, identity)
    for class_index in range(5):
        assert np.isclose(weight[event == class_index].sum(), 0.2)
    assert np.isclose(weight[(identity == 10)].sum(), weight[(identity == 11)].sum())
