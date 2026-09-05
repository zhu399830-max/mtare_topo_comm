from __future__ import annotations

import numpy as np
import torch

from mtare_topo.semantics.gse_composer_learning import (
    fractional_backprojection_distribution,
    fuse_factorized_event_logits,
    identity_coverage,
    macro_f1,
    structural_acceptance_metrics,
)


def test_factorized_fusion_preserves_physical_exclusivity() -> None:
    action=torch.tensor([[5.0,1.0,0.0],[0.0,5.0,1.0],[0.0,0.0,5.0]])
    metric=torch.tensor([[5.0,0.0,0.0],[5.0,0.0,0.0],[0.0,5.0,1.0]])
    fused=fuse_factorized_event_logits(action,metric)
    assert fused.argmax(-1).tolist()==[0,1,2]
    torch.testing.assert_close(torch.exp(fused).sum(-1),torch.ones(3))


def test_fractional_backprojection_preserves_physical_delay() -> None:
    steps=torch.tensor([2.5,0.0,4.0,3.25]);valid=torch.tensor([True,True,True,False])
    target=fractional_backprojection_distribution(steps,valid)
    torch.testing.assert_close(target[0],torch.tensor([0.0,0.5,0.5,0.0,0.0]))
    torch.testing.assert_close(target[1],torch.tensor([0.0,0.0,0.0,0.0,1.0]))
    torch.testing.assert_close(target[2],torch.tensor([1.0,0.0,0.0,0.0,0.0]))
    assert target[3].sum()==0


def test_selective_and_identity_metrics_use_objective_denominators() -> None:
    truth=np.array([1,1,2,3,4,0]);pred=np.array([1,0,2,3,1,1]);score=np.array([.9,.1,.8,.7,.95,.99])
    metrics=structural_acceptance_metrics(truth,pred,score,.7)
    assert metrics["true_positive"]==3 and metrics["false_positive"]==2
    assert metrics["recall"]==3/5
    coverage=identity_coverage(truth,pred,score,.7,["j0","j1","t0","u0","g0",""])
    assert coverage["junction"]["covered"]==1 and coverage["junction"]["total"]==2
    result=macro_f1(truth,pred)
    assert 0.0 < result["macro_f1"] < 1.0
