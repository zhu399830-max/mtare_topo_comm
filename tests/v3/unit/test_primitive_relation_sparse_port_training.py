from __future__ import annotations

import pytest
import torch

from mtare_topo.representation.primitive_relation_sparse_port_training import (
    staged_training_loss,
    training_stage,
)


def families() -> dict[str, torch.Tensor]:
    result = {
        "primitive_set_parameters": torch.tensor(1.0),
        "surface_reconstruction": torch.tensor(2.0),
        "ray_free_space": torch.tensor(3.0),
        "port_relations": torch.tensor(4.0),
        "temporal_equivariance": torch.tensor(5.0),
        "uncertainty_calibration": torch.tensor(6.0),
    }
    result["total"] = torch.stack(tuple(result.values())).mean()
    return result


def test_frozen_seven_epoch_schedule_restores_all_five_stages() -> None:
    assert [training_stage(epoch) for epoch in range(7)] == [
        "geometry_pretraining", "sparse_set_mdl_pretraining",
        "endpoint_relation_pretraining", "temporal_uncertainty_pretraining",
        "joint_finetuning", "joint_finetuning", "joint_finetuning",
    ]
    with pytest.raises(ValueError):
        training_stage(7)


def test_sparse_stage_stays_inside_registered_six_families() -> None:
    value = families()
    assert float(staged_training_loss(value, epoch=0)) == 2.0
    assert float(staged_training_loss(value, epoch=1)) == 1.0
    assert float(staged_training_loss(value, epoch=2)) == 4.0
    assert float(staged_training_loss(value, epoch=3)) == 5.5
    assert torch.equal(staged_training_loss(value, epoch=4), value["total"])


def test_schedule_rejects_loss_family_drift() -> None:
    value = families(); value["seventh_unregistered_loss"] = torch.tensor(0.0)
    with pytest.raises(ValueError, match="inventory"):
        staged_training_loss(value, epoch=0)
