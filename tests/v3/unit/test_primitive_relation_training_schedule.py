from __future__ import annotations

import pytest
import torch

from mtare_topo.representation.primitive_relation_training import (
    staged_training_loss,
    training_stage,
)


def _families() -> dict[str, torch.Tensor]:
    values = {
        "primitive_set_parameters": 1.0,
        "surface_reconstruction": 2.0,
        "ray_free_space": 3.0,
        "port_relations": 4.0,
        "temporal_equivariance": 5.0,
        "uncertainty_calibration": 6.0,
    }
    result = {name: torch.tensor(value) for name, value in values.items()}
    result["total"] = torch.stack(tuple(result.values())).mean()
    return result


def test_frozen_six_epoch_schedule() -> None:
    assert [training_stage(index) for index in range(6)] == [
        "geometry_pretraining", "relation_pretraining",
        "temporal_uncertainty_pretraining", "joint_finetuning",
        "joint_finetuning", "joint_finetuning",
    ]
    with pytest.raises(ValueError):
        training_stage(6)


def test_stage_objectives_do_not_add_new_loss_families() -> None:
    families = _families()
    assert float(staged_training_loss(families, epoch=0)) == 2.0
    assert float(staged_training_loss(families, epoch=1)) == 4.0
    assert float(staged_training_loss(families, epoch=2)) == 5.5
    assert torch.equal(staged_training_loss(families, epoch=3), families["total"])

