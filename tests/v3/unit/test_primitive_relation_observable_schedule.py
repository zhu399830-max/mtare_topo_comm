import pytest
import torch

from mtare_topo.representation.primitive_relation_observable_schedule import (
    TRAINING_EPOCHS,
    relation_training_objective,
    training_stage,
)


def _families() -> dict[str, torch.Tensor]:
    values = {
        "primitive_set_parameters": torch.tensor(1.0),
        "surface_reconstruction": torch.tensor(2.0),
        "ray_free_space": torch.tensor(3.0),
        "port_relations": torch.tensor(4.0),
        "temporal_equivariance": torch.tensor(5.0),
        "uncertainty_calibration": torch.tensor(6.0),
    }
    values["total"] = torch.stack(tuple(values.values())).mean()
    return values


def test_three_epoch_relation_only_schedule_is_fixed() -> None:
    assert TRAINING_EPOCHS == 3
    assert [training_stage(epoch) for epoch in range(3)] == [
        "observable_relation_finetuning",
        "observable_relation_finetuning",
        "observable_relation_finetuning",
    ]
    with pytest.raises(ValueError):
        training_stage(3)


def test_relation_objective_uses_only_relation_and_uncertainty_families() -> None:
    values = _families()
    assert float(relation_training_objective(values)) == 5.0
    values["primitive_set_parameters"] = torch.tensor(1000.0)
    assert float(relation_training_objective(values)) == 5.0


def test_relation_objective_rejects_family_drift() -> None:
    values = _families(); values["extra"] = torch.tensor(0.0)
    with pytest.raises(ValueError, match="inventory"):
        relation_training_objective(values)
