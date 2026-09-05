import torch

from mtare_topo.representation.primitive_relation_observable_initialization import (
    initialize_observable_relation_from_state,
    is_relation_trainable_parameter,
    set_observable_relation_training_mode,
)
from mtare_topo.representation.primitive_relation_sparse_port_model import SparsePortRelationNet


def _checkpoint(seed: int = 3) -> dict:
    torch.manual_seed(19)
    return {
        "schema_version": "primitive_relation_sparse_port_checkpoint_v1",
        "seed": seed,
        "epoch": 4,
        "model_state_dict": SparsePortRelationNet().state_dict(),
    }


def test_geometry_transfer_is_exact_and_only_relations_train() -> None:
    checkpoint = _checkpoint()
    model, report = initialize_observable_relation_from_state(checkpoint, initialization_seed=7)
    source = checkpoint["model_state_dict"]
    for name, parameter in model.named_parameters():
        assert parameter.requires_grad == is_relation_trainable_parameter(name)
        if not is_relation_trainable_parameter(name):
            assert torch.equal(parameter.detach(), source[name])
    assert report.checkpoint_seed == 3 and report.checkpoint_epoch == 4
    assert report.transferred_tensors > 100
    assert report.reset_relation_tensors > 20
    assert 0 < report.trainable_parameters < report.total_parameters
    assert report.frozen_parameters + report.trainable_parameters == report.total_parameters


def test_relation_reset_is_deterministic_and_seed_sensitive() -> None:
    checkpoint = _checkpoint()
    first, _ = initialize_observable_relation_from_state(checkpoint, initialization_seed=11)
    second, _ = initialize_observable_relation_from_state(checkpoint, initialization_seed=11)
    third, _ = initialize_observable_relation_from_state(checkpoint, initialization_seed=12)
    relation_names = [name for name in first.state_dict() if is_relation_trainable_parameter(name)]
    assert all(torch.equal(first.state_dict()[name], second.state_dict()[name]) for name in relation_names)
    assert any(not torch.equal(first.state_dict()[name], third.state_dict()[name]) for name in relation_names)


def test_relation_training_mode_keeps_transferred_modules_in_eval() -> None:
    model, _ = initialize_observable_relation_from_state(_checkpoint(), initialization_seed=13)
    set_observable_relation_training_mode(model)
    for name, module in model.named_modules():
        if not name:
            assert module.training is False
        elif is_relation_trainable_parameter(name + "."):
            assert module.training is True
