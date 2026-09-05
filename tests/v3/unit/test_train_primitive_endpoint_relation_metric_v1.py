from pathlib import Path

import torch

from train_primitive_endpoint_relation_metric_v1 import (
    HEAD_PARAMETERS,
    HEAD_TENSORS,
    LOSS_NAMES,
    TOTAL_PARAMETERS,
    save_checkpoint,
)


def test_endpoint_metric_training_boundary_is_frozen() -> None:
    assert HEAD_PARAMETERS == 426_818
    assert HEAD_TENSORS == 34
    assert TOTAL_PARAMETERS == 3_062_449
    assert LOSS_NAMES[-1] == "total"


def test_checkpoint_contract_has_no_arbitrary_slots_or_dustbin(tmp_path: Path) -> None:
    model = torch.nn.Linear(2, 2); optimizer = torch.optim.AdamW(model.parameters())
    source = tmp_path / "source.pt"; torch.save({"source": True}, source)
    target = tmp_path / "metric.pt"
    save_checkpoint(
        target, model=model, optimizer=optimizer, seed=1, epoch=2,
        source_checkpoint=source, training_metrics={}, selection_metrics={},
    )
    payload = torch.load(target, weights_only=False)
    contract = payload["training_contract"]
    assert payload["schema_version"] == "primitive_endpoint_relation_metric_checkpoint_v1"
    assert contract["arbitrary_slots"] is False
    assert contract["dustbin_class"] is False
    assert contract["selection_metric"] == "mean_C07_endpoint_relation_metric_total_loss"


def test_training_uses_all_four_relation_objectives() -> None:
    assert set(LOSS_NAMES[:-1]) == {
        "balanced_pair_logistic", "same_composition_compactness",
        "different_composition_separation", "disconnected_overlap_rejection",
    }
