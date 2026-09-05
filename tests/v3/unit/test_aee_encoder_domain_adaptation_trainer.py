"""CPU contracts for full AEE encoder-domain adaptation."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[3]
TOOLS = ROOT / "tools/v3"
if str(TOOLS) not in sys.path:
    sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "train_aee_encoder_domain_adaptation_v2",
    TOOLS / "train_aee_encoder_domain_adaptation_v2.py",
)
assert spec is not None and spec.loader is not None
trainer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(trainer)


def _sample(frame_id: str, domain: str, invalid_period: int | None = None):
    student = np.full((2, 16, 720), 0.2, dtype=np.float32)
    student[1] = 1.0
    if invalid_period is not None:
        student[1, :, ::invalid_period] = 0.0
        student[0, :, ::invalid_period] = 1.0
    target = np.zeros(720, dtype=np.float32)
    target[0] = 1.0
    return {
        "student": student,
        "direction_target": target,
        "count_target": np.int64(0),
        "role_target": np.int64(0),
        "frame_id": frame_id,
        "headings_robot_deg": (0.0,),
        "domain": domain,
    }


def test_full_model_contract_trains_encoder_embedding_and_all_heads() -> None:
    model = trainer.StructuralSemanticNet()
    contract = trainer.make_full_model_trainable(model)
    assert contract["frozen_parameters"] == []
    names = contract["trainable_parameters"]
    assert all(parameter.requires_grad for parameter in model.parameters())
    for prefix in trainer.MODEL_PREFIXES:
        assert any(name.startswith(prefix) for name in names)


def test_collator_expands_views_with_balanced_effective_weights() -> None:
    samples = [
        _sample("cano:0", "cano"),
        _sample("aee:0", "aee", 2),
        _sample("cano:1", "cano"),
        _sample("aee:1", "aee", 3),
    ]
    collator = trainer.DomainAdaptationCollator(seed=2)
    with pytest.raises(RuntimeError, match="set_epoch"):
        collator(samples)
    collator.set_epoch(1)
    first = collator(samples)
    second = collator(samples)
    assert first["student"].shape == (6, 2, 16, 720)
    assert first["domain"].count("cano_dense") == 2
    assert first["domain"].count("cano_mask_matched") == 2
    assert first["domain"].count("aee") == 2
    assert first["loss_weight"].sum().item() == pytest.approx(4.0)
    assert first["mask_pairing"] == second["mask_pairing"]
    assert torch.equal(first["student"], second["student"])


def test_weighted_loss_gives_two_cano_views_one_independent_sample_weight() -> None:
    outputs = {
        "direction_logits": torch.zeros((3, 720)),
        "count_logits": torch.zeros((3, 6)),
        "role_logits": torch.zeros((3, 3)),
    }
    direction = torch.zeros((3, 720))
    count = torch.zeros(3, dtype=torch.long)
    role = torch.zeros(3, dtype=torch.long)
    result = trainer.weighted_multitask_loss(
        outputs, direction, count, role, torch.tensor([0.5, 0.5, 1.0])
    )
    assert result["direction"].item() == pytest.approx(np.log(2.0), rel=1e-6)
    assert torch.isfinite(result["total"])


def test_weighted_role_loss_preserves_cross_entropy_class_weight_normalization() -> None:
    torch.manual_seed(3)
    outputs = {
        "direction_logits": torch.randn((4, 720), requires_grad=True),
        "count_logits": torch.randn((4, 6), requires_grad=True),
        "role_logits": torch.randn((4, 3), requires_grad=True),
    }
    direction = torch.zeros((4, 720))
    count = torch.tensor([0, 1, 2, 3])
    role = torch.tensor([0, 1, 1, 2])
    role_weights = torch.tensor([0.5, 2.0, 4.0])
    result = trainer.weighted_multitask_loss(
        outputs, direction, count, role, torch.ones(4), role_weights
    )
    expected = torch.nn.functional.cross_entropy(
        outputs["role_logits"], role, weight=role_weights
    )
    assert result["role"].item() == pytest.approx(expected.item(), rel=1e-7)


def test_one_optimizer_step_propagates_into_every_declared_model_group() -> None:
    torch.manual_seed(4)
    model = trainer.StructuralSemanticNet()
    trainer.make_full_model_trainable(model)
    before = {
        prefix: trainer.selected_state_hashes(model, (prefix,))
        for prefix in trainer.MODEL_PREFIXES
    }
    student = torch.rand((2, 2, 16, 720))
    outputs = model(student)
    loss = trainer.weighted_multitask_loss(
        outputs,
        torch.zeros((2, 720)),
        torch.tensor([0, 1]),
        torch.tensor([0, 2]),
        torch.ones(2),
    )["total"]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    for prefix in trainer.MODEL_PREFIXES:
        parameters = [parameter for name, parameter in model.named_parameters() if name.startswith(prefix)]
        assert parameters and any(parameter.grad is not None for parameter in parameters)
    optimizer.step()
    for prefix in trainer.MODEL_PREFIXES:
        assert before[prefix] != trainer.selected_state_hashes(model, (prefix,))


def test_count_gate_uses_1_to_4_and_keeps_5_to_6_diagnostic_only() -> None:
    metrics = {
        "count": {
            "f1": [0.8, 0.7, 0.6, 0.9, 0.0, 0.0],
            "support": [10, 10, 10, 10, 3, 2],
        }
    }
    result = trainer.attach_count_contract(metrics)["count"]
    assert result["macro_f1_count_1_to_4"] == pytest.approx(0.75)
    assert result["gate_branch_counts"] == [1, 2, 3, 4]
    assert result["rare_branch_counts_diagnostic_only"]["5"] == {"support": 3, "f1": 0.0}
    assert result["rare_branch_counts_diagnostic_only"]["6"] == {"support": 2, "f1": 0.0}


def test_validation_collate_has_no_mask_matched_expansion() -> None:
    batch = trainer.collate([_sample("aee:0", "aee", 2), _sample("aee:1", "aee", 3)])
    assert batch["student"].shape[0] == 2
    assert batch["domain"] == ["aee", "aee"]
    assert "loss_weight" not in batch


def test_model_finite_contract_detects_nonfinite_tensor() -> None:
    model = trainer.StructuralSemanticNet()
    assert trainer.all_state_tensors_finite(model)
    with torch.no_grad():
        next(model.parameters()).view(-1)[0] = float("nan")
    assert not trainer.all_state_tensors_finite(model)
