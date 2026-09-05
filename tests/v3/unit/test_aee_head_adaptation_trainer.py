"""CPU-only helper tests for the AEE head-adaptation trainer."""

from __future__ import annotations

import importlib.util
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader


ROOT = Path(__file__).resolve().parents[3]


def _load_trainer():
    import sys

    tools = ROOT / "tools/v3"
    if str(tools) not in sys.path:
        sys.path.insert(0, str(tools))
    spec = importlib.util.spec_from_file_location(
        "train_aee_head_adaptation_v1", tools / "train_aee_head_adaptation_v1.py"
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


trainer = _load_trainer()


def _sample(index: int = 0) -> dict[str, object]:
    direction = np.zeros(720, dtype=np.float32)
    direction[0] = 1.0
    return {
        "student": np.full((2, 16, 720), 0.2, dtype=np.float32),
        "direction_target": direction,
        "count_target": np.int64(0),
        "role_target": np.int64(2),
        "frame_id": f"fixture:{index}",
        "headings_robot_deg": (0.0,),
        "domain": "aee",
    }


def test_freeze_representation_only_heads_train_and_hashes_track_head_change() -> None:
    model = trainer.StructuralSemanticNet()
    contract = trainer.freeze_representation(model)
    assert contract["trainable_parameters"]
    assert all(name.startswith(trainer.TRAINABLE_PREFIXES) for name in contract["trainable_parameters"])
    assert all(not parameter.requires_grad for name, parameter in model.named_parameters() if name.startswith(trainer.FROZEN_PREFIXES))
    frozen_before = trainer.selected_state_hashes(model, trainer.FROZEN_PREFIXES)
    heads_before = trainer.selected_state_hashes(model, trainer.TRAINABLE_PREFIXES)
    with torch.no_grad():
        parameter = next(parameter for name, parameter in model.named_parameters() if name.startswith("direction_head."))
        parameter.view(-1)[0].add_(1.0)
    frozen_after = trainer.selected_state_hashes(model, trainer.FROZEN_PREFIXES)
    heads_after = trainer.selected_state_hashes(model, trainer.TRAINABLE_PREFIXES)
    assert frozen_before == frozen_after
    assert heads_before != heads_after


def test_collate_preserves_domains_and_tensor_contract() -> None:
    batch = trainer.collate([_sample(0), {**_sample(1), "domain": "cano"}])
    assert batch["student"].shape == (2, 2, 16, 720)
    assert batch["direction_target"].dtype == torch.float32
    assert batch["count_target"].dtype == torch.long
    assert batch["role_target"].dtype == torch.long
    assert batch["domain"] == ["aee", "cano"]
    assert batch["frame_id"] == ["fixture:0", "fixture:1"]


def test_evaluate_cpu_returns_finite_direction_count_role_and_empty_metrics() -> None:
    model = trainer.StructuralSemanticNet()
    loader = DataLoader([_sample(0), _sample(1)], batch_size=2, collate_fn=trainer.collate)
    result = trainer.evaluate(
        model,
        loader,
        torch.device("cpu"),
        torch.ones(3),
    )
    assert result["frames"] == 2
    assert 0.0 <= result["direction"]["f1"] <= 1.0
    assert 0.0 <= result["direction"]["empty_rate"] <= 1.0
    assert np.isfinite(result["loss"]["total"])
    assert np.isfinite(result["count"]["macro_f1_present"])
    assert np.isfinite(result["role"]["macro_f1_present"])


def test_evaluate_b0_runs_on_small_dataset() -> None:
    class SmallDataset:
        def __len__(self):
            return 2

        def __getitem__(self, index):
            return _sample(index)

    result = trainer.evaluate_b0(SmallDataset())
    assert result["frames"] == 2
    for key in ("f1", "precision", "recall", "empty_rate"):
        assert np.isfinite(result[key])


def test_trainer_source_contains_frozen_contract_and_all_eight_gates() -> None:
    source = (ROOT / "tools/v3/train_aee_head_adaptation_v1.py").read_text(encoding="utf-8")
    assert "args.epochs != 10" in source
    assert "args.batch_size != 128" in source
    assert "math.isclose(args.learning_rate, 1e-4)" in source
    assert "math.isclose(args.weight_decay, 1e-4)" in source
    assert 'os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8"' in source
    assert 'domain_samples != {"cano": 3000, "aee": 3000}' in source
    assert '"aee_direction_vs_b0"' in source
    assert '"aee_empty_rate"' in source
    assert '"aee_count_macro_f1"' in source
    assert '"aee_role_macro_f1"' in source
    assert '"cano_direction_retention"' in source
    assert '"frozen_tensor_identity"' in source
    assert '"fixed_probe_z_role_identity"' in source
    assert '"semantic_heads_changed"' in source
    assert '"cano_c09_validation_frames_read": len(cano_validation)' in source
    assert '"c10_frames_read": 0' in source
    assert "def main" in source
    assert "main()" in source
